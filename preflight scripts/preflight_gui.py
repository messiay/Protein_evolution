"""
preflight_gui.py
----------------
Upload a PDB file → automatically get ALL inputs needed for:
  • ProteinMPNN (HuggingFace Gradio webapp)
  • VIdock / AutoDock Vina

Active-site auto-detection strategy (in priority order):
  1. PDB SITE records  → explicitly annotated active/binding site residues
  2. Ligand proximity  → non-water HETATM residues + protein residues within 5 Å
  3. Fallback          → show a warning and let user enter residue IDs manually

ProteinMPNN outputs generated:
  • Designed chain   → all chain IDs in the structure (comma-separated)
  • Fixed positions  → "within <radius> of resid <active site IDs>"

VIdock outputs generated:
  • Grid Center      → centroid of active-site atoms (X, Y, Z)
  • Grid Size        → bounding-box of active-site atoms (X, Y, Z)
"""

import warnings
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np

from Bio import BiopythonWarning
warnings.simplefilter("ignore", BiopythonWarning)
from Bio.PDB import PDBParser


# ─── Core logic ──────────────────────────────────────────────────────────────

def _all_atoms(structure):
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue.get_atoms():
                    yield chain.id, residue, atom


def detect_active_site_from_site_records(pdb_path: str):
    """
    Parse SITE records from a PDB file.
    Returns list of (chain_id, res_seq_id) tuples, or [] if none found.
    """
    # SITE lines look like:
    # SITE     1 AC1  3 HIS A  64  HIS A  94  CYS A  97
    pattern = re.compile(
        r"([A-Z]{3})\s+([A-Z])\s+(-?\d+)",   # resname  chain  seqnum
    )
    sites = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("SITE"):
                for m in pattern.finditer(line[11:]):          # skip first 11 chars
                    chain_id = m.group(2)
                    res_id   = int(m.group(3))
                    sites.append((chain_id, res_id))
    return list(dict.fromkeys(sites))   # deduplicate preserving order


def detect_active_site_from_ligand(structure, radius=5.0, chain_id="A"):
    """
    Find non-water HETATM residues (ligands) then collect all protein residues
    in `chain_id` whose atoms are within `radius` Å of any ligand atom.
    Returns list of (chain_id, res_seq_id) or [] if no ligand found.
    """
    # Gather ligand atom coordinates (from the target chain only)
    ligand_coords = []
    water_resnames = {"HOH", "WAT", "SOL", "H2O"}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                het_flag = residue.id[0].strip()
                if het_flag.startswith("H") and residue.resname not in water_resnames:
                    for atom in residue.get_atoms():
                        ligand_coords.append(atom.get_vector().get_array())

    if not ligand_coords:
        return []

    lig_arr = np.array(ligand_coords)

    pocket = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:   # only look at the requested chain
                continue
            for residue in chain:
                if residue.id[0].strip():    # skip non-standard / HETATM
                    continue
                res_id   = residue.id[1]
                cand_arr = np.array([a.get_vector().get_array() for a in residue.get_atoms()])
                if cand_arr.size == 0:
                    continue
                diff = cand_arr[:, np.newaxis, :] - lig_arr[np.newaxis, :, :]
                if np.any(np.linalg.norm(diff, axis=2) <= radius):
                    pocket.append((chain.id, res_id))

    return sorted(set(pocket))


def calc_center_and_size(structure, chain_res_pairs, chain_id="A"):
    """
    Returns (center_xyz tuple, size_xyz tuple) for the given (chain, res_id) pairs.
    Only considers atoms that belong to chain_id.
    """
    target_set = set(chain_res_pairs)
    coords = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:   # ignore all other chains
                continue
            for residue in chain:
                if (chain.id, residue.id[1]) in target_set:
                    for atom in residue.get_atoms():
                        coords.append(atom.get_vector().get_array())

    if not coords:
        raise ValueError("No atoms found for the detected active-site residues.")

    arr    = np.array(coords)
    center = np.mean(arr, axis=0)
    size   = np.max(arr, axis=0) - np.min(arr, axis=0)
    return tuple(center.tolist()), tuple(size.tolist())


def get_chain_ids(structure):
    ids = []
    for model in structure:
        for chain in model:
            if chain.id.strip():
                ids.append(chain.id)
    return sorted(set(ids))


def analyse_pdb(pdb_path: str, radius: float = 5.0, chain_id: str = "A"):
    """
    Main analysis function.
    All geometry is computed using ONLY atoms from `chain_id`.
    Returns a dict with all fields ready for display.
    """
    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    # 1. Detect active site (filtered to chain_id)
    method = ""
    site_pairs = detect_active_site_from_site_records(pdb_path)
    # Keep only pairs that belong to the selected chain
    site_pairs = [(c, r) for c, r in site_pairs if c == chain_id]
    if site_pairs:
        method = "PDB SITE records"
    else:
        site_pairs = detect_active_site_from_ligand(structure, radius, chain_id)
        if site_pairs:
            method = f"Ligand proximity (≤ {radius} Å)"

    if not site_pairs:
        raise ValueError(
            f"Could not auto-detect active site on chain '{chain_id}'.\n\n"
            "The PDB has no SITE records and no bound ligand on this chain.\n"
            "Try a different chain ID, or check your PDB file."
        )

    # 2. Centre + size  (only chain_id atoms used)
    center, size = calc_center_and_size(structure, site_pairs, chain_id)

    # 3. Chain info
    all_chains     = get_chain_ids(structure)
    designed_chain = ",".join(all_chains)

    # 4. ProteinMPNN Fixed positions
    #    Format: not (chain A and resid 40 41 42)
    res_ids_only = sorted(set(r for _, r in site_pairs))
    resid_str    = " ".join(str(r) for r in res_ids_only)
    mpnn_fixed   = f"not (chain {chain_id} and resid {resid_str})"

    return {
        "method":          method,
        "active_res_ids":  res_ids_only,
        "active_pairs":    site_pairs,
        "center":          center,
        "size":            size,
        "designed_chain":  designed_chain,
        "mpnn_fixed":      mpnn_fixed,
    }


# ─── GUI ─────────────────────────────────────────────────────────────────────

BG   = "#1e1e2e"
BG2  = "#2a2a3e"
BG3  = "#181825"
FG   = "#cdd6f4"
ACC  = "#89b4fa"
GRN  = "#a6e3a1"
MUTED= "#6c7086"
FONT = ("Segoe UI", 9)
MONO = ("Consolas", 9)


class PreflightApp:
    DEFAULT_RADIUS   = "5.0"
    DEFAULT_CHAIN_ID = "A"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pre-Flight PDB Tool")
        self.root.geometry("660x700")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self.pdb_path = None
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build(self):
        # Title
        tk.Label(self.root, text="⚗  Pre-Flight PDB Analysis",
                 bg=BG, fg=ACC, font=("Segoe UI", 13, "bold")).pack(pady=(14, 2))
        tk.Label(self.root,
                 text="Upload a PDB → auto-detect active site → copy outputs into VIdock & ProteinMPNN",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(pady=(0, 10))

        # ── File picker ───────────────────────────────────────────────────
        frm_file = self._lf("Step 1 · Upload PDB File")
        frm_file.pack(fill="x", padx=14, pady=4)

        tk.Button(frm_file, text="Browse PDB…",
                  bg=ACC, fg=BG2, activebackground="#74c7ec",
                  relief="flat", cursor="hand2",
                  font=("Segoe UI", 9, "bold"),
                  command=self.browse).grid(row=0, column=0, padx=10, pady=8)

        self.lbl_file = tk.Label(frm_file, text="No file selected",
                                 bg=BG2, fg=MUTED, font=FONT)
        self.lbl_file.grid(row=0, column=1, sticky="w", padx=4)

        # ── Options ───────────────────────────────────────────────────────
        frm_opt = self._lf("Step 2 · Options")
        frm_opt.pack(fill="x", padx=14, pady=4)

        # Chain ID
        tk.Label(frm_opt, text="Chain ID:", bg=BG2, fg=FG,
                 font=FONT).grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.ent_chain_id = tk.Entry(frm_opt, width=5, bg=BG3, fg=FG,
                                     insertbackground=FG, relief="flat", font=MONO)
        self.ent_chain_id.insert(0, self.DEFAULT_CHAIN_ID)
        self.ent_chain_id.grid(row=0, column=1, sticky="w", padx=6)
        tk.Label(frm_opt, text="Only atoms from this chain are used for all calculations",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 7)).grid(
            row=0, column=2, sticky="w", padx=4)

        # Radius
        tk.Label(frm_opt, text="Search radius (Å):", bg=BG2, fg=FG,
                 font=FONT).grid(row=1, column=0, sticky="w", padx=10, pady=6)
        self.ent_radius = tk.Entry(frm_opt, width=8, bg=BG3, fg=FG,
                                   insertbackground=FG, relief="flat", font=MONO)
        self.ent_radius.insert(0, self.DEFAULT_RADIUS)
        self.ent_radius.grid(row=1, column=1, sticky="w", padx=6)
        tk.Label(frm_opt,
                 text="Used for ligand-proximity detection",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 7)).grid(
            row=1, column=2, sticky="w", padx=4)

        # ── Run button ────────────────────────────────────────────────────
        self.btn_run = tk.Button(
            self.root, text="▶  Analyse PDB",
            bg=GRN, fg=BG2, activebackground="#94e2d5",
            relief="flat", font=("Segoe UI", 11, "bold"),
            cursor="hand2", command=self.run)
        self.btn_run.pack(pady=8)

# ── Results ─────────────────────────────────────────────────────
        frm_res = self._lf("Step 3 · Results  –  Click Copy to paste into your tool")
        frm_res.pack(fill="x", padx=14, pady=4)

        def ro_box(parent, width=11):
            e = tk.Entry(parent, width=width, bg=BG3, fg=GRN,
                         relief="flat", font=MONO, state="readonly",
                         readonlybackground=BG3)
            return e

        def wide_ro(parent, width=44):
            e = tk.Entry(parent, width=width, bg=BG3, fg=GRN,
                         relief="flat", font=MONO, state="readonly",
                         readonlybackground=BG3)
            return e

        def copy_btn(parent, fn):
            return tk.Button(parent, text="Copy",
                             bg="#45475a", fg=FG,
                             activebackground=ACC, activeforeground=BG,
                             relief="flat", cursor="hand2",
                             font=("Segoe UI", 8),
                             command=lambda: self._copy(fn()))

        row = 0

        # Detection method
        tk.Label(frm_res, text="Detection method:", bg=BG2, fg=FG,
                 font=FONT, anchor="w").grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self.lbl_method = tk.Label(frm_res, text="—", bg=BG2, fg=MUTED,
                                   font=MONO, width=44, anchor="w")
        self.lbl_method.grid(row=row, column=1, columnspan=5, sticky="w", padx=4)
        row += 1

        # Active-site residue IDs
        tk.Label(frm_res, text="Active-site residue IDs:", bg=BG2, fg=FG,
                 font=FONT, anchor="w").grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self.ent_resids = wide_ro(frm_res, width=40)
        self.ent_resids.grid(row=row, column=1, columnspan=4, sticky="w", padx=4)
        copy_btn(frm_res, self.ent_resids.get).grid(row=row, column=5, padx=6)
        row += 1

        # VIdock Grid Center — three separate boxes
        tk.Label(frm_res, text="VIdock Grid Center:", bg=BG2, fg=FG,
                 font=FONT, anchor="w").grid(row=row, column=0, sticky="w", padx=10, pady=4)
        tk.Label(frm_res, text="X", bg=BG2, fg=MUTED, font=FONT).grid(row=row, column=1, sticky="e")
        self.ent_cx = ro_box(frm_res); self.ent_cx.grid(row=row, column=2, padx=(2, 6))
        tk.Label(frm_res, text="Y", bg=BG2, fg=MUTED, font=FONT).grid(row=row, column=3, sticky="e")
        self.ent_cy = ro_box(frm_res); self.ent_cy.grid(row=row, column=4, padx=(2, 6))
        tk.Label(frm_res, text="Z", bg=BG2, fg=MUTED, font=FONT).grid(row=row, column=5, sticky="e")
        self.ent_cz = ro_box(frm_res); self.ent_cz.grid(row=row, column=6, padx=(2, 6))
        copy_btn(frm_res,
                 lambda: f"{self.ent_cx.get()}  {self.ent_cy.get()}  {self.ent_cz.get()}"
                 ).grid(row=row, column=7, padx=6)
        row += 1

        # VIdock Grid Size — three separate boxes
        tk.Label(frm_res, text="VIdock Grid Size:", bg=BG2, fg=FG,
                 font=FONT, anchor="w").grid(row=row, column=0, sticky="w", padx=10, pady=4)
        tk.Label(frm_res, text="X", bg=BG2, fg=MUTED, font=FONT).grid(row=row, column=1, sticky="e")
        self.ent_sx = ro_box(frm_res); self.ent_sx.grid(row=row, column=2, padx=(2, 6))
        tk.Label(frm_res, text="Y", bg=BG2, fg=MUTED, font=FONT).grid(row=row, column=3, sticky="e")
        self.ent_sy = ro_box(frm_res); self.ent_sy.grid(row=row, column=4, padx=(2, 6))
        tk.Label(frm_res, text="Z", bg=BG2, fg=MUTED, font=FONT).grid(row=row, column=5, sticky="e")
        self.ent_sz = ro_box(frm_res); self.ent_sz.grid(row=row, column=6, padx=(2, 6))
        copy_btn(frm_res,
                 lambda: f"{self.ent_sx.get()}  {self.ent_sy.get()}  {self.ent_sz.get()}"
                 ).grid(row=row, column=7, padx=6)
        row += 1

        # ProteinMPNN Designed chain
        tk.Label(frm_res, text="ProteinMPNN Designed chain:", bg=BG2, fg=FG,
                 font=FONT, anchor="w").grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self.ent_chain = wide_ro(frm_res, width=40)
        self.ent_chain.grid(row=row, column=1, columnspan=4, sticky="w", padx=4)
        copy_btn(frm_res, self.ent_chain.get).grid(row=row, column=5, padx=6)
        row += 1

        # ProteinMPNN Fixed positions
        tk.Label(frm_res, text='ProteinMPNN Fixed positions:', bg=BG2, fg=FG,
                 font=FONT, anchor="w").grid(row=row, column=0, sticky="w", padx=10, pady=4)
        self.ent_mpnn = wide_ro(frm_res, width=40)
        self.ent_mpnn.grid(row=row, column=1, columnspan=4, sticky="w", padx=4)
        copy_btn(frm_res, self.ent_mpnn.get).grid(row=row, column=5, padx=6)

        # Status bar
        self.lbl_status = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                                   font=("Segoe UI", 8))
        self.lbl_status.pack(pady=(4, 10))

    def _lf(self, title):
        return tk.LabelFrame(self.root, text=title, bg=BG2, fg=ACC,
                             font=("Segoe UI", 9, "bold"),
                             bd=1, relief="groove")

    # ── Actions ───────────────────────────────────────────────────────────

    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")])
        if path:
            self.pdb_path = path
            short = path if len(path) <= 55 else "…" + path[-53:]
            self.lbl_file.config(text=short, fg=FG)

    def run(self):
        if not self.pdb_path:
            messagebox.showerror("No file", "Please select a PDB file first.")
            return

        try:
            radius = float(self.ent_radius.get())
            if radius <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Bad input", "Radius must be a positive number.")
            return

        chain_id = self.ent_chain_id.get().strip().upper()
        if not chain_id:
            messagebox.showerror("Bad input", "Chain ID cannot be empty.")
            return

        self.btn_run.config(text="Analysing…", state="disabled")
        self.lbl_status.config(text="")
        self.root.update()

        try:
            data = analyse_pdb(self.pdb_path, radius, chain_id)

            # Detection method label
            self.lbl_method.config(
                text=f"✔  {data['method']}", fg=GRN)

            # Active-site residue IDs
            res_str = ", ".join(str(r) for r in data["active_res_ids"])
            self._set(self.ent_resids,  res_str)

            # VIdock fields — fill 6 individual boxes
            c = data["center"]
            s = data["size"]
            self._set(self.ent_cx, f"{c[0]:.3f}")
            self._set(self.ent_cy, f"{c[1]:.3f}")
            self._set(self.ent_cz, f"{c[2]:.3f}")
            self._set(self.ent_sx, f"{s[0]:.3f}")
            self._set(self.ent_sy, f"{s[1]:.3f}")
            self._set(self.ent_sz, f"{s[2]:.3f}")

            # ProteinMPNN fields
            self._set(self.ent_chain, data["designed_chain"])
            self._set(self.ent_mpnn,  data["mpnn_fixed"])

            n = len(data["active_res_ids"])
            self.lbl_status.config(
                text=f"✔  Detected {n} active-site residue(s) via {data['method']}",
                fg=GRN)

        except Exception as exc:
            messagebox.showerror("Analysis failed", str(exc))
            self.lbl_status.config(text="⚠  Analysis failed — see error dialog", fg="#f38ba8")
        finally:
            self.btn_run.config(text="▶  Analyse PDB", state="normal")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _set(widget: tk.Entry, value: str):
        widget.config(state="normal")
        widget.delete(0, tk.END)
        widget.insert(0, value)
        widget.config(state="readonly")

    def _copy(self, text: str):
        if text.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
            self.lbl_status.config(text="📋  Copied to clipboard!", fg=ACC)


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    PreflightApp(root)
    root.mainloop()
