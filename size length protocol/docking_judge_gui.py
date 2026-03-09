"""
docking_judge_gui.py
--------------------
GUI wrapper for the Docking Judge tool.

Upload a .pdbqt docked ligand file, enter the active-site centre
(copy from Pre-Flight GUI) and get an instant VALID / FAILED verdict.

No external libraries — only built-in `math` and `tkinter`.
"""

import math
import tkinter as tk
from tkinter import filedialog, messagebox


# ─── Core logic (no Biopython, only math) ────────────────────────────────────

def check_docking_distance(pdbqt_file: str, target_coords: tuple, threshold: float = 4.0):
    """
    Parse a PDBQT file, find the shortest atom→centre distance, return results dict.
    """
    tx, ty, tz = target_coords
    ligand_atoms = []

    # TASK 1: Parse PDBQT — strict PDB column slicing
    with open(pdbqt_file, "r") as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ligand_atoms.append((x, y, z))
                except ValueError:
                    continue

    if not ligand_atoms:
        raise ValueError("No ATOM/HETATM lines with valid coordinates found in the file.")

    # TASK 2: Shortest Euclidean distance
    shortest = math.inf
    for (ax, ay, az) in ligand_atoms:
        d = math.sqrt((ax - tx)**2 + (ay - ty)**2 + (az - tz)**2)
        if d < shortest:
            shortest = d

    # TASK 3: Verdict
    verdict = "VALID" if shortest <= threshold else "FAILED"
    return {
        "n_atoms":   len(ligand_atoms),
        "distance":  shortest,
        "threshold": threshold,
        "verdict":   verdict,
    }


# ─── GUI ─────────────────────────────────────────────────────────────────────

BG    = "#1e1e2e"
BG2   = "#2a2a3e"
BG3   = "#181825"
FG    = "#cdd6f4"
ACC   = "#89b4fa"
GRN   = "#a6e3a1"
RED   = "#f38ba8"
MUTED = "#6c7086"
FONT  = ("Segoe UI", 9)
MONO  = ("Consolas", 9)


def _ro_entry(parent, width=14):
    return tk.Entry(parent, width=width, bg=BG3, fg=GRN,
                    relief="flat", font=MONO,
                    state="readonly", readonlybackground=BG3)

def _entry(parent, width=12, default=""):
    e = tk.Entry(parent, width=width, bg=BG3, fg=FG,
                 insertbackground=FG, relief="flat", font=MONO)
    e.insert(0, default)
    return e

def _lf(parent, title):
    return tk.LabelFrame(parent, text=title, bg=BG2, fg=ACC,
                         font=("Segoe UI", 9, "bold"), bd=1, relief="groove")

def _lbl(parent, text, fg=None, **kw):
    return tk.Label(parent, text=text, bg=BG2, fg=fg or FG, font=FONT, **kw)


class DockingJudgeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Docking Judge")
        self.root.geometry("620x520")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self.pdbqt_path = None
        self._build()

    def _build(self):
        # ── Title ─────────────────────────────────────────────────────────
        tk.Label(self.root, text="⚖  Docking Judge",
                 bg=BG, fg=ACC, font=("Segoe UI", 13, "bold")).pack(pady=(14, 2))
        tk.Label(self.root,
                 text="Upload your Vina/VIdock .pdbqt output → get an instant docking verdict",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(pady=(0, 10))

        # ── Step 1: File ──────────────────────────────────────────────────
        f1 = _lf(self.root, "Step 1 · Docked Ligand File (.pdbqt)")
        f1.pack(fill="x", padx=14, pady=4)

        tk.Button(f1, text="Browse .pdbqt…",
                  bg=ACC, fg=BG2, activebackground="#74c7ec",
                  relief="flat", cursor="hand2",
                  font=("Segoe UI", 9, "bold"),
                  command=self.browse).grid(row=0, column=0, padx=10, pady=8)

        self.lbl_file = tk.Label(f1, text="No file selected",
                                 bg=BG2, fg=MUTED, font=FONT)
        self.lbl_file.grid(row=0, column=1, sticky="w", padx=4)

        # ── Step 2: Parameters ────────────────────────────────────────────
        f2 = _lf(self.root, "Step 2 · Active-Site Centre  (paste from Pre-Flight GUI)")
        f2.pack(fill="x", padx=14, pady=4)

        # X Y Z inputs on one row
        _lbl(f2, "X:").grid(row=0, column=0, sticky="e", padx=(10, 2), pady=8)
        self.ent_x = _entry(f2, width=10, default="17.770")
        self.ent_x.grid(row=0, column=1, padx=(0, 8))

        _lbl(f2, "Y:").grid(row=0, column=2, sticky="e", padx=(0, 2))
        self.ent_y = _entry(f2, width=10, default="11.395")
        self.ent_y.grid(row=0, column=3, padx=(0, 8))

        _lbl(f2, "Z:").grid(row=0, column=4, sticky="e", padx=(0, 2))
        self.ent_z = _entry(f2, width=10, default="14.893")
        self.ent_z.grid(row=0, column=5, padx=(0, 8))

        # Threshold
        _lbl(f2, "Threshold (Å):").grid(row=1, column=0, columnspan=2,
                                         sticky="w", padx=10, pady=(0, 8))
        self.ent_thresh = _entry(f2, width=8, default="4.0")
        self.ent_thresh.grid(row=1, column=2, sticky="w", pady=(0, 8))
        tk.Label(f2, text="Max acceptable distance from active site",
                 bg=BG2, fg=MUTED, font=("Segoe UI", 7)).grid(
            row=1, column=3, columnspan=3, sticky="w", padx=4)

        # ── Run button ─────────────────────────────────────────────────────
        self.btn_run = tk.Button(
            self.root, text="▶  Judge Docking",
            bg=GRN, fg=BG2, activebackground="#94e2d5",
            relief="flat", font=("Segoe UI", 11, "bold"),
            cursor="hand2", command=self.run)
        self.btn_run.pack(pady=8)

        # ── Step 3: Results ────────────────────────────────────────────────
        f3 = _lf(self.root, "Step 3 · Results")
        f3.pack(fill="x", padx=14, pady=4)

        def ro(w=14): return _ro_entry(f3, width=w)

        _lbl(f3, "Ligand atoms parsed:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.ent_atoms = ro(8)
        self.ent_atoms.grid(row=0, column=1, sticky="w", padx=4)

        _lbl(f3, "Closest distance (Å):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.ent_dist = ro(10)
        self.ent_dist.grid(row=1, column=1, sticky="w", padx=4)

        _lbl(f3, "Threshold (Å):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.ent_thresh_out = ro(10)
        self.ent_thresh_out.grid(row=2, column=1, sticky="w", padx=4)

        # Verdict banner
        self.lbl_verdict = tk.Label(
            f3, text="—", bg=BG2, fg=MUTED,
            font=("Segoe UI", 12, "bold"), anchor="center", width=45)
        self.lbl_verdict.grid(row=3, column=0, columnspan=3,
                              padx=10, pady=(8, 10), sticky="ew")

        # Status bar
        self.lbl_status = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                                   font=("Segoe UI", 8))
        self.lbl_status.pack(pady=(4, 10))

    # ── Actions ───────────────────────────────────────────────────────────

    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("PDBQT files", "*.pdbqt"), ("All files", "*.*")])
        if path:
            self.pdbqt_path = path
            short = path if len(path) <= 55 else "…" + path[-53:]
            self.lbl_file.config(text=short, fg=FG)

    def run(self):
        if not self.pdbqt_path:
            messagebox.showerror("No file", "Please select a .pdbqt file first.")
            return

        # Validate numeric inputs
        try:
            tx = float(self.ent_x.get())
            ty = float(self.ent_y.get())
            tz = float(self.ent_z.get())
            threshold = float(self.ent_thresh.get())
            if threshold <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Bad input",
                                 "X, Y, Z and Threshold must all be valid numbers.")
            return

        self.btn_run.config(text="Judging…", state="disabled")
        self.lbl_status.config(text="")
        self.root.update()

        try:
            result = check_docking_distance(
                pdbqt_file    = self.pdbqt_path,
                target_coords = (tx, ty, tz),
                threshold     = threshold,
            )

            # Fill result fields
            self._set(self.ent_atoms,      str(result["n_atoms"]))
            self._set(self.ent_dist,       f"{result['distance']:.2f} Å")
            self._set(self.ent_thresh_out, f"{result['threshold']:.2f} Å")

            # Verdict banner
            if result["verdict"] == "VALID":
                self.lbl_verdict.config(
                    text="✔  [VERDICT: VALID]  Ligand successfully docked inside the active site pocket.",
                    fg=GRN)
                self.lbl_status.config(
                    text=f"Closest atom is {result['distance']:.2f} Å — within the {threshold} Å threshold.",
                    fg=GRN)
            else:
                self.lbl_verdict.config(
                    text="✘  [VERDICT: FAILED]  Ligand drifted away from the target coordinates.",
                    fg=RED)
                self.lbl_status.config(
                    text=f"Closest atom is {result['distance']:.2f} Å — exceeds the {threshold} Å threshold.",
                    fg=RED)

        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.lbl_status.config(text="⚠  Error — see dialog.", fg=RED)
        finally:
            self.btn_run.config(text="▶  Judge Docking", state="normal")

    @staticmethod
    def _set(widget: tk.Entry, value: str):
        widget.config(state="normal")
        widget.delete(0, tk.END)
        widget.insert(0, value)
        widget.config(state="readonly")


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    DockingJudgeApp(root)
    root.mainloop()
