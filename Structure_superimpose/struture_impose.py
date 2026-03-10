#!/usr/bin/env python3
"""
Structure Aligner - The Superimposer
=====================================
Aligns a mutant .pdb structure onto a parent .pdb structure using
only Carbon-Alpha (CA) atoms and the Kabsch algorithm.

Usage:
    python structure_aligner.py <parent.pdb> <mutant.pdb>

Output:
    mutant_aligned.pdb  — the mutant structure superimposed onto the parent frame
"""

import sys
import os
import numpy as np


# ─────────────────────────────────────────────
#  PDB I/O
# ─────────────────────────────────────────────

def parse_pdb(filepath: str):
    """
    Read a PDB file and return:
      - atom_lines : list of raw ATOM/HETATM lines (strings, no newline)
      - coords     : numpy array (N, 3) of ALL atom XYZ coordinates
      - ca_indices : list of indices into atom_lines that are CA atoms
    """
    atom_lines = []
    coords = []

    with open(filepath, "r") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_lines.append(line.rstrip("\n"))
                coords.append([x, y, z])

    if not atom_lines:
        raise ValueError(f"No ATOM/HETATM records found in {filepath}")

    coords = np.array(coords, dtype=np.float64)

    # Identify CA atoms (Carbon-Alpha of protein backbone)
    ca_indices = [
        i for i, ln in enumerate(atom_lines)
        if ln[12:16].strip() == "CA" and ln[:6].strip() == "ATOM"
    ]

    if not ca_indices:
        raise ValueError(f"No CA atoms found in {filepath}. "
                         "Ensure this is a standard protein PDB file.")

    return atom_lines, coords, ca_indices


def write_pdb(filepath: str, atom_lines: list, coords: np.ndarray):
    """
    Write atom_lines back to a PDB file, replacing XYZ columns with
    the updated coordinates from `coords`.
    """
    with open(filepath, "w") as fh:
        for i, line in enumerate(atom_lines):
            x, y, z = coords[i]
            # PDB fixed-width format: cols 31-38, 39-46, 47-54 (1-indexed)
            new_line = (
                line[:30]
                + f"{x:8.3f}"
                + f"{y:8.3f}"
                + f"{z:8.3f}"
                + line[54:]
            )
            fh.write(new_line + "\n")
        fh.write("END\n")


# ─────────────────────────────────────────────
#  Kabsch Algorithm
# ─────────────────────────────────────────────

def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute the optimal rotation matrix R such that:
        RMSD( Q - (P @ R.T) ) is minimised

    Both P and Q must already be centred (mean = 0).

    Parameters
    ----------
    P : (N, 3)  — mobile points (mutant CA, centred)
    Q : (N, 3)  — target points (parent CA, centred)

    Returns
    -------
    R : (3, 3) rotation matrix  (apply as:  P_rotated = P @ R.T )
    """
    # Covariance / cross-correlation matrix
    H = P.T @ Q                          # shape (3, 3)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection (ensure proper rotation, det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    # Optimal rotation
    R = Vt.T @ D @ U.T
    return R


def compute_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Root-Mean-Square Deviation between two (N,3) arrays."""
    diff = A - B
    return float(np.sqrt((diff ** 2).sum() / len(A)))


# ─────────────────────────────────────────────
#  Main alignment routine
# ─────────────────────────────────────────────

def align_structures(parent_pdb: str, mutant_pdb: str, output_pdb: str = "mutant_aligned.pdb"):
    print(f"\n{'='*55}")
    print("  Structure Aligner  —  The Superimposer")
    print(f"{'='*55}")
    print(f"  Parent  : {parent_pdb}")
    print(f"  Mutant  : {mutant_pdb}")
    print(f"  Output  : {output_pdb}")
    print(f"{'='*55}\n")

    # ── Step 1: Parse both PDB files ──────────────────────────
    print("[1/5]  Parsing PDB files …")
    parent_lines, parent_coords, parent_ca_idx = parse_pdb(parent_pdb)
    mutant_lines, mutant_coords, mutant_ca_idx = parse_pdb(mutant_pdb)

    print(f"       Parent : {len(parent_lines):>5} atoms  |  {len(parent_ca_idx):>4} CA atoms")
    print(f"       Mutant : {len(mutant_lines):>5} atoms  |  {len(mutant_ca_idx):>4} CA atoms")

    # ── Step 2: Extract CA spines ─────────────────────────────
    print("\n[2/5]  Extracting Carbon-Alpha spines …")
    parent_ca = parent_coords[parent_ca_idx]   # shape (Np, 3)
    mutant_ca = mutant_coords[mutant_ca_idx]   # shape (Nm, 3)

    # Handle chains of different lengths — align on the shared N-terminal segment
    n_common = min(len(parent_ca), len(mutant_ca))
    if len(parent_ca) != len(mutant_ca):
        print(f"  ⚠  CA count mismatch ({len(parent_ca)} vs {len(mutant_ca)}). "
              f"Aligning on first {n_common} residues.")
    parent_ca = parent_ca[:n_common]
    mutant_ca = mutant_ca[:n_common]

    # ── Step 3: Centre both spines on their centroids ─────────
    print("\n[3/5]  Calculating rotation & translation matrix (Kabsch) …")
    parent_centroid = parent_ca.mean(axis=0)
    mutant_centroid = mutant_ca.mean(axis=0)

    parent_ca_c = parent_ca - parent_centroid
    mutant_ca_c = mutant_ca - mutant_centroid

    # ── Kabsch rotation ───────────────────────────────────────
    R = kabsch_rotation(mutant_ca_c, parent_ca_c)

    # Translation vector (after rotation, shift to parent centroid)
    # Full transform: coords_new = (coords - mutant_centroid) @ R.T + parent_centroid
    t = parent_centroid - mutant_centroid @ R.T

    # Verify RMSD on CA atoms
    mutant_ca_aligned = mutant_ca @ R.T + t
    rmsd = compute_rmsd(parent_ca, mutant_ca_aligned)
    print(f"       CA RMSD after alignment : {rmsd:.4f} Å")

    # ── Step 4: Apply the transformation to ALL mutant atoms ──
    print("\n[4/5]  Applying transformation to entire mutant structure …")
    # Formula:  new_xyz = old_xyz @ R.T + t
    mutant_coords_aligned = mutant_coords @ R.T + t

    # ── Step 5: Save the aligned structure ────────────────────
    print(f"\n[5/5]  Writing aligned structure to '{output_pdb}' …")
    write_pdb(output_pdb, mutant_lines, mutant_coords_aligned)

    print(f"\n{'='*55}")
    print(f"  ✅  Done!  CA RMSD = {rmsd:.4f} Å")
    print(f"  Output saved → {os.path.abspath(output_pdb)}")
    print(f"{'='*55}\n")

    return rmsd


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("ERROR: Please provide both parent and mutant PDB file paths.")
        print("Usage: python structure_aligner.py <parent.pdb> <mutant.pdb>")
        sys.exit(1)

    parent_file = sys.argv[1]
    mutant_file = sys.argv[2]

    for f in (parent_file, mutant_file):
        if not os.path.isfile(f):
            print(f"ERROR: File not found: {f}")
            sys.exit(1)

    align_structures(parent_file, mutant_file, output_pdb="mutant_aligned.pdb")
