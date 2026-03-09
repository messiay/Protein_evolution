"""
preflight_prep.py
-----------------
Pre-Flight preparation tool for a protein engineering pipeline.

Given a PDB file and a list of active-site residue IDs, this script:
  1. Calculates the docking grid center (centroid of active-site atoms).
  2. Finds every other residue whose atoms lie within `radius` Å of the
     active-site atoms and prints a VMD selection string.

Dependencies:
    pip install biopython
"""

import warnings
import sys
from typing import List

import numpy as np

# Suppress the common PDBConstructionWarning that Biopython emits for
# slightly non-standard PDB files (missing occupancy, SEQATOM mismatches, etc.)
from Bio import BiopythonWarning
warnings.simplefilter("ignore", BiopythonWarning)

from Bio.PDB import PDBParser


# ---------------------------------------------------------------------------
# TASK 1 – Docking-centre calculation
# ---------------------------------------------------------------------------

def calculate_docking_center(structure, target_residues: List[int]) -> np.ndarray:
    """
    Return the geometric centre (centroid) of all atoms belonging to the
    residues listed in `target_residues`.

    Parameters
    ----------
    structure      : Bio.PDB Structure object
    target_residues: list of integer residue sequence IDs

    Returns
    -------
    np.ndarray of shape (3,) – [X, Y, Z] centroid coordinates
    """
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # residue.id is a tuple: (hetflag, seq_id, insertion_code)
                if residue.id[1] in target_residues:
                    for atom in residue.get_atoms():
                        coords.append(atom.get_vector().get_array())

    if not coords:
        raise ValueError(
            f"No atoms found for residue IDs {target_residues}. "
            "Check that the IDs exist in the PDB file."
        )

    centroid = np.mean(coords, axis=0)
    return centroid


# ---------------------------------------------------------------------------
# TASK 2 – Mutation-pocket VMD command generation
# ---------------------------------------------------------------------------

def find_pocket_residues(
    structure, target_residues: List[int], radius: float
) -> List[int]:
    """
    Identify residues (excluding the active-site residues themselves) that
    have at least one atom within `radius` Å of any active-site atom.

    Parameters
    ----------
    structure      : Bio.PDB Structure object
    target_residues: list of integer residue sequence IDs (active site)
    radius         : float, distance cut-off in Angstroms

    Returns
    -------
    Sorted list of integer residue sequence IDs that fall inside the pocket.
    """

    # ---- Collect coordinates of all active-site atoms --------------------
    active_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[1] in target_residues:
                    for atom in residue.get_atoms():
                        active_coords.append(atom.get_vector().get_array())

    active_coords = np.array(active_coords)  # shape (N_active, 3)

    if active_coords.size == 0:
        raise ValueError(
            f"No atoms found for active-site residues {target_residues}."
        )

    # ---- Screen every non-active-site residue ----------------------------
    pocket_residues = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.id[1]

                # Skip the active-site residues themselves
                if res_id in target_residues:
                    continue

                # Collect coordinates for this candidate residue
                candidate_coords = np.array(
                    [atom.get_vector().get_array() for atom in residue.get_atoms()]
                )

                if candidate_coords.size == 0:
                    continue

                # Vectorised pairwise distance calculation:
                #   diff[i, j] = candidate_coords[i] - active_coords[j]
                #   dist[i, j] = Euclidean distance between atom i and ref atom j
                diff = (
                    candidate_coords[:, np.newaxis, :]   # (C, 1, 3)
                    - active_coords[np.newaxis, :, :]    # (1, A, 3)
                )                                        # → (C, A, 3)
                dist = np.linalg.norm(diff, axis=2)      # (C, A)

                # If *any* atom pair is within the radius, include residue
                if np.any(dist <= radius):
                    pocket_residues.append(res_id)

    return sorted(pocket_residues)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preflight(pdb_file: str, target_residues: List[int], radius: float = 5.0):
    """
    Run the full pre-flight analysis.

    Parameters
    ----------
    pdb_file       : str  – path to the .pdb file
    target_residues: list – residue IDs of the enzyme's active site
    radius         : float – search radius in Å (default 5.0)
    """

    # Parse the structure (QUIET=True suppresses per-atom console noise)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # ------------------------------------------------------------------
    # TASK 1: Docking centre
    # ------------------------------------------------------------------
    center = calculate_docking_center(structure, target_residues)
    print(
        f"VIdock Grid Center: "
        f"X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}"
    )

    # ------------------------------------------------------------------
    # TASK 2: Pocket residues & VMD command
    # ------------------------------------------------------------------
    pocket = find_pocket_residues(structure, target_residues, radius)

    # Format as a space-separated string of residue IDs
    pocket_str = " ".join(str(r) for r in pocket)
    print(f"not resid {pocket_str}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Edit the values below to match your actual PDB file and active site.
    # -----------------------------------------------------------------
    PDB_FILE = "example.pdb"          # <- replace with your .pdb path
    ACTIVE_SITE_RESIDUES = [40, 42]   # <- residue IDs of the active site
    SEARCH_RADIUS = 5.0               # <- Å; default pocket search radius

    # Optionally accept a PDB path as the first command-line argument:
    if len(sys.argv) > 1:
        PDB_FILE = sys.argv[1]

    preflight(
        pdb_file=PDB_FILE,
        target_residues=ACTIVE_SITE_RESIDUES,
        radius=SEARCH_RADIUS,
    )
