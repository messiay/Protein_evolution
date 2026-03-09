"""
docking_judge.py
----------------
"Docking Judge" — measures the shortest distance between a docked ligand
and an enzyme's active-site centre.

Dependencies: NONE  (only Python built-in `math` module used)

Usage:
    python docking_judge.py

Or import and call directly:
    from docking_judge import check_docking_distance
    check_docking_distance("output.pdbqt", (17.770, 11.395, 14.893))
"""

import math


def check_docking_distance(
    pdbqt_file: str,
    target_coords: tuple,
    threshold: float = 4.0
) -> float:
    """
    Measure the shortest distance from any ligand atom (in a PDBQT file)
    to the enzyme's active-site centre, then print a pass/fail verdict.

    Parameters
    ----------
    pdbqt_file   : str   – path to the docked ligand .pdbqt file (Vina/VIdock output)
    target_coords: tuple – (X, Y, Z) float coordinates of the active-site centre
    threshold    : float – maximum acceptable distance in Å (default 4.0)

    Returns
    -------
    shortest_dist: float – the closest atom-to-centre distance found (Å)
    """

    tx, ty, tz = target_coords   # unpack active-site centre

    # ── TASK 1: Parse the PDBQT file ─────────────────────────────────────────
    # Only ATOM / HETATM lines carry coordinate data.
    # PDB/PDBQT column layout (1-indexed in the standard, 0-indexed in Python):
    #   cols 31-38  →  X  →  line[30:38]
    #   cols 39-46  →  Y  →  line[38:46]
    #   cols 47-54  →  Z  →  line[46:54]

    ligand_atoms = []   # list of (x, y, z) tuples

    with open(pdbqt_file, "r") as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ligand_atoms.append((x, y, z))
                except ValueError:
                    # Skip any malformed coordinate lines gracefully
                    continue

    if not ligand_atoms:
        raise ValueError(
            f"No ATOM/HETATM lines with valid coordinates found in: {pdbqt_file}"
        )

    # ── TASK 2: Calculate shortest Euclidean distance ─────────────────────────
    # 3D distance formula: d = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)

    shortest_dist = math.inf   # start with an impossibly large value

    for (ax, ay, az) in ligand_atoms:
        dist = math.sqrt(
            (ax - tx) ** 2 +
            (ay - ty) ** 2 +
            (az - tz) ** 2
        )
        if dist < shortest_dist:
            shortest_dist = dist   # keep only the minimum

    # ── TASK 3: Print the verdict ─────────────────────────────────────────────
    print(f"\nDocking Judge Results")
    print(f"  Target coords    : X={tx}, Y={ty}, Z={tz}")
    print(f"  Ligand atoms     : {len(ligand_atoms)} parsed from {pdbqt_file}")
    print(f"  Closest distance : {shortest_dist:.2f} Å  (threshold: {threshold} Å)")
    print()

    if shortest_dist <= threshold:
        print("[VERDICT: VALID] - Ligand successfully docked inside the active site pocket.")
    else:
        print("[VERDICT: FAILED] - Ligand drifted away from the target coordinates.")

    return shortest_dist


# ── Example usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Replace these values with your actual Vina/VIdock output and
    # the Grid Center you copied from the Pre-Flight tool.
    # -----------------------------------------------------------------------
    PDBQT_FILE    = "docked_ligand.pdbqt"   # path to your docked output file
    ACTIVE_SITE   = (17.770, 11.395, 14.893) # X, Y, Z from preflight_gui
    MAX_DIST_ANG  = 4.0                      # Å — adjust if needed

    check_docking_distance(
        pdbqt_file    = PDBQT_FILE,
        target_coords = ACTIVE_SITE,
        threshold     = MAX_DIST_ANG,
    )
