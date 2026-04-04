#!/usr/bin/env python3
"""
Kim-Hummer coarse-grained protein forcefield implementation.

Uses parameters from: https://github.com/bio-phys/complexespp
File: pycomplexes/pycomplexes/forcefields/KimHummer

References:
- Kim & Hummer (2008) J. Mol. Biol. 375, 1416-1433
  https://doi.org/10.1016/j.jmb.2007.11.063
- Miyazawa & Jernigan (1996) J. Mol. Biol. 256, 623-644
  https://doi.org/10.1006/jmbi.1996.0114
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Amino acid order (standard)
AA_ORDER = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# Van der Waals diameters (Å) from Kim & Hummer Table 5
# Note: These are DIAMETERS, not radii
VDW_DIAMETER = {
    'ALA': 5.0, 'ARG': 6.6, 'ASN': 5.7, 'ASP': 5.6, 'CYS': 5.5,
    'GLN': 6.0, 'GLU': 5.9, 'GLY': 4.5, 'HIS': 6.1, 'ILE': 6.2,
    'LEU': 6.2, 'LYS': 6.4, 'MET': 6.2, 'PHE': 6.4, 'PRO': 5.6,
    'SER': 5.2, 'THR': 5.6, 'TRP': 6.8, 'TYR': 6.5, 'VAL': 5.9
}

# Pre-computed epsilon_ij values (kT) from KimHummer forcefield file
# These are already scaled: ε_ij = λ(e_ij - e_0) with λ=0.159, e_0=-2.27 kT
# Only upper triangle stored in file; matrix is symmetric
ENERGIES_RAW = {
    'ALA': {'ALA': -0.07155, 'ARG': 0.06996, 'ASN': 0.06837, 'ASP': 0.09063,
            'GLN': 0.06042, 'GLU': 0.12084, 'GLY': -0.00636, 'HIS': -0.02226,
            'LYS': 0.15264, 'PRO': 0.03816, 'SER': 0.04134, 'THR': -0.00795},
    'ARG': {'ARG': 0.11448, 'LYS': 0.26712, 'PRO': 0.09063},
    'ASN': {'ARG': 0.10017, 'ASN': 0.09381, 'ASP': 0.09381, 'GLN': 0.08904,
            'GLU': 0.12084, 'HIS': 0.03021, 'LYS': 0.16854, 'PRO': 0.11766},
    'ASP': {'ARG': -0.00318, 'ASP': 0.16854, 'GLU': 0.19875, 'HIS': -0.00795,
            'LYS': 0.09381, 'PRO': 0.14946},
    'CYS': {'ALA': -0.2067, 'ARG': -0.0477, 'ASN': -0.05088, 'ASP': -0.02226,
            'CYS': -0.50403, 'GLN': -0.09222, 'GLU': 0.0, 'GLY': -0.14151,
            'HIS': -0.21147, 'ILE': -0.51357, 'LEU': -0.56604, 'LYS': 0.05088,
            'MET': -0.43248, 'PHE': -0.56127, 'PRO': -0.1272, 'SER': -0.09381,
            'THR': -0.13356, 'TRP': -0.42612, 'TYR': -0.30051, 'VAL': -0.42771},
    'GLN': {'ARG': 0.07473, 'ASP': 0.12879, 'GLN': 0.11607, 'GLU': 0.13515,
            'HIS': 0.04611, 'LYS': 0.15582, 'PRO': 0.08586},
    'GLU': {'ARG': 0.0, 'GLU': 0.21624, 'HIS': 0.01908, 'LYS': 0.07473, 'PRO': 0.16059},
    'GLY': {'ARG': 0.08745, 'ASN': 0.08427, 'ASP': 0.10812, 'GLN': 0.09699,
            'GLU': 0.16695, 'GLY': 0.00477, 'HIS': 0.01908, 'LYS': 0.17808,
            'PRO': 0.0636, 'SER': 0.07155, 'THR': 0.03021},
    'HIS': {'ARG': 0.01749, 'HIS': -0.12402, 'LYS': 0.14628, 'PRO': 0.00318},
    'ILE': {'ALA': -0.36729, 'ARG': -0.21624, 'ASN': -0.15423, 'ASP': -0.1431,
            'GLN': -0.2226, 'GLU': -0.159, 'GLY': -0.24009, 'HIS': -0.29733,
            'ILE': -0.67893, 'LEU': -0.75843, 'LYS': -0.11766, 'PRO': -0.23691,
            'SER': -0.19875, 'THR': -0.27984, 'TRP': -0.55809, 'TYR': -0.47382,
            'VAL': -0.60102},
    'LEU': {'ALA': -0.41976, 'ARG': -0.27984, 'ASN': -0.23373, 'ASP': -0.17967,
            'GLN': -0.28143, 'GLU': -0.20988, 'GLY': -0.30051, 'HIS': -0.36093,
            'LEU': -0.8109, 'LYS': -0.1749, 'PRO': -0.30687, 'SER': -0.26235,
            'THR': -0.32913, 'TRP': -0.61533, 'TYR': -0.5406, 'VAL': -0.66939},
    'LYS': {'LYS': 0.34185, 'PRO': 0.2067},
    'MET': {'ALA': -0.26553, 'ARG': -0.13515, 'ASN': -0.10812, 'ASP': -0.0477,
            'GLN': -0.16377, 'GLU': -0.09858, 'GLY': -0.17808, 'HIS': -0.27189,
            'ILE': -0.59625, 'LEU': -0.65826, 'LYS': -0.03339, 'MET': -0.50721,
            'PHE': -0.68211, 'PRO': -0.18762, 'SER': -0.12084, 'THR': -0.19716,
            'TRP': -0.52152, 'TYR': -0.41976, 'VAL': -0.48495},
    'PHE': {'ALA': -0.40386, 'ARG': -0.27189, 'ASN': -0.23532, 'ASP': -0.19239,
            'GLN': -0.29097, 'GLU': -0.20511, 'GLY': -0.29574, 'HIS': -0.3975,
            'ILE': -0.72663, 'LEU': -0.79659, 'LYS': -0.17331, 'PHE': -0.79341,
            'PRO': -0.31482, 'SER': -0.27825, 'THR': -0.31959, 'TRP': -0.61851,
            'TYR': -0.53901, 'VAL': -0.63918},
    'PRO': {'PRO': 0.08268},
    'SER': {'ARG': 0.10335, 'ASN': 0.10971, 'ASP': 0.10176, 'GLN': 0.12402,
            'GLU': 0.12561, 'HIS': 0.02544, 'LYS': 0.19398, 'PRO': 0.1113, 'SER': 0.0954},
    'THR': {'ARG': 0.05883, 'ASN': 0.06201, 'ASP': 0.07473, 'GLN': 0.05883,
            'GLU': 0.08427, 'HIS': -0.02385, 'LYS': 0.15264, 'PRO': 0.05883,
            'SER': 0.04929, 'THR': 0.02385},
    'TRP': {'ALA': -0.24645, 'ARG': -0.18126, 'ASN': -0.1272, 'ASP': -0.09063,
            'GLN': -0.13356, 'GLU': -0.11448, 'GLY': -0.18285, 'HIS': -0.27189,
            'LYS': -0.06678, 'PRO': -0.23214, 'SER': -0.11448, 'THR': -0.15105,
            'TRP': -0.44361, 'TYR': -0.38001},
    'TYR': {'ALA': -0.17331, 'ARG': -0.14151, 'ASN': -0.07791, 'ASP': -0.07791,
            'GLN': -0.1113, 'GLU': -0.08268, 'GLY': -0.11766, 'HIS': -0.19875,
            'LYS': -0.05247, 'PRO': -0.14628, 'SER': -0.08109, 'THR': -0.11766,
            'TYR': -0.3021},
    'VAL': {'ALA': -0.28143, 'ARG': -0.1272, 'ASN': -0.08904, 'ASP': -0.03339,
            'GLN': -0.1272, 'GLU': -0.0636, 'GLY': -0.17649, 'HIS': -0.20829,
            'LYS': -0.03498, 'PRO': -0.16695, 'SER': -0.12402, 'THR': -0.18921,
            'TRP': -0.46269, 'TYR': -0.37365, 'VAL': -0.51675},
}

# Pre-computed sigma_ij (diameter) values (Å) from KimHummer forcefield file
# σ_ij = (σ_i + σ_j) / 2
DIAMETERS_RAW = {
    'ALA': {'ALA': 5.0, 'ARG': 5.85, 'ASN': 5.35, 'ASP': 5.3, 'CYS': 5.25,
            'GLN': 5.5, 'GLU': 5.45, 'GLY': 4.75, 'HIS': 5.55, 'ILE': 5.6,
            'LEU': 5.6, 'LYS': 5.7, 'MET': 5.6, 'PHE': 5.7, 'PRO': 5.3,
            'SER': 5.1, 'THR': 5.3, 'TRP': 5.9, 'TYR': 5.75, 'VAL': 5.45},
    'ARG': {'ARG': 6.7, 'ASN': 6.2, 'ASP': 6.15, 'CYS': 6.1, 'GLN': 6.35,
            'GLU': 6.3, 'GLY': 5.6, 'HIS': 6.4, 'ILE': 6.45, 'LEU': 6.45,
            'LYS': 6.55, 'MET': 6.45, 'PHE': 6.55, 'PRO': 6.15, 'SER': 5.95,
            'THR': 6.15, 'TRP': 6.75, 'TYR': 6.6, 'VAL': 6.3},
    'ASN': {'ASN': 5.7, 'ASP': 5.65, 'CYS': 5.6, 'GLN': 5.85, 'GLU': 5.8,
            'GLY': 5.1, 'HIS': 5.9, 'ILE': 5.95, 'LEU': 5.95, 'LYS': 6.05,
            'MET': 5.95, 'PHE': 6.05, 'PRO': 5.65, 'SER': 5.45, 'THR': 5.65,
            'TRP': 6.25, 'TYR': 6.1, 'VAL': 5.8},
    'ASP': {'ASP': 5.6, 'CYS': 5.55, 'GLN': 5.8, 'GLU': 5.75, 'GLY': 5.05,
            'HIS': 5.85, 'ILE': 5.9, 'LEU': 5.9, 'LYS': 6.0, 'MET': 5.9,
            'PHE': 6.0, 'PRO': 5.6, 'SER': 5.4, 'THR': 5.6, 'TRP': 6.2,
            'TYR': 6.05, 'VAL': 5.75},
    'CYS': {'CYS': 5.5, 'GLN': 5.75, 'GLU': 5.7, 'GLY': 5.0, 'HIS': 5.8,
            'ILE': 5.85, 'LEU': 5.85, 'LYS': 5.95, 'MET': 5.85, 'PHE': 5.95,
            'PRO': 5.55, 'SER': 5.35, 'THR': 5.55, 'TRP': 6.15, 'TYR': 6.0, 'VAL': 5.7},
    'GLN': {'GLN': 6.0, 'GLU': 5.95, 'GLY': 5.25, 'HIS': 6.05, 'ILE': 6.1,
            'LEU': 6.1, 'LYS': 6.2, 'MET': 6.1, 'PHE': 6.2, 'PRO': 5.8,
            'SER': 5.6, 'THR': 5.8, 'TRP': 6.4, 'TYR': 6.25, 'VAL': 5.95},
    'GLU': {'GLU': 5.9, 'GLY': 5.2, 'HIS': 6.0, 'ILE': 6.05, 'LEU': 6.05,
            'LYS': 6.15, 'MET': 6.05, 'PHE': 6.15, 'PRO': 5.75, 'SER': 5.55,
            'THR': 5.75, 'TRP': 6.35, 'TYR': 6.2, 'VAL': 5.9},
    'GLY': {'GLY': 4.5, 'HIS': 5.3, 'ILE': 5.35, 'LEU': 5.35, 'LYS': 5.45,
            'MET': 5.35, 'PHE': 5.45, 'PRO': 5.05, 'SER': 4.85, 'THR': 5.05,
            'TRP': 5.65, 'TYR': 5.5, 'VAL': 5.2},
    'HIS': {'HIS': 6.1, 'ILE': 6.15, 'LEU': 6.15, 'LYS': 6.25, 'MET': 6.15,
            'PHE': 6.25, 'PRO': 5.85, 'SER': 5.65, 'THR': 5.85, 'TRP': 6.45,
            'TYR': 6.3, 'VAL': 6.0},
    'ILE': {'ILE': 6.2, 'LEU': 6.2, 'LYS': 6.3, 'MET': 6.2, 'PHE': 6.3,
            'PRO': 5.9, 'SER': 5.7, 'THR': 5.9, 'TRP': 6.5, 'TYR': 6.35, 'VAL': 6.05},
    'LEU': {'LEU': 6.2, 'LYS': 6.3, 'MET': 6.2, 'PHE': 6.3, 'PRO': 5.9,
            'SER': 5.7, 'THR': 5.9, 'TRP': 6.5, 'TYR': 6.35, 'VAL': 6.05},
    'LYS': {'LYS': 6.4, 'MET': 6.3, 'PHE': 6.4, 'PRO': 6.0, 'SER': 5.8,
            'THR': 6.0, 'TRP': 6.6, 'TYR': 6.45, 'VAL': 6.15},
    'MET': {'MET': 6.2, 'PHE': 6.3, 'PRO': 5.9, 'SER': 5.7, 'THR': 5.9,
            'TRP': 6.5, 'TYR': 6.35, 'VAL': 6.05},
    'PHE': {'PHE': 6.4, 'PRO': 6.0, 'SER': 5.8, 'THR': 6.0, 'TRP': 6.6,
            'TYR': 6.45, 'VAL': 6.15},
    'PRO': {'PRO': 5.6, 'SER': 5.4, 'THR': 5.6, 'TRP': 6.2, 'TYR': 6.05, 'VAL': 5.75},
    'SER': {'SER': 5.2, 'THR': 5.4, 'TRP': 6.0, 'TYR': 5.85, 'VAL': 5.55},
    'THR': {'THR': 5.6, 'TRP': 6.2, 'TYR': 6.05, 'VAL': 5.75},
    'TRP': {'TRP': 6.8, 'TYR': 6.65, 'VAL': 6.35},
    'TYR': {'TYR': 6.5, 'VAL': 6.2},
    'VAL': {'VAL': 5.9},
}


def build_symmetric_matrix(raw_dict: dict, aa_order: list) -> pd.DataFrame:
    """Build symmetric DataFrame from upper-triangle dictionary."""
    n = len(aa_order)
    matrix = np.zeros((n, n))
    
    for i, aa1 in enumerate(aa_order):
        for j, aa2 in enumerate(aa_order):
            # Try both orderings
            if aa1 in raw_dict and aa2 in raw_dict[aa1]:
                matrix[i, j] = raw_dict[aa1][aa2]
                matrix[j, i] = raw_dict[aa1][aa2]
            elif aa2 in raw_dict and aa1 in raw_dict[aa2]:
                matrix[i, j] = raw_dict[aa2][aa1]
                matrix[j, i] = raw_dict[aa2][aa1]
    
    return pd.DataFrame(matrix, index=aa_order, columns=aa_order)


def get_epsilon(aa1: str, aa2: str) -> float:
    """Get epsilon_ij (kT) for amino acid pair."""
    aa1, aa2 = aa1.upper(), aa2.upper()
    if aa1 in ENERGIES_RAW and aa2 in ENERGIES_RAW[aa1]:
        return ENERGIES_RAW[aa1][aa2]
    elif aa2 in ENERGIES_RAW and aa1 in ENERGIES_RAW[aa2]:
        return ENERGIES_RAW[aa2][aa1]
    return 0.0


def get_sigma(aa1: str, aa2: str) -> float:
    """Get sigma_ij (Å) for amino acid pair."""
    aa1, aa2 = aa1.upper(), aa2.upper()
    if aa1 in DIAMETERS_RAW and aa2 in DIAMETERS_RAW[aa1]:
        return DIAMETERS_RAW[aa1][aa2]
    elif aa2 in DIAMETERS_RAW and aa1 in DIAMETERS_RAW[aa2]:
        return DIAMETERS_RAW[aa2][aa1]
    # Fallback: compute from individual diameters
    return (VDW_DIAMETER.get(aa1, 5.0) + VDW_DIAMETER.get(aa2, 5.0)) / 2


def kim_hummer_potential(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """
    Compute Kim-Hummer LJ-like potential.
    
    From theory.tex eq. 5.1:
    U_LJ(r, σ, ε) = 
      4ε[(σ/r)¹² - (σ/r)⁶]           if ε < 0  (attractive)
      4ε[(σ/r)¹² - (σ/r)⁶] + 2ε      if ε > 0 and r < 2^(1/6)σ  (repulsive, inner)
     -4ε[(σ/r)¹² - (σ/r)⁶]           if ε > 0 and r ≥ 2^(1/6)σ  (repulsive, outer)
      0.01(σ/r)¹²                    if ε = 0  (neutral)
    
    Parameters
    ----------
    r : array-like
        Distance (Å)
    epsilon : float
        Interaction strength (kT)
    sigma : float
        Contact distance (Å)
        
    Returns
    -------
    U : ndarray
        Potential energy (kT)
    """
    r = np.atleast_1d(r).astype(float)
    U = np.zeros_like(r)
    
    # Avoid division by zero
    mask_valid = r > 0
    sr6 = np.zeros_like(r)
    sr6[mask_valid] = (sigma / r[mask_valid]) ** 6
    sr12 = sr6 ** 2
    
    r0 = 2 ** (1/6) * sigma  # LJ minimum position
    
    if epsilon < 0:
        # Attractive: standard LJ with well depth |ε|
        # Kim & Hummer eq. (2): u_ij = 4|ε_ij|[(σ/r)^12 - (σ/r)^6]
        U[mask_valid] = 4 * abs(epsilon) * (sr12[mask_valid] - sr6[mask_valid])
    elif epsilon > 0:
        # Repulsive: two branches
        inner = mask_valid & (r < r0)
        outer = mask_valid & (r >= r0)
        U[inner] = 4 * epsilon * (sr12[inner] - sr6[inner]) + 2 * epsilon
        U[outer] = -4 * epsilon * (sr12[outer] - sr6[outer])
    else:
        # Neutral: soft wall
        U[mask_valid] = 0.01 * sr12[mask_valid]
    
    # Handle r=0 (set to large value)
    U[~mask_valid] = 1e10
    
    return U


def generate_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate full epsilon_ij and sigma_ij tables."""
    epsilon_df = build_symmetric_matrix(ENERGIES_RAW, AA_ORDER)
    sigma_df = build_symmetric_matrix(DIAMETERS_RAW, AA_ORDER)
    return epsilon_df, sigma_df


def plot_potentials(pairs: list[tuple[str, str]], rmin: float = 0.8, 
                    rmax: float = 2.5, npts: int = 500) -> plt.Figure:
    """
    Plot potential curves for selected amino acid pairs with r/σ normalization.
    
    Parameters
    ----------
    pairs : list of (str, str)
        Amino acid pairs to plot
    rmin : float
        Minimum r/σ_ij (default: 0.8)
    rmax : float
        Maximum r/σ_ij (default: 2.5)
    npts : int
        Number of points
    """
    r_norm = np.linspace(rmin, rmax, npts)  # r/σ
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for aa1, aa2 in pairs:
        eps = get_epsilon(aa1, aa2)
        sig = get_sigma(aa1, aa2)
        r = r_norm * sig  # Convert to absolute distance
        U = kim_hummer_potential(r, eps, sig)
        
        # Clip for plotting
        U_plot = np.clip(U, -2, 5)
        label = f"{aa1}-{aa2} (ε={eps:.3f})"
        ax.plot(r_norm, U_plot, label=label, lw=1.5)
    
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axvline(2**(1/6), color='gray', ls=':', lw=0.5, label=r'$2^{1/6}$')
    ax.set_xlabel(r'$r/\sigma_{ij}$', fontsize=12)
    ax.set_ylabel('U(r) (kT)', fontsize=12)
    ax.set_title('Kim-Hummer Potential', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(-1.5, 3)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


R_KJ_PER_MOL = 8.314462618e-3  # kJ/(mol·K)
KT_TO_KJ_MOL = R_KJ_PER_MOL * 300.0  # at T_ref = 300 K

# Charges at pH 7 and residue masses (Da)
CHARGES = {
    'ALA': 0.0, 'ARG': 1.0, 'ASN': 0.0, 'ASP': -1.0, 'CYS': 0.0,
    'GLN': 0.0, 'GLU': -1.0, 'GLY': 0.0, 'HIS': 0.5, 'ILE': 0.0,
    'LEU': 0.0, 'LYS': 1.0, 'MET': 0.0, 'PHE': 0.0, 'PRO': 0.0,
    'SER': 0.0, 'THR': 0.0, 'TRP': 0.0, 'TYR': 0.0, 'VAL': 0.0
}

MASSES = {
    'ALA': 71.07, 'ARG': 156.19, 'ASN': 114.1, 'ASP': 115.09, 'CYS': 103.14,
    'GLN': 128.13, 'GLU': 129.11, 'GLY': 57.05, 'HIS': 137.14, 'ILE': 113.16,
    'LEU': 113.16, 'LYS': 128.17, 'MET': 131.2, 'PHE': 147.18, 'PRO': 97.12,
    'SER': 87.08, 'THR': 101.11, 'TRP': 186.22, 'TYR': 163.18, 'VAL': 99.13
}


def generate_faunus_yaml(output: str | None = None) -> str:
    """Generate Faunus YAML force field file for Kim-Hummer.

    Epsilon values are converted from kT (T_ref=300K) to kJ/mol.
    Sigma values are pairwise arithmetic means of VDW diameters.
    Pairs are placed under `append:` so that a Coulomb `default:`
    interaction (defined in the input file) is inherited.
    """
    lines = [
        "# Kim-Hummer coarse-grained amino acid force field for Faunus.",
        "# Charges at pH 7. Residue masses in Da.",
        "#",
        "# Epsilon values are pre-scaled: eps_ij = lambda*(e_ij - e_0)",
        "# with lambda=0.159, e_0=-2.27 kT (Miyazawa-Jernigan contact energies).",
        f"# Converted from kT (T_ref=300K) to kJ/mol: eps(kJ/mol) = eps(kT) * R * 300.",
        "# Sigma values are pairwise effective diameters in Angstrom.",
        "#",
        "# References:",
        "#   Kim & Hummer, https://doi.org/10.1016/j.jmb.2007.11.063",
        "#   Miyazawa & Jernigan, https://doi.org/10.1006/jmbi.1996.0114",
        "#   Parameters from complexespp: https://github.com/bio-phys/complexespp",
        "",
    ]

    # Atoms section — sigma is the self-pair diameter σ_ii
    lines.append("atoms:")
    for aa in AA_ORDER:
        sig_ii = get_sigma(aa, aa)
        lines.append(
            f"  - {{name: {aa}, charge: {CHARGES[aa]}, "
            f"mass: {MASSES[aa]}, sigma: {sig_ii}}}"
        )

    # Nonbonded: Coulomb as default, KH pairs under append: to inherit it
    lines.append("energy:")
    lines.append("  nonbonded:")
    lines.append("    default:")
    lines.append("      - !Coulomb {cutoff: 1000}")
    lines.append("    append:")
    for i, aa1 in enumerate(AA_ORDER):
        for j in range(i, len(AA_ORDER)):
            aa2 = AA_ORDER[j]
            eps = get_epsilon(aa1, aa2)
            sig = get_sigma(aa1, aa2)
            eps_kj = eps * KT_TO_KJ_MOL
            lines.append(
                f"      [{aa1}, {aa2}]:"
                f"\n        - !KimHummer {{sigma: {sig}, epsilon: {eps_kj:.4f}}}"
            )

    text = "\n".join(lines) + "\n"
    if output:
        with open(output, "w") as f:
            f.write(text)
        print(f"Saved: {output}")
    return text


def plot_epsilon_heatmap() -> plt.Figure:
    """Plot heatmap of epsilon_ij matrix."""
    epsilon_df, _ = generate_tables()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = max(abs(epsilon_df.values.min()), abs(epsilon_df.values.max()))
    im = ax.imshow(epsilon_df.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(AA_ORDER)))
    ax.set_yticks(range(len(AA_ORDER)))
    ax.set_xticklabels(AA_ORDER, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(AA_ORDER, fontsize=8)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('ε_ij (kT)', fontsize=12)
    
    ax.set_title('Kim-Hummer ε_ij Matrix\n(blue=attractive, red=repulsive)', fontsize=12)
    
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import sys
    
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        import unittest
        
        class TestKimHummerPotential(unittest.TestCase):
            """Unit tests for Kim-Hummer potential implementation."""
            
            def setUp(self):
                """Set up test parameters."""
                self.sigma = 6.0  # Å
                self.r0 = 2**(1/6) * self.sigma  # LJ minimum position
                self.tol = 1e-10  # Numerical tolerance
            
            # --- Attractive pairs (ε < 0) ---
            
            def test_attractive_minimum_value(self):
                """At r = 2^(1/6)σ, U should equal ε for attractive pairs."""
                epsilon = -0.5
                r = np.array([self.r0])
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertAlmostEqual(U[0], epsilon, places=8)
            
            def test_attractive_at_sigma(self):
                """At r = σ, U should be 0 for attractive LJ."""
                epsilon = -0.5
                r = np.array([self.sigma])
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertAlmostEqual(U[0], 0.0, places=8)
            
            def test_attractive_repulsive_core(self):
                """For r < σ, U should be positive (repulsive core)."""
                epsilon = -0.5
                r = np.array([0.9 * self.sigma])
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertGreater(U[0], 0)
            
            def test_attractive_well_shape(self):
                """U should be negative between σ and large r for attractive pairs."""
                epsilon = -0.5
                r = np.array([1.5 * self.sigma])
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertLess(U[0], 0)
                self.assertGreater(U[0], epsilon)  # Above well minimum
            
            # --- Repulsive pairs (ε > 0) ---
            
            def test_repulsive_cusp_value(self):
                """At r = 2^(1/6)σ, U should equal ε for repulsive pairs."""
                epsilon = 0.3
                r = np.array([self.r0])
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertAlmostEqual(U[0], epsilon, places=8)
            
            def test_repulsive_always_positive(self):
                """Repulsive potential should be positive everywhere."""
                epsilon = 0.3
                r = np.linspace(0.8 * self.sigma, 3 * self.sigma, 100)
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertTrue(np.all(U > 0))
            
            def test_repulsive_inner_branch(self):
                """Test inner branch (r < r0): U = 4ε[(σ/r)^12 - (σ/r)^6] + 2ε."""
                epsilon = 0.3
                r = np.array([0.95 * self.r0])  # Just inside r0
                U = kim_hummer_potential(r, epsilon, self.sigma)
                sr6 = (self.sigma / r[0])**6
                sr12 = sr6**2
                expected = 4 * epsilon * (sr12 - sr6) + 2 * epsilon
                self.assertAlmostEqual(U[0], expected, places=8)
            
            def test_repulsive_outer_branch(self):
                """Test outer branch (r >= r0): U = -4ε[(σ/r)^12 - (σ/r)^6]."""
                epsilon = 0.3
                r = np.array([1.5 * self.sigma])  # Outside r0
                U = kim_hummer_potential(r, epsilon, self.sigma)
                sr6 = (self.sigma / r[0])**6
                sr12 = sr6**2
                expected = -4 * epsilon * (sr12 - sr6)
                self.assertAlmostEqual(U[0], expected, places=8)
            
            def test_repulsive_continuity_at_r0(self):
                """Potential should be continuous at r = r0."""
                epsilon = 0.3
                delta = 1e-8
                r_below = np.array([self.r0 - delta])
                r_above = np.array([self.r0 + delta])
                U_below = kim_hummer_potential(r_below, epsilon, self.sigma)
                U_above = kim_hummer_potential(r_above, epsilon, self.sigma)
                self.assertAlmostEqual(U_below[0], U_above[0], places=4)
            
            # --- Neutral pairs (ε = 0) ---
            
            def test_neutral_soft_wall(self):
                """For ε = 0, U = 0.01(σ/r)^12."""
                epsilon = 0.0
                r = np.array([self.sigma])
                U = kim_hummer_potential(r, epsilon, self.sigma)
                expected = 0.01 * (self.sigma / r[0])**12
                self.assertAlmostEqual(U[0], expected, places=8)
            
            def test_neutral_always_positive(self):
                """Neutral potential should be positive everywhere."""
                epsilon = 0.0
                r = np.linspace(0.8 * self.sigma, 3 * self.sigma, 100)
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertTrue(np.all(U > 0))
            
            def test_neutral_monotonic_decrease(self):
                """Neutral potential should decrease monotonically with r."""
                epsilon = 0.0
                r = np.linspace(0.8 * self.sigma, 3 * self.sigma, 100)
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertTrue(np.all(np.diff(U) < 0))
            
            # --- Edge cases ---
            
            def test_large_r_approaches_zero(self):
                """Potential should approach 0 at large r."""
                for epsilon in [-0.5, 0.0, 0.3]:
                    r = np.array([100 * self.sigma])
                    U = kim_hummer_potential(r, epsilon, self.sigma)
                    self.assertAlmostEqual(U[0], 0.0, places=6)
            
            def test_r_zero_large_value(self):
                """At r = 0, potential should be very large."""
                for epsilon in [-0.5, 0.0, 0.3]:
                    r = np.array([0.0])
                    U = kim_hummer_potential(r, epsilon, self.sigma)
                    self.assertGreater(U[0], 1e6)
            
            def test_array_input(self):
                """Function should handle array inputs correctly."""
                epsilon = -0.5
                r = np.array([self.sigma, self.r0, 2 * self.sigma])
                U = kim_hummer_potential(r, epsilon, self.sigma)
                self.assertEqual(len(U), 3)
            
            # --- Parameter lookup tests ---
            
            def test_get_epsilon_symmetric(self):
                """ε_ij should equal ε_ji."""
                eps1 = get_epsilon('LEU', 'PHE')
                eps2 = get_epsilon('PHE', 'LEU')
                self.assertEqual(eps1, eps2)
            
            def test_get_sigma_symmetric(self):
                """σ_ij should equal σ_ji."""
                sig1 = get_sigma('LEU', 'PHE')
                sig2 = get_sigma('PHE', 'LEU')
                self.assertEqual(sig1, sig2)
            
            def test_known_epsilon_values(self):
                """Test specific ε values from forcefield file."""
                # LEU-LEU: most attractive
                self.assertAlmostEqual(get_epsilon('LEU', 'LEU'), -0.8109, places=4)
                # LYS-LYS: most repulsive
                self.assertAlmostEqual(get_epsilon('LYS', 'LYS'), 0.34185, places=4)
                # CYS-GLU: neutral
                self.assertAlmostEqual(get_epsilon('CYS', 'GLU'), 0.0, places=4)
            
            def test_known_sigma_values(self):
                """Test specific σ values from forcefield file."""
                self.assertAlmostEqual(get_sigma('ALA', 'ALA'), 5.0, places=2)
                self.assertAlmostEqual(get_sigma('TRP', 'TRP'), 6.8, places=2)
                self.assertAlmostEqual(get_sigma('GLY', 'GLY'), 4.5, places=2)
        
        # Run tests
        unittest.main(argv=[''], exit=True, verbosity=2)
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'yaml':
        # Generate Faunus YAML force field file
        outfile = sys.argv[2] if len(sys.argv) > 2 else 'kimhummer.yaml'
        generate_faunus_yaml(outfile)

    else:
        # Normal execution: generate tables and plots
        epsilon_df, sigma_df = generate_tables()

        print("Epsilon_ij matrix (kT):")
        print(epsilon_df.round(4).to_string())
        print("\nSigma_ij matrix (Å):")
        print(sigma_df.round(2).to_string())

        epsilon_df.to_csv('epsilon_ij_table.csv')
        sigma_df.to_csv('sigma_ij_table.csv')
        print("\nSaved: epsilon_ij_table.csv, sigma_ij_table.csv")

        pairs = [
            ('LEU', 'LEU'),   # Most attractive
            ('PHE', 'PHE'),   # Aromatic
            ('CYS', 'CYS'),   # Disulfide-like
            ('LYS', 'LYS'),   # Most repulsive (charge-charge)
            ('LYS', 'GLU'),   # Opposite charges
            ('ALA', 'ALA'),   # Weak attractive
            ('CYS', 'GLU'),   # Neutral (ε=0)
        ]

        fig1 = plot_potentials(pairs)
        fig1.savefig('kim_hummer_potentials.png', dpi=150)
        print("Saved: kim_hummer_potentials.png")

        fig2 = plot_epsilon_heatmap()
        fig2.savefig('kim_hummer_epsilon_heatmap.png', dpi=150)
        print("Saved: kim_hummer_epsilon_heatmap.png")

        plt.show()
