// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! Residue classification constants for selection keywords.

/// Standard protein residue names (including terminal patches).
pub const PROTEIN_RESIDUES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HIE", "HID", "HIP", "CYX", "ASH", "GLH",
    "LYN", "NTR", "CTR",
];

/// DNA residue names.
pub const DNA_RESIDUES: &[&str] = &["DA", "DT", "DG", "DC", "DU"];

/// RNA residue names.
pub const RNA_RESIDUES: &[&str] = &["A", "U", "G", "C", "RA", "RU", "RG", "RC"];

/// Backbone atom names.
pub const BACKBONE_ATOMS: &[&str] = &["C", "CA", "N", "O"];

/// Hydrophobic residues (nonpolar sidechains).
pub const HYDROPHOBIC_RESIDUES: &[&str] = &[
    "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "GLY",
];

/// Aromatic residues.
pub const AROMATIC_RESIDUES: &[&str] = &["PHE", "TYR", "TRP", "HIS", "HIE", "HID", "HIP"];

/// Acidic (negatively charged) residues.
pub const ACIDIC_RESIDUES: &[&str] = &["ASP", "GLU", "ASH", "GLH", "CTR"];

/// Basic (positively charged) residues.
pub const BASIC_RESIDUES: &[&str] = &["ARG", "LYS", "HIS", "HIE", "HID", "HIP", "LYN", "NTR"];

/// Polar (uncharged) residues.
pub const POLAR_RESIDUES: &[&str] = &["SER", "THR", "ASN", "GLN", "CYS", "CYX", "TYR"];

/// Check if residue name is in a set.
pub fn resname_in(resname: &str, set: &[&str]) -> bool {
    set.contains(&resname)
}
