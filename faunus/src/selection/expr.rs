// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! AST expression types for the selection language.

use super::glob::GlobPattern;

/// Expression AST node for the selection language.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Match chain/segment identifier.
    Chain(Vec<GlobPattern>),
    /// Match residue name.
    Resname(Vec<GlobPattern>),
    /// Match residue number (inclusive ranges).
    Resid(Vec<(i32, i32)>),
    /// Match per-instance atom name (`MoleculeKind.atom_names`).
    Name(Vec<GlobPattern>),
    /// Match chemical element symbol (`AtomKind.element`).
    Element(Vec<GlobPattern>),
    /// Match force-field atom type name (`AtomKind.name`).
    Atomtype(Vec<GlobPattern>),
    /// Match atom kind id (inclusive ranges).
    Atomid(Vec<(i32, i32)>),
    /// Match molecule kind name (`MoleculeKind.name`).
    Molecule(Vec<GlobPattern>),
    /// Protein residues.
    Protein,
    /// Backbone atoms in protein residues.
    Backbone,
    /// Sidechain atoms in protein residues.
    Sidechain,
    /// Nucleic acid residues (DNA/RNA).
    Nucleic,
    /// Hydrophobic residues.
    Hydrophobic,
    /// Aromatic residues.
    Aromatic,
    /// Acidic (negatively charged) residues.
    Acidic,
    /// Basic (positively charged) residues.
    Basic,
    /// Polar (uncharged) residues.
    Polar,
    /// Charged residues (acidic or basic).
    Charged,
    /// Select all atoms.
    All,
    /// Select no atoms.
    None,
    /// Boolean AND.
    And(Box<Self>, Box<Self>),
    /// Boolean OR.
    Or(Box<Self>, Box<Self>),
    /// Boolean NOT.
    Not(Box<Self>),
}
