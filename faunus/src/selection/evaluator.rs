// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! Evaluator: resolve selection expressions against topology and groups.

use crate::group::Group;
use crate::topology::{AtomKind, Chain, IndexRange, MoleculeKind, Residue, Topology};

use super::constants::*;
use super::expr::Expr;

/// Per-atom context gathered during evaluation.
struct AtomContext<'a> {
    atom_kind: &'a AtomKind,
    atom_name: Option<&'a str>,
    residue: Option<&'a Residue>,
    chain: Option<&'a Chain>,
    mol_kind: &'a MoleculeKind,
}

impl<'a> AtomContext<'a> {
    /// Build context for an atom at `abs_idx` within `group`.
    fn new(
        abs_idx: usize,
        group: &Group,
        mol_kind: &'a MoleculeKind,
        topology: &'a Topology,
    ) -> Self {
        let rel_idx = abs_idx - group.start();
        let atom_kind_id = mol_kind.atom_indices()[rel_idx];
        let atom_kind = &topology.atomkinds()[atom_kind_id];
        let atom_name = mol_kind
            .atom_names()
            .get(rel_idx)
            .and_then(|n| n.as_deref());
        let residue = mol_kind
            .residues()
            .iter()
            .find(|r| r.range().contains(&rel_idx));
        let chain = mol_kind
            .chains()
            .iter()
            .find(|c| c.range().contains(&rel_idx));
        Self {
            atom_kind,
            atom_name,
            residue,
            chain,
            mol_kind,
        }
    }
}

impl<'a> AtomContext<'a> {
    /// Check if the residue name (or atom type as fallback) is in the given list.
    /// Allows coarse-grained models without residue info to use residue-category keywords.
    fn residue_or_atomtype_in(&self, names: &[&str]) -> bool {
        self.residue.map_or_else(
            || names.contains(&self.atom_kind.name()),
            |r| resname_in(r.name(), names),
        )
    }
}

impl Expr {
    /// Evaluate whether a single atom matches this expression.
    fn matches(&self, ctx: &AtomContext) -> bool {
        match self {
            Self::Chain(patterns) => ctx
                .chain
                .is_some_and(|c| patterns.iter().any(|p| p.matches(c.name()))),
            Self::Resname(patterns) => ctx
                .residue
                .is_some_and(|r| patterns.iter().any(|p| p.matches(r.name()))),
            Self::Resid(ranges) => ctx.residue.is_some_and(|r| {
                r.number().is_some_and(|num| {
                    let num = num as i32;
                    ranges.iter().any(|(lo, hi)| num >= *lo && num <= *hi)
                })
            }),
            Self::Name(patterns) => ctx
                .atom_name
                .is_some_and(|name| patterns.iter().any(|p| p.matches(name))),
            Self::Element(patterns) => ctx
                .atom_kind
                .element()
                .is_some_and(|elem| patterns.iter().any(|p| p.matches(elem))),
            Self::Atomtype(patterns) => patterns.iter().any(|p| p.matches(ctx.atom_kind.name())),
            Self::Atomid(ranges) => {
                let id = ctx.atom_kind.id() as i32;
                ranges.iter().any(|(lo, hi)| id >= *lo && id <= *hi)
            }
            Self::Molecule(patterns) => patterns.iter().any(|p| p.matches(ctx.mol_kind.name())),
            Self::Protein => ctx.residue_or_atomtype_in(PROTEIN_RESIDUES),
            Self::Backbone => ctx.residue.is_some_and(|r| {
                resname_in(r.name(), PROTEIN_RESIDUES)
                    && ctx
                        .atom_name
                        .is_some_and(|name| BACKBONE_ATOMS.contains(&name))
            }),
            Self::Sidechain => ctx.residue.is_some_and(|r| {
                resname_in(r.name(), PROTEIN_RESIDUES)
                    && ctx
                        .atom_name
                        .is_none_or(|name| !BACKBONE_ATOMS.contains(&name))
            }),
            Self::Nucleic => {
                ctx.residue_or_atomtype_in(DNA_RESIDUES) || ctx.residue_or_atomtype_in(RNA_RESIDUES)
            }
            Self::Hydrophobic => ctx.residue_or_atomtype_in(HYDROPHOBIC_RESIDUES),
            Self::Aromatic => ctx.residue_or_atomtype_in(AROMATIC_RESIDUES),
            Self::Acidic => ctx.residue_or_atomtype_in(ACIDIC_RESIDUES),
            Self::Basic => ctx.residue_or_atomtype_in(BASIC_RESIDUES),
            Self::Polar => ctx.residue_or_atomtype_in(POLAR_RESIDUES),
            Self::Charged => {
                ctx.residue_or_atomtype_in(ACIDIC_RESIDUES)
                    || ctx.residue_or_atomtype_in(BASIC_RESIDUES)
            }
            Self::All => true,
            Self::None => false,
            Self::And(a, b) => a.matches(ctx) && b.matches(ctx),
            Self::Or(a, b) => a.matches(ctx) || b.matches(ctx),
            Self::Not(a) => !a.matches(ctx),
        }
    }
}

/// Resolve an expression to absolute particle indices (sorted, deduplicated).
pub fn resolve_atoms(expr: &Expr, topology: &Topology, groups: &[Group]) -> Vec<usize> {
    let mut result: Vec<usize> = groups
        .iter()
        .filter(|g| !g.is_empty())
        .flat_map(|group| {
            let mol_kind = &topology.moleculekinds()[group.molecule()];
            group.iter_active().filter(|&abs_idx| {
                expr.matches(&AtomContext::new(abs_idx, group, mol_kind, topology))
            })
        })
        .collect();

    result.sort_unstable();
    result.dedup();
    result
}

/// Resolve an expression to group indices where ANY active atom matches.
pub fn resolve_groups(expr: &Expr, topology: &Topology, groups: &[Group]) -> Vec<usize> {
    groups
        .iter()
        .enumerate()
        .filter(|(_, group)| !group.is_empty())
        .filter(|(_, group)| {
            let mol_kind = &topology.moleculekinds()[group.molecule()];
            group
                .iter_active()
                .any(|abs_idx| expr.matches(&AtomContext::new(abs_idx, group, mol_kind, topology)))
        })
        .map(|(idx, _)| idx)
        .collect()
}
