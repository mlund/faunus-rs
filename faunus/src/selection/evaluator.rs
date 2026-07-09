// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! AST and evaluator for the selection language.

use crate::group::Group;
use crate::topology::{AtomKind, Chain, IndexRange, MoleculeKind, Residue, Topology};

use super::constants::{
    ACIDIC_RESIDUES, AROMATIC_RESIDUES, BACKBONE_ATOMS, BASIC_RESIDUES, DNA_RESIDUES,
    HYDROPHOBIC_RESIDUES, POLAR_RESIDUES, PROTEIN_RESIDUES, RNA_RESIDUES,
};
use super::glob::GlobPattern;

/// Expression AST node for the selection language.
#[derive(Debug, Clone)]
pub(super) enum Expr {
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
    /// Match absolute atom index (inclusive ranges).
    Index(Vec<(i32, i32)>),
    /// Match group index (inclusive ranges).
    Group(Vec<(i32, i32)>),
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

impl Expr {
    /// Whether the outcome can change when an atom's kind changes, as a titration or speciation
    /// move does.
    ///
    /// A selection that cannot must never be re-resolved on a kind swap, or every energy
    /// evaluation during titration would pay for it. The residue-category keywords count as
    /// dependent because they fall back to the atom-type name whenever a model carries no residue
    /// records (see [`AtomContext::residue_or_atomtype_in`]).
    pub(super) fn depends_on_atom_kind(&self) -> bool {
        match self {
            // Read `atom_kind` directly.
            Self::Element(_) | Self::Atomtype(_) | Self::Atomid(_) => true,
            // Read the residue name, falling back to the atom-type name in residue-less models.
            Self::Protein
            | Self::Nucleic
            | Self::Hydrophobic
            | Self::Aromatic
            | Self::Acidic
            | Self::Basic
            | Self::Polar
            | Self::Charged => true,
            // Read only the molecule template: its residues, chains, and atom names. `name` is an
            // atom's template name, not its kind, and `backbone`/`sidechain` derive from the
            // residue plus that name — none of which a kind swap can change.
            Self::Chain(_)
            | Self::Resname(_)
            | Self::Resid(_)
            | Self::Name(_)
            | Self::Backbone
            | Self::Sidechain
            | Self::Index(_)
            | Self::Group(_)
            | Self::Molecule(_)
            | Self::All
            | Self::None => false,
            Self::And(a, b) | Self::Or(a, b) => {
                a.depends_on_atom_kind() || b.depends_on_atom_kind()
            }
            Self::Not(inner) => inner.depends_on_atom_kind(),
        }
    }
}

/// Per-atom context gathered during evaluation.
struct AtomContext<'a> {
    abs_idx: usize,
    group_index: usize,
    atom_kind: &'a AtomKind,
    atom_name: Option<&'a str>,
    residue: Option<&'a Residue>,
    chain: Option<&'a Chain>,
    mol_kind: &'a MoleculeKind,
}

impl<'a> AtomContext<'a> {
    /// Build context for an atom at `abs_idx` within `group`.
    ///
    /// `atom_kind_index` indexes `topology.atomkinds()` and is the atom's *current* kind, which a
    /// titration or speciation swap can move away from the molecule template's.
    fn new(
        abs_idx: usize,
        group: &Group,
        mol_kind: &'a MoleculeKind,
        topology: &'a Topology,
        atom_kind_index: usize,
    ) -> Self {
        let rel_idx = abs_idx - group.start();
        let topo_rel = mol_kind.topology_index(rel_idx);
        let atom_kind = &topology.atomkinds()[atom_kind_index];
        let atom_name = mol_kind
            .atom_names()
            .get(topo_rel)
            .and_then(|n| n.as_deref());
        let residue = mol_kind
            .residues()
            .iter()
            .find(|r| r.range().contains(&topo_rel));
        let chain = mol_kind
            .chains()
            .iter()
            .find(|c| c.range().contains(&topo_rel));
        Self {
            abs_idx,
            group_index: group.index(),
            atom_kind,
            atom_name,
            residue,
            chain,
            mol_kind,
        }
    }

    /// Check if the residue name (or atom type as fallback) is in the given list.
    /// Allows coarse-grained models without residue info to use residue-category keywords.
    fn residue_or_atomtype_in(&self, names: &[&str]) -> bool {
        self.residue.map_or_else(
            || names.contains(&self.atom_kind.name()),
            |r| names.contains(&r.name()),
        )
    }
}

/// Check if `value` falls within any of the inclusive `(lo, hi)` ranges.
fn in_ranges(value: usize, ranges: &[(i32, i32)]) -> bool {
    let v = value as i32;
    ranges.iter().any(|(lo, hi)| v >= *lo && v <= *hi)
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
            Self::Resid(ranges) => ctx
                .residue
                .is_some_and(|r| r.number().is_some_and(|n| in_ranges(n, ranges))),
            Self::Name(patterns) => ctx
                .atom_name
                .is_some_and(|name| patterns.iter().any(|p| p.matches(name))),
            Self::Element(patterns) => ctx
                .atom_kind
                .element()
                .is_some_and(|elem| patterns.iter().any(|p| p.matches(elem))),
            Self::Atomtype(patterns) => patterns.iter().any(|p| p.matches(ctx.atom_kind.name())),
            Self::Atomid(ranges) => in_ranges(ctx.atom_kind.id(), ranges),
            Self::Index(ranges) => in_ranges(ctx.abs_idx, ranges),
            Self::Group(ranges) => in_ranges(ctx.group_index, ranges),
            Self::Molecule(patterns) => patterns.iter().any(|p| p.matches(ctx.mol_kind.name())),
            Self::Protein => ctx.residue_or_atomtype_in(PROTEIN_RESIDUES),
            Self::Backbone => ctx.residue.is_some_and(|r| {
                PROTEIN_RESIDUES.contains(&r.name())
                    && ctx
                        .atom_name
                        .is_some_and(|name| BACKBONE_ATOMS.contains(&name))
            }),
            Self::Sidechain => ctx.residue.is_some_and(|r| {
                PROTEIN_RESIDUES.contains(&r.name())
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
///
pub(super) fn resolve_atoms(
    expr: &Expr,
    topology: &Topology,
    groups: &[Group],
    get_atom_kind: &dyn Fn(usize) -> usize,
) -> Vec<usize> {
    let mut result: Vec<usize> = groups
        .iter()
        .filter(|g| !g.is_empty())
        .flat_map(|group| {
            let mol_kind = &topology.moleculekinds()[group.molecule()];
            group.iter_active().filter(|&abs_idx| {
                expr.matches(&AtomContext::new(
                    abs_idx,
                    group,
                    mol_kind,
                    topology,
                    get_atom_kind(abs_idx),
                ))
            })
        })
        .collect();

    result.sort_unstable();
    result.dedup();
    result
}

/// Resolve an expression to group indices where ANY active atom matches.
///
pub(super) fn resolve_groups(
    expr: &Expr,
    topology: &Topology,
    groups: &[Group],
    get_atom_kind: &dyn Fn(usize) -> usize,
) -> Vec<usize> {
    groups
        .iter()
        .enumerate()
        .filter(|(_, group)| !group.is_empty())
        .filter(|(_, group)| {
            let mol_kind = &topology.moleculekinds()[group.molecule()];
            group.iter_active().any(|abs_idx| {
                expr.matches(&AtomContext::new(
                    abs_idx,
                    group,
                    mol_kind,
                    topology,
                    get_atom_kind(abs_idx),
                ))
            })
        })
        .map(|(idx, _)| idx)
        .collect()
}
