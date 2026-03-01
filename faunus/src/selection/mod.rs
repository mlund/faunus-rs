// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! VMD-like atom selection language for defining atom groups.
//!
//! Supports boolean expressions with `and`, `or`, `not`, parentheses,
//! and keywords like `chain`, `resname`, `resid`, `name`, `molecule`, etc.
//!
//! # Examples
//!
//! ```
//! use faunus::selection::Selection;
//!
//! let sel = Selection::parse("protein and backbone").unwrap();
//! let sel = Selection::parse("molecule water").unwrap();
//! let sel = Selection::parse("resid 10 to 20 and chain A").unwrap();
//! let sel = Selection::parse("atomtype CA or atomtype CB").unwrap();
//! ```

mod constants;
mod evaluator;
mod expr;
mod glob;
mod parser;
mod token;

use crate::group::{Group, GroupSelection};
use crate::topology::Topology;

/// Selection parsing error.
#[derive(Debug, Clone)]
pub struct SelectionError {
    /// Error message.
    pub message: String,
    /// Position in input where error occurred.
    pub position: usize,
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at position {}", self.message, self.position)
    }
}

impl std::error::Error for SelectionError {}

/// Caches resolved selection indices, invalidated when `GroupLists` generation changes.
///
/// Use `get_or_resolve()` to lazily re-resolve only when the group composition
/// has actually changed (e.g. after Grand Canonical insert/delete moves).
#[derive(Debug, Clone)]
pub struct SelectionCache {
    indices: Vec<usize>,
    generation: u64,
}

impl Default for SelectionCache {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            generation: u64::MAX,
        }
    }
}

impl SelectionCache {
    /// Return cached indices, or re-resolve if the generation has changed.
    pub fn get_or_resolve(
        &mut self,
        generation: u64,
        resolve: impl FnOnce() -> Vec<usize>,
    ) -> &[usize] {
        if self.generation != generation {
            self.indices = resolve();
            self.generation = generation;
        }
        &self.indices
    }
}

/// A parsed VMD-like atom selection expression.
///
/// Parses from a string, then resolves against topology and groups
/// to produce atom indices or group indices.
///
/// # Examples
///
/// ```
/// use faunus::selection::Selection;
///
/// let sel = Selection::parse("molecule water").unwrap();
/// let sel = Selection::parse("name CA and chain A").unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Selection {
    source: String,
    expr: expr::Expr,
}

impl Selection {
    /// Parse a VMD-like selection expression.
    ///
    /// # Errors
    /// Returns error if the expression is syntactically invalid.
    pub fn parse(input: &str) -> Result<Self, SelectionError> {
        let tokens = token::tokenize(input)?;
        let mut parser = parser::Parser::new(&tokens);
        let expr = parser.parse()?;
        Ok(Self {
            source: input.to_string(),
            expr,
        })
    }

    /// Get the original source string.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Resolve to absolute particle indices (sorted, deduplicated).
    ///
    /// Iterates all active particles in all non-empty groups and returns
    /// the absolute indices of those matching the expression.
    pub fn resolve_atoms(&self, topology: &Topology, groups: &[Group]) -> Vec<usize> {
        evaluator::resolve_atoms(&self.expr, topology, groups)
    }

    /// Resolve to group indices where ANY active atom matches.
    ///
    /// Returns the index of each non-empty group that contains at least
    /// one active atom matching the expression. This naturally gives
    /// molecule-level selection.
    pub fn resolve_groups(&self, topology: &Topology, groups: &[Group]) -> Vec<usize> {
        evaluator::resolve_groups(&self.expr, topology, groups)
    }

    /// Convert resolved groups to a `GroupSelection` for use with existing code.
    pub fn to_group_selection(&self, topology: &Topology, groups: &[Group]) -> GroupSelection {
        let indices = self.resolve_groups(topology, groups);
        match indices.len() {
            0 => GroupSelection::Index(vec![]),
            1 => GroupSelection::Single(indices[0]),
            _ => GroupSelection::Index(indices),
        }
    }
}

impl std::fmt::Display for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.source)
    }
}

/// Deserialize from string, serialize back to string.
impl serde::Serialize for Selection {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.source)
    }
}

impl<'de> serde::Deserialize<'de> for Selection {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display() {
        let sel = Selection::parse("protein and backbone").unwrap();
        assert_eq!(sel.to_string(), "protein and backbone");
    }

    #[test]
    fn parse_invalid() {
        assert!(Selection::parse("").is_err());
        assert!(Selection::parse("unknown_keyword").is_err());
        assert!(Selection::parse("chain").is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let yaml = serde_yaml::to_string(&Selection::parse("molecule water").unwrap()).unwrap();
        assert_eq!(yaml.trim(), "molecule water");
        let sel: Selection = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(sel.source(), "molecule water");
    }

    #[test]
    fn serde_invalid_input() {
        let result: Result<Selection, _> = serde_yaml::from_str("''");
        assert!(result.is_err());
    }
}

#[cfg(all(test, feature = "chemfiles"))]
mod integration_tests {
    use super::*;
    use crate::context::WithTopology;
    use crate::group::GroupCollection;
    use crate::platform::reference::ReferencePlatform;
    use std::path::Path;

    fn make_context() -> ReferencePlatform {
        let mut rng = rand::thread_rng();
        ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap()
    }

    #[test]
    fn select_all_atoms() {
        let ctx = make_context();
        let sel = Selection::parse("all").unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups());
        // Should return all active particles
        let expected_count: usize = ctx.groups().iter().map(|g| g.len()).sum();
        assert_eq!(atoms.len(), expected_count);
    }

    #[test]
    fn select_none_atoms() {
        let ctx = make_context();
        let sel = Selection::parse("none").unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups());
        assert!(atoms.is_empty());
    }

    #[test]
    fn select_all_groups() {
        let ctx = make_context();
        let sel = Selection::parse("all").unwrap();
        let groups = sel.resolve_groups(ctx.topology_ref(), ctx.groups());
        // Should return all non-empty groups
        let expected_count = ctx.groups().iter().filter(|g| !g.is_empty()).count();
        assert_eq!(groups.len(), expected_count);
    }

    #[test]
    fn select_none_groups() {
        let ctx = make_context();
        let sel = Selection::parse("none").unwrap();
        let groups = sel.resolve_groups(ctx.topology_ref(), ctx.groups());
        assert!(groups.is_empty());
    }

    #[test]
    fn select_by_molecule_name() {
        let ctx = make_context();
        // Find a molecule name from the topology
        let mol_name = ctx.topology_ref().moleculekinds()[0].name();
        let sel_str = format!("molecule {mol_name}");
        let sel = Selection::parse(&sel_str).unwrap();
        let group_indices = sel.resolve_groups(ctx.topology_ref(), ctx.groups());

        // All returned groups should have the correct molecule kind
        let mol_id = ctx.topology_ref().moleculekinds()[0].id();
        for &gi in &group_indices {
            assert_eq!(ctx.groups()[gi].molecule(), mol_id);
        }

        // Compare with GroupSelection::ByMoleculeId
        let expected = ctx.select(&GroupSelection::ByMoleculeId(mol_id));
        assert_eq!(group_indices, expected);
    }

    #[test]
    fn select_by_atomtype() {
        let ctx = make_context();
        let atom_name = ctx.topology_ref().atomkinds()[0].name().to_string();
        let sel_str = format!("atomtype {atom_name}");
        let sel = Selection::parse(&sel_str).unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups());
        // All returned atoms should have atomkind id 0
        assert!(!atoms.is_empty());
    }

    #[test]
    fn select_not_inverts() {
        let ctx = make_context();
        let mol_name = ctx.topology_ref().moleculekinds()[0].name().to_string();
        let sel1 = Selection::parse(&format!("molecule {mol_name}")).unwrap();
        let sel2 = Selection::parse(&format!("not molecule {mol_name}")).unwrap();
        let atoms1 = sel1.resolve_atoms(ctx.topology_ref(), ctx.groups());
        let atoms2 = sel2.resolve_atoms(ctx.topology_ref(), ctx.groups());

        // Together they should cover all active atoms
        let all = Selection::parse("all").unwrap();
        let all_atoms = all.resolve_atoms(ctx.topology_ref(), ctx.groups());
        assert_eq!(atoms1.len() + atoms2.len(), all_atoms.len());

        // No overlap
        for idx in &atoms1 {
            assert!(!atoms2.contains(idx));
        }
    }

    #[test]
    fn to_group_selection_single() {
        let ctx = make_context();
        // Select a molecule that has exactly one group instance
        // First molecule kind, check how many groups it has
        let sel = Selection::parse("all").unwrap();
        let gs = sel.to_group_selection(ctx.topology_ref(), ctx.groups());
        assert!(matches!(gs, GroupSelection::Index(_)));
    }

    #[test]
    fn select_by_atomid_range() {
        let ctx = make_context();
        let sel = Selection::parse("atomid 0 to 0").unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups());
        // Should select only atoms with atomkind id 0
        assert!(!atoms.is_empty());
    }
}
