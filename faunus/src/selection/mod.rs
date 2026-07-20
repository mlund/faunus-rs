// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! VMD-like atom selection language for defining atom groups.
//!
//! Supports boolean expressions with `and`, `or`, `not`, parentheses,
//! and keywords like `chain`, `resname`, `resid`, `name`, `molecule`, etc.
//!
//! Selections are written as strings and parsed by [`Selection::parse`], for example
//! `protein and backbone`, `resid 10 to 20 and chain A`, or `atomtype CA or atomtype CB`.
//! See `parses_the_selection_language` for the full set exercised by the tests.

mod constants;
mod evaluator;
mod glob;
mod parser;
mod token;

use crate::group::{AbsIndex, Group, GroupIndex, GroupSize};
use crate::topology::{MoleculeKind, Topology};

/// Return the canonical reserved selection keyword matching `name`, if any.
pub(crate) fn reserved_keyword_name(name: &str) -> Option<&'static str> {
    token::reserved_keyword_name(name)
}

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

/// A snapshot of everything a selection's outcome can depend on.
///
/// `groups` counts changes to group composition (insertions, deletions, resizes); `atom_kinds`
/// counts in-place changes of atom identity (titration and speciation swaps).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generation {
    /// Group composition, from `GroupCollection::group_lists_generation`.
    pub groups: u64,
    /// Atom identities, from `GroupCollection::atom_kinds_generation`.
    pub atom_kinds: u64,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Atoms {}
    impl Sealed for super::Groups {}
}

/// What a [`CachedSelection`] resolves to, and hence which index space it yields.
///
/// Sealed: atoms and groups are the only two spaces a selection can name.
pub trait Target: sealed::Sealed + std::fmt::Debug + Clone {
    /// The index space [`CachedSelection::resolve`] yields.
    type Index: Copy + Clone + std::fmt::Debug + PartialEq;

    #[doc(hidden)]
    fn resolve(context: &impl crate::ObserveContext, selection: &Selection) -> Vec<Self::Index>;
}

/// Resolves to the absolute indices of matching atoms.
#[derive(Clone, Copy, Debug)]
pub struct Atoms;

/// Resolves to the indices of groups holding at least one matching atom.
#[derive(Clone, Copy, Debug)]
pub struct Groups;

impl Target for Atoms {
    type Index = AbsIndex;
    fn resolve(context: &impl crate::ObserveContext, selection: &Selection) -> Vec<AbsIndex> {
        context
            .resolve_atoms(selection)
            .into_iter()
            .map(AbsIndex::new)
            .collect()
    }
}

impl Target for Groups {
    type Index = GroupIndex;
    fn resolve(context: &impl crate::ObserveContext, selection: &Selection) -> Vec<GroupIndex> {
        context
            .resolve_groups(selection)
            .into_iter()
            .map(GroupIndex::new)
            .collect()
    }
}

/// A selection together with its resolved indices, re-resolved only when the system has changed
/// in a way this particular selection can see.
///
/// The cache key is derived internally, so a consumer cannot supply a key that misses a change.
/// This matters: an atom-kind swap leaves group composition untouched, and a cache keyed on
/// composition alone would keep serving the pre-swap atoms of an `atomtype` selection forever.
///
/// The target is part of the type, so the indices it yields cannot be spent in the wrong space.
/// `CachedSelection::<Groups>::resolve` yields `GroupIndex`, which `group()` accepts. Passing it to
/// `position()`, which indexes the particle array by `usize`, is a type error: `GroupIndex` has no
/// conversion to `usize` other than the explicit `get()`, so reaching into the wrong array is a
/// visible act rather than a silent one. See `resolved_group_indices_address_the_group_array`.
#[derive(Debug, Clone)]
pub struct CachedSelection<T: Target> {
    selection: Selection,
    indices: Vec<T::Index>,
    /// `None` until first resolved.
    generation: Option<Generation>,
}

impl CachedSelection<Atoms> {
    /// Resolve to the absolute indices of matching atoms.
    pub fn atoms(selection: Selection) -> Self {
        Self::new(selection)
    }
}

impl CachedSelection<Groups> {
    /// Resolve to the indices of groups holding at least one matching atom.
    pub fn groups(selection: Selection) -> Self {
        Self::new(selection)
    }
}

impl<T: Target> CachedSelection<T> {
    fn new(selection: Selection) -> Self {
        Self {
            selection,
            indices: Vec::new(),
            generation: None,
        }
    }

    /// The underlying selection, for logging and reporting.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Currently matching indices, re-resolved only if the system has changed relevantly.
    pub fn resolve(&mut self, context: &impl crate::ObserveContext) -> &[T::Index] {
        self.resolve_with_selection(context).1
    }

    /// Like [`resolve`](Self::resolve), but also hands back the selection.
    ///
    /// A caller that must report on an empty result would otherwise need a second `resolve` just
    /// to borrow the selection again — wasteful on the energy hot path.
    pub fn resolve_with_selection(
        &mut self,
        context: &impl crate::ObserveContext,
    ) -> (&Selection, &[T::Index]) {
        let generation = self.selection.generation(context);
        if self.generation != Some(generation) {
            self.indices = T::resolve(context, &self.selection);
            self.generation = Some(generation);
        }
        (&self.selection, &self.indices)
    }
}

/// A selection whose target follows a consumer's `com` flag.
///
/// The flag decides both what the consumer iterates over and what its selection must resolve to;
/// deriving one from the other here keeps the two from drifting apart, and matching on this enum
/// hands each branch the index space it can actually use.
#[derive(Debug, Clone)]
pub enum ComSelection {
    /// Individual atoms, for `com: false`.
    Atoms(CachedSelection<Atoms>),
    /// Molecular mass centers, one per group, for `com: true`.
    Groups(CachedSelection<Groups>),
}

impl ComSelection {
    /// Groups when `com` acts on molecular mass centers, atoms otherwise.
    pub fn new(selection: Selection, com: bool) -> Self {
        if com {
            Self::Groups(CachedSelection::groups(selection))
        } else {
            Self::Atoms(CachedSelection::atoms(selection))
        }
    }

    /// The underlying selection, for logging and reporting.
    pub fn selection(&self) -> &Selection {
        match self {
            Self::Atoms(cache) => cache.selection(),
            Self::Groups(cache) => cache.selection(),
        }
    }
}

/// Find a group the selection could match but the caller cannot handle.
///
/// An analysis usually works only on molecules carrying some property — a mass centre, say, or a
/// rigid-body orientation. Describe that property with `supported`, and any group returned here is
/// one the selection names and the analysis cannot use, ready to be refused while the run is still
/// starting up. `None` means every group the selection can reach is usable.
///
/// The search runs against a fully populated copy of the groups, because a selection matches only
/// the atoms that are currently present. A species that starts out empty — a grand-canonical
/// reservoir, for instance — therefore matches nothing, passes a check made against the initial
/// configuration, and breaks the run much later, once its first particle is inserted.
pub(crate) fn first_unsupported_group(
    context: &impl crate::ObserveContext,
    selection: &Selection,
    supported: impl Fn(&MoleculeKind) -> bool,
) -> anyhow::Result<Option<GroupIndex>> {
    let mut groups = context.groups().to_vec();
    for group in &mut groups {
        group.resize(GroupSize::Full)?;
    }
    let topology = context.topology_ref();
    let kinds = topology.moleculekinds();
    Ok(selection
        .resolve_groups(topology, &groups, &|index| context.atom_kind(index))
        .into_iter()
        .find(|&index| !supported(&kinds[groups[index].molecule().get()]))
        .map(GroupIndex::new))
}

/// A parsed VMD-like atom selection expression.
///
/// Parses from a string, then resolves against topology and groups
/// to produce atom indices or group indices.
#[derive(Debug, Clone)]
pub struct Selection {
    source: String,
    expr: evaluator::Expr,
    /// Whether a titration or speciation swap can change what this selection matches.
    depends_on_atom_kind: bool,
}

impl Selection {
    /// Whether an in-place change of an atom's kind can change what this selection matches.
    pub fn depends_on_atom_kind(&self) -> bool {
        self.depends_on_atom_kind
    }

    /// The state this selection's outcome depends on, used as a cache key by [`CachedSelection`].
    ///
    /// Selections that cannot see atom identities ignore that counter, so a titration move does
    /// not force, say, `molecule water` to be re-resolved on every energy evaluation.
    fn generation(&self, context: &impl crate::ObserveContext) -> Generation {
        Generation {
            groups: context.group_lists_generation(),
            atom_kinds: if self.depends_on_atom_kind {
                context.atom_kinds_generation()
            } else {
                0
            },
        }
    }

    /// Parse a VMD-like selection expression.
    ///
    /// # Errors
    /// Returns error if the expression is syntactically invalid.
    pub fn parse(input: &str) -> Result<Self, SelectionError> {
        let tokens = token::tokenize(input)?;
        let mut parser = parser::Parser::new(&tokens);
        let expr = parser.parse()?;
        Ok(Self {
            depends_on_atom_kind: expr.depends_on_atom_kind(),
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
    /// Iterates all active particles in all non-empty groups and returns the absolute indices of
    /// those matching the expression.
    ///
    /// `get_atom_kind(abs_index)` returns the atom's *current* kind as an index into
    /// [`Topology::atomkinds`], which a titration or speciation swap can move away from the
    /// molecule template's. Note this is the position in that slice, not `AtomKind::id()` — the
    /// two happen to coincide, but the `atomid` selection keyword matches on the latter.
    pub fn resolve_atoms(
        &self,
        topology: &Topology,
        groups: &[Group],
        get_atom_kind: &dyn Fn(usize) -> crate::group::AtomKindId,
    ) -> Vec<usize> {
        evaluator::resolve_atoms(&self.expr, topology, groups, get_atom_kind)
    }

    /// Resolve to the index of each non-empty group holding at least one matching active atom,
    /// which naturally gives molecule-level selection.
    ///
    /// See [`resolve_atoms`](Self::resolve_atoms) for `get_atom_kind`.
    pub fn resolve_groups(
        &self,
        topology: &Topology,
        groups: &[Group],
        get_atom_kind: &dyn Fn(usize) -> crate::group::AtomKindId,
    ) -> Vec<usize> {
        evaluator::resolve_groups(&self.expr, topology, groups, get_atom_kind)
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
mod atom_kind_dependence_tests {
    use super::*;

    fn depends(expression: &str) -> bool {
        Selection::parse(expression).unwrap().depends_on_atom_kind()
    }

    #[test]
    fn selections_reading_the_atom_kind_are_flagged() {
        for expression in [
            "atomtype Na",
            "element H",
            "atomid 0 to 2",
            "charged",
            "acidic",
        ] {
            assert!(depends(expression), "{expression} reads the atom kind");
        }
    }

    #[test]
    fn selections_reading_only_static_topology_are_not_flagged() {
        // `name` resolves against the molecule template, not the atom kind, and
        // `backbone`/`sidechain` derive from the residue plus that name — a swap changes none.
        for expression in [
            "molecule water",
            "resid 1 to 5",
            "name CA",
            "backbone",
            "sidechain",
            "index 0",
            "all",
            "none",
        ] {
            assert!(!depends(expression), "{expression} ignores the atom kind");
        }
    }

    #[test]
    fn dependence_propagates_through_boolean_operators() {
        assert!(depends("molecule water and atomtype OW"));
        assert!(depends("not element H"));
        assert!(depends("resid 1 or charged"));
        assert!(!depends("molecule water and not name CA"));
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

    /// Was the module doc example, before `selection` became crate-private and rustdoc stopped
    /// compiling it.
    #[test]
    fn parses_the_selection_language() {
        for expression in [
            "protein and backbone",
            "molecule water",
            "resid 10 to 20 and chain A",
            "atomtype CA or atomtype CB",
            "name CA and chain A",
        ] {
            assert!(Selection::parse(expression).is_ok(), "{expression}");
        }
    }

    #[test]
    fn parse_invalid() {
        assert!(Selection::parse("").is_err());
        assert!(Selection::parse("unknown_keyword").is_err());
        assert!(Selection::parse("chain").is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let yaml = yaml_serde::to_string(&Selection::parse("molecule water").unwrap()).unwrap();
        assert_eq!(yaml.trim(), "molecule water");
        let sel: Selection = yaml_serde::from_str(&yaml).unwrap();
        assert_eq!(sel.source(), "molecule water");
    }

    #[test]
    fn serde_invalid_input() {
        let result: Result<Selection, _> = yaml_serde::from_str("''");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::backend::Backend;
    use crate::context::WithTopology;
    use crate::group::GroupCollection;
    use crate::group::GroupSelection;
    use std::path::Path;

    fn make_context() -> Backend {
        let mut rng = rand::thread_rng();
        Backend::new(
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
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        // Should return all active particles
        let expected_count: usize = ctx.groups().iter().map(|g| g.len()).sum();
        assert_eq!(atoms.len(), expected_count);
    }

    #[test]
    fn select_none_atoms() {
        let ctx = make_context();
        let sel = Selection::parse("none").unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        assert!(atoms.is_empty());
    }

    #[test]
    fn select_all_groups() {
        let ctx = make_context();
        let sel = Selection::parse("all").unwrap();
        let groups = sel.resolve_groups(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        // Should return all non-empty groups
        let expected_count = ctx.groups().iter().filter(|g| !g.is_empty()).count();
        assert_eq!(groups.len(), expected_count);
    }

    #[test]
    fn select_none_groups() {
        let ctx = make_context();
        let sel = Selection::parse("none").unwrap();
        let groups = sel.resolve_groups(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        assert!(groups.is_empty());
    }

    #[test]
    fn select_by_molecule_name() {
        let ctx = make_context();
        // Find a molecule name from the topology
        let mol_name = ctx.topology_ref().moleculekinds()[0].name();
        let sel_str = format!("molecule {mol_name}");
        let sel = Selection::parse(&sel_str).unwrap();
        let group_indices =
            sel.resolve_groups(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));

        // All returned groups should have the correct molecule kind
        let mol_id = crate::group::MoleculeId::new(ctx.topology_ref().moleculekinds()[0].id());
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
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        assert!(!atoms.is_empty());
        for &i in &atoms {
            let kind = ctx.atom_kind(i).get();
            assert_eq!(ctx.topology_ref().atomkinds()[kind].name(), atom_name);
        }
    }

    #[test]
    fn select_not_inverts() {
        let ctx = make_context();
        let mol_name = ctx.topology_ref().moleculekinds()[0].name().to_string();
        let sel1 = Selection::parse(&format!("molecule {mol_name}")).unwrap();
        let sel2 = Selection::parse(&format!("not molecule {mol_name}")).unwrap();
        let atoms1 = sel1.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        let atoms2 = sel2.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));

        // Together they should cover all active atoms
        let all = Selection::parse("all").unwrap();
        let all_atoms = all.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        assert_eq!(atoms1.len() + atoms2.len(), all_atoms.len());

        // No overlap
        for idx in &atoms1 {
            assert!(!atoms2.contains(idx));
        }
    }

    #[test]
    fn select_by_index() {
        let ctx = make_context();
        let sel = Selection::parse("index 0").unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        assert_eq!(atoms, vec![0]);
    }

    #[test]
    fn select_by_group() {
        let ctx = make_context();
        let sel = Selection::parse("group 0").unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        // Should return all active atoms in group 0
        let g0 = &ctx.groups()[0];
        let expected: Vec<usize> = g0.iter_active().collect();
        assert_eq!(atoms, expected);
    }

    #[test]
    fn select_by_group_range() {
        let ctx = make_context();
        let all_groups: Vec<usize> = ctx
            .groups()
            .iter()
            .filter(|g| !g.is_empty())
            .map(|g| g.index())
            .collect();
        if all_groups.len() >= 2 {
            let last = *all_groups.last().unwrap() as i32;
            let sel_str = format!("group 0 to {last}");
            let sel = Selection::parse(&sel_str).unwrap();
            let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
            let all_sel = Selection::parse("all").unwrap();
            let all_atoms =
                all_sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
            assert_eq!(atoms, all_atoms);
        }
    }

    #[test]
    fn select_by_atomid_range() {
        let ctx = make_context();
        let sel = Selection::parse("atomid 0 to 0").unwrap();
        let atoms = sel.resolve_atoms(ctx.topology_ref(), ctx.groups(), &|i| ctx.atom_kind(i));
        assert!(!atoms.is_empty());
        for &i in &atoms {
            let kind = ctx.atom_kind(i).get();
            // `atomid` filters on `AtomKind::id()`, not the raw array index — check that field
            // directly rather than the index it happens to coincide with.
            assert_eq!(ctx.topology_ref().atomkinds()[kind].id(), 0);
        }
    }
}
