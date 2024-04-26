// Copyright 2023 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Topology module for storing information about atoms, residues, bonds, etc.
//!
//! The topology describes which _kind_ of atoms and residues that are present in the system,
//! and how they are connected.
//! Notably it _does not_
//! - include state information such as positions, velocities, etc.
//! - know how many atoms or residues are present in the system.
//!
//! The [`Topology`] is constructed using the following building blocks:
//!
//! - [`AtomKind`] is the smallest unit, but need not to be a chemical element.
//! - [`ResidueKind`] is a collection of atoms, e.g. a protein residue or a water molecule.
//! - [`ChainKind`] is a collection of residues, e.g. a polymer or protein chain.
//!
//! # Examples
//! ~~~
//! use faunus::topology::*;
//! let mut top = Topology::default();
//! top.add_atom(AtomKind::new("Ow")).unwrap();
//! top.add_atom(AtomKind::new("Hw")).unwrap();
//! assert!(top.add_atom(AtomKind::new("Hw")).is_err()); // error: duplicate name
//! assert_eq!(top.atoms().len(), 2);
//!
//! let mut water = ResidueKind::new("Water", &[0, 1, 1]);
//! assert_eq!(water.len(), 3);
//! let bond1 = Bond::new([0, 1], BondKind::Harmonic{ k: 100.0, req: 1.0 }, None);
//! let bond2 = Bond::new([1, 2], BondKind::Harmonic{ k: 100.0, req: 1.0 }, None);
//! water.add_bond(bond1).unwrap();
//! water.add_bond(bond2).unwrap();
//! assert_eq!(water.connectivity.bonds().len(), 2);
//!
//! top.add_residue(water).unwrap();
//! assert_eq!(top.residues().len(), 1);
//! assert_eq!(top.residues().last().unwrap().id, 0);
//! ~~~
mod atom;
mod bond;
mod chain;
//pub mod chemfiles;
mod dihedral;
mod molecule;
mod residue;
mod torsion;
use std::fmt::Debug;
use std::ops::Range;

use anyhow::Ok;
pub use atom::*;
pub use bond::*;
pub use chain::*;
pub use dihedral::*;
pub use residue::*;
pub use torsion::*;
use validator::ValidationError;

use crate::Point;
use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};

/// Trait implemented by collections of atoms that should not overlap (e.g., residues, chains).
pub(super) trait NonOverlapping {
    /// Get the indices of atoms in the collection.
    fn atoms(&self) -> &Range<usize>;

    /// Check whether two collections overlap.
    ///
    /// ## Return
    /// Returns `true` if the collections overlap, else returns `false`.
    fn overlap(&self, other: &Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.atoms().start < other.atoms().end
            && other.atoms().start < self.atoms().end
    }

    /// Check whether the collection is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.atoms().is_empty()
    }

    // TODO! tests
    /// Validate that collections in a list do not overlap.
    fn validate(collection: &[impl NonOverlapping]) -> Result<(), ValidationError> {
        if collection.iter().enumerate().any(|(i, item_i)| {
            collection
                .iter()
                .skip(i + 1)
                .any(|item_j| item_i.overlap(item_j))
        }) {
            Err(ValidationError::new("overlap between collections"))
        } else {
            core::result::Result::Ok(())
        }
    }
}

#[test]
fn collections_overlap() {
    let residue1 = Residue::new("ALA".to_owned(), None, 2..5);
    let residue2 = Residue::new("LYS".to_owned(), None, 7..11);
    assert!(!residue1.overlap(&residue2));

    let residue2 = Residue::new("LYS".to_owned(), None, 5..11);
    assert!(!residue1.overlap(&residue2));

    let residue2 = Residue::new("LYS".to_owned(), None, 0..2);
    assert!(!residue1.overlap(&residue2));

    let residue2 = Residue::new("LYS".to_owned(), None, 2..5);
    assert!(residue1.overlap(&residue2));

    let residue2 = Residue::new("LYS".to_owned(), None, 1..11);
    assert!(residue1.overlap(&residue2));

    let residue2 = Residue::new("LYS".to_owned(), None, 3..4);
    assert!(residue1.overlap(&residue2));

    let residue2 = Residue::new("LYS".to_owned(), None, 1..3);
    assert!(residue1.overlap(&residue2));

    let residue2 = Residue::new("LYS".to_owned(), None, 4..11);
    assert!(residue1.overlap(&residue2));

    let chain1 = Chain::new("A", 2..5);
    let chain2 = Chain::new("B", 7..11);
    assert!(!chain1.overlap(&chain2));

    let chain2 = Chain::new("B", 4..11);
    assert!(chain1.overlap(&chain2));
}

/// Trait for identifying elements in a collection, e.g. index
trait Indices {
    /// Positional index of element in a collection
    fn index(&self) -> usize;
    /// Set positional index of element in a collection
    fn set_index(&mut self, index: usize);
}

/// Enum to store custom data for atoms, residues, molecules etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Bool(bool),
    Int(i32),
    Float(f64),
    // Point must be placed before Vector for correct classification by serde
    Point(Point),
    Vector(Vec<f64>),
}

// Test Value conversions to f64 and bool
#[test]
fn test_value_conversions() {
    let v = Value::Float(1.0);
    assert_eq!(f64::try_from(v.clone()).unwrap(), 1.0);
    assert!(bool::try_from(v).is_err());
    let v = Value::Bool(true);
    assert!(f64::try_from(v.clone()).is_err());
    assert!(bool::try_from(v).unwrap());
}

impl core::convert::TryFrom<Value> for f64 {
    type Error = anyhow::Error;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(f) => Ok(f),
            _ => Err(anyhow::anyhow!("Could not convert value to f64")),
        }
    }
}

impl core::convert::TryFrom<Value> for bool {
    type Error = anyhow::Error;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Bool(b) => Ok(b),
            _ => Err(anyhow::anyhow!("Could not convert value to bool")),
        }
    }
}

/// A custom property for atoms, residues, chains etc.
pub trait CustomProperty {
    /// Set a custom, property associated with a unique `key`.
    ///
    /// The key could e.g. be a converted discriminant from a field-less enum.
    fn set_property(&mut self, key: &str, value: Value) -> anyhow::Result<()>;
    /// Get property assosiated with a `key`.
    fn get_property(&self, key: &str) -> Option<Value>;
}

/// A selection of atoms or residues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Selection<T> {
    /// A list of names
    Vec(Vec<T>),
    /// A repeated element
    Repeat(T, usize),
    /// Vector of identifiers like atom or residue ids
    Ids(Vec<usize>),
}

impl<T> Selection<T> {
    /// Number of elements in selection
    pub fn len(&self) -> usize {
        match self {
            Selection::Vec(v) => v.len(),
            Selection::Ids(v) => v.len(),
            Selection::Repeat(_, n) => *n,
        }
    }
    /// Check if selection is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Iterate over selection
    pub fn iter(&self) -> anyhow::Result<Box<dyn Iterator<Item = &T> + '_>> {
        match self {
            Selection::Vec(v) => Ok(Box::new(v.iter())),
            Selection::Repeat(t, n) => Ok(Box::new(std::iter::repeat(t).take(*n))),
            _ => anyhow::bail!("Cannot iterate over selection"),
        }
    }
}
#[test]
fn test_selection() {
    let s = Selection::Vec(vec!["a", "b", "c"]);
    assert_eq!(s.len(), 3);
    assert_eq!(
        s.iter().unwrap().collect::<Vec<_>>(),
        vec![&"a", &"b", &"c"]
    );
    let s = Selection::Repeat("a", 3);
    assert_eq!(s.len(), 3);
    assert_eq!(
        s.iter().unwrap().collect::<Vec<_>>(),
        vec![&"a", &"a", &"a"]
    );
}

/// Describes the internal degrees of freedom of a system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum DegreesOfFreedom {
    /// All degrees of freedom are free
    #[default]
    Free,
    /// All degrees of freedom are frozen
    Frozen,
    /// Rigid body where only rotations and translations are free
    Rigid,
    /// Rigid body where alchemical degrees of freedom are free
    RigidAlchemical,
}

/// A topology is a collection of atoms, residues, bonds, etc.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Topology {
    /// List of all possible atom types.
    ///
    /// Atoms are identified either by their name or by their index in this list.
    atoms: Vec<AtomKind>,
    /// List of all possible residue types.
    ///
    /// Residues are identified either by their name or by their index in this list.
    residues: Vec<ResidueKind>,

    /// List of all possible chain types.
    ///
    /// Chains are identified either by their name or by their index in this list.
    chains: Vec<ChainKind>,
}

impl Topology {
    /// List of all possible atom types.
    ///
    /// Atoms are identified either by their name or by their index in this list.
    pub fn atoms(&self) -> &[AtomKind] {
        self.atoms.as_slice()
    }

    /// Find atom with given name
    ///
    /// # Examples
    /// ~~~
    /// use faunus::topology::*;
    /// let mut top = Topology::default();
    /// top.add_atom(AtomKind::new("Au")).unwrap();
    /// assert_eq!(top.find_atom("Au").unwrap().id, 0);
    /// assert_eq!(top.find_atom("Pb"), None);
    /// ~~~
    pub fn find_atom(&self, name: &str) -> Option<&AtomKind> {
        self.atoms.iter().find(|a| a.name == name)
    }

    /// Find residue with given name
    pub fn find_residue(&self, name: &str) -> Option<&ResidueKind> {
        self.residues.iter().find(|r| r.name == name)
    }

    /// List of all possible residue types.
    ///
    /// Residues are identified either by their name or by their index in this list.
    pub fn residues(&self) -> &[ResidueKind] {
        self.residues.as_slice()
    }
    /// List of all possible chain types.
    ///
    /// Chains are identified either by their name or by their index in this list.
    pub fn chains(&self) -> &[ChainKind] {
        self.chains.as_slice()
    }

    /// Append a new atom.
    ///
    /// The `id` will be overwritten with the last index in the atom list.
    /// Will error if the atom name already exists.
    pub fn add_atom(&mut self, atom: AtomKind) -> anyhow::Result<()> {
        // Ensure that the atom name does not already exist
        if self.atoms.iter().any(|a| a.name == atom.name) {
            anyhow::bail!("Atom with name '{}' already exists", atom.name);
        }
        self.atoms.push(atom);
        let index = self.atoms.len() - 1;
        self.atoms.last_mut().unwrap().set_id(index);
        Ok(())
    }

    /// Append a new residue.
    ///
    /// The `id` will be set to the last index in the residue list.
    /// Will error if:
    /// 1. the residue name already exists.
    /// 2. if any of the atom ids are not defined in the [`Topology::atoms()`] list.
    pub fn add_residue(&mut self, residue: ResidueKind) -> anyhow::Result<()> {
        // Ensure that the residue name does not already exist
        if self.residues.iter().any(|r| r.name == residue.name) {
            anyhow::bail!("Residue with name '{}' already exists", residue.name);
        }
        // Ensure that the atom ids are valid
        if residue.atoms.iter().any(|i| i >= &self.atoms.len()) {
            anyhow::bail!("Atom id in residue '{}' is out of bounds", residue.name);
        }
        self.residues.push(residue);
        let index = self.residues.len() - 1;
        self.residues.last_mut().unwrap().set_id(index);
        Ok(())
    }
}

/// Connectivity information such as bonds, dihedrals, etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct Connectivity {
    /// Bonds between atoms
    #[serde(default)]
    bonds: Vec<Bond>,
    /// Dihedrals
    #[serde(default)]
    dihedrals: Vec<Dihedral>,
    /// Dihedrals between bonds
    #[serde(default)]
    torsions: Vec<Torsion>,
}

impl Connectivity {
    /// Bonds
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }

    /// Add a new bond.
    pub fn add_bond(&mut self, bond: &Bond) {
        self.bonds.push(bond.clone())
    }

    /// Diherals
    pub fn dihedrals(&self) -> &[Dihedral] {
        &self.dihedrals
    }

    /// Add a new dihedral
    pub fn add_dihedral(&mut self, dihedral: &Dihedral) {
        self.dihedrals.push(dihedral.clone())
    }

    /// Torsions
    pub fn torsions(&self) -> &[Torsion] {
        &self.torsions
    }

    /// Add a new torsion
    pub fn add_torsion(&mut self, torsion: &Torsion) {
        self.torsions.push(torsion.clone())
    }

    /// Find dihedrals based on bonds
    pub fn find_dihedrals(&mut self) -> Vec<&Dihedral> {
        todo!()
    }

    /// Shift all indices by a given offset
    pub fn shift(&mut self, offset: isize) {
        for bond in &mut self.bonds {
            bond.shift(offset);
        }
        for dihedral in &mut self.dihedrals {
            dihedral.shift(offset);
        }
        for torsion in &mut self.torsions {
            torsion.shift(offset);
        }
    }
}

/// Serialize std::ops::Range as an array.
fn serialize_range_as_array<S>(
    range: &std::ops::Range<usize>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    [range.start, range.end].serialize(serializer)
}

/// Deserialize std::ops::Range from an array.
/// This allows the range to be defined as `[start, end]`.
fn deserialize_range_from_array<'de, D>(deserializer: D) -> Result<std::ops::Range<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let arr: [usize; 2] = Deserialize::deserialize(deserializer)?;
    core::result::Result::Ok(std::ops::Range {
        start: arr[0],
        end: arr[1],
    })
}

/// Check that all items of a collection are unique.
///
/// ## Parameters
/// - `collection` collection of items to compare
/// - `compare_fn` function/closure used for comparing the items
fn are_unique<T, F>(collection: &[T], compare_fn: F) -> bool
where
    F: Fn(&T, &T) -> bool,
{
    !collection.iter().enumerate().any(|(i, item_i)| {
        collection
            .iter()
            .skip(i + 1)
            .any(|item_j| compare_fn(item_i, item_j))
    })
}

/// Validate that the provided atom indices are unique.
/// Used e.g. to validate that a bond does not connect one and the same atom.
fn validate_unique_indices(indices: &[usize]) -> Result<(), ValidationError> {
    if are_unique(indices, |i: &usize, j: &usize| i == j) {
        core::result::Result::Ok(())
    } else {
        Err(ValidationError::new("non-unqiue atom indices"))
    }
}

/*
/// See stackoverflow workaround: <https://stackoverflow.com/questions/61446984/impl-iterator-failing-for-iterator-with-multiple-lifetime-parameters>
pub(crate) trait Captures<'a> {}
impl<'a, T: ?Sized> Captures<'a> for T {}

impl<'a> Topology<'a> {
    /// Find atom type by name in the list of atom types
    pub fn atom_kind(&self, name: &str) -> Option<&AtomKind> {
        self.atom_kinds.iter().find(|at| at.name() == name)
    }
    /// Find residue type by name in the list of residue types
    pub fn residue_kind(&self, name: &str) -> Option<&ResidueKind> {
        self.residue_kinds.iter().find(|rt| rt.name() == name)
    }
    /// Check if residue type and atom types are defined before adding to topology
    fn check_residue(&self, residue: &Residue) -> anyhow::Result<()> {
        // check if residue type is defined
        if self.atom_kind(residue.kind().name()).is_none() {
            anyhow::bail!(
                "Residue type '{}' is not defined in topology",
                residue.kind().name()
            );
        }
        // check of atom types are defined
        for atom in residue.atoms().iter() {
            if self.atom_kind(atom.kind().name()).is_none() {
                anyhow::bail!(
                    "Atom type '{}' is not defined in topology",
                    atom.kind().name()
                );
            }
        }
        Ok(())
    }

    /// Add a new residue to the system
    ///
    /// This will automatically:
    /// 1. Set the `Residue::index` to the last index in the list
    /// 2. Set all `Atom::index()` to the absolute positions in the system atom list
    /// 3. Copy bonds from the `ResidueType` if not already set
    /// 4. Shift all bonds to match the new atom indices.
    pub fn add_residue(&mut self, residue: Residue<'a>) -> anyhow::Result<()> {
        self.check_residue(&residue)?;
        let first_atom = self.len();
        let res_index = self.residues.len();
        self.residues.push(residue);
        let residue = self.residues.last_mut().unwrap();
        residue.set_index(res_index);
        residue.shift_indices(first_atom);
        Ok(())
    }
    /// All residues in the system
    pub fn residues(&self) -> &[Residue] {
        &self.residues
    }
    /// All residues in the system
    pub fn residues_mut(&mut self) -> &mut [Residue<'a>] {
        &mut self.residues
    }
    /// All atoms in the system
    pub fn atoms(&self) -> impl Iterator<Item = &Atom> {
        self.residues.iter().flat_map(|r| r.atoms())
    }
    /// All atoms in the system
    pub fn atoms_mut<'b>(&'b mut self) -> impl Iterator<Item = &mut Atom> + Captures<'a> + 'b {
        self.residues.iter_mut().flat_map(|r| r.atoms_mut())
    }
    /// List of all bonds in the system (intra- and inter-residue)
    pub fn bonds(&self) -> impl Iterator<Item = &Bond> {
        self.residues
            .iter()
            .flat_map(|r| r.bonds())
            .chain(self.inter_residue_bonds.iter())
    }
    /// Total number of atoms in the system
    pub fn len(&self) -> usize {
        self.residues.iter().map(|r| r.len()).sum()
    }
    /// Check if system is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a new atom type or "kind" to the system
    ///
    /// The atom type will be assigned a unique `AtomType::id()`, starting from 0 and increasing by 1 for each new atom type.
    /// Will fail if an atom type with the same name already exists.
    pub fn add_atom_kind(&mut self, atom_type: AtomKind) -> anyhow::Result<()> {
        if self
            .atom_kinds
            .iter()
            .any(|at| at.name() == atom_type.name())
        {
            anyhow::bail!("Atom type with name '{}' already exists", atom_type.name());
        }
        self.atom_kinds.push(atom_type);
        let len = self.atom_kinds.len();
        self.atom_kinds.last_mut().unwrap().set_id(len - 1);
        Ok(())
    }

    /// Add a new kind of residue to the system
    ///
    /// The residue type or "kind" will be assigned a unique `ResidueType::id()`, starting from 0 and increasing by 1 for each new residue type.
    /// Will fail if a residue type with the same name already exists.
    pub fn add_residue_kind(&mut self, kind: ResidueKind) -> anyhow::Result<()> {
        if self.residue_kinds.iter().any(|rt| rt.name() == kind.name()) {
            anyhow::bail!("Residue type with name '{}' already exists", kind.name());
        }
        self.residue_kinds.push(kind);
        let len = self.residue_kinds.len();
        self.residue_kinds.last_mut().unwrap().set_id(len - 1);
        Ok(())
    }
    /// List of all possible atom types in the system
    pub fn atom_types(&self) -> &[AtomKind] {
        &self.atom_kinds
    }
    /// List of all possible residue types in the system
    pub fn residue_types(&self) -> &[ResidueKind] {
        &self.residue_kinds
    }
}

#[test]
/// Test adding a new atom type
fn test_add_atom_kind() {
    let mut top = Topology::default();
    let atom_type = AtomKind::new("C");
    top.add_atom_kind(atom_type.clone()).unwrap();
    assert_eq!(top.atom_types(), &[atom_type]);
    assert!(top.atom_kind("C").is_some());
    assert_eq!(top.atom_kind("C").unwrap().id(), 0);
    top.add_atom_kind(AtomKind::new("O")).unwrap();
    assert_eq!(top.atom_types().len(), 2);
    assert_eq!(top.atom_kind("O").unwrap().id(), 1);
}

#[test]
fn test_add_residue_kind() {
    let mut top = Topology::default();
    let residue_type = ResidueKind::new("ALA", Selection::Vec(vec!["CA".to_string()]), None, None);
    top.add_residue_kind(residue_type.clone()).unwrap();
    assert_eq!(top.residue_types(), &[residue_type]);
    assert!(top.residue_kind("ALA").is_some());
    assert_eq!(top.residue_kind("ALA").unwrap().id(), 0);
    top.add_residue_kind(ResidueKind::new(
        "GLY",
        Selection::Vec(vec!["CA".to_string()]),
        None,
        None,
    ))
    .unwrap();
    assert_eq!(top.residue_types().len(), 2);
    assert_eq!(top.residue_kind("GLY").unwrap().id(), 1);
}

impl core::convert::From<chemfiles::Topology> for Topology<'_> {
    fn from(value: chemfiles::Topology) -> Self {
        let mut _atom_types: Vec<AtomKind> = (0..value.size())
            .map(|i| value.atom(i))
            .unique_by(|atom| atom.name())
            .map(|atom| atom.into())
            .collect();

        for (i, atom) in _atom_types.iter_mut().enumerate() {
            atom.set_id(i);
        }

        // let mut _atoms: Vec<Atom> = (0..value.size())
        //     .map(|i| value.atom(i))
        //     .map(|atom| Atom::new(atom.into())).collect();
        // for (i, atom) in _atoms.iter_mut().enumerate() {
        //     atom.set_index(i);
        //     let atom_kind = _atom_types.iter().find(|at| at.name() == atom.kind().name()).unwrap();
        //     atom.kind().
        // }

        let mut _residue_types: Vec<ResidueKind> = (0..value.residues_count())
            .map(|i| value.residue(i).unwrap())
            .unique_by(|residue| residue.name())
            .map(|residue| residue.into())
            .collect();

        for (i, residue) in _residue_types.iter_mut().enumerate() {
            residue.set_id(i);
        }

        let _bonds: Vec<Bond> = value
            .bonds()
            .iter()
            .map(|bond| Bond::new([bond[0], bond[1]], BondKind::None, BondOrder::None))
            .collect();

        unimplemented!()
    }
}

*/
