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
//! - [`MoleculeKind`] is a collection of atoms, e.g. a protein or a water molecule.
//! - [`MoleculeBlock`] is a collection of molecules of the same type.
//!
//! Topology is read from a file in yaml format using:
//! ```
//! # use faunus::topology::Topology;
//! let top = Topology::from_file("tests/files/topology_input.yaml");
//! ```
mod atom;
mod bond;
mod chain;
//pub mod chemfiles;
mod block;
mod dihedral;
mod molecule;
mod residue;
mod torsion;
use std::fmt::Debug;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::Ok;
pub use atom::*;
pub use bond::*;
pub use chain::*;
pub use dihedral::*;
pub use residue::*;
pub use torsion::*;
use validator::{Validate, ValidationError};

use crate::Point;
use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};

use self::block::{InsertionPolicy, MoleculeBlock};
use self::molecule::MoleculeKind;

/// Trait implemented by collections of atoms that should not overlap (e.g., residues, chains).
pub(super) trait NonOverlapping {
    /// Get the indices of atoms in the collection.
    fn range(&self) -> &Range<usize>;

    /// Check whether two collections overlap.
    ///
    /// ## Return
    /// Returns `true` if the collections overlap, else returns `false`.
    fn overlap(&self, other: &Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.range().start < other.range().end
            && other.range().start < self.range().end
    }

    /// Check whether the collection is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.range().is_empty()
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

/// Trait implemented by any structure resembling a Topology.
pub trait TopologyLike {
    /// Get atoms of the topology.
    fn atoms(&self) -> &[AtomKind];
    /// Add atom to the topology.
    fn add_atom(&mut self, atom: AtomKind);
    /// Get molecules of the topology.
    fn molecules(&self) -> &[MoleculeKind];
    /// Add molecule to the topology.
    fn add_molecule(&mut self, molecule: MoleculeKind);

    fn find_atom(&self, name: &str) -> Option<&AtomKind> {
        self.atoms().iter().find(|a| a.name() == name)
    }

    /// Find molecule with given name
    fn find_molecule(&self, name: &str) -> Option<&MoleculeKind> {
        self.molecules().iter().find(|r| r.name() == name)
    }

    /// Add atom kinds into a topology. In case an AtomKind with the same name already
    /// exists in the Topology, it is NOT overwritten and a warning is raised.
    fn include_atoms(&mut self, atoms: Vec<AtomKind>) {
        for atom in atoms.into_iter() {
            if self.atoms().iter().any(|x| x.name() == atom.name()) {
                log::warn!(
                    "Atom kind '{}' redefinition in included topology.",
                    atom.name()
                )
            } else {
                self.add_atom(atom);
            }
        }
    }

    /// Add molecule kinds into a toplogy. In case a MoleculeKind with the same name
    /// already exists in the Topology, it is NOT overwritten and a warning is raised.
    fn include_molecules(&mut self, molecules: Vec<MoleculeKind>) {
        for molecule in molecules.into_iter() {
            if self.molecules().iter().any(|x| x.name() == molecule.name()) {
                log::warn!(
                    "Molecule kind '{}' redefinition in included topology.",
                    molecule.name()
                )
            } else {
                self.add_molecule(molecule);
            }
        }
    }

    /// Read additional topologies into topology.
    ///
    /// ## Parameters
    /// - `parent_path` path to the directory containing the parent topology file
    /// - `topologies` paths to the topologies to be included (absolute or relative to the `parent_path`)
    fn include_topologies(
        &mut self,
        parent_path: impl AsRef<Path>,
        topologies: Vec<String>,
    ) -> Result<(), anyhow::Error> {
        for file in topologies.iter() {
            // if the path to the included directory is absolute, use it
            // otherwise consider it to be relative to the `parent_path`
            let mut path = PathBuf::from(file);
            if !path.is_absolute() {
                path = parent_path.as_ref().parent().unwrap().join(path);
            }

            let included_top = IncludedTopology::from_file(path)?;
            self.include_atoms(included_top.atoms);
            self.include_molecules(included_top.molecules);
        }

        Ok(())
    }
}

/// Topology of the molecular system.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Topology {
    /// Other yaml files that should be included in the topology.
    #[serde(skip_serializing, default)]
    include: Vec<String>,
    /// All possible atom types.
    #[serde(default)] // can be defined in an include
    atoms: Vec<AtomKind>,
    /// All possible molecule types.
    #[serde(default)] // can be defined in an include
    molecules: Vec<MoleculeKind>,
    /// Molecules of the system.
    /// Must always be provided.
    system: Vec<MoleculeBlock>,
}

impl Topology {
    /// Parse a yaml file as Topology.
    pub fn from_file(filename: impl AsRef<Path> + Clone) -> anyhow::Result<Topology> {
        let yaml = std::fs::read_to_string(filename.clone())?;
        let mut topology = serde_yaml::from_str::<Topology>(&yaml)?;

        // parse included files
        topology.include_topologies(filename, topology.include.clone())?;

        topology.finalize_atoms()?;
        topology.finalize_molecules()?;
        topology.finalize_blocks()?;

        Ok(topology)
    }

    /// Get molecule blocks of the system.
    pub fn system(&self) -> &[MoleculeBlock] {
        &self.system
    }

    /// Set ids for atom kinds in the topology and make sure that the atom names are unique.
    fn finalize_atoms(&mut self) -> anyhow::Result<()> {
        self.atoms
            .iter_mut()
            .enumerate()
            .for_each(|(i, atom): (usize, &mut AtomKind)| {
                atom.set_id(i);
            });

        if are_unique(&self.atoms, |i: &AtomKind, j: &AtomKind| {
            i.name() == j.name()
        }) {
            Ok(())
        } else {
            Err(anyhow::Error::msg("atoms have non-unique names"))
        }
    }

    /// Set ids for molecule kinds in the topology, validate the molecules and
    /// set indices of atom kinds forming each molecule.
    fn finalize_molecules(&mut self) -> anyhow::Result<()> {
        for (i, molecule) in self.molecules.iter_mut().enumerate() {
            // set atom names
            if molecule.atom_names().is_empty() {
                molecule.empty_atom_names();
            }

            // validate the molecule
            molecule.validate()?;

            // set index
            molecule.set_id(i);

            // set atom indices
            let indices = molecule
                .atoms()
                .iter()
                .map(|atom| {
                    self.atoms
                        .iter()
                        .position(|x| x.name() == atom)
                        .ok_or_else(|| anyhow::Error::msg("undefined atom kind in a molecule"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            molecule.set_atom_indices(indices);
        }

        // check that all molecule names are unique
        if are_unique(&self.molecules, |i: &MoleculeKind, j: &MoleculeKind| {
            i.name() == j.name()
        }) {
            Ok(())
        } else {
            Err(anyhow::Error::msg("molecules have non-unique names"))
        }
    }

    /// Set molecule indices for blocks and validate them.
    fn finalize_blocks(&mut self) -> anyhow::Result<()> {
        for block in self.system.iter_mut() {
            block.finalize()?;

            let index = self
                .molecules
                .iter()
                .position(|x| x.name() == block.molecule())
                .ok_or(anyhow::Error::msg("undefined molecule kind in a block"))?;
            block.set_molecule_index(index);

            // check that if positions are provided manually, they are consistent with the topology
            if let Some(InsertionPolicy::Manual(positions)) = block.insert() {
                if positions.len() != (*block.number() * self.molecules[index].atom_indices().len())
                {
                    return Err(anyhow::Error::msg(
                        "the number of manually provided positions does not match the number of atoms",
                    ));
                }
            }
        }

        Ok(())
    }
}

impl TopologyLike for Topology {
    fn atoms(&self) -> &[AtomKind] {
        &self.atoms
    }

    fn molecules(&self) -> &[MoleculeKind] {
        &self.molecules
    }

    fn add_atom(&mut self, atom: AtomKind) {
        self.atoms.push(atom)
    }

    fn add_molecule(&mut self, molecule: MoleculeKind) {
        self.molecules.push(molecule)
    }
}

/// Partial topology that can be included in other topology files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct IncludedTopology {
    /// Other yaml files that should be included in the topology.
    #[serde(skip_serializing)]
    #[serde(default)]
    include: Vec<String>,
    /// All possible atom types.
    #[serde(default)]
    atoms: Vec<AtomKind>,
    /// All possible molecule types.
    #[serde(default)]
    molecules: Vec<MoleculeKind>,
}

impl IncludedTopology {
    /// Parse a yaml file as IncludedTopology.
    fn from_file(filename: impl AsRef<Path> + Clone) -> anyhow::Result<IncludedTopology> {
        let yaml = std::fs::read_to_string(filename.clone())?;
        let mut topology = serde_yaml::from_str::<IncludedTopology>(&yaml)?;
        // parse included files
        topology.include_topologies(filename, topology.include.clone())?;

        Ok(topology)
    }
}

impl TopologyLike for IncludedTopology {
    fn atoms(&self) -> &[AtomKind] {
        &self.atoms
    }

    fn molecules(&self) -> &[MoleculeKind] {
        &self.molecules
    }

    fn add_atom(&mut self, atom: AtomKind) {
        self.atoms.push(atom)
    }

    fn add_molecule(&mut self, molecule: MoleculeKind) {
        self.molecules.push(molecule)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_topology() {
        let topology = Topology::from_file("tests/files/topology_input.yaml").unwrap();

        println!("{:?}\n", topology.atoms());
        println!("{:?}\n", topology.molecules());
        println!("{:?}\n", topology.system);

        //println!("{}", serde_yaml::to_string(&topology).unwrap());
    }
}
