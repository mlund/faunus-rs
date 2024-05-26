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
pub(crate) mod block;
mod bond;
mod chain;
#[cfg(feature = "chemfiles")]
pub mod chemfiles_interface;
mod dihedral;
mod molecule;
mod residue;
mod structure;
mod torsion;
use std::ffi::OsString;
use std::fmt::Debug;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::Ok;
pub use atom::*;
pub use bond::*;
pub use chain::*;
use derive_getters::Getters;
pub use dihedral::*;
use itertools::Itertools;
use rand::rngs::ThreadRng;
pub use residue::*;
pub use torsion::*;
use validator::{Validate, ValidationError};

use crate::{Context, Point};
use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};

use self::block::{InsertionPolicy, MoleculeBlock};
use self::molecule::MoleculeKind;
use self::structure::positions_from_structure_file;

/// Trait implemented by collections of atoms that should not overlap (e.g., residues, chains).
pub(super) trait IndexRange {
    /// Get the indices of atoms in the collection.
    fn range(&self) -> Range<usize>;

    /// Check if elements are in union / shared with `other` range
    fn is_union(&self, other: &Self) -> bool
    where
        Self: Sized,
    {
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

    /// Validate that ranges in a list do not overlap.
    fn validate(collection: &[impl IndexRange]) -> Result<(), ValidationError>
    where
        Self: Sized,
    {
        let overlap = collection
            .iter()
            .permutations(2)
            .any(|v| v[0].is_union(v[1]));
        if overlap {
            Err(ValidationError::new("").with_message("overlap between collections".into()))
        } else {
            core::result::Result::Ok(())
        }
    }
}

#[test]
fn collections_overlap() {
    let residue1 = Residue::new("ALA", None, 2..5);
    let residue2 = Residue::new("LYS", None, 7..11);
    assert!(!residue1.is_union(&residue2));

    let residue2 = Residue::new("LYS", None, 5..11);
    assert!(!residue1.is_union(&residue2));

    let residue2 = Residue::new("LYS", None, 0..2);
    assert!(!residue1.is_union(&residue2));

    let residue2 = Residue::new("LYS", None, 2..5);
    assert!(residue1.is_union(&residue2));

    let residue2 = Residue::new("LYS", None, 1..11);
    assert!(residue1.is_union(&residue2));

    let residue2 = Residue::new("LYS", None, 3..4);
    assert!(residue1.is_union(&residue2));

    let residue2 = Residue::new("LYS", None, 1..3);
    assert!(residue1.is_union(&residue2));

    let residue2 = Residue::new("LYS", None, 4..11);
    assert!(residue1.is_union(&residue2));

    let chain1 = Chain::new("A", 2..5);
    let chain2 = Chain::new("B", 7..11);
    assert!(!chain1.is_union(&chain2));

    let chain2 = Chain::new("B", 4..11);
    assert!(chain1.is_union(&chain2));
}

#[test]
fn collections_validate() {
    let residue1 = Residue::new("ALA", None, 2..5);
    let residue2 = Residue::new("GLY", None, 5..7);
    let residue3 = Residue::new("LYS", None, 7..11);
    let residue4 = Residue::new("ALA", None, 11..14);

    assert!(Residue::validate(&[
        residue1.clone(),
        residue2.clone(),
        residue3.clone(),
        residue4.clone()
    ])
    .is_ok());

    let residue1b = Residue::new("ALA", None, 2..6);
    assert!(Residue::validate(&[
        residue1b.clone(),
        residue2.clone(),
        residue3.clone(),
        residue4.clone()
    ])
    .is_err());

    let residue2b = Residue::new("GLY", None, 11..13);
    assert!(Residue::validate(&[
        residue1.clone(),
        residue2b.clone(),
        residue3.clone(),
        residue4.clone()
    ])
    .is_err());

    let residue3b = Residue::new("LYS", None, 0..4);
    assert!(Residue::validate(&[
        residue1.clone(),
        residue2.clone(),
        residue3b.clone(),
        residue4.clone()
    ])
    .is_err());

    let chain1 = Chain::new("A", 2..5);
    let chain2 = Chain::new("B", 5..7);
    let chain3 = Chain::new("C", 7..11);
    let chain4 = Chain::new("D", 11..14);

    assert!(Chain::validate(&[
        chain1.clone(),
        chain2.clone(),
        chain3.clone(),
        chain4.clone()
    ])
    .is_ok());

    let chain2b = Chain::new("B", 11..13);
    assert!(Chain::validate(&[
        chain1.clone(),
        chain2b.clone(),
        chain3.clone(),
        chain4.clone()
    ])
    .is_err());
}

/// Trait implemented by collections where atoms are provided as indices (e.g., bonds, torsions, dihedrals)
pub(super) trait Indexed {
    /// Get indices of atoms forming the collection.
    fn index(&self) -> &[usize];

    /// Check that all the indices are lower than the provided value.
    fn lower(&self, value: usize) -> bool {
        self.index().iter().all(|&index| index < value)
    }
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Copy)]
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
    fn atomkinds(&self) -> &[AtomKind];
    /// Add atom to the topology.
    fn add_atomkind(&mut self, atom: AtomKind);
    /// Get molecules of the topology.
    fn moleculekinds(&self) -> &[MoleculeKind];
    /// Add molecule to the topology.
    fn add_moleculekind(&mut self, molecule: MoleculeKind);

    /// Find atom with given name.
    fn find_atom(&self, name: &str) -> Option<&AtomKind> {
        self.atomkinds().iter().find(|a| a.name() == name)
    }

    /// Find molecule with given name.
    fn find_molecule(&self, name: &str) -> Option<&MoleculeKind> {
        self.moleculekinds().iter().find(|r| r.name() == name)
    }

    /// Add atom kinds into a topology. In case an AtomKind with the same name already
    /// exists in the Topology, it is NOT overwritten and a warning is raised.
    fn include_atoms(&mut self, atoms: &[AtomKind]) {
        for atom in atoms.iter() {
            if self.atomkinds().iter().any(|x| x.name() == atom.name()) {
                log::warn!(
                    "Atom kind '{}' redefinition in included topology.",
                    atom.name()
                )
            } else {
                self.add_atomkind(atom.clone());
            }
        }
    }

    /// Add molecule kinds into a toplogy. In case a MoleculeKind with the same name
    /// already exists in the Topology, it is NOT overwritten and a warning is raised.
    fn include_molecules(&mut self, molecules: Vec<MoleculeKind>) {
        for molecule in molecules.into_iter() {
            if self
                .moleculekinds()
                .iter()
                .any(|x| x.name() == molecule.name())
            {
                log::warn!(
                    "Molecule kind '{}' redefinition in included topology.",
                    molecule.name()
                )
            } else {
                self.add_moleculekind(molecule);
            }
        }
    }

    /// Read additional topologies into topology.
    ///
    /// ## Parameters
    /// - `parent_path` path to the directory containing the parent topology file
    /// - `topologies` paths to the topologies to be included (absolute or relative to the `parent_path`)
    fn include_topologies(&mut self, topologies: Vec<InputPath>) -> Result<(), anyhow::Error> {
        for file in topologies.iter() {
            let included_top = IncludedTopology::from_file(file.path().unwrap())?;
            self.include_atoms(&included_top.atomkinds);
            self.include_molecules(included_top.moleculekinds);
        }

        Ok(())
    }
}

/// Topology of the molecular system.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
pub struct Topology {
    /// Other yaml files that should be included in the topology.
    #[serde(skip_serializing, default)]
    include: Vec<InputPath>,
    /// All possible atom types.
    #[serde(default, rename = "atoms")] // can be defined in an include
    atomkinds: Vec<AtomKind>,
    /// All possible molecule types.
    #[serde(default, rename = "molecules")] // can be defined in an include
    moleculekinds: Vec<MoleculeKind>,
    /// Properties of the system.
    /// Must always be provided.
    #[validate(nested)]
    system: Option<System>,
}

impl Topology {
    pub fn system(&self) -> &System {
        &self.system.as_ref().unwrap()
    }
    /// Parse a yaml file as Topology.
    pub fn from_file(filename: impl AsRef<Path>) -> anyhow::Result<Topology> {
        let yaml = std::fs::read_to_string(&filename)?;
        let mut topology: Topology = serde_yaml::from_str(&yaml)?;

        let Some(ref system) = topology.system else {
            anyhow::bail!("missing field `system`");
        };
        if system.blocks.is_empty() {
            anyhow::bail!("missing field `blocks`");
        }

        // finalize includes
        for file in topology.include.iter_mut() {
            file.finalize(&filename);
        }

        // parse included files
        topology.include_topologies(topology.include.clone())?;

        topology.finalize_atoms()?;
        topology.finalize_molecules()?;
        topology.finalize_blocks(&filename)?;
        topology.validate_intermolecular()?;

        topology.validate()?;

        Ok(topology)
    }

    /// Get molecule blocks of the system.
    pub fn blocks(&self) -> &[MoleculeBlock] {
        &self.system().blocks
    }

    /// Get intermolecular bonded interactions of the system.
    pub fn intermolecular(&self) -> &IntermolecularBonded {
        &self.system().intermolecular
    }

    /// Get the total number of particules in the topology.
    pub fn num_particles(&self) -> usize {
        self.system()
            .blocks
            .iter()
            .map(|block| block.num_atoms(&self.moleculekinds))
            .sum()
    }

    /// Create a new Topology structure. This function performs no sanity checks.
    #[allow(dead_code)]
    pub(crate) fn new(
        atoms: Vec<AtomKind>,
        molecules: Vec<MoleculeKind>,
        intermolecular: IntermolecularBonded,
        blocks: Vec<MoleculeBlock>,
    ) -> Topology {
        Topology {
            include: vec![],
            atomkinds: atoms,
            moleculekinds: molecules,
            system: Some(System {
                intermolecular,
                blocks,
            }),
        }
    }

    /// Create groups and populate target Context-implementing structure with particles.
    pub(crate) fn insert_groups(
        &self,
        context: &mut impl Context,
        structure: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<()> {
        // current index of the coordinate in the external structure file to use
        let mut curr_start = 0;

        let positions = match structure {
            Some(x) => Some(positions_from_structure_file(&x, Some(context.cell()))?),
            None => None,
        };

        // create groups
        for block in self.blocks() {
            if block.insert_policy().is_none() {
                let atoms_in_block = block.num_atoms(self.moleculekinds());

                match positions {
                    None => {
                        anyhow::bail!(
                            "molecule block requires external structure that was not provided"
                        )
                    }
                    Some(ref positions) => {
                        let positions = match positions
                            .get(curr_start..(curr_start + atoms_in_block))
                        {
                            None => anyhow::bail!("external structure does not match topology - not enough coordinates"),
                            Some(pos) => pos,
                        };

                        block.insert_block(context, positions, rng)?;

                        curr_start += atoms_in_block;
                    }
                };
            } else {
                block.insert_block(context, &[], rng)?;
            }
        }

        // check that all coordinates from the structure file have been used
        match positions {
            Some(positions) if positions.len() != curr_start => {
                anyhow::bail!("external structure does not match topology - too many coordinates")
            }
            _ => (),
        }

        Ok(())
    }

    /// Set ids for atom kinds in the topology and make sure that the atom names are unique.
    fn finalize_atoms(&mut self) -> anyhow::Result<()> {
        self.atomkinds
            .iter_mut()
            .enumerate()
            .for_each(|(i, atom): (usize, &mut AtomKind)| {
                atom.set_id(i);
            });

        if are_unique(&self.atomkinds, |i: &AtomKind, j: &AtomKind| {
            i.name() == j.name()
        }) {
            Ok(())
        } else {
            anyhow::bail!("atoms have non-unique names")
        }
    }

    /// Set ids for molecule kinds in the topology, validate the molecules and
    /// set indices of atom kinds forming each molecule.
    fn finalize_molecules(&mut self) -> anyhow::Result<()> {
        for (i, molecule) in self.moleculekinds.iter_mut().enumerate() {
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
                    self.atomkinds
                        .iter()
                        .position(|x| x.name() == atom)
                        .ok_or_else(|| anyhow::Error::msg("undefined atom kind in a molecule"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            molecule.set_atom_indices(indices);

            // expand exclusions (must be done after validation - the bonds must be valid)
            molecule.generate_exclusions();
        }

        // check that all molecule names are unique
        if are_unique(&self.moleculekinds, |i: &MoleculeKind, j: &MoleculeKind| {
            i.name() == j.name()
        }) {
            Ok(())
        } else {
            anyhow::bail!("molecules have non-unique names")
        }
    }

    /// Set molecule indices for blocks and validate them.
    fn finalize_blocks(&mut self, filename: impl AsRef<Path> + Clone) -> anyhow::Result<()> {
        for block in self.system.as_mut().unwrap().blocks.iter_mut() {
            block.finalize(filename.clone())?;

            let index = self
                .moleculekinds
                .iter()
                .position(|x| x.name() == block.molecule())
                .ok_or(anyhow::Error::msg("undefined molecule kind in a block"))?;
            block.set_molecule_id(index);

            // check that if positions are provided manually, they are consistent with the topology
            if let Some(InsertionPolicy::Manual(positions)) = block.insert_policy() {
                if positions.len() != block.num_atoms(&self.moleculekinds) {
                    anyhow::bail!(
                        "the number of manually provided positions does not match the number of atoms",
                    );
                }
            }
        }

        Ok(())
    }

    /// Validate intermolecular bonded interactions.
    fn validate_intermolecular(&mut self) -> anyhow::Result<()> {
        let num_particles = self.num_particles();

        #[inline(always)]
        fn check_intermolecular_items<T: Indexed>(
            items: &[T],
            num_particles: usize,
            error_msg: &'static str,
        ) -> anyhow::Result<()> {
            if !items.iter().all(|item| item.lower(num_particles)) {
                anyhow::bail!(error_msg);
            }

            Ok(())
        }
        check_intermolecular_items(
            &self.system().intermolecular.bonds,
            num_particles,
            "intermolecular bond between undefined atoms",
        )?;
        check_intermolecular_items(
            &self.system().intermolecular.torsions,
            num_particles,
            "intermolecular torsion between undefined atoms",
        )?;
        check_intermolecular_items(
            &self.system().intermolecular.dihedrals,
            num_particles,
            "intermolecular dihedral between undefined atoms",
        )?;

        Ok(())
    }
}

impl TopologyLike for Topology {
    fn atomkinds(&self) -> &[AtomKind] {
        &self.atomkinds
    }

    fn moleculekinds(&self) -> &[MoleculeKind] {
        &self.moleculekinds
    }

    fn add_atomkind(&mut self, atom: AtomKind) {
        self.atomkinds.push(atom)
    }

    fn add_moleculekind(&mut self, molecule: MoleculeKind) {
        self.moleculekinds.push(molecule)
    }
}

/// Partial topology that can be included in other topology files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct IncludedTopology {
    /// Other yaml files that should be included in the topology.
    #[serde(skip_serializing)]
    #[serde(default)]
    include: Vec<InputPath>,
    /// All possible atom types.
    #[serde(default, rename = "atoms")]
    atomkinds: Vec<AtomKind>,
    /// All possible molecule types.
    #[serde(default, rename = "molecules")]
    moleculekinds: Vec<MoleculeKind>,
}

impl IncludedTopology {
    /// Parse a yaml file as IncludedTopology.
    fn from_file(filename: impl AsRef<Path> + Clone) -> anyhow::Result<IncludedTopology> {
        let yaml = std::fs::read_to_string(filename.clone())?;
        let mut topology = serde_yaml::from_str::<IncludedTopology>(&yaml)?;

        // parse included files
        for file in topology.include.iter_mut() {
            file.finalize(filename.clone());
        }
        topology.include_topologies(topology.include.clone())?;

        Ok(topology)
    }
}

impl TopologyLike for IncludedTopology {
    fn atomkinds(&self) -> &[AtomKind] {
        &self.atomkinds
    }

    fn moleculekinds(&self) -> &[MoleculeKind] {
        &self.moleculekinds
    }

    fn add_atomkind(&mut self, atom: AtomKind) {
        self.atomkinds.push(atom)
    }

    fn add_moleculekind(&mut self, molecule: MoleculeKind) {
        self.moleculekinds.push(molecule)
    }
}

/// Fields of the topology related to specific molecular system.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
struct System {
    /// Intermolecular bonded interactions.
    #[serde(default)]
    #[validate(nested)]
    intermolecular: IntermolecularBonded,
    /// Molecules of the system.
    /// Must always be provided.
    blocks: Vec<MoleculeBlock>,
}

/// Intermolecular bonded interactions. Global atom indices have to be provided.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate, Getters)]
pub struct IntermolecularBonded {
    /// Intermolecular bonds between the atoms.
    #[serde(default)]
    #[validate(nested)]
    bonds: Vec<Bond>,
    /// Intermolecular dihedrals.
    #[serde(default)]
    #[validate(nested)]
    dihedrals: Vec<Dihedral>,
    /// Intermolecular torsions.
    #[serde(default)]
    #[validate(nested)]
    torsions: Vec<Torsion>,
}

impl IntermolecularBonded {
    /// Create a new IntermolecularBonded structure. This function does not perform any sanity checks.
    #[allow(dead_code)]
    pub(crate) fn new(
        bonds: Vec<Bond>,
        dihedrals: Vec<Dihedral>,
        torsions: Vec<Torsion>,
    ) -> IntermolecularBonded {
        IntermolecularBonded {
            bonds,
            dihedrals,
            torsions,
        }
    }

    /// Returns `true` if the `IntermolecularBonded` structure is empty (contains no bonds, no torsions and no dihedrals).
    /// Otherwise returns `false`.
    pub(crate) fn is_empty(&self) -> bool {
        self.bonds.is_empty() && self.torsions.is_empty() && self.dihedrals.is_empty()
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
        Err(ValidationError::new("").with_message("non-unique atom indices".into()))
    }
}

/// Path to input topology or structure file.
#[derive(Debug, PartialEq, Clone)]
pub struct InputPath {
    /// Raw path to the input file. Treated either as absolute
    /// or as relative to the parent directory.
    /// Used to construct the `path`.
    raw_path: OsString,
    /// Absolute path to the input file.
    path: Option<PathBuf>,
}

impl InputPath {
    /// Create new InputPath.
    #[allow(dead_code)]
    pub(crate) fn new(raw_path: OsString, parent_file: impl AsRef<Path>) -> InputPath {
        let mut path = InputPath {
            raw_path,
            path: None,
        };

        path.finalize(parent_file);
        path
    }

    /// Get path to file.
    pub(crate) fn path(&self) -> Option<&PathBuf> {
        self.path.as_ref()
    }

    /// Convert the raw_path to absolute path to the file.
    fn finalize(&mut self, parent_file: impl AsRef<Path>) {
        let path = PathBuf::from(&self.raw_path);
        let parent_path = parent_file.as_ref().parent().unwrap();

        if path.is_absolute() {
            self.path = Some(path);
        } else {
            self.path = Some(parent_path.join(path));
        }
    }
}

impl Serialize for InputPath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.raw_path.to_str().unwrap())
    }
}

impl<'de> Deserialize<'de> for InputPath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let path: String = Deserialize::deserialize(deserializer)?;

        std::result::Result::Ok(InputPath {
            raw_path: path.into(),
            path: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use self::block::BlockActivationStatus;
    use crate::dimension::Dimension;
    use float_cmp::assert_approx_eq;
    use std::collections::{HashMap, HashSet};
    use unordered_pair::UnorderedPair;

    use super::*;

    /// Compare the fields of AtomKind with the expected values.
    fn compare_atom_kind(
        atom: &AtomKind,
        name: &str,
        id: usize,
        mass: f64,
        charge: f64,
        element: Option<&str>,
        sigma: Option<f64>,
        epsilon: Option<f64>,
        hydrophobicity: Option<Hydrophobicity>,
        custom: &HashMap<String, Value>,
    ) {
        assert_eq!(atom.name(), name);
        assert_eq!(atom.id(), id);
        assert_approx_eq!(f64, atom.mass(), mass);
        assert_approx_eq!(f64, atom.charge(), charge);
        assert_eq!(atom.element(), element);
        assert_eq!(atom.sigma(), sigma);
        assert_eq!(atom.epsilon(), epsilon);
        assert_eq!(atom.hydrophobicity(), hydrophobicity);

        compare_custom(atom.custom(), custom);
    }

    /// Compare two hashmaps of custom properties.
    fn compare_custom(custom1: &HashMap<String, Value>, custom2: &HashMap<String, Value>) {
        assert_eq!(custom1.len(), custom2.len());

        for (key1, val1) in custom1 {
            let val2 = custom2.get(key1).expect("Custom properties do not match.");
            match (val1, val2) {
                (Value::Bool(x), Value::Bool(y)) => assert_eq!(x, y),
                (Value::Int(x), Value::Int(y)) => assert_eq!(x, y),
                (Value::Float(x), Value::Float(y)) => assert_approx_eq!(f64, *x, *y),
                (Value::Point(x), Value::Point(y)) => {
                    assert_approx_eq!(f64, x[0], y[0]);
                    assert_approx_eq!(f64, x[1], y[1]);
                    assert_approx_eq!(f64, x[2], y[2]);
                }
                (Value::Vector(x), Value::Vector(y)) => {
                    assert_eq!(x.len(), y.len());

                    for (i, j) in x.iter().zip(y.iter()) {
                        assert_approx_eq!(f64, *i, *j);
                    }
                }
                _ => panic!("Custom properties do not match."),
            }
        }
    }

    /// Compare the fields of molecule kind.
    fn compare_molecule_kind(
        molecule: &MoleculeKind,
        name: &str,
        id: usize,
        atoms: &[&str],
        indices: &[usize],
        bonds: &[Bond],
        torsions: &[Torsion],
        dihedrals: &[Dihedral],
        excluded_neighbours: usize,
        exclusions: &HashSet<UnorderedPair<usize>>,
        dof: DegreesOfFreedom,
        atom_names: &[Option<&str>],
        residues: &[Residue],
        chains: &[Chain],
        com: bool,
        custom: &HashMap<String, Value>,
    ) {
        assert_eq!(molecule.name(), name);
        assert_eq!(molecule.id(), id);
        assert_eq!(molecule.atoms(), atoms);
        assert_eq!(molecule.atom_indices(), indices);
        assert_eq!(molecule.bonds(), bonds);
        assert_eq!(molecule.torsions(), torsions);
        assert_eq!(molecule.dihedrals(), dihedrals);
        assert_eq!(molecule.degrees_of_freedom(), dof);
        assert_eq!(molecule.atom_names().len(), atom_names.len());
        for (name1, name2) in molecule.atom_names().iter().zip(atom_names.iter()) {
            assert_eq!(name1.as_deref(), *name2);
        }
        assert_eq!(molecule.residues(), residues);
        assert_eq!(molecule.chains(), chains);
        assert_eq!(molecule.has_com(), com);
        assert_eq!(*molecule.excluded_neighbours(), excluded_neighbours);
        assert_eq!(molecule.exclusions(), exclusions);

        compare_custom(molecule.custom(), custom);
    }

    /// Compare the intermolecular bonded interactions.
    fn compare_intermolecular(
        intermolecular: &IntermolecularBonded,
        bonds: &[Bond],
        torsions: &[Torsion],
        dihedrals: &[Dihedral],
    ) {
        assert_eq!(intermolecular.bonds(), bonds);
        assert_eq!(intermolecular.torsions(), torsions);
        assert_eq!(intermolecular.dihedrals(), dihedrals);
    }

    /// Compare the fields of a molecule block.
    fn compare_block(
        block: &MoleculeBlock,
        molecule_name: &str,
        molecule_index: usize,
        number: usize,
        active: BlockActivationStatus,
        insert: Option<&InsertionPolicy>,
    ) {
        assert_eq!(block.molecule(), molecule_name);
        assert_eq!(block.molecule_index(), molecule_index);
        assert_eq!(block.num_molecules(), number);
        assert_eq!(block.active(), active);
        assert_eq!(block.insert_policy(), insert);
    }

    #[test]
    fn read_topology_pass() {
        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();

        assert_eq!(topology.atomkinds().len(), 5);

        compare_atom_kind(
            &topology.atomkinds()[0],
            "OW",
            0,
            16.0,
            -1.0,
            Some("O"),
            Some(3.4),
            Some(1.8),
            Some(Hydrophobicity::SurfaceTension(1.0)),
            &HashMap::new(),
        );

        compare_atom_kind(
            &topology.atomkinds()[1],
            "HW",
            1,
            1.0,
            0.0,
            None,
            Some(1.0),
            Some(0.5),
            None,
            &HashMap::new(),
        );

        compare_atom_kind(
            &topology.atomkinds()[2],
            "X",
            2,
            12.0,
            1.0,
            None,
            None,
            None,
            Some(Hydrophobicity::Hydrophilic),
            &HashMap::new(),
        );

        let custom = HashMap::from([("unused".to_owned(), Value::Bool(true))]);

        compare_atom_kind(
            &topology.atomkinds()[3],
            "O",
            3,
            16.0,
            0.0,
            None,
            None,
            None,
            None,
            &custom,
        );

        compare_atom_kind(
            &topology.atomkinds()[4],
            "C",
            4,
            12.0,
            0.0,
            None,
            None,
            None,
            None,
            &custom,
        );

        let atoms = ["OW", "HW", "HW", "HW", "OW", "OW", "OW"];
        let indices = [0, 1, 1, 1, 0, 0, 0];
        let bonds = [
            Bond::new(
                [0, 1],
                BondKind::Harmonic(interatomic::twobody::Harmonic::new(1.0, 100.0)),
                BondOrder::Single,
            ),
            Bond::new(
                [1, 2],
                BondKind::Morse(interatomic::twobody::Morse::new(1.0, 10.0, 100.0)),
                BondOrder::Unspecified,
            ),
            Bond::new([2, 3], BondKind::default(), BondOrder::Unspecified),
        ];
        let torsions = [
            Torsion::new(
                [2, 3, 4],
                TorsionKind::Cosine(interatomic::threebody::CosineTorsion::new(45.0, 50.0)),
            ),
            Torsion::new([1, 2, 3], TorsionKind::default()),
        ];
        let dihedrals = [
            Dihedral::new(
                [0, 1, 2, 3],
                DihedralKind::ImproperHarmonic(interatomic::fourbody::HarmonicDihedral::new(
                    90.0, 100.0,
                )),
                Some(0.5),
                Some(0.5),
            ),
            Dihedral::new([3, 4, 5, 6], DihedralKind::default(), None, None),
        ];
        let names = [
            Some("O1"),
            None,
            Some("H1"),
            Some("H2"),
            None,
            Some("O1"),
            Some("O2"),
        ];
        let residues = [
            Residue::new("ALA", Some(2), 0..3),
            Residue::new("GLY", None, 1..1),
            Residue::new("ALA", Some(4), 4..6),
        ];
        let chains = [Chain::new("A", 0..7), Chain::new("Chain2", 14..0)];
        let custom = HashMap::from([
            ("bool".to_owned(), Value::Bool(false)),
            ("int".to_owned(), Value::Int(13)),
            ("float".to_owned(), Value::Float(76.3)),
            (
                "vector".to_owned(),
                Value::Vector(vec![13.1, 18.9, -13.4, 12.0]),
            ),
            ("point".to_owned(), Value::Point([1.4, 2.2, -0.71].into())),
        ]);

        assert_eq!(topology.moleculekinds().len(), 2);

        compare_molecule_kind(
            &topology.moleculekinds()[0],
            "MOL",
            0,
            &atoms,
            &indices,
            &bonds,
            &torsions,
            &dihedrals,
            1,
            &HashSet::from([
                UnorderedPair(0, 1),
                UnorderedPair(1, 2),
                UnorderedPair(2, 3),
                UnorderedPair(0, 4),
                UnorderedPair(5, 6),
            ]),
            DegreesOfFreedom::RigidAlchemical,
            &names,
            &residues,
            &chains,
            false,
            &custom,
        );

        compare_molecule_kind(
            &topology.moleculekinds()[1],
            "MOL2",
            1,
            &["OW", "OW", "X"],
            &[0, 0, 2],
            &[],
            &[],
            &[],
            0,
            &HashSet::new(),
            DegreesOfFreedom::Free,
            &[None, None, None],
            &[],
            &[],
            true,
            &HashMap::new(),
        );

        let bonds = vec![
            Bond::new(
                [0, 220],
                BondKind::Harmonic(interatomic::twobody::Harmonic::new(3.0, 50.0)),
                BondOrder::Unspecified,
            ),
            Bond::new(
                [52, 175],
                BondKind::FENE(interatomic::twobody::FENE::new(1.5, 5.0, 25.0)),
                BondOrder::Triple,
            ),
        ];
        let torsions = vec![Torsion::new(
            [1, 75, 128],
            TorsionKind::Harmonic(interatomic::threebody::HarmonicTorsion::new(120.0, 100.0)),
        )];
        let dihedrals = vec![
            Dihedral::new(
                [1, 35, 75, 128],
                DihedralKind::ProperHarmonic(interatomic::fourbody::HarmonicDihedral::new(
                    105.0, 27.5,
                )),
                None,
                Some(0.9),
            ),
            Dihedral::new([17, 45, 125, 215], DihedralKind::default(), None, None),
        ];

        compare_intermolecular(topology.intermolecular(), &bonds, &torsions, &dihedrals);

        compare_block(
            &topology.blocks()[0],
            "MOL",
            0,
            3,
            BlockActivationStatus::All,
            None,
        );

        compare_block(
            &topology.blocks()[1],
            "MOL2",
            1,
            50,
            BlockActivationStatus::Partial(30),
            Some(&InsertionPolicy::RandomCOM {
                filename: InputPath::new(
                    "mol2.xyz".to_owned().into(),
                    "tests/files/topology_pass.yaml",
                ),
                rotate: false,
                directions: Dimension::default(),
            }),
        );

        compare_block(
            &topology.blocks()[2],
            "MOL2",
            1,
            6,
            BlockActivationStatus::All,
            Some(&InsertionPolicy::RandomCOM {
                filename: InputPath::new(
                    "mol2.xyz".to_owned().into(),
                    "tests/files/topology_pass.yaml",
                ),
                rotate: true,
                directions: Dimension::X,
            }),
        );

        compare_block(
            &topology.blocks()[3],
            "MOL2",
            1,
            1,
            BlockActivationStatus::All,
            Some(&InsertionPolicy::Manual(vec![
                Point::from([1.43, 3.21, 2.65]),
                Point::from([0.65, 1.19, 2.34]),
                Point::from([2.1, 3.9, 0.8]),
            ])),
        );

        compare_block(
            &topology.blocks()[4],
            "MOL",
            0,
            2,
            BlockActivationStatus::All,
            Some(&InsertionPolicy::RandomAtomPos {
                directions: Dimension::XY,
            }),
        );

        compare_block(
            &topology.blocks()[5],
            "MOL2",
            1,
            5,
            BlockActivationStatus::All,
            Some(&InsertionPolicy::FromFile(InputPath::new(
                "mol2_absolute.xyz".to_owned().into(),
                "tests/files/topology_pass.yaml",
            ))),
        );
    }

    #[test]
    fn read_topology_fail_nonexistent_file() {
        let error = Topology::from_file("tests/files/this_file_does_not_exist.yaml").unwrap_err();
        match error.downcast_ref::<std::io::Error>().unwrap().kind() {
            std::io::ErrorKind::NotFound => (),
            _ => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn read_topology_fail_invalid_include() {
        let error = Topology::from_file("tests/files/topology_invalid_include.yaml").unwrap_err();
        match error.downcast_ref::<std::io::Error>().unwrap().kind() {
            std::io::ErrorKind::NotFound => (),
            _ => panic!("Incorrect error type returned."),
        }
    }

    #[test]
    fn read_topology_fail_missing_system() {
        let error = Topology::from_file("tests/files/topology_missing_system.yaml").unwrap_err();
        assert!(error.to_string().contains("missing field `system`"));
    }

    #[test]
    fn read_topology_fail_missing_blocks() {
        let res = Topology::from_file("tests/files/topology_missing_blocks.yaml");
        assert!(res.is_err());
    }

    #[test]
    fn read_topology_fail_nonunique_atoms() {
        let error = Topology::from_file("tests/files/topology_nonunique_atoms.yaml").unwrap_err();
        assert_eq!(&error.to_string(), "atoms have non-unique names");
    }

    #[test]
    fn read_topology_fail_nonunique_molecules() {
        let error =
            Topology::from_file("tests/files/topology_nonunique_molecules.yaml").unwrap_err();
        assert_eq!(&error.to_string(), "molecules have non-unique names");
    }

    #[test]
    fn read_topology_fail_nonexistent_atom() {
        let error = Topology::from_file("tests/files/topology_nonexistent_atom.yaml").unwrap_err();
        assert_eq!(&error.to_string(), "undefined atom kind in a molecule");
    }

    #[test]
    fn read_topology_fail_nonexistent_molecule() {
        let error =
            Topology::from_file("tests/files/topology_nonexistent_molecule.yaml").unwrap_err();
        assert_eq!(&error.to_string(), "undefined molecule kind in a block");
    }

    #[test]
    fn read_topology_fail_bond_undefined_atoms() {
        let error =
            Topology::from_file("tests/files/topology_bond_undefined_atoms.yaml").unwrap_err();
        assert!(error.to_string().contains("bond between undefined atoms"));
    }

    #[test]
    fn read_topology_fail_intermolecular_bond_undefined_atoms() {
        let error =
            Topology::from_file("tests/files/topology_intermolecular_bond_undefined_atoms.yaml")
                .unwrap_err();
        assert!(error.to_string().contains("bond between undefined atoms"));
    }

    #[test]
    fn read_topology_fail_torsion_undefined_atoms() {
        let error =
            Topology::from_file("tests/files/topology_torsion_undefined_atoms.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("torsion between undefined atoms"));
    }

    #[test]
    fn read_topology_fail_intermolecular_torsion_undefined_atoms() {
        let error =
            Topology::from_file("tests/files/topology_intermolecular_torsion_undefined_atoms.yaml")
                .unwrap_err();
        assert!(error
            .to_string()
            .contains("torsion between undefined atoms"));
    }

    #[test]
    fn read_topology_fail_dihedral_undefined_atoms() {
        let error =
            Topology::from_file("tests/files/topology_dihedral_undefined_atoms.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("dihedral between undefined atoms"));
    }

    #[test]
    fn read_topology_fail_intermolecular_dihedral_undefined_atoms() {
        let error = Topology::from_file(
            "tests/files/topology_intermolecular_dihedral_undefined_atoms.yaml",
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("dihedral between undefined atoms"));
    }

    #[test]
    fn read_topology_fail_residues_undefined_atoms() {
        let error =
            Topology::from_file("tests/files/topology_residues_undefined_atoms.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("residue contains undefined atoms"));
    }

    #[test]
    fn read_topology_fail_chains_undefined_atoms() {
        let error =
            Topology::from_file("tests/files/topology_chains_undefined_atoms.yaml").unwrap_err();
        assert!(error.to_string().contains("chain contains undefined atoms"));
    }

    #[test]
    fn read_topology_fail_residues_overlap() {
        let error = Topology::from_file("tests/files/topology_residues_overlap.yaml").unwrap_err();
        assert!(error.to_string().contains("overlap between collections"));
    }

    #[test]
    fn read_topology_fail_chains_overlap() {
        let error = Topology::from_file("tests/files/topology_chains_overlap.yaml").unwrap_err();
        assert!(error.to_string().contains("overlap between collections"));
    }

    #[test]
    fn read_topology_fail_too_few_names() {
        let error = Topology::from_file("tests/files/topology_too_few_names.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("the number of atom names does not match the number of atoms in a molecule"));
    }

    #[test]
    fn read_topology_fail_too_many_names() {
        let error = Topology::from_file("tests/files/topology_too_many_names.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("the number of atom names does not match the number of atoms in a molecule"));
    }

    #[test]
    fn read_topology_fail_bond_nonunique_atoms() {
        let error =
            Topology::from_file("tests/files/topology_bond_nonunique_atoms.yaml").unwrap_err();
        assert!(error.to_string().contains("non-unique atom indices"));
    }

    #[test]
    fn read_topology_fail_intermolecular_bond_nonunique_atoms() {
        let error =
            Topology::from_file("tests/files/topology_intermolecular_bond_nonunique_atoms.yaml")
                .unwrap_err();
        assert!(error.to_string().contains("non-unique atom indices"));
    }

    #[test]
    fn read_topology_fail_torsion_nonunique_atoms() {
        let error =
            Topology::from_file("tests/files/topology_torsion_nonunique_atoms.yaml").unwrap_err();
        assert!(error.to_string().contains("non-unique atom indices"));
    }

    #[test]
    fn read_topology_fail_intermolecular_torsion_nonunique_atoms() {
        let error =
            Topology::from_file("tests/files/topology_intermolecular_torsion_nonunique_atoms.yaml")
                .unwrap_err();
        assert!(error.to_string().contains("non-unique atom indices"));
    }

    #[test]
    fn read_topology_fail_dihedral_nonunique_atoms() {
        let error =
            Topology::from_file("tests/files/topology_dihedral_nonunique_atoms.yaml").unwrap_err();
        assert!(error.to_string().contains("non-unique atom indices"));
    }

    #[test]
    fn read_topology_fail_intermolecular_dihedral_nonunique_atoms() {
        let error = Topology::from_file(
            "tests/files/topology_intermolecular_dihedral_nonunique_atoms.yaml",
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-unique atom indices"));
    }

    #[test]
    fn read_topology_fail_block_too_many_active() {
        let error =
            Topology::from_file("tests/files/topology_block_too_many_active.yaml").unwrap_err();
        assert!(error.to_string().contains("the specified number of active molecules in a block is higher than the total number of molecules"))
    }

    #[test]
    fn read_topology_fail_block_too_few_manual_positions() {
        let error = Topology::from_file("tests/files/topology_block_too_few_manual_positions.yaml")
            .unwrap_err();
        assert_eq!(
            &error.to_string(),
            "the number of manually provided positions does not match the number of atoms"
        );
    }

    #[test]
    fn read_topology_fail_block_too_many_manual_positions() {
        let error =
            Topology::from_file("tests/files/topology_block_too_many_manual_positions.yaml")
                .unwrap_err();
        assert_eq!(
            &error.to_string(),
            "the number of manually provided positions does not match the number of atoms"
        );
    }

    #[test]
    fn read_topology_fail_atom_unknown_field() {
        let error =
            Topology::from_file("tests/files/topology_atom_unknown_field.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `nonexistent_field`"))
    }

    #[test]
    fn read_topology_fail_molecule_unknown_field() {
        let error =
            Topology::from_file("tests/files/topology_molecule_unknown_field.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `nonexistent_field`"))
    }

    #[test]
    fn read_topology_fail_block_unknown_field() {
        let error =
            Topology::from_file("tests/files/topology_block_unknown_field.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `nonexistent_field`"))
    }

    #[test]
    fn read_topology_fail_bond_unknown_field() {
        let error =
            Topology::from_file("tests/files/topology_bond_unknown_field.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `nonexistent_field`"))
    }

    #[test]
    fn read_topology_fail_torsion_unknown_field() {
        let error =
            Topology::from_file("tests/files/topology_torsion_unknown_field.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `nonexistent_field`"))
    }

    #[test]
    fn read_topology_fail_dihedral_unknown_field() {
        let error =
            Topology::from_file("tests/files/topology_dihedral_unknown_field.yaml").unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `nonexistent_field`"))
    }

    #[test]
    fn read_topology_fail_exclusions_nonunique_atoms() {
        let error = Topology::from_file("tests/files/topology_exclusions_nonunique_atoms.yaml")
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("exclusion between the same atom"))
    }

    #[test]
    fn read_topology_fail_exclusions_undefined_atoms() {
        let error = Topology::from_file("tests/files/topology_exclusions_undefined_atoms.yaml")
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("exclusion between undefined atoms"))
    }
}
