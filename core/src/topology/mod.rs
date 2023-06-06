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

mod atom;
mod bond;
//pub mod chemfiles;
mod dihedral;
mod residue;
mod torsion;
use std::fmt::Debug;

use anyhow::Ok;
pub use atom::*;
pub use bond::*;
pub use dihedral::*;
pub use residue::*;
pub use torsion::*;

use crate::Point;
use chemfiles;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// Trait for identifying elements in a collection, e.g. index
pub trait Indices {
    /// Positional index of element in a collection
    fn index(&self) -> usize;
    /// Set positional index of element in a collection
    fn set_index(&mut self, index: usize);
}

/// Enum to store custom data for atoms, residues, molecules etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i32),
    Float(f64),
    Vector(Vec<f64>),
    Point(Point),
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

/// A topology is a collection of atoms, residues, bonds, etc.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Topology<'a> {
    /// List of all possible atom types in the system
    ///
    /// Each type is uniquely numbered, starting from 0. Duplicate names are not allowed.
    atom_kinds: Vec<AtomKind>,
    /// List of all possible residue types in the system
    ///
    /// Each type is uniquely numbered, starting from 0. Duplicate names are not allowed.
    residue_kinds: Vec<ResidueKind>,
    /// List of all residues in the system
    residues: Vec<Residue<'a>>,
    /// Bonds between residues. Indices are relative to the residues in the system.
    inter_residue_bonds: Vec<Bond>,
}

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
