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

pub mod atom;
mod bond;
pub mod chemfiles;
mod dihedral;
pub mod residue;
mod torsion;
use std::fmt::Debug;

use anyhow::Ok;
pub use bond::*;
pub use dihedral::*;
pub use torsion::*;

use crate::Point;
use serde::{Deserialize, Serialize};

/// Trait for identifying elements in a collection, e.g. index
pub trait Identity {
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

/// Enum to store hydrophobicity information of an atom or residue
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy)]
pub enum Hydrophobicity {
    Hydrophobic,
    Hydrophilic,
    /// Stores information about surface tension
    SurfaceTension(f64),
}

/// Information about atoms
///
/// Atoms need not be chemical elements, but can be custom atoms representing interaction sites.
/// The [`AtomType`] does _not_ include positions; indices etc., but is rather
/// used to represent static properties used for templating atoms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AtomType {
    /// Unique name
    name: String,
    /// Unique identifier
    id: usize,
    /// Atomic mass
    mass: f64,
    /// Atomic charge
    charge: f64,
    /// Atomic number
    atomic_number: Option<usize>,
    /// Atomic symbol (He, C, O, Fe, etc.)
    element: Option<String>,
    /// Lennard-Jones diameter
    sigma: Option<f64>,
    /// Lennaard-Jones well depth
    epsilon: Option<f64>,
    /// Hydrophobicity information
    hydrophobicity: Option<Hydrophobicity>,
    /// Map of custom properties
    properties: std::collections::HashMap<String, Value>,
}

impl AtomType {
    /// New atom type with given name but with otherwise default values
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }
    /// Unique identifier
    pub fn id(&self) -> usize {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl CustomProperty for AtomType {
    fn set_property(&mut self, key: &str, value: Value) -> anyhow::Result<()> {
        self.properties.insert(key.to_string(), value);
        Ok(())
    }
    fn get_property(&self, key: &str) -> Option<Value> {
        self.properties.get(key).cloned()
    }
}

/// An atom is the smallest particle entity.
///
/// It does not have to be a real chemical element, but can be a dummy or custom atom.
/// The type or name of the atom is given by an [`AtomType`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Atom {
    /// Unique index
    index: usize,
    /// Atom type
    kind: AtomType,
    /// Poisition of atom (x, y, z)
    pos: Point,
}

impl Atom {
    /// New atom with given name but with otherwise default values
    pub fn new(atomtype: AtomType) -> Self {
        Self {
            kind: atomtype,
            ..Default::default()
        }
    }
    /// Atom type
    pub fn kind(&self) -> &AtomType {
        &self.kind
    }
    /// Position of atom (x, y, z)
    pub fn pos(&self) -> &Point {
        &self.pos
    }
    /// Position of atom (x, y, z)
    pub fn pos_mut(&mut self) -> &mut Point {
        &mut self.pos
    }
    /// Unique identifier
    pub fn id(&self) -> usize {
        self.kind.id
    }
}

impl Identity for Atom {
    fn index(&self) -> usize {
        self.index
    }
    fn set_index(&mut self, index: usize) {
        self.index = index;
    }
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

/// Information about a residue
///
/// The [`ResidueType`] does _not_ include positions; indices etc., but is rather
/// used to represent static properties used for templating residues.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResidueType {
    /// Unique name, e.g. "GLU", "SOL", etc.
    name: String,
    /// Unique identifier
    id: usize,
    /// List of atom names in the residue
    atom_names: Selection<String>,
    /// Bonds between atoms in the residue. Indices are relative to the atoms in the residue.
    bonds: Vec<Bond>,
    /// Map of custom properties
    properties: std::collections::HashMap<String, Value>,
}

impl ResidueType {
    pub fn new(
        name: &str,
        atom_names: Selection<String>,
        id: Option<usize>,
        bonds: Option<&[Bond]>,
    ) -> Self {
        Self {
            name: name.to_string(),
            atom_names,
            id: id.unwrap_or(0),
            bonds: bonds.map(|b| b.to_vec()).unwrap_or_default(),
            properties: std::collections::HashMap::new(),
        }
    }
}

impl ResidueType {
    /// Short, one-letter code for residue (A, G, etc.). This follows the PDB standard.
    pub fn short_name(&self) -> Option<char> {
        residue_name_to_letter(&self.name)
    }
    /// Residue name
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Unique identifier
    pub fn id(&self) -> usize {
        self.id
    }
}

/// Atom and residue definitions used to build up a topology
pub struct BuildingBlocks {
    pub atom_types: Vec<AtomType>,
    pub residue_types: Vec<ResidueType>,
}

impl BuildingBlocks {
    pub fn find_atom_type(&self, name: &str) -> Option<&AtomType> {
        self.atom_types.iter().find(|at| at.name == name)
    }
    pub fn find_residue_type(&self, name: &str) -> Option<&ResidueType> {
        self.residue_types.iter().find(|rt| rt.name == name)
    }
    pub fn has_atom_type(&self, name: &str) -> bool {
        self.find_atom_type(name).is_some()
    }
    pub fn has_residue_type(&self, name: &str) -> bool {
        self.find_residue_type(name).is_some()
    }
}

/// Residue based on a [`ResidueType`] but with additional information such as atom positions; bonds; and indices relative to the whole system.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Residue {
    /// Index; automatically set when added to a topology
    index: usize,
    /// Residue type
    kind: ResidueType,
    /// List of atoms in the residue; indices are automatically set when added to a topology
    atoms: Vec<Atom>,
    /// List of bonds in the residue; indices are automatically set when added to a topology
    bonds: Vec<Bond>,
    dihedrals: Vec<Dihedral>,
    torsions: Vec<Torsion>,
}

impl Residue {
    /// New residue with given kind but with otherwise default values
    pub fn new(kind: ResidueType, atoms: &[Atom], bonds: Option<&[Bond]>) -> Self {
        Self {
            kind,
            atoms: atoms.to_vec(),
            index: 0,
            bonds: bonds.map(|b| b.to_vec()).unwrap_or_default(),
            dihedrals: Vec::new(),
            torsions: Vec::new(),
        }
    }
    /// Check if atom index is part of residue
    pub fn contains(&self, index: usize) -> bool {
        index.checked_sub(self.first_atom()).unwrap() < self.atoms.len()
    }

    /// Set bonds and ensure that they match to ones in the `ResidueType`
    pub fn set_bonds(&mut self, bonds: &[Bond]) -> anyhow::Result<()> {
        if self.kind.bonds.len() != bonds.len() {
            anyhow::bail!(
                "Number of bonds ({}) does not match number of bonds in residue type ({})",
                bonds.len(),
                self.kind.bonds.len()
            );
        }
        // find max index in bonds (relative to residue)
        let max_index: usize = bonds.iter().flat_map(|b| b.index).max().unwrap_or(0);
        if max_index.checked_sub(self.first_atom()).unwrap() >= self.atoms.len() {
            anyhow::bail!(
                "Bond index ({}) doesn't fit number of atoms in residue ({})",
                max_index,
                self.atoms.len()
            );
        }
        Ok(())
    }

    /// Residue type
    pub fn kind(&self) -> &ResidueType {
        &self.kind
    }
    /// Index of first atom in the residue
    pub fn first_atom(&self) -> usize {
        self.atoms.first().map(|a| a.index()).unwrap_or(0)
    }
    /// Number of atoms in the residue
    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    /// Check if residue is empty
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
    /// Unique identifier
    pub fn id(&self) -> usize {
        self.kind.id
    }
    /// List of atoms in the residue
    ///
    /// Each atom has a unique index relative to the sytem, i.e. an absolute index.
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }
    /// List of atoms in the residue
    ///
    /// Each atom has a unique index relative to the sytem, i.e. an absolute index.
    pub fn atoms_mut(&mut self) -> &mut [Atom] {
        &mut self.atoms
    }
    /// Bonds with absolute index, i.e. indices are relative to the whole system
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }
    /// Sets the index of the first atom in the residue
    ///
    /// This will set the correct bond indices (shifted by `first_index`) as well as
    /// make sure that all atom indices are correct.
    pub fn shift_indices(&mut self, first_index: usize) {
        // set atom indices
        self.atoms
            .iter_mut()
            .enumerate()
            .for_each(|(i, a)| a.set_index(i + first_index));
        assert_eq!(self.first_atom(), first_index);

        // if empty, copy bonds from residue type which acts as a template
        if self.bonds.is_empty() {
            self.bonds = self.kind.bonds.clone();
        }
        // shift bond indices
        for (bond, ref_bond) in std::iter::zip(self.bonds.iter_mut(), self.kind.bonds.iter()) {
            bond.index = ref_bond.shift_index(first_index as isize).index;
        }
    }
}

impl Identity for Residue {
    fn index(&self) -> usize {
        self.index
    }
    fn set_index(&mut self, index: usize) {
        self.index = index;
    }
}

/// A topology is a collection of atoms, residues, bonds, etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Topology {
    /// List of all possible atom types in the system
    ///
    /// Each type is uniquely numbered, starting from 0. Duplicate names are not allowed.
    atom_types: Vec<AtomType>,
    /// List of all possible residue types in the system
    ///
    /// Each type is uniquely numbered, starting from 0. Duplicate names are not allowed.
    residue_types: Vec<ResidueType>,
    /// List of all residues in the system
    residues: Vec<Residue>,
    /// Bonds between residues. Indices are relative to the residues in the system.
    inter_residue_bonds: Vec<Bond>,
}

impl Topology {
    /// Check if an atom type of given name exists
    fn atom_type_defined(&self, name: &str) -> bool {
        self.atom_types.iter().any(|at| at.name == name)
    }
    /// Check if a residue type of given name exists
    fn residue_type_defined(&self, name: &str) -> bool {
        self.residue_types.iter().any(|rt| rt.name == name)
    }
    /// Check if residue type and atom types are defined before adding to topology
    fn check_residue(&self, residue: &Residue) -> anyhow::Result<()> {
        // check if residue type is defined
        if !self.residue_type_defined(&residue.kind.name) {
            anyhow::bail!(
                "Residue type '{}' is not defined in topology",
                residue.kind.name
            );
        }
        // check of atom types are defined
        for atom in residue.atoms.iter() {
            if !self.atom_type_defined(&atom.kind.name) {
                anyhow::bail!("Atom type '{}' is not defined in topology", atom.kind.name);
            }
        }
        Ok(())
    }

    /// Add a new residue to the system
    ///
    /// This will automatically:
    /// 1. Set the [`Residue::index`] to the last index in the list
    /// 2. Set all [`Atom::index`] to the absolute positions in the system atom list
    /// 3. Copy bonds from the [`ResidueType`] if not already set
    /// 4. Shift all bonds to match the new atom indices.
    pub fn add_residue(&mut self, residue: Residue) -> anyhow::Result<()> {
        self.check_residue(&residue)?;
        let first_atom = self.len();
        self.residues.push(residue);
        let res_index = self.residues.len() - 1;
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
    pub fn residues_mut(&mut self) -> &mut [Residue] {
        &mut self.residues
    }
    /// All atoms in the system
    pub fn atoms(&self) -> impl Iterator<Item = &Atom> {
        self.residues.iter().flat_map(|r| r.atoms())
    }
    /// All atoms in the system
    pub fn atoms_mut(&mut self) -> impl Iterator<Item = &mut Atom> {
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

    /// Add a new atom type to the system
    ///
    /// The atom type will be assigned a unique [`AtomType::id()`], starting from 0 and increasing by 1 for each new atom type.
    /// Will fail if an atom type with the same name already exists.
    pub fn add_atom_type(&mut self, atom_type: AtomType) -> anyhow::Result<()> {
        if self.atom_types.iter().any(|at| at.name == atom_type.name) {
            anyhow::bail!("Atom type with name '{}' already exists", atom_type.name);
        }
        self.atom_types.push(atom_type);
        self.atom_types.last_mut().unwrap().id = self.atom_types.len() - 1;
        Ok(())
    }

    /// Add a new residue type to the system
    ///
    /// The residue type will be assigned a unique [`ResidueType::id()`], starting from 0 and increasing by 1 for each new residue type.
    /// Will fail if a residue type with the same name already exists.
    pub fn add_residue_type(&mut self, residue: Residue) -> anyhow::Result<()> {
        if self
            .residue_types
            .iter()
            .any(|rt| rt.name == residue.kind.name)
        {
            anyhow::bail!(
                "Residue type with name '{}' already exists",
                residue.kind.name
            );
        }
        self.residues.push(residue);
        self.residue_types.last_mut().unwrap().id = self.residue_types.len() - 1;
        Ok(())
    }
    /// List of all possible atom types in the system
    pub fn atom_types(&self) -> &[AtomType] {
        &self.atom_types
    }
    /// List of all possible residue types in the system
    pub fn residue_types(&self) -> &[ResidueType] {
        &self.residue_types
    }
}

/*
/// An atom is the smallest particle entity.
///
/// An atom can be a part of a molecule, or it can be a free atom.
/// It does not have to be a real chemical element, but can be a dummy or custom atom.
pub trait Atom: AtomType + Identity {
    /// Position of atom (x, y, z)
    fn pos(&self) -> Option<&Point> {
        None
    }
}

/// A residue is a continuous collection of atoms that can represent a single molecule, or used to build up a larger chain.
pub trait ResidueType: CustomProperty {
    /// Unique name for the residue type
    fn name(&self) -> String;
    /// Unique identifier for the residue type
    fn id(&self) -> usize;
    /// Short, one-letter code for residue (A, G, etc.). This follows the PDB standard.
    fn short_name(&self) -> Option<char> {
        residue_name_to_letter(&self.name())
    }
    /// List of atom ids in the residue
    fn atom_ids(&self) -> &[usize];
    /// Bonds between atoms in the residue. Indices are relative to the atoms in the residue.
    fn bonds(&self) -> &[Bond];
    /// Dihedrals between atoms in the residue. Indices are relative to the atoms in the residue.
    fn dihedrals(&self) -> &[Dihedral];
    /// Torsions between atoms in the residue. Indices are relative to the atoms in the residue.
    fn torsions(&self) -> &[Torsion];
    /// Check if residue is empty
    fn is_empty(&self) -> bool {
        self.atom_ids().is_empty()
    }
}

pub trait Residue<T: Atom>: ResidueType + Identity {
    /// Atoms in the residue matching [`ResidueType::atom_ids`]
    fn atoms(&self) -> &[T];
    /// Atoms in the residue matching [`ResidueType::atom_ids`]
    fn atoms_mut(&mut self) -> &mut [T];
}

pub struct Top<A: Atom, R: Residue<A>> {
    _t: core::marker::PhantomData<A>,
    /// All atoms in the system
    atoms: Vec<A>,
    residues: Vec<R>,
    inter_residue_bonds: Vec<Bond>,
}

impl<A: Atom, R: Residue<A>> Top<A, R> {
    pub fn residues(&self) -> &[R] {
        &self.residues
    }
    pub fn residues_mut(&mut self) -> &mut [R] {
        &mut self.residues
    }
    pub fn atoms(&self) -> impl Iterator<Item = &A> {
        self.residues.iter().flat_map(|r| r.atoms())
    }
    pub fn atoms_mut(&mut self) -> impl Iterator<Item = &mut A> {
        self.residues.iter_mut().flat_map(|r| r.atoms_mut())
    }
    /// List of all bonds in the system (intra- and inter-residue)
    pub fn bonds(&self) -> impl Iterator<Item = &Bond> {
        self.residues
            .iter()
            .flat_map(|r| r.bonds())
            .chain(self.inter_residue_bonds.iter())
    }
}

pub trait Topology<T: Atom, R: Residue<T>>: Debug {
    /// Residues in the system
    fn residues(&self) -> &[R];
    /// Bonds between residues. Indices are relative to the residues in the system.
    fn bonds(&self) -> &[Bond];
    fn len(&self) -> usize {
        self.residues().len()
    }
    /// Check if system is empty
    fn is_empty(&self) -> bool {
        self.residues().is_empty()
    }
}
*/

/// Function to convert an amino acid residue name to a one-letter code.
/// This follows the PDB standard and handles the 20 standard amino acids and nucleic acids (A, G, C, T, U).
fn residue_name_to_letter(name: &str) -> Option<char> {
    let letter = match name.to_uppercase().as_str() {
        // Amino acids
        "ALA" => 'A',
        "ARG" => 'R',
        "LYS" => 'K',
        "ASP" => 'D',
        "GLU" => 'E',
        "GLN" => 'Q',
        "ASN" => 'N',
        "HIS" => 'H',
        "TRP" => 'W',
        "PHE" => 'F',
        "TYR" => 'Y',
        "THR" => 'T',
        "SER" => 'S',
        "GLY" => 'G',
        "PRO" => 'P',
        "CYS" => 'C',
        "MET" => 'M',
        "VAL" => 'V',
        "LEU" => 'L',
        "ILE" => 'I',
        "MSE" => 'M',
        "UNK" => 'X',
        // DNA
        "DA" => 'A',
        "DG" => 'G',
        "DT" => 'T',
        "DC" => 'C',
        // RNA
        "A" => 'A',
        "G" => 'G',
        "U" => 'U',
        "C" => 'C',
        _ => return None,
    };
    Some(letter)
}

#[test]
fn test_info() {}
