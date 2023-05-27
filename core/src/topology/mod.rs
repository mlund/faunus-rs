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

mod bond;
mod dihedral;
mod torsion;
use std::fmt::Debug;

pub use bond::*;
pub use dihedral::*;
pub use torsion::*;

use serde::{Deserialize, Serialize};

/// Enum to store a custom data for atoms, residues, molecules etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Property {
    Bool(bool),
    Int(i32),
    Float(f64),
    Vector(Vec<f64>),
}

/// Defines a unique atom type used to template atoms in a molecule. Each type is uniquely identified by its name and
/// accompaning identifier (`id`).
pub trait AtomType: Debug {
    /// Chemical symbol (He, C, O, Fe, etc.)
    fn element(&self) -> Option<String> {
        None
    }
    /// Unique name of atom type (opls_138, etc.)
    fn name(&self) -> String;
    /// Unique identifier of atom type
    fn id(&self) -> usize;
    /// Class of atom type (C138, etc.)
    fn class(&self) -> Option<String> {
        None
    }
    /// Atomic mass
    fn mass(&self) -> f64;
    /// Atomic charge
    fn charge(&self) -> f64;
    /// Lennard-Jones like well depth
    fn epsilon(&self) -> f64;
    /// Lennard-Jones like diameter
    fn sigma(&self) -> f64;
    /// Set a custom, named property
    fn set_property(&mut self, name: &str, property: Property) -> anyhow::Result<()>;
    /// Get named property
    fn get_property(&self, name: &str) -> Option<&Property>;
}

/// Various ways to specify a collection of atoms in e.g. a residue
pub enum AtomList<'a> {
    ById(Vec<usize>),
    ByType(Vec<Box<&'a dyn AtomType>>),
    ByRepeat(Box<&'a dyn AtomType>, usize),
}

impl AtomList<'_> {
    pub fn is_empty(&self) -> bool {
        match self {
            AtomList::ById(v) => v.is_empty(),
            AtomList::ByType(v) => v.is_empty(),
            AtomList::ByRepeat(_, n) => *n == 0,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            AtomList::ById(v) => v.len(),
            AtomList::ByType(v) => v.len(),
            AtomList::ByRepeat(_, n) => *n,
        }
    }
}

/// A residue is a collection of atoms that can represent a single molecule, or used to build up a larger molecule.
pub trait ResidueType {
    /// Name of residue (ALA, GLY, etc.)
    fn name(&self) -> String;
    /// Unique identifier of the residue type. This is typically an integer reflecting the insertion order of the atom type
    /// in the topology file, but the only requirement is that it is unique.
    fn id(&self) -> usize;
    /// Atoms in the residue
    fn atoms(&self) -> AtomList;
    /// Bonds between atoms in the residue. Indices are relative to the atoms in the residue.
    fn internal_bonds(&self) -> Vec<&Bond>;
    /// Short, one-letter code for residue (A, G, etc.). This follows the PDB standard.
    fn short_name(&self) -> Option<char> {
        residue_name_to_letter(&self.name())
    }
    /// One-letter chain identifier (A, B, etc.)
    fn chain(&self) -> Option<char> {
        None
    }
    /// Set a custom, named property
    fn set_property(&mut self, name: &str, property: Property) -> anyhow::Result<()>;
    /// Get named property
    fn get_property(&self, name: &str) -> Option<Property>;
    /// Number of atoms in residue
    fn len(&self) -> usize {
        self.atoms().len()
    }
    /// Check if residue is empty
    fn is_empty(&self) -> bool {
        self.atoms().is_empty()
    }
}

// pub struct Atom<'a> {
//     r#type: &'a dyn AtomType,
//     pub index: usize,
// }

// impl Atom<'_> {
//     pub fn new(r#type: &dyn AtomType, index: usize) -> Atom {
//         Atom { r#type, index }
//     }
//     pub fn get_type(&self) -> &dyn AtomType {
//         self.r#type
//     }
//     pub fn get_index(&self) -> usize {
//         self.index
//     }
// }

pub trait System {
    /// Unique atom types in the system
    fn atom_types(&self) -> Vec<Box<&dyn AtomType>>;
    /// Unique residue types in the system
    fn residue_types(&self) -> Vec<Box<&dyn ResidueType>>;
    /// All atoms in the system
    fn atoms(&self) -> Vec<Box<&dyn AtomType>>;
    /// All residues in the system
    fn residues(&self) -> Vec<Box<&dyn ResidueType>>;
    /// All bonds in the system. Automatically updated when adding residues.
    fn bonds(&self) -> Vec<&Bond>;
    /// All dihedrals in the system. Automatically updated when adding residues.
    fn dihedrals(&self) -> Vec<&Dihedral>;
    /// All torsions in the system. Automatically updated when adding residues.
    fn torsions(&self) -> Vec<&Torsion>;

    /// Add a new residue to the system. If the residue _type_ is new, it will be registered in `residue_types()` and
    /// new atom types will be registered in `atom_types()`.
    /// Bonds and dihedrals are automatically updated.
    fn add_residue(&mut self, residue: Box<&dyn ResidueType>) -> anyhow::Result<()>;
}

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
