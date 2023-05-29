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
//pub mod chemfiles;
mod dihedral;
mod torsion;
use std::fmt::Debug;

pub use bond::*;
pub use dihedral::*;
pub use torsion::*;

use crate::Point;
use serde::{Deserialize, Serialize};

/// Enum to store hydrophobicity information of an atom or residue
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy, Default)]
pub enum Hydrophobicity {
    Hydrophobic,
    Hydrophilic,
    /// Stores information about surface tension
    SurfaceTension(f64),
    /// Unknown hydrophobicity
    #[default]
    None,
}

/// Information about an atom type.
///
/// Atoms need not be chemical elements, but can be custom atoms representing interaction sites.
/// This does _not_ include:
/// - index or id information, other than the name.
/// - position information
pub trait AtomType: CustomProperty {
    /// Unique name for the atom type
    fn name(&self) -> String;
    /// Unique identifier for the atom type
    fn id(&self) -> usize;
    /// Atomic mass
    fn mass(&self) -> f64;
    /// Atomic charge
    fn charge(&self) -> f64;
    /// Atomic number
    fn atomic_number(&self) -> Option<usize> {
        None
    }
    /// Atomic symbol (He, C, O, Fe, etc.)
    fn element(&self) -> Option<String> {
        None
    }
    /// Lennard-Jones diameter and well-depth (sigma, epsilon)
    fn sigma_epsilon(&self) -> Option<(f64, f64)> {
        None
    }
    /// Hydrophobicity information
    fn hydrophobicity(&self) -> Hydrophobicity {
        Hydrophobicity::None
    }
    /// Intrinsic solvation radius
    fn solvent_radius(&self) -> Option<f64> {
        None
    }
    /// Generalized Born screening factor
    fn gb_screening(&self) -> Option<f64> {
        None
    }
    /// Generate new atom type
    fn new(name: &str, id: usize) -> Self;
}

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

/// A custom property for atoms, residues, chains etc.
pub trait CustomProperty {
    /// Set a custom, property associated with a unique `key`.
    ///
    /// The key could e.g. be a converted discriminant from a field-less enum.
    fn set_property(&mut self, key: &str, value: Value) -> anyhow::Result<()>;
    /// Get property assosiated with a `key`.
    fn get_property(&self, key: &str) -> Option<Value>;
}

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
