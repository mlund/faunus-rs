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

use serde::{Deserialize, Serialize};

/// Enum to store a bool, int, or float for atoms, molecules etc.
pub enum Property {
    Bool(bool),
    Int(i32),
    Float(f64),
    Vector(Vec<f64>),
}
/// Enum to store bond type, e.g. harmonic, FENE, Morse, etc.
/// Each varient stores the parameters for the bond type, like force constant, equilibrium distance, etc.
/// For more information see:
/// - Morse: https://en.wikipedia.org/wiki/Morse_potential
/// - Harmonic: https://en.wikipedia.org/wiki/Harmonic_oscillator
/// - FENE: https://en.wikipedia.org/wiki/Finitely_extensible_nonlinear_elastic_potential
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum BondKind {
    /// Harmonic bond type (force constant, equilibrium distance)
    Harmonic(f64, f64),
    /// Finite extensible nonlinear elastic bond type (force constant, equilibrium distance, maximum distance)
    FENE(f64, f64),
    /// Morse bond type (force constant, equilibrium distance, depth of potential well)
    Morse(f64, f64, f64),
    /// Undefined bond type
    #[default]
    None,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum BondOrder {
    None,
    Single,
    Double,
    Triple,
    Quadruple,
    Quintuplet,
    Amide,
    Aromatic,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Bond {
    /// Indices of the two atoms in the bond
    pub index: [usize; 2],
    /// Kind of bond, e.g. harmonic, FENE, Morse, etc.
    pub kind: BondKind,
}

impl Bond {
    /// Create new bond
    pub fn new(index: [usize; 2], kind: BondKind) -> Self {
        Self { index, kind }
    }
}

/// Angle potential between three atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Angle {
    /// Indices of the three atoms in the angle
    pub index: [usize; 3],
}

pub trait Atom {
    /// Chemical symbol (He, C, O, Fe, etc.)
    fn element(&self) -> Option<String> {
        None
    }
    /// Name of atom type (opls_138, etc.)
    fn name(&self) -> String;
    /// Class of atom type (C138, etc.)
    fn class(&self) -> String;
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

/// A residue is a collection of atoms and typically used to represent proteins, nucleic acids, etc.
pub trait Residue {
    /// Name of residue (ALA, GLY, etc.)
    fn name(&self) -> String;
    /// Atoms in the residue
    fn atoms(&self) -> Vec<Box<&dyn Atom>>;
    /// Bonds between atoms in the residue. Indices are relative to the atoms in the residue.
    fn bonds(&self) -> Vec<&Bond>;
    /// Angles between atoms in the residue. Indices are relative to the atoms in the residue.
    fn angles(&self) -> Vec<&Angle>;
    /// Short, one-letter code for residue (A, G, etc.). This follows the PDB standard.
    fn short_name(&self) -> Option<char> {
        match self.name().to_uppercase().as_str() {
            // Amino acids
            "ALA" => Some('A'),
            "ARG" => Some('R'),
            "LYS" => Some('K'),
            "ASP" => Some('D'),
            "GLU" => Some('E'),
            "GLN" => Some('Q'),
            "ASN" => Some('N'),
            "HIS" => Some('H'),
            "TRP" => Some('W'),
            "PHE" => Some('F'),
            "TYR" => Some('Y'),
            "THR" => Some('T'),
            "SER" => Some('S'),
            "GLY" => Some('G'),
            "PRO" => Some('P'),
            "CYS" => Some('C'),
            "MET" => Some('M'),
            "VAL" => Some('V'),
            "LEU" => Some('L'),
            "ILE" => Some('I'),
            "MSE" => Some('M'),
            "UNK" => Some('X'),
            // DNA
            "DA" => Some('A'),
            "DG" => Some('G'),
            "DT" => Some('T'),
            "DC" => Some('C'),
            // RNA
            "A" => Some('A'),
            "G" => Some('G'),
            "U" => Some('U'),
            "C" => Some('C'),
            _ => None,
        }
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

#[test]
fn test_info() {}
