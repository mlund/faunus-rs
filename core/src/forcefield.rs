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

use float_cmp::approx_eq;
use serde::{Deserialize, Serialize};

/// Enum to store a custom data for atoms, residues, molecules etc.
pub enum Property {
    Bool(bool),
    Int(i32),
    Float(f64),
    Vector(Vec<f64>),
}

/// Variants of bond types, e.g. harmonic, FENE, Morse, etc.
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
    FENE(f64, f64, f64),
    /// Morse bond type (force constant, equilibrium distance, depth of potential well)
    Morse(f64, f64, f64),
    /// Harmonic Urey-Bradley bond type (force constant, equilibrium distance)
    /// See https://manual.gromacs.org/documentation/current/reference-manual/functions/bonded-interactions.html#urey-bradley-potential
    /// for more information.
    UreyBradley(f64, f64),
    /// Undefined bond type
    #[default]
    None,
}

/// Bond order decribing the multiplicity of a bond between two atoms.
/// See https://en.wikipedia.org/wiki/Bond_order for more information.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum BondOrder {
    #[default]
    None,
    Single,
    Double,
    Triple,
    Quadruple,
    Quintuple,
    Sextuple,
    Amide,
    Aromatic,
    Custom(f64),
}

impl From<BondOrder> for f64 {
    fn from(value: BondOrder) -> Self {
        match value {
            BondOrder::None => 0.0,
            BondOrder::Single => 1.0,
            BondOrder::Double => 2.0,
            BondOrder::Triple => 3.0,
            BondOrder::Quadruple => 4.0,
            BondOrder::Quintuple => 5.0,
            BondOrder::Sextuple => 6.0,
            BondOrder::Amide => 1.25,
            BondOrder::Aromatic => 1.5,
            BondOrder::Custom(value) => value,
        }
    }
}

impl From<f64> for BondOrder {
    fn from(value: f64) -> Self {
        match value {
            x if approx_eq!(f64, x, 0.0) => BondOrder::None,
            x if approx_eq!(f64, x, 1.0) => BondOrder::Single,
            x if approx_eq!(f64, x, 2.0) => BondOrder::Double,
            x if approx_eq!(f64, x, 3.0) => BondOrder::Triple,
            x if approx_eq!(f64, x, 4.0) => BondOrder::Quadruple,
            x if approx_eq!(f64, x, 5.0) => BondOrder::Quintuple,
            x if approx_eq!(f64, x, 6.0) => BondOrder::Sextuple,
            x if approx_eq!(f64, x, 1.25) => BondOrder::Amide,
            x if approx_eq!(f64, x, 1.5) => BondOrder::Aromatic,
            _ => BondOrder::Custom(value),
        }
    }
}

/// Describes a bond between two atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Bond {
    /// Indices of the two atoms in the bond
    pub index: [usize; 2],
    /// Kind of bond, e.g. harmonic, FENE, Morse, etc.
    pub kind: BondKind,
    /// Bond order
    pub order: BondOrder,
}

impl Bond {
    /// Create new bond
    pub fn new(index: [usize; 2], kind: BondKind, order: BondOrder) -> Self {
        Self { index, kind, order }
    }

    /// Check if bond contains atom with index
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }
}

/// Variants of bond angles, e.g. harmonic, cosine, etc.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum TorsionKind {
    /// Harmonic torsion (force constant, equilibrium angle)
    Harmonic(f64, f64),
    /// Cosine angle as used in e.g. GROMOS-96 (force constant, equilibrium angle)
    Cosine(f64, f64),
    /// Unspecified torsion type
    #[default]
    None,
}

/// Torsion potential between three atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Torsion {
    /// Indices of the three atoms in the angle
    pub index: [usize; 3],
    /// Kind of torsion, e.g. harmonic, cosine, etc.
    pub kind: TorsionKind,
}

impl Torsion {
    /// Create new torsion
    pub fn new(index: [usize; 3], kind: TorsionKind) -> Self {
        Self { index, kind }
    }

    /// Check if torsion contains atom with index
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }
}

/// Variants of dihedral angle potentials between four atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum DihedralKind {
    /// Harmonic dihedral type (force constant, equilibrium angle)
    Harmonic(f64, f64),
    /// Proper periodic dihedral type (force constant, periodicity, phase)
    ProperPeriodic(f64, f64, f64),
    /// Improper harmonic dihedral type (force constant, equilibrium angle)
    ImproperHarmonic(f64, f64),
    /// Amber-style improper torsion, where atom3 is the central atom bonded to atoms 1, 2, and 4.
    /// Atoms 1, 2, and 4 are only bonded to atom 3 in this instance.
    /// (force constant, periodicity, phase)
    ImproperAmber(f64, f64, f64),
    /// CHARMM-style improper torsion between 4 atoms. The first atom must be the central atom. (force constant, periodicity, phase)
    ImproperCHARMM(f64, f64, f64),
    /// Unspecified dihedral type
    #[default]
    None,
}

impl DihedralKind {
    /// Check if dihedral is improper
    pub fn is_improper(&self) -> bool {
        matches!(
            self,
            DihedralKind::ImproperHarmonic(..) | DihedralKind::ImproperAmber(..) | DihedralKind::ImproperCHARMM(..)
        )
    }
}

/// Valence dihedral between four atoms separated by three covalent bonds.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Dihedral {
    /// Indices of the four atoms in the dihedral.
    /// The indices are bonded as 1-2-3-4.
    pub index: [usize; 4],
    /// Kind of dihedral, e.g. harmonic, proper periodic, improper harmonic, etc.
    pub kind: DihedralKind,
    /// Optional 1-4 electrostatic scaling factor
    pub electrostatic_scaling: Option<f64>,
    /// Optional 1-4 Lennard-Jones scaling factor
    pub lj_scaling: Option<f64>,
}

impl Dihedral {
    /// Create new dihedral
    pub fn new(index: [usize; 4], kind: DihedralKind) -> Self {
        Self {
            index,
            kind,
            electrostatic_scaling: None,
            lj_scaling: None,
        }
    }

    /// Determines if the dihedral contains the given atom index
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }

    /// Determines if the dihedral is improper
    pub fn is_improper(&self) -> bool {
        self.kind.is_improper()
    }
}

/// Properties of an atom, sans positions.
/// This is used to generate atom types for force fields.
pub trait AtomProperty {
    /// Chemical symbol (He, C, O, Fe, etc.)
    fn element(&self) -> Option<String> {
        None
    }
    /// Name of atom type (opls_138, etc.)
    fn name(&self) -> String;
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

/// A residue is a collection of atoms that can represent a single molecule, or used to build up a larger molecule.
pub trait ResidueProperty {
    /// Name of residue (ALA, GLY, etc.)
    fn name(&self) -> String;
    /// Atoms in the residue
    fn atoms(&self) -> Vec<Box<&dyn AtomProperty>>;
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
