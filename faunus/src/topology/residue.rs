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

use crate::topology::{bond::Bond, Connectivity, Value};
use serde::{Deserialize, Serialize};

use std::ops::Range;

use super::DegreesOfFreedom;

/// Non-overlapping collection of atoms with a non-unique name and number.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Residue {
    /// Residue name
    name: String,
    /// Residue number
    number: Option<usize>,
    /// Atoms forming the residue.
    /// Range of indices relating to the atoms of a molecule.
    #[serde(
        serialize_with = "crate::topology::serialize_range_as_array",
        deserialize_with = "crate::topology::deserialize_range_from_array"
    )]
    atoms: Range<usize>,
}

impl Residue {
    #[inline(always)]
    pub fn new(name: String, number: Option<usize>, atoms: Range<usize>) -> Self {
        Residue {
            name,
            number,
            atoms,
        }
    }

    #[inline(always)]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline(always)]
    pub fn number(&self) -> Option<usize> {
        self.number
    }
}

impl crate::topology::NonOverlapping for Residue {
    #[inline(always)]
    fn atoms(&self) -> &Range<usize> {
        &self.atoms
    }
}

/// Collection of connected atoms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ResidueKind {
    /// Unique name, e.g. _GLU_, _SOL_, etc.
    pub name: String,
    /// Unique identifier
    pub id: usize,
    /// List of atom ids in the residue
    pub atoms: Vec<usize>,
    /// Map of custom properties
    pub custom: std::collections::HashMap<String, Value>,
    /// Internal connections between atoms in the residue.
    pub connectivity: Connectivity,
    /// Internal degrees of freedom
    pub dof: DegreesOfFreedom,
}

impl ResidueKind {
    pub fn new(name: &str, atoms: &[usize]) -> Self {
        Self {
            name: name.to_string(),
            atoms: atoms.to_vec(),
            ..Default::default()
        }
    }

    /// Number of atoms in the residue
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Check if residue is empty
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Short, one-letter code for residue (A, G, etc.). This follows the PDB standard.
    pub fn short_name(&self) -> Option<char> {
        residue_name_to_letter(&self.name)
    }

    /// Set unique identifier
    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    /// Add bond between atoms
    pub fn add_bond(&mut self, bond: Bond) -> anyhow::Result<()> {
        if bond.index.iter().any(|i| i >= &self.len()) || bond.index[0] == bond.index[1] {
            anyhow::bail!("Invalid index in bond {:?} for residue {}", bond, self.name);
        }
        self.connectivity.bonds.push(bond);
        Ok(())
    }

    /// Append atom to residue
    pub fn add_atom(&mut self, atom: usize) {
        self.atoms.push(atom);
    }
}

// Convert a chemfiles residue to a topology residue
impl core::convert::From<chemfiles::ResidueRef<'_>> for ResidueKind {
    fn from(residue: chemfiles::ResidueRef) -> Self {
        ResidueKind {
            name: residue.name(),
            id: residue.id().unwrap() as usize,
            atoms: residue.atoms(),
            ..Default::default()
        }
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
