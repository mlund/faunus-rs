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

use std::ops::Range;

/// Continuous range of atoms with a non-unique name and number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Residue {
    /// Residue name.
    name: String,
    /// Residue number.
    number: Option<usize>,
    /// Atoms indices forming the residue.
    /// Range of indices relating to the atoms of a molecule.
    #[serde(
        serialize_with = "crate::topology::serialize_range_as_array",
        deserialize_with = "crate::topology::deserialize_range_from_array"
    )]
    range: Range<usize>,
}

impl Residue {
    #[inline(always)]
    pub fn new(name: &str, number: Option<usize>, range: Range<usize>) -> Self {
        Residue {
            name: name.to_owned(),
            number,
            range,
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

impl crate::topology::IndexRange for Residue {
    #[inline(always)]
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// Function to convert an amino acid residue name to a one-letter code.
/// This follows the PDB standard and handles the 20 standard amino acids and nucleic acids (A, G, C, T, U).
#[allow(dead_code)]
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
