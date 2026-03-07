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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    #[allow(dead_code)] // used in tests
    pub(crate) fn new(name: &str, number: Option<usize>, range: Range<usize>) -> Self {
        Self {
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
    pub const fn number(&self) -> Option<usize> {
        self.number
    }
}

impl crate::topology::IndexRange for Residue {
    #[inline(always)]
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// Convert an amino acid residue name to a one-letter code.
/// Follows the PDB standard for the 20 amino acids and nucleic acids (A, G, C, T, U).
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

/// Convert a FASTA one-letter code to a three-letter residue name.
///
/// Supports the 20 standard IUPAC amino acids (uppercase) and
/// Faunus-specific codes: `n` → NTR, `c` → CTR, `a` → ANK.
///
/// See [IUPAC-IUB 1983 nomenclature](https://doi.org/10.1111/j.1432-1033.1984.tb07877.x).
pub(crate) fn fasta_letter_to_residue_name(letter: char) -> Option<&'static str> {
    let name = match letter {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'E' => "GLU",
        'Q' => "GLN",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        // Faunus-specific
        'n' => "NTR",
        'c' => "CTR",
        'a' => "ANK",
        _ => return None,
    };
    Some(name)
}

/// Parse a FASTA sequence string into a vector of three-letter residue names.
///
/// Whitespace and newlines are skipped. A `*` terminates the sequence.
/// Returns an error if an unknown letter is encountered.
pub(crate) fn fasta_to_residue_names(sequence: &str) -> anyhow::Result<Vec<&'static str>> {
    let mut names = Vec::with_capacity(sequence.len());
    for ch in sequence.chars() {
        if ch == '*' {
            break;
        }
        if ch.is_whitespace() {
            continue;
        }
        let name = fasta_letter_to_residue_name(ch)
            .ok_or_else(|| anyhow::anyhow!("invalid FASTA letter '{ch}'"))?;
        names.push(name);
    }
    Ok(names)
}

/// Read sequence from a FASTA file, skipping header (`>`) and comment (`;`) lines.
///
/// Multiple sequences are concatenated. `*` terminates a sequence.
/// See [FASTA format](https://doi.org/10.1073/pnas.85.8.2444).
pub(crate) fn read_fasta_file(path: &std::path::Path) -> anyhow::Result<String> {
    use anyhow::Context;
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading FASTA file '{}'", path.display()))?;
    let is_sequence_line = |line: &&str| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('>') && !trimmed.starts_with(';')
    };
    Ok(content.lines().filter(is_sequence_line).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fasta_letter_roundtrip() {
        for letter in "ARNDCEQGHILKMFPSTWYV".chars() {
            let name = fasta_letter_to_residue_name(letter).unwrap();
            assert_eq!(residue_name_to_letter(name), Some(letter));
        }
    }

    #[test]
    fn test_fasta_to_residue_names() {
        let names = fasta_to_residue_names("nAGKc").unwrap();
        assert_eq!(names, vec!["NTR", "ALA", "GLY", "LYS", "CTR"]);
    }

    #[test]
    fn test_fasta_terminates_at_star() {
        let names = fasta_to_residue_names("AG*KK").unwrap();
        assert_eq!(names, vec!["ALA", "GLY"]);
    }

    #[test]
    fn test_fasta_skips_whitespace() {
        let names = fasta_to_residue_names("A G\nK").unwrap();
        assert_eq!(names, vec!["ALA", "GLY", "LYS"]);
    }

    #[test]
    fn test_fasta_invalid_letter() {
        assert!(fasta_to_residue_names("AXG").is_err());
    }

    #[test]
    fn test_read_fasta_file() {
        let path = std::path::Path::new("tests/files/test.fasta");
        let sequence = read_fasta_file(path).unwrap();
        assert_eq!(sequence, "DSHAKRHHGYKRKFHEKHHSHRGY");
    }

    #[test]
    fn test_read_fasta_file_not_found() {
        let path = std::path::Path::new("nonexistent.fasta");
        assert!(read_fasta_file(path).is_err());
    }
}
