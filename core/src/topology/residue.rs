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

use crate::topology::{
    atom::Atom, bond::Bond, dihedral::Dihedral, torsion::Torsion, Indices, Value,
};
use chemfiles;
use serde::{Deserialize, Serialize};

/// Information about a residue
///
/// The `ResidueType` does _not_ include positions; indices etc., but is rather
/// used to represent static properties used for templating residues.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResidueKind {
    /// Unique name, e.g. "GLU", "SOL", etc.
    name: String,
    /// Unique identifier
    id: usize,
    /// List of atom names in the residue
    atom_names: super::Selection<String>,
    /// Bonds between atoms in the residue. Indices are relative to the atoms in the residue.
    bonds: Vec<Bond>,
    /// Map of custom properties
    properties: std::collections::HashMap<String, Value>,
}

impl ResidueKind {
    pub fn new(
        name: &str,
        atom_names: super::Selection<String>,
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
    /// Set unique identifier
    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }
}

/// Residue based on a `ResidueType` but with additional information such as atom positions; bonds; and indices relative to the whole system.
///
/// The `ResidueKind` is not owned by the `Residue` but is rather a reference to a `ResidueType` which acts as a template.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Residue<'a> {
    /// Index; automatically set when added to a topology
    index: usize,
    /// Residue type
    kind: &'a ResidueKind,
    /// List of atoms in the residue; indices are automatically set when added to a topology
    atoms: Vec<Atom>,
    /// List of bonds in the residue; indices are automatically set when added to a topology
    bonds: Vec<Bond>,
    dihedrals: Vec<Dihedral>,
    torsions: Vec<Torsion>,
}

impl<'a> Residue<'a> {
    /// New residue with given kind but with otherwise default values
    pub fn new(kind: &'a ResidueKind, atoms: &[Atom], bonds: Option<&[Bond]>) -> Self {
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
    pub fn kind(&self) -> &ResidueKind {
        self.kind
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

impl Indices for Residue<'_> {
    fn index(&self) -> usize {
        self.index
    }
    fn set_index(&mut self, index: usize) {
        self.index = index;
    }
}

/// Convert a chemfiles residue to a topology residue
impl core::convert::From<chemfiles::ResidueRef<'_>> for ResidueKind {
    fn from(residue: chemfiles::ResidueRef) -> Self {
        ResidueKind {
            name: residue.name(),
            id: residue.id().unwrap() as usize,
            atom_names: super::Selection::Ids(residue.atoms()),
            bonds: Default::default(),
            properties: Default::default(),
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
