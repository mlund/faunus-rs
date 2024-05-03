// Copyright 2023-2024 Mikael Lund
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

use std::collections::HashMap;

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use crate::topology::{Chain, DegreesOfFreedom, Residue, Value};
use validator::{Validate, ValidationError};

use super::{Bond, CustomProperty, Dihedral, Indexed, NonOverlapping, Torsion};

#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate, Getters)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_molecule"))]
pub struct MoleculeKind {
    /// Unique name.
    name: String,
    /// Unique identifier.
    /// Only defined if the MoleculeKind is inside of Topology.
    #[serde(skip)]
    #[getter(skip)]
    id: usize,
    /// Names of atom kinds forming the molecule.
    atoms: Vec<String>,
    /// Indices of atom kinds forming the molecule.
    /// Populated once the molecule is added to a topology.
    #[serde(skip)]
    atom_indices: Vec<usize>,
    /// Intramolecular bonds between the atoms.
    #[serde(default)]
    #[validate(nested)]
    bonds: Vec<Bond>,
    /// Intramolecular dihedrals.
    #[serde(default)]
    #[validate(nested)]
    dihedrals: Vec<Dihedral>,
    /// Intramolecular torsions.
    #[serde(default)]
    #[validate(nested)]
    torsions: Vec<Torsion>,
    /// Internal degrees of freedom.
    #[serde(default)]
    #[getter(skip)]
    degrees_of_freedom: DegreesOfFreedom,
    /// Names of atoms forming the molecule.
    #[serde(default)]
    atom_names: Vec<Option<String>>,
    /// Residues forming the molecule.
    #[validate(custom(function = "super::Residue::validate"))]
    #[serde(default)]
    residues: Vec<Residue>,
    /// Chains forming the molecule.
    #[validate(custom(function = "super::Chain::validate"))]
    #[serde(default)]
    chains: Vec<Chain>,
    /// Does it make sense to calculate center of mass for the molecule?
    #[serde(default = "default_true")]
    #[getter(skip)]
    has_com: bool,
    /// Map of custom properties.
    #[serde(default)]
    custom: HashMap<String, Value>,
}

fn default_true() -> bool {
    true
}

impl MoleculeKind {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn degrees_of_freedom(&self) -> DegreesOfFreedom {
        self.degrees_of_freedom
    }

    pub fn has_com(&self) -> bool {
        self.has_com
    }

    /// Set indices of atom types.
    pub(super) fn set_atom_indices(&mut self, indices: Vec<usize>) {
        self.atom_indices = indices;
    }

    /// Set names of all atoms of the molecule to None.
    pub(super) fn empty_atom_names(&mut self) {
        self.atom_names = vec![None; self.atoms.len()]
    }

    /// Set molecule id
    pub(super) fn set_id(&mut self, id: usize) {
        self.id = id;
    }
}

fn validate_molecule(molecule: &MoleculeKind) -> Result<(), ValidationError> {
    let n_atoms = molecule.atoms.len();

    // bonds must only exist between defined atoms
    if !molecule.bonds.iter().all(|x| x.lower(n_atoms)) {
        return Err(ValidationError::new("").with_message("bond between undefined atoms".into()));
    }

    // torsions must only exist between defined atoms
    if !molecule.torsions.iter().all(|x| x.lower(n_atoms)) {
        return Err(ValidationError::new("").with_message("torsion between undefined atoms".into()));
    }

    // dihedrals must only exist between defined atoms
    if !molecule.dihedrals.iter().all(|x| x.lower(n_atoms)) {
        return Err(
            ValidationError::new("").with_message("dihedral between undefined atoms".into())
        );
    }

    // residues can't contain undefined atoms
    for residue in molecule.residues.iter() {
        // empty residues can contain any indices
        if !residue.is_empty() && residue.range().end > n_atoms {
            return Err(
                ValidationError::new("").with_message("residue contains undefined atoms".into())
            );
        }
    }

    // chains can't contain undefined atoms
    for chain in molecule.chains.iter() {
        if !chain.is_empty() && chain.range().end > n_atoms {
            return Err(
                ValidationError::new("").with_message("chain contains undefined atoms".into())
            );
        }
    }

    // vector of atom names must correspond to the number of atoms (or be empty)
    if molecule.atom_names.len() != n_atoms {
        return Err(ValidationError::new("").with_message(
            "the number of atom names does not match the number of atoms in a molecule".into(),
        ));
    }

    Ok(())
}

impl CustomProperty for MoleculeKind {
    fn get_property(&self, key: &str) -> Option<Value> {
        self.custom.get(key).cloned()
    }

    fn set_property(&mut self, key: &str, value: Value) -> anyhow::Result<()> {
        self.custom.insert(key.to_string(), value);
        Ok(())
    }
}
