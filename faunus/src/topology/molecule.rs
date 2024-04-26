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

use serde::{Deserialize, Serialize};

use crate::topology::{Chain, DegreesOfFreedom, Residue, Value};
use validator::{Validate, ValidationError};

use super::{AtomKind, Bond, Dihedral, NonOverlapping, Torsion};

#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_molecule_kind"))]
pub struct MoleculeKind {
    /// Unique name
    name: String,
    /// Unique identifier
    /// Only defined once the MoleculeKind is added to the system
    #[serde(skip_deserializing)]
    id: usize,
    /// Names of atom kinds forming the molecule
    atoms: Vec<String>,
    /// Indices of atom kinds forming the molecule
    /// Populated once the molecule is added to a system
    #[serde(skip)]
    atom_indices: Vec<usize>,
    /// Intramolecular bonds between the atoms
    #[serde(default)]
    bonds: Vec<Bond>,
    /// Intramolecular dihedrals
    #[serde(default)]
    dihedrals: Vec<Dihedral>,
    /// Intramolecular torsions
    #[serde(default)]
    torsions: Vec<Torsion>,
    /// Internal degrees of freedom
    #[serde(default)]
    degrees_of_freedom: DegreesOfFreedom,
    /// Names of atoms forming the molecule
    #[serde(default)]
    atom_names: Vec<Option<String>>,
    /// Residues forming the molecule
    #[validate(custom(function = "validate_overlap"))]
    #[serde(default)]
    residues: Vec<Residue>,
    /// Chains forming the molecule
    #[validate(custom(function = "validate_overlap"))]
    #[serde(default)]
    chains: Vec<Chain>,
    /// Map of custom properties
    /// TODO! Converting values
    #[serde(default)]
    custom: HashMap<String, Value>,
}

impl MoleculeKind {
    /// Convert a yaml-formatted string into a MoleculeKind.
    pub fn from_str(string: &str) -> Result<MoleculeKind, anyhow::Error> {
        let mut molecule = serde_yaml::from_str::<MoleculeKind>(string)?;

        if molecule.atom_names.is_empty() {
            molecule.atom_names = vec![None; molecule.atoms.len()];
        }

        molecule.validate()?;
        Ok(molecule)
    }
}

fn validate_molecule_kind(molecule: &MoleculeKind) -> Result<(), ValidationError> {
    let n_atoms = molecule.atoms.len();

    // bonds must only exist between defined atoms
    for bond in molecule.bonds.iter() {
        if bond.index.iter().any(|&index| index >= n_atoms) {
            return Err(ValidationError::new("bond between undefined atoms"));
        }
    }

    // torsions must only exist between defined atoms
    for torsion in molecule.torsions.iter() {
        if torsion.index.iter().any(|&index| index >= n_atoms) {
            return Err(ValidationError::new("torsion between undefined atoms"));
        }
    }

    // dihedrals must only exist between defined atoms
    for dihedral in molecule.dihedrals.iter() {
        if dihedral.index.iter().any(|&index| index >= n_atoms) {
            return Err(ValidationError::new("dihedral between undefined atoms"));
        }
    }

    // residues can't contain undefined atoms
    for residue in molecule.residues.iter() {
        // empty residues can contain any indices
        if !residue.atoms().is_empty() && residue.atoms().end > n_atoms {
            return Err(ValidationError::new("residue contains undefined atoms"));
        }
    }

    // chains can't contain undefined atoms
    for chain in molecule.chains.iter() {
        if !chain.atoms().is_empty() && chain.atoms().end > n_atoms {
            return Err(ValidationError::new("chain contains undefined atoms"));
        }
    }

    // vector of atom names must correspond to the number of atoms (or be empty)
    if molecule.atom_names.len() != n_atoms {
        return Err(ValidationError::new("invalid number of atom names"));
    }

    Ok(())
}

// TODO! tests
fn validate_overlap(collection: &[impl NonOverlapping]) -> Result<(), ValidationError> {
    if collection.iter().enumerate().any(|(i, item_i)| {
        collection
            .iter()
            .skip(i + 1)
            .any(|item_j| item_i.overlap(item_j))
    }) {
        Err(ValidationError::new("overlap between collections"))
    } else {
        Ok(())
    }
}

/// TODO! Should not be placed here.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
#[validate(schema(function = "validate_topology"))]
pub struct Topology {
    /// All possible atom types
    atoms: Vec<AtomKind>,
    /// All possible molecule types
    #[validate(nested)]
    molecules: Vec<MoleculeKind>,
}

// TODO!
fn validate_topology(topology: &Topology) -> Result<(), ValidationError> {
    Ok(())
}

#[cfg(test)]
mod tests {

    use std::{
        fs::File,
        io::{BufWriter, Write},
    };

    use crate::topology::Bond;

    use super::*;

    #[test]
    fn write_yaml() {
        let chains = vec![Chain::new("A", 1..3), Chain::new("B", 0..0)];

        let bond = Bond::new(
            [0, 1],
            crate::topology::BondKind::Harmonic { k: 100.0, req: 1.0 },
            Some(crate::topology::BondOrder::Single),
        );
        let molecule = MoleculeKind {
            name: "test".to_owned(),
            id: 0,
            atoms: vec![String::from("OW"), String::from("HW"), String::from("HW")],
            atom_indices: vec![],
            bonds: vec![bond],
            dihedrals: vec![],
            torsions: vec![],
            degrees_of_freedom: DegreesOfFreedom::Free,
            atom_names: vec![],
            residues: vec![],
            chains,
            custom: HashMap::new(),
        };

        let serialized = serde_yaml::to_string(&molecule).unwrap();
        let file = File::create("tests/files/molecule.yaml").unwrap();
        let mut writer = BufWriter::new(file);

        write!(writer, "{}", serialized).unwrap();
    }

    #[test]
    fn read_yaml() {
        let string = std::fs::read_to_string("tests/files/molecule_input.yaml").unwrap();
        let molecule = MoleculeKind::from_str(&string).unwrap();

        println!("{:?}", molecule);
    }

    #[test]
    fn minimal_molecule() {
        let string = std::fs::read_to_string("tests/files/minimal_molecule.yaml").unwrap();
        let molecule = MoleculeKind::from_str(&string).unwrap();

        assert_eq!(&molecule.name, "minimal_molecule");
        assert_eq!(molecule.atoms, vec!["OW", "HW", "HW"]);
        assert!(molecule.atom_indices.is_empty());
        assert_eq!(molecule.atom_names.len(), molecule.atoms.len());
        assert!(molecule.atom_names.iter().all(|x| *x == None));
        assert!(molecule.residues.is_empty());
        assert!(molecule.chains.is_empty());
        assert!(molecule.custom.is_empty());
        assert_eq!(molecule.degrees_of_freedom, DegreesOfFreedom::Free);
        assert_eq!(molecule.id, 0);
        assert!(molecule.bonds.is_empty());
        assert!(molecule.torsions.is_empty());
        assert!(molecule.dihedrals.is_empty());
    }
}
