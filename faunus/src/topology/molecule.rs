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

use serde::{Deserialize, Deserializer, Serialize};

use crate::topology::{Chain, DegreesOfFreedom, Residue, Value};
use validator::{Validate, ValidationError};

use super::{AtomKind, Bond, Dihedral, NonOverlapping, Torsion};

#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_molecule"))]
pub struct MoleculeKind {
    /// Unique name.
    name: String,
    /// Unique identifier.
    /// Only defined if the MoleculeKind is inside of Topology.
    #[serde(skip_deserializing)]
    id: usize,
    /// Names of atom kinds forming the molecule.
    atoms: Vec<String>,
    /// Indices of atom kinds forming the molecule.
    /// Populated once the molecule is added to a system.
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
    /// Map of custom properties.
    #[serde(default)]
    custom: HashMap<String, Value>,
}

impl MoleculeKind {
    /// Convert a yaml-formatted string into a MoleculeKind.
    /// This performs sanity checks and always returns either a valid MoleculeKind or an error.
    pub fn from_str(string: &str) -> Result<MoleculeKind, anyhow::Error> {
        let mut molecule = serde_yaml::from_str::<MoleculeKind>(string)?;

        if molecule.atom_names.is_empty() {
            molecule.atom_names = vec![None; molecule.atoms.len()];
        }

        molecule.validate()?;
        Ok(molecule)
    }
}

fn validate_molecule(molecule: &MoleculeKind) -> Result<(), ValidationError> {
    let n_atoms = molecule.atoms.len();

    // bonds must only exist between defined atoms
    for bond in molecule.bonds.iter() {
        if bond.index().iter().any(|&index| index >= n_atoms) {
            return Err(ValidationError::new("bond between undefined atoms"));
        }
    }

    // torsions must only exist between defined atoms
    for torsion in molecule.torsions.iter() {
        if torsion.index().iter().any(|&index| index >= n_atoms) {
            return Err(ValidationError::new("torsion between undefined atoms"));
        }
    }

    // dihedrals must only exist between defined atoms
    for dihedral in molecule.dihedrals.iter() {
        if dihedral.index().iter().any(|&index| index >= n_atoms) {
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

/// TODO! Should not be placed here.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
pub struct Topology {
    /// All possible atom types.
    #[serde(deserialize_with = "deserialize_atoms")]
    atoms: Vec<AtomKind>,
    /// All possible molecule types.
    #[serde(deserialize_with = "deserialize_molecules")]
    molecules: Vec<MoleculeKind>,
}

impl Topology {
    /// Convert a yaml-formatted string into Topology.
    pub fn from_str(string: &str) -> Result<Topology, anyhow::Error> {
        let mut topology = serde_yaml::from_str::<Topology>(string)?;

        // get indices of atom kinds forming each molecule
        for molecule in topology.molecules.iter_mut() {
            for atom in molecule.atoms.iter() {
                let index = topology
                    .atoms
                    .iter()
                    .position(|x| &x.name == atom)
                    .ok_or(anyhow::Error::msg("undefined atom kind in a molecule"))?;
                molecule.atom_indices.push(index);
            }
        }

        Ok(topology)
    }
}

/// Deserialize atoms in Topology and set their IDs.
/// Makes sure that the atom names are unique.
fn deserialize_atoms<'de, D>(deserializer: D) -> Result<Vec<AtomKind>, D::Error>
where
    D: Deserializer<'de>,
{
    let atoms: Vec<_> = Vec::deserialize(deserializer)?
        .into_iter()
        .enumerate()
        .map(|(i, mut atom): (usize, AtomKind)| {
            atom.id = i;
            atom
        })
        .collect();

    // check for duplicate atom names
    if super::are_unique(&atoms, |i: &AtomKind, j: &AtomKind| i.name == j.name) {
        Ok(atoms)
    } else {
        Err(serde::de::Error::custom("atoms have non-unique names"))
    }
}

/// Deserialize molecules in Topology and set their IDs.
/// Makes sure that the molecule names are unique.
fn deserialize_molecules<'de, D>(deserializer: D) -> Result<Vec<MoleculeKind>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut molecules: Vec<MoleculeKind> = Vec::deserialize(deserializer)?;

    for (i, molecule) in molecules.iter_mut().enumerate() {
        // set atom names
        if molecule.atom_names.is_empty() {
            molecule.atom_names = vec![None; molecule.atoms.len()];
        }

        molecule.validate().map_err(serde::de::Error::custom)?;

        // set index
        molecule.id = i;
    }

    // check for duplicate molecule names
    if super::are_unique(&molecules, |i: &MoleculeKind, j: &MoleculeKind| {
        i.name == j.name
    }) {
        Ok(molecules)
    } else {
        Err(serde::de::Error::custom("molecules have non-unique names"))
    }
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

        // TODO! write a proper test

        write!(writer, "{}", serialized).unwrap();
    }

    #[test]
    fn read_yaml() {
        let string = std::fs::read_to_string("tests/files/molecule_input.yaml").unwrap();
        let molecule = MoleculeKind::from_str(&string).unwrap();

        // TODO! write a proper test
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

    #[test]
    fn read_topology() {
        let string = std::fs::read_to_string("tests/files/topology_input.yaml").unwrap();
        let topology = Topology::from_str(&string).unwrap();

        println!("{:?}", topology);
    }
}
