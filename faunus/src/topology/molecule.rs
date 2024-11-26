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

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
};

use super::chemfiles_interface;

use derive_builder::Builder;
use derive_getters::Getters;
use serde::{Deserialize, Serialize};
use unordered_pair::UnorderedPair;

use crate::topology::{Chain, DegreesOfFreedom, Residue, Value};
use validator::{Validate, ValidationError};

use super::{Bond, CustomProperty, Dihedral, IndexRange, Indexed, Torsion};

/// Description of molecule properties.
///
/// Molecule is a collection of atoms that can (but not do not have to be) connected by bonds.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate, Getters, Builder)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_molecule"))]
#[builder(default)]
pub struct MoleculeKind {
    /// Unique name.
    #[builder(setter(into))]
    name: String,
    /// Unique identifier.
    /// Only defined if the MoleculeKind is inside of Topology.
    #[serde(skip)]
    #[getter(skip)]
    id: usize,
    /// Names of atom kinds forming the molecule.
    #[serde(default)]
    atoms: Vec<String>,
    /// Indices / atom ids of atom kinds forming the molecule.
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
    /// Generate an exclusions list from bonds.
    /// Add all atom pairs which are excluded_neighbours or less bonds apart.
    #[serde(default)] // default value is 0
    excluded_neighbours: usize,
    /// List of atom pairs which nonbonded interactions are excluded.
    #[serde(default)]
    exclusions: HashSet<UnorderedPair<usize>>,
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
    /// Construct molecule from existing structure file (xyz, pdb, etc.)
    #[serde(default)]
    #[builder(setter(strip_option, custom), default)]
    from_structure: Option<PathBuf>,
}

impl MoleculeKindBuilder {
    /// Populate `atoms` from structure file.
    ///
    /// # Panics
    /// Panics of the file doesn't exist or is of unknown format.
    pub fn from_structure(&mut self, filename: impl AsRef<Path>) -> &mut Self {
        let atom_names = chemfiles_interface::frame_from_file(&filename)
            .unwrap()
            .iter_atoms()
            .map(|a| a.name())
            .collect();
        self.atoms(atom_names)
    }
}

fn default_true() -> bool {
    true
}

impl MoleculeKind {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub const fn degrees_of_freedom(&self) -> DegreesOfFreedom {
        self.degrees_of_freedom
    }

    pub const fn has_com(&self) -> bool {
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

    /// Generate exclusions for the molecule based on value of the `excluded_neighbours`
    /// and on the bonds of the molecule.
    pub(super) fn generate_exclusions(&mut self) {
        let mut edges = vec![Vec::new(); self.atoms.len()];

        for bond in self.bonds.iter() {
            let [i, j] = *bond.index();
            edges[i].push(j);
            edges[j].push(i);
        }

        // do BFS and get indices in range
        let mut exclusions = HashSet::new();
        for (index, _) in self.atoms.iter().enumerate() {
            let mut queue = VecDeque::new();
            let mut distances = HashMap::new();

            queue.push_back(index);
            distances.insert(index, 0);
            while let Some(current) = queue.pop_front() {
                let current_distance = *distances.get(&current).unwrap();

                // create exclusion
                if current_distance <= self.excluded_neighbours && current != index {
                    exclusions.insert(UnorderedPair(index, current));
                }

                // continue if we are still in range
                if current_distance < self.excluded_neighbours {
                    for &neighbour in edges[current].iter() {
                        distances.entry(neighbour).or_insert_with(|| {
                            queue.push_back(neighbour);
                            current_distance + 1
                        });
                    }
                }
            }
        }

        // add the obtained exclusions
        self.exclusions.extend(exclusions.iter())
    }

    /// Get number of atoms in the molecule.
    pub fn len(&self) -> usize {
        self.atom_indices().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    // exclusions can't contain undefined atoms or the same atom twice
    for (i, j) in molecule.exclusions.iter().map(|e| e.into_ordered_tuple()) {
        if i == j {
            return Err(
                ValidationError::new("").with_message("exclusion between the same atom".into())
            );
        }

        if i >= n_atoms || j >= n_atoms {
            return Err(
                ValidationError::new("").with_message("exclusion between undefined atoms".into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{BondKind, BondOrder};

    #[test]
    fn generate_exclusions_n1() {
        let mut molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .atoms("ABCDEFG".chars().map(|c| c.to_string()).collect())
            .atom_indices((0..8).collect())
            .bonds(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [5, 6],
                    [3, 6],
                    [4, 6],
                ]
                .iter()
                .map(|&ij| Bond::new(ij, BondKind::Unspecified, BondOrder::Unspecified))
                .collect(),
            )
            .degrees_of_freedom(DegreesOfFreedom::Free)
            .excluded_neighbours(1)
            .build()
            .unwrap();

        molecule.generate_exclusions();

        assert_eq!(molecule.exclusions.len(), 8);
        assert!(molecule.exclusions.contains(&UnorderedPair(0, 1)));
        assert!(molecule.exclusions.contains(&UnorderedPair(1, 2)));
        assert!(molecule.exclusions.contains(&UnorderedPair(2, 3)));
        assert!(molecule.exclusions.contains(&UnorderedPair(3, 4)));
        assert!(molecule.exclusions.contains(&UnorderedPair(3, 5)));
        assert!(molecule.exclusions.contains(&UnorderedPair(5, 6)));
        assert!(molecule.exclusions.contains(&UnorderedPair(4, 6)));
        assert!(molecule.exclusions.contains(&UnorderedPair(3, 6)));
    }

    #[test]
    fn generate_exclusions_n2() {
        let mut molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .atoms("ABCDEFG".chars().map(|c| c.to_string()).collect())
            .atom_indices((0..8).collect())
            .bonds(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [5, 6],
                    [3, 6],
                    [4, 6],
                ]
                .iter()
                .map(|&ij| Bond::new(ij, BondKind::Unspecified, BondOrder::Unspecified))
                .collect(),
            )
            .degrees_of_freedom(DegreesOfFreedom::Free)
            .excluded_neighbours(2)
            .build()
            .unwrap();

        molecule.generate_exclusions();

        assert_eq!(molecule.exclusions.len(), 14);
        let expected = [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 5],
            [4, 6],
            [5, 6],
        ];

        for pair in expected {
            assert!(molecule
                .exclusions
                .contains(&UnorderedPair(pair[0], pair[1])));
        }
    }

    #[test]
    fn generate_exclusions_n3() {
        let mut molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .atoms("ABCDEFG".chars().map(|c| c.to_string()).collect())
            .atom_indices((0..8).collect())
            .bonds(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [5, 6],
                    [3, 6],
                    [4, 6],
                ]
                .iter()
                .map(|&ij| Bond::new(ij, BondKind::Unspecified, BondOrder::Unspecified))
                .collect(),
            )
            .degrees_of_freedom(DegreesOfFreedom::Free)
            .excluded_neighbours(3)
            .build()
            .unwrap();

        molecule.generate_exclusions();

        assert_eq!(molecule.exclusions.len(), 18);
        let expected = [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 5],
            [4, 6],
            [5, 6],
        ];

        for pair in expected {
            assert!(molecule
                .exclusions
                .contains(&UnorderedPair(pair[0], pair[1])));
        }
    }

    #[test]
    fn test_load_structure() {
        let molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .from_structure("tests/files/mol2.xyz")
            .build()
            .unwrap();

        assert_eq!(molecule.atoms.len(), 3);
        assert_eq!(
            molecule.atoms.as_slice(),
            ["OW".to_owned(), "OW".to_owned(), "X".to_owned()]
        );
    }
}
