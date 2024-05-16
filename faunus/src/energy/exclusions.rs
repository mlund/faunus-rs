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

//! Implementation of the exclusions.

use crate::{topology::TopologyLike, Topology};

/// Matrix of exclusions based on particle ids.
/// Pairs of particle indices which should not interact via nonbonded interactions
/// are assigned a value of 0. Pairs of particle indices which should interact
/// are assigned a value of 1.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExclusionMatrix(Vec<Vec<u8>>);

impl ExclusionMatrix {
    /// Create a new ExclusionMatrix based on the Topology of the system.
    pub fn new(topology: &Topology) -> Self {
        let num_particles = topology.num_particles();
        let mut exclusions = ExclusionMatrix(vec![vec![1; num_particles]; num_particles]);

        // atoms should not interact with themselves
        for i in 0..num_particles {
            exclusions.0[i][i] = 0;
        }

        // read the exclusions for the individual atoms
        let mut atom_counter = 0;
        for block in topology.blocks() {
            let molecule = &topology.molecules()[block.molecule_index()];
            // loop through the specific number of molecules in the block
            for _ in 0..block.number() {
                for exclusion in molecule.exclusions() {
                    let (relative1, relative2) = exclusion.into_ordered_tuple();
                    let absolute1 = relative1 + atom_counter;
                    let absolute2 = relative2 + atom_counter;

                    exclusions.0[absolute1][absolute2] = 0;
                    exclusions.0[absolute2][absolute1] = 0;
                }

                atom_counter += molecule.atoms().len();
            }
        }

        exclusions
    }

    /// Get exclusion status for the specified pair of particle indices.
    /// - 1 => particles interact via nonbonded interactions.
    /// - 0 => particles do NOT interact via nonbonded interactions.
    ///
    /// Thus, the result can be simply cast to f64 and
    /// be used to multiply the calculated interaction energy.
    ///
    /// Panics if the indices are out of range.
    pub fn get(&self, i: usize, j: usize) -> u8 {
        self.0[i][j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclusion_matrix() {
        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();
        let exclusions = ExclusionMatrix::new(&topology);

        let num_particles = topology.num_particles();
        assert_eq!(exclusions.0.len(), num_particles);
        for x in exclusions.0.iter() {
            assert_eq!(x.len(), num_particles);
        }

        let expected_exclusions = [
            (0, 1),
            (2, 3),
            (1, 2),
            (0, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (8, 9),
            (7, 11),
            (12, 13),
            (14, 15),
            (16, 17),
            (15, 16),
            (14, 18),
            (19, 20),
            (192, 193),
            (193, 194),
            (194, 195),
            (192, 196),
            (197, 198),
            (199, 200),
            (200, 201),
            (201, 202),
            (199, 203),
            (204, 205),
        ];

        for i in 0..num_particles {
            for j in 0..num_particles {
                if expected_exclusions.contains(&(i, j))
                    || expected_exclusions.contains(&(j, i))
                    || i == j
                {
                    assert_eq!(exclusions.get(i, j), 0);
                } else {
                    assert_eq!(exclusions.get(i, j), 1);
                }
            }
        }
    }
}
