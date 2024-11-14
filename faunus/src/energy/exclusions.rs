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

use crate::{DMatrix, Topology};

/// Matrix of exclusions based on particle ids.
/// Pairs of particle indices which should not interact via nonbonded interactions
/// are assigned a value of 0. Pairs of particle indices which should interact
/// are assigned a value of 1.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ExclusionMatrix(DMatrix<u8>);

impl ExclusionMatrix {
    /// Create exclusions based on topology.
    pub fn from_topology(topology: &Topology) -> Self {
        let n = topology.num_particles();
        let exclude_self = |i, j| if i == j { 0 } else { 1 };
        let mut exclusions = ExclusionMatrix(DMatrix::from_fn(n, n, exclude_self));

        // read the exclusions for the individual atoms
        let mut atom_cnt = 0;
        for block in topology.blocks() {
            let molecule = &topology.moleculekinds()[block.molecule_index()];
            // loop through the specific number of molecules in the block
            for _ in 0..block.num_molecules() {
                for exclusion in molecule.exclusions() {
                    let rel = exclusion.into_ordered_tuple();
                    let abs = (rel.0 + atom_cnt, rel.1 + atom_cnt);
                    exclusions.set(abs, 0);
                }
                atom_cnt += molecule.atoms().len();
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
    #[inline]
    pub fn get(&self, indices: (usize, usize)) -> u8 {
        self.0[indices]
    }

    /// Set exclusion status for the specified pair of particle indices.
    /// Sets both `(i,j)` and `(j,i)` in the matrix.
    /// Panics if the indices are out of range.
    pub fn set(&mut self, indices: (usize, usize), value: u8) {
        self.0[(indices.0, indices.1)] = value;
        self.0[(indices.1, indices.0)] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclusion_matrix() {
        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();
        let exclusions = ExclusionMatrix::from_topology(&topology);

        let num_particles = topology.num_particles();
        assert_eq!(exclusions.0.row(0).len(), num_particles);
        assert_eq!(exclusions.0.column(0).len(), num_particles);
        assert_eq!(exclusions.0.len(), num_particles * num_particles);

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
                    assert_eq!(exclusions.get((i, j)), 0);
                } else {
                    assert_eq!(exclusions.get((i, j)), 1);
                }
            }
        }
    }
}
