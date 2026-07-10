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

use super::square_matrix::SquareMatrix;
use crate::Topology;

/// Which particle pairs interact via nonbonded interactions.
///
/// Values: 1 = particles interact, 0 = excluded pair.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ExclusionMatrix(SquareMatrix<u8>);

impl ExclusionMatrix {
    /// Create exclusions based on topology.
    pub fn from_topology(topology: &Topology) -> Self {
        let n = topology.num_particles();
        // Everything interacts except a particle with itself.
        let mut exclusions = Self(SquareMatrix::from_fn(n, |i, j| u8::from(i != j)));

        let mut atom_cnt = 0;
        for block in topology.blocks() {
            let molecule = &topology.moleculekinds()[block.molecule_index()];
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
    #[inline]
    pub fn get(&self, indices: (usize, usize)) -> u8 {
        self.0[indices]
    }

    /// Contiguous row slice so the inner loop can use `get_unchecked(j)`
    /// on a single slice instead of recomputing the offset each iteration.
    #[inline]
    pub fn row(&self, i: usize) -> &[u8] {
        self.0.row(i)
    }

    /// Set exclusion status for the specified pair of particle indices.
    /// Sets both `(i,j)` and `(j,i)` in the matrix.
    pub fn set(&mut self, (i, j): (usize, usize), value: u8) {
        self.0[(i, j)] = value;
        self.0[(j, i)] = value;
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
        assert_eq!(exclusions.row(0).len(), num_particles);
        assert_eq!(exclusions.0.order(), num_particles);
        assert_eq!(exclusions.0.iter().count(), num_particles * num_particles);

        // MOL (RigidAlchemical, 7 atoms) excludes all intra-molecular pairs.
        // MOL2 (Free, 3 atoms, no bonds) has no exclusions.
        // Blocks: 3×MOL @0, 50×MOL2 @21, 6×MOL2 @171, 1×MOL2 @189, 2×MOL @192, 5×MOL2 @206
        let mol_offsets = [0, 7, 14, 192, 199]; // start of each MOL instance
        let mut expected_exclusions = Vec::new();
        for &offset in &mol_offsets {
            for i in 0..7 {
                for j in (i + 1)..7 {
                    expected_exclusions.push((offset + i, offset + j));
                }
            }
        }

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

    /// Verify the GPU exclusion CSR against the CPU ExclusionMatrix.
    ///
    /// Rigid molecules must produce empty CSR rows (kernel handles them via
    /// `mol_is_rigid`), while flexible molecules must match the CPU exclusions.
    /// Verify the GPU exclusion CSR against the CPU ExclusionMatrix.
    ///
    /// Rigid molecules must produce empty CSR rows (kernel handles them via
    /// `mol_is_rigid`), while flexible molecules must match the CPU exclusions.
    #[test]
    #[cfg(feature = "gpu")]
    fn repack_exclusions_matches_cpu_matrix() {
        use crate::group::{Group, MoleculeId};
        use std::collections::HashSet;

        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();

        // Build groups from topology blocks (mirroring Backend initialization)
        let mut groups = Vec::new();
        let mut offset = 0usize;
        for block in topology.blocks() {
            let mol_idx = block.molecule_index();
            let n_atoms = topology.moleculekinds()[mol_idx].atoms().len();
            for _ in 0..block.num_molecules() {
                groups.push(Group::new(
                    groups.len(),
                    MoleculeId::new(mol_idx),
                    offset..offset + n_atoms,
                ));
                offset += n_atoms;
            }
        }

        let (offsets, atoms) = crate::energy::bonded::kernel::repack_exclusions(&topology, &groups);

        let n: usize = groups.iter().map(|g| g.capacity()).sum();
        assert_eq!(offsets.len(), n + 1);

        let excl = ExclusionMatrix::from_topology(&topology);

        for i in 0..n {
            let csr_neighbors: HashSet<u32> = atoms[offsets[i] as usize..offsets[i + 1] as usize]
                .iter()
                .copied()
                .collect();

            let group = groups
                .iter()
                .find(|g: &&Group| i >= g.start() && i < g.start() + g.capacity())
                .unwrap();
            let mol = topology.moleculekind(group.molecule());

            if mol.degrees_of_freedom().is_rigid() {
                assert!(
                    csr_neighbors.is_empty(),
                    "rigid atom {i} should have empty CSR row"
                );
            } else {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let cpu_excluded = excl.get((i, j)) == 0;
                    let csr_excluded = csr_neighbors.contains(&(j as u32));
                    assert_eq!(
                        cpu_excluded, csr_excluded,
                        "mismatch at ({i}, {j}): cpu={cpu_excluded}, csr={csr_excluded}"
                    );
                }
            }
        }
    }
}
