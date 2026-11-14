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
/// Exclusions only pair atoms of the same molecule, so each molecule kind keeps a small local
/// table and particles in different molecules always interact: memory is O(N + Σ n_kind²).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ExclusionMatrix {
    /// Per molecule kind: 1 = atoms interact, 0 = excluded; the diagonal is 0.
    kinds: Vec<SquareMatrix<u8>>,
    /// Per particle: the molecule it belongs to.
    owners: Vec<Owner>,
}

/// Where a particle's molecule starts and which kind table applies to it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Owner {
    start: usize,
    kind: usize,
}

impl ExclusionMatrix {
    /// Create exclusions based on topology.
    ///
    /// Particles are laid out block by block, each molecule a contiguous run of its atoms,
    /// as in `Block::num_atoms`.
    pub fn from_topology(topology: &Topology) -> Self {
        let kinds = topology
            .moleculekinds()
            .iter()
            .map(|molecule| {
                let mut local =
                    SquareMatrix::from_fn(molecule.atom_indices().len(), |i, j| u8::from(i != j));
                for exclusion in molecule.exclusions() {
                    let (i, j) = exclusion.into_ordered_tuple();
                    local[(i, j)] = 0;
                    local[(j, i)] = 0;
                }
                local
            })
            .collect();

        let mut owners = Vec::with_capacity(topology.num_particles());
        for block in topology.blocks() {
            let kind = block.molecule_index();
            let size = topology.moleculekinds()[kind].atom_indices().len();
            for _ in 0..block.num_molecules() {
                let start = owners.len();
                owners.extend(std::iter::repeat_n(Owner { start, kind }, size));
            }
        }
        Self { kinds, owners }
    }

    /// Whether particles `i` and `j` interact via nonbonded interactions.
    pub fn interacts(&self, i: usize, j: usize) -> bool {
        self.row(i).interacts(j)
    }

    /// Particle `i`'s exclusions, resolved once for a loop over partners `j`.
    #[inline]
    pub fn row(&self, i: usize) -> ExclusionRow<'_> {
        let Owner { start, kind } = self.owners[i];
        ExclusionRow {
            start,
            local: self.kinds[kind].row(i - start),
        }
    }
}

/// One particle's view of [`ExclusionMatrix`]: partners outside `start..start + local.len()`
/// always interact, partners inside are looked up in `local`.
#[derive(Debug, Clone, Copy)]
pub(super) struct ExclusionRow<'a> {
    start: usize,
    local: &'a [u8],
}

impl ExclusionRow<'_> {
    /// Whether this particle interacts with particle `j`.
    #[inline(always)]
    pub fn interacts(&self, j: usize) -> bool {
        // Wrapping turns "j before start" into a huge offset, so one compare covers both sides.
        let k = j.wrapping_sub(self.start);
        // SAFETY: k < local.len() is checked first.
        k >= self.local.len() || unsafe { *self.local.get_unchecked(k) != 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two rigid dimers, then three flexible 4-bead chains with `excluded_neighbours: 1`:
    /// directly bonded beads are excluded, and only within their own copy.
    #[test]
    fn flexible_chain_excludes_only_bonded_neighbours() {
        let yaml = "\
atoms:
  - {name: A, mass: 1.0, sigma: 1.0}
molecules:
  - name: dimer
    from_structure: [{A: [0, 0, 0]}, {A: [1, 0, 0]}]
    degrees_of_freedom: Rigid
  - name: chain
    from_structure: [{A: [0, 0, 0]}, {A: [1, 0, 0]}, {A: [2, 0, 0]}, {A: [3, 0, 0]}]
    bonds:
      - {index: [0, 1]}
      - {index: [1, 2]}
      - {index: [2, 3]}
    excluded_neighbours: 1
system:
  cell: !Cuboid [50.0, 50.0, 50.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: []
  blocks:
    - {molecule: dimer, N: 2, insert: !RandomCOM {}}
    - {molecule: chain, N: 3, insert: !RandomCOM {}}
";
        let topology = Topology::from_str(yaml).unwrap();
        let exclusions = ExclusionMatrix::from_topology(&topology);
        // Particles: dimers @0, @2; chains @4, @8, @12.
        let excluded = |i: usize, j: usize| {
            let (i, j) = (i.min(j), i.max(j));
            match (i, j) {
                _ if i == j => true,
                (0, 1) | (2, 3) => true,
                _ if i >= 4 && (i - 4) / 4 == (j - 4) / 4 => j - i == 1,
                _ => false,
            }
        };
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(
                    exclusions.interacts(i, j),
                    !excluded(i, j),
                    "pair ({i}, {j})"
                );
            }
        }
    }

    /// Hot loops resolve particle `i` once via `row(i)`; it must answer like `interacts(i, j)`
    /// for partners before, inside and after `i`'s own molecule.
    #[test]
    fn row_agrees_with_pairwise_lookup() {
        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();
        let exclusions = ExclusionMatrix::from_topology(&topology);
        let n = topology.num_particles();
        for i in 0..n {
            let row = exclusions.row(i);
            for j in 0..n {
                assert_eq!(
                    row.interacts(j),
                    exclusions.interacts(i, j),
                    "pair ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_exclusion_matrix() {
        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();
        let exclusions = ExclusionMatrix::from_topology(&topology);

        let num_particles = topology.num_particles();

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
                let excluded = expected_exclusions.contains(&(i, j))
                    || expected_exclusions.contains(&(j, i))
                    || i == j;
                assert_eq!(exclusions.interacts(i, j), !excluded, "pair ({i}, {j})");
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
                    let cpu_excluded = !excl.interacts(i, j);
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
