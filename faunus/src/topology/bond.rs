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

//! Bonds between atoms

use std::collections::{HashSet, VecDeque};

use derive_getters::Getters;
use interatomic::twobody::IsotropicTwobodyEnergy;
use interatomic::Cutoff;
use serde::{Deserialize, Serialize};
use unordered_pair::UnorderedPair;
use validator::Validate;

use crate::{group::Group, Context};

use super::Indexed;

/// Force field definition for bonds, e.g. harmonic, FENE, Morse, etc.
///
/// Each varient stores the parameters for the bond type, like force constant, equilibrium distance, etc.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub(crate) enum BondKind {
    /// Harmonic bond type.
    /// See <https://en.wikipedia.org/wiki/Harmonic_oscillator>.
    Harmonic(interatomic::twobody::Harmonic),
    /// Finitely extensible nonlinear elastic bond type,
    /// See <https://en.wikipedia.org/wiki/FENE>.
    FENE(interatomic::twobody::FENE),
    /// Morse bond type.
    /// See <https://en.wikipedia.org/wiki/Morse_potential>.
    Morse(interatomic::twobody::Morse),
    /// Harmonic Urey-Bradley bond type.
    /// See <https://manual.gromacs.org/documentation/current/reference-manual/functions/bonded-interactions.html#urey-bradley-potential>
    /// for more information.
    UreyBradley(interatomic::twobody::UreyBradley),
    /// Undefined bond type.
    #[default]
    Unspecified,
}

/// Bond order describing the multiplicity of a bond between two atoms.
///
/// See <https://en.wikipedia.org/wiki/Bond_order> for more information.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub(crate) enum BondOrder {
    #[default]
    /// Undefined bond order
    Unspecified,
    /// Single bond, e.g. diatomic hydrogen, H–H
    Single,
    /// Double bond, e.g. diatomic oxygen, O=O
    Double,
    /// Triple bond, e.g. diatomic nitrogen, N≡N
    Triple,
    Quadruple,
    Quintuple,
    Sextuple,
    Amide,
    Aromatic,
    Custom(f64),
}

impl From<BondOrder> for f64 {
    fn from(value: BondOrder) -> Self {
        match value {
            BondOrder::Unspecified => 0.0,
            BondOrder::Single => 1.0,
            BondOrder::Double => 2.0,
            BondOrder::Triple => 3.0,
            BondOrder::Quadruple => 4.0,
            BondOrder::Quintuple => 5.0,
            BondOrder::Sextuple => 6.0,
            BondOrder::Amide => 1.25,
            BondOrder::Aromatic => 1.5,
            BondOrder::Custom(value) => value,
        }
    }
}

impl From<f64> for BondOrder {
    fn from(value: f64) -> Self {
        match value {
            0.0 => Self::Unspecified,
            1.0 => Self::Single,
            2.0 => Self::Double,
            3.0 => Self::Triple,
            4.0 => Self::Quadruple,
            5.0 => Self::Quintuple,
            6.0 => Self::Sextuple,
            1.25 => Self::Amide,
            1.5 => Self::Aromatic,
            _ => Self::Custom(value),
        }
    }
}

/// Describes a bond between two atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Validate, Getters)]
#[serde(deny_unknown_fields)]
pub struct Bond {
    /// Indices of the two atoms in the bond
    #[validate(custom(function = "super::validate_unique_indices"))]
    index: [usize; 2],
    /// Kind of bond, e.g. harmonic, FENE, Morse, etc.
    #[serde(default)]
    kind: BondKind,
    /// Bond order
    #[serde(default)]
    order: BondOrder,
}

impl Bond {
    /// Create new bond. This function performs no sanity checks.
    #[allow(dead_code)]
    pub(crate) const fn new(index: [usize; 2], kind: BondKind, order: BondOrder) -> Self {
        Self { index, kind, order }
    }

    /// Check if the bond contains atom with index.
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }

    /// Calculate energy of a bond in a specific group.
    /// Returns 0.0 if any of the bonded particles is inactive.
    pub fn energy(&self, context: &impl Context, group: &Group) -> f64 {
        let to_abs_index = |i| group.to_absolute_index(i);
        let [Ok(i), Ok(j)] = self.index.map(to_abs_index) else {
            return 0.0;
        };

        let distance_squared = context.get_distance_squared(i, j);
        self.isotropic_twobody_energy(distance_squared)
    }

    /// Calculate energy of an intermolecular bond.
    /// Returns 0.0 if any of the bonded particles is inactive.
    pub fn energy_intermolecular(
        &self,
        context: &impl Context,
        term: &crate::energy::IntermolecularBonded,
    ) -> f64 {
        // one or both particles are inactive
        if self.index.iter().any(|&i| !term.is_active(i)) {
            return 0.0;
        }

        let distance_squared = context.get_distance_squared(self.index[0], self.index[1]);
        self.isotropic_twobody_energy(distance_squared)
    }
}

impl Indexed for Bond {
    fn index(&self) -> &[usize] {
        &self.index
    }
}

impl Cutoff for Bond {
    fn cutoff(&self) -> f64 {
        f64::INFINITY
    }
}

/// Adjacency list built from intramolecular bonds.
///
/// Used for bond-walking algorithms such as PBC-aware COM calculation
/// and pivot moves on branched molecules.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BondGraph {
    neighbors: Vec<Vec<usize>>,
}

impl BondGraph {
    pub fn from_bonds(bonds: &[Bond], num_atoms: usize) -> Self {
        let mut neighbors = vec![Vec::new(); num_atoms];
        for bond in bonds {
            let [i, j] = *bond.index();
            neighbors[i].push(j);
            neighbors[j].push(i);
        }
        Self { neighbors }
    }

    pub fn neighbors(&self, index: usize) -> &[usize] {
        &self.neighbors[index]
    }

    pub const fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    pub const fn num_atoms(&self) -> usize {
        self.neighbors.len()
    }

    /// BFS from `start`, treating `excluded` as a barrier.
    /// Returns all reachable nodes including `start`.
    pub fn connected_from(&self, start: usize, excluded: usize) -> Vec<usize> {
        let mut visited = vec![false; self.neighbors.len()];
        visited[excluded] = true;
        visited[start] = true;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        let mut result = vec![start];
        while let Some(current) = queue.pop_front() {
            for &neighbor in &self.neighbors[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                    result.push(neighbor);
                }
            }
        }
        result
    }

    /// Uses BFS from each atom to find all pairs within `max_distance` bonds.
    pub fn pairs_within(&self, max_distance: usize) -> HashSet<UnorderedPair<usize>> {
        let n = self.num_atoms();
        let mut pairs = HashSet::new();
        let mut distances = vec![None; n];
        let mut queue = VecDeque::new();

        for start in 0..n {
            distances.fill(None);
            distances[start] = Some(0);
            queue.push_back(start);

            while let Some(current) = queue.pop_front() {
                let d = distances[current].unwrap();

                if current != start {
                    pairs.insert(UnorderedPair(start, current));
                }

                if d < max_distance {
                    for &neighbour in &self.neighbors[current] {
                        if distances[neighbour].is_none() {
                            distances[neighbour] = Some(d + 1);
                            queue.push_back(neighbour);
                        }
                    }
                }
            }
        }
        pairs
    }
}

impl IsotropicTwobodyEnergy for Bond {
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        match &self.kind {
            BondKind::Harmonic(x) => x.isotropic_twobody_energy(distance_squared),
            BondKind::FENE(x) => x.isotropic_twobody_energy(distance_squared),
            BondKind::Morse(x) => x.isotropic_twobody_energy(distance_squared),
            BondKind::UreyBradley(x) => x.isotropic_twobody_energy(distance_squared),
            BondKind::Unspecified => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bond(i: usize, j: usize) -> Bond {
        Bond::new([i, j], BondKind::Unspecified, BondOrder::Unspecified)
    }

    #[test]
    fn bond_graph_branched() {
        // Branched molecule: 0-1-2-3(-4,-5-6), plus 3-6 and 4-6
        let bonds: Vec<Bond> = [
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
        .map(|&[i, j]| make_bond(i, j))
        .collect();

        let graph = BondGraph::from_bonds(&bonds, 7);
        assert_eq!(graph.num_atoms(), 7);
        assert!(!graph.is_empty());

        // Verify neighbor counts
        assert_eq!(graph.neighbors(0).len(), 1); // 0 -> [1]
        assert_eq!(graph.neighbors(1).len(), 2); // 1 -> [0, 2]
        assert_eq!(graph.neighbors(2).len(), 2); // 2 -> [1, 3]
        assert_eq!(graph.neighbors(3).len(), 4); // 3 -> [2, 4, 5, 6]
        assert_eq!(graph.neighbors(4).len(), 2); // 4 -> [3, 6]
        assert_eq!(graph.neighbors(5).len(), 2); // 5 -> [3, 6]
        assert_eq!(graph.neighbors(6).len(), 3); // 6 -> [5, 3, 4]

        // Verify symmetry: if j in neighbors(i), then i in neighbors(j)
        for i in 0..graph.num_atoms() {
            for &j in graph.neighbors(i) {
                assert!(
                    graph.neighbors(j).contains(&i),
                    "asymmetry: {j} in neighbors({i}) but {i} not in neighbors({j})"
                );
            }
        }
    }

    #[test]
    fn bond_graph_empty() {
        let graph = BondGraph::from_bonds(&[], 0);
        assert!(graph.is_empty());
        assert_eq!(graph.num_atoms(), 0);

        // No bonds but some atoms
        let graph = BondGraph::from_bonds(&[], 3);
        assert!(!graph.is_empty());
        assert_eq!(graph.num_atoms(), 3);
        assert!(graph.neighbors(0).is_empty());
        assert!(graph.neighbors(1).is_empty());
        assert!(graph.neighbors(2).is_empty());
    }

    #[test]
    fn bond_graph_default() {
        let graph = BondGraph::default();
        assert!(graph.is_empty());
        assert_eq!(graph.num_atoms(), 0);
    }

    #[test]
    fn connected_from_branched() {
        // Branched: 0-1-2-3(-4,-5-6), plus 3-6, 4-6
        let bonds: Vec<Bond> = [
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
        .map(|&[i, j]| make_bond(i, j))
        .collect();
        let graph = BondGraph::from_bonds(&bonds, 7);

        // Pivot at node 2, walk toward node 3 (away from 0-1 side)
        let mut result = graph.connected_from(3, 2);
        result.sort();
        assert_eq!(result, vec![3, 4, 5, 6]);

        // Pivot at node 2, walk toward node 1 (away from 3-6 side)
        let mut result = graph.connected_from(1, 2);
        result.sort();
        assert_eq!(result, vec![0, 1]);

        // Pivot must not appear in result
        let result = graph.connected_from(3, 2);
        assert!(!result.contains(&2));
    }

    #[test]
    fn connected_from_linear() {
        // Linear chain: 0-1-2-3-4
        let bonds: Vec<Bond> = [[0, 1], [1, 2], [2, 3], [3, 4]]
            .iter()
            .map(|&[i, j]| make_bond(i, j))
            .collect();
        let graph = BondGraph::from_bonds(&bonds, 5);

        let mut result = graph.connected_from(0, 1);
        result.sort();
        assert_eq!(result, vec![0]);

        let mut result = graph.connected_from(2, 1);
        result.sort();
        assert_eq!(result, vec![2, 3, 4]);
    }
}
