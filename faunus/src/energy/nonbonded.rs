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

//! Implementation of the Nonbonded energy terms.

use interatomic::twobody::IsotropicTwobodyEnergy;
use itertools::iproduct;
use std::fmt::Debug;

use crate::{
    energy::{builder::NonbondedBuilder, EnergyTerm},
    topology::{Topology, TopologyLike},
    Change, Context, Group, GroupChange, SyncFrom,
};

use super::exclusions::ExclusionMatrix;

/// Energy term for computing nonbonded interactions
/// using a matrix of `IsotropicTwobodyEnergy` trait objects.
#[derive(Debug, Clone)]
pub struct NonbondedMatrix {
    /// Matrix of pair potentials based on particle ids.
    potentials: Vec<Vec<Box<dyn IsotropicTwobodyEnergy>>>,
    /// Matrix of excluded interactions.
    exclusions: ExclusionMatrix,
}
impl NonbondedMatrix {
    /// Create a new NonbondedReference structure wrapped in an EnergyTerm enum.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(
        nonbonded: &NonbondedBuilder,
        topology: &Topology,
    ) -> anyhow::Result<EnergyTerm> {
        let atoms = topology.atoms();

        let potentials: Vec<Vec<_>> = atoms
            .iter()
            .map(|type1| {
                atoms
                    .iter()
                    .map(|type2| nonbonded.get_interaction(type1, type2))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(EnergyTerm::NonbondedMatrix(NonbondedMatrix {
            potentials,
            exclusions: ExclusionMatrix::default(),
        }))
    }

    /// Compute the energy change due to a change in the system.
    pub(super) fn energy_change(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything => self.all_with_all(context),
            Change::SingleGroup(group_index, group_change) => {
                self.single_group_change(context, *group_index, group_change)
            }
            Change::None => 0.0,
            _ => todo!("implement other changes"),
        }
    }

    /// Matches all possible single group perturbations and returns the energy.
    fn single_group_change(
        &self,
        context: &impl Context,
        group_index: usize,
        change: &GroupChange,
    ) -> f64 {
        match change {
            GroupChange::RigidBody => self.group_with_all(context, group_index),
            GroupChange::None => 0.0,
            _ => todo!("implement other group changes"),
        }
    }

    /// Calculates the energy between two particles given by indices.
    ///
    /// ## Parameters
    /// - `context` Context to work with.
    /// - `i, j` Indices of the particles in the `context`.
    /// - `atom_kind_i, atom_kind_j` Indices of the atom kinds in the `context`.
    #[inline(always)]
    fn particle_with_particle(
        &self,
        context: &impl Context,
        i: usize,
        j: usize,
        atom_kind_i: usize,
        atom_kind_j: usize,
    ) -> f64 {
        let distance_squared = context.get_distance_squared(i, j);
        self.exclusions.get(i, j) as f64
            * self.potentials[atom_kind_i][atom_kind_j].isotropic_twobody_energy(distance_squared)
    }

    /// Single particle with all remaining active particles.
    fn particle_with_all(
        &self,
        context: &impl Context,
        group_index: usize,
        rel_index: usize,
    ) -> f64 {
        let groups = &context.groups();
        let index = groups[group_index].absolute_index(rel_index).unwrap();
        let atomkind_index = context.get_atomkind(index);
        groups
            .iter()
            .flat_map(|group| group.iter_active())
            .filter(|other| *other != index)
            .fold(0.0, |sum, other| {
                let atomkind_other = context.get_atomkind(other);
                sum + self.particle_with_particle(
                    context,
                    index,
                    other,
                    atomkind_index,
                    atomkind_other,
                )
            })
    }

    /// Calculates the energy between a single group and all other groups.
    pub fn group_with_all(&self, context: &impl Context, group_index: usize) -> f64 {
        let group = &context.groups()[group_index];
        context
            .groups()
            .iter()
            .filter(|other_group| other_group.index() != group_index)
            .fold(0.0, |sum, other_group| {
                sum + self.group_to_group(context, group, other_group)
            })
    }

    /// Calculates the energy between two groups.
    pub fn group_to_group(&self, context: &impl Context, group1: &Group, group2: &Group) -> f64 {
        let particles1 = group1.iter_active();
        let particles2 = group2.iter_active();
        iproduct!(particles1, particles2).fold(0.0, |sum, (i, j)| {
            let atomkind_i = context.get_atomkind(i);
            let atomkind_j = context.get_atomkind(j);
            sum + self.particle_with_particle(context, i, j, atomkind_i, atomkind_j)
        })
    }

    /// Calculates the full energy of the system by summing over
    /// all group-to-group interactions.
    pub fn all_with_all(&self, context: &impl Context) -> f64 {
        let groups = context.groups();
        iproduct!(groups.iter(), groups.iter())
            .fold(0.0, |sum, (i, j)| sum + self.group_to_group(context, i, j))
    }
}

impl SyncFrom for NonbondedMatrix {
    fn sync_from(&mut self, other: &NonbondedMatrix, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything => self.potentials = other.potentials.clone(),
            Change::None => (),
            _ => todo!("Implement other changes."),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use crate::{energy::builder::HamiltonianBuilder, topology::Topology};

    use super::*;

    /// Compare behavior of two `IsotropicTwobodyEnergy` trait objects.
    fn assert_behavior(
        obj1: &Box<dyn IsotropicTwobodyEnergy>,
        obj2: &Box<dyn IsotropicTwobodyEnergy>,
    ) {
        let testing_distances = [0.00201, 0.7, 12.3, 12457.6];

        for &dist in testing_distances.iter() {
            assert_approx_eq!(
                f64,
                obj1.isotropic_twobody_energy(dist),
                obj2.isotropic_twobody_energy(dist)
            );
        }
    }

    #[test]
    fn test_nonbonded_reference_new() {
        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml")
            .unwrap()
            .nonbonded;

        let nonbonded = NonbondedMatrix::new(&builder, &topology).unwrap();
        let nonbonded = match nonbonded {
            EnergyTerm::NonbondedMatrix(x) => x,
            _ => panic!("Incorrect Energy Term constructed."),
        };

        assert_eq!(nonbonded.potentials.len(), topology.atoms().len());
        for potential in nonbonded.potentials.iter() {
            assert_eq!(potential.len(), topology.atoms().len());
        }

        for i in 0..topology.atoms().len() {
            for j in (i + 1)..topology.atoms().len() {
                assert_behavior(&nonbonded.potentials[i][j], &nonbonded.potentials[j][i]);
            }
        }

        // O, C with anything: default interaction
        let o_index = topology
            .atoms()
            .iter()
            .position(|x| x.name() == "O")
            .unwrap();
        let c_index = topology
            .atoms()
            .iter()
            .position(|x| x.name() == "C")
            .unwrap();

        let default = &nonbonded.potentials[o_index][o_index];

        for i in [o_index, c_index] {
            for j in 0..topology.atoms().len() {
                assert_behavior(&nonbonded.potentials[i][j], default);
            }
        }

        // X interacts slightly differently with charged atoms because it is itself charged
        let x_index = topology
            .atoms()
            .iter()
            .position(|x| x.name() == "X")
            .unwrap();
        let ow_index = topology
            .atoms()
            .iter()
            .position(|x| x.name() == "OW")
            .unwrap();

        for i in 0..topology.atoms().len() {
            if i == x_index || i == ow_index {
                continue;
            }

            assert_behavior(&nonbonded.potentials[x_index][i], default);
        }
    }
}
