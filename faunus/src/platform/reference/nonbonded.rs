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

use interatomic::twobody::IsotropicTwobodyEnergy;
use itertools::iproduct;
use std::fmt::Debug;

use crate::{
    energy::{builder::NonbondedBuilder, EnergyTerm},
    platform::reference::ReferencePlatform,
    topology::{Topology, TopologyLike},
    Change, Group, GroupChange, GroupCollection, Particle, SyncFrom,
};

/// Energy term for computing nonbonding interactions implemented for the reference platform.
#[derive(Debug, Clone)]
pub struct NonbondedReference {
    /// Matrix of pair potentials based on particle ids.
    potentials: Vec<Vec<Box<dyn IsotropicTwobodyEnergy>>>,
}
impl NonbondedReference {
    /// Create a new NonbondedReference structure wrapped in an EnergyTerm enum.
    pub(crate) fn new(
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

        Ok(EnergyTerm::NonbondedReference(NonbondedReference {
            potentials,
        }))
    }

    /// Compute the energy change due to a change in the system.
    pub(crate) fn energy_change(&self, context: &ReferencePlatform, change: &Change) -> f64 {
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
        context: &ReferencePlatform,
        group_index: usize,
        change: &GroupChange,
    ) -> f64 {
        match change {
            GroupChange::RigidBody => self.group_with_all(context, group_index),
            GroupChange::None => 0.0,
            _ => todo!("implement other group changes"),
        }
    }

    /// Calculates the energy between two particles.
    #[inline(always)]
    fn particle_with_particle(&self, particle1: &Particle, particle2: &Particle) -> f64 {
        let distance_squared = 0.0;
        // let distance_squared = self
        //     .platform
        //     .distance_squared(particle1.pos(), particle2.pos());
        self.potentials[particle1.atom_id][particle2.atom_id]
            .isotropic_twobody_energy(distance_squared)
    }

    /// Single particle with all remaining active particles.
    fn particle_with_all(
        &self,
        context: &ReferencePlatform,
        group_index: usize,
        rel_index: usize,
    ) -> f64 {
        let groups = &context.groups();
        let index = groups[group_index].absolute_index(rel_index).unwrap();
        groups
            .iter()
            .flat_map(|group| group.iter_active())
            .filter(|other| *other != index)
            .fold(0.0, |sum, other| {
                let particle1 = &context.particles()[index];
                let particle2 = &context.particles()[other];
                sum + self.particle_with_particle(particle1, particle2)
            })
    }

    /// Calculates the energy between a single group and all other groups.
    pub(crate) fn group_with_all(&self, context: &ReferencePlatform, group_index: usize) -> f64 {
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
    pub(crate) fn group_to_group(
        &self,
        context: &ReferencePlatform,
        group1: &Group,
        group2: &Group,
    ) -> f64 {
        let particles1 = context.particles()[group1.iter_active()].iter();
        let particles2 = context.particles()[group2.iter_active()].iter();
        iproduct!(particles1, particles2)
            .fold(0.0, |sum, (i, j)| sum + self.particle_with_particle(i, j))
    }

    /// Calculates the full energy of the system by summing over
    /// all group-to-group interactions.
    pub(crate) fn all_with_all(&self, context: &ReferencePlatform) -> f64 {
        let groups = context.groups();
        iproduct!(groups.iter(), groups.iter())
            .fold(0.0, |sum, (i, j)| sum + self.group_to_group(context, i, j))
    }
}

impl SyncFrom for NonbondedReference {
    fn sync_from(&mut self, other: &NonbondedReference, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything => self.potentials = other.potentials.clone(),
            Change::None => (),
            _ => todo!("Implement other changes."),
        }

        Ok(())
    }
}
