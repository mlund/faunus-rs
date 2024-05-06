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

use anyhow::Ok;
use as_any::{AsAny, Downcast};
use interatomic::twobody::IsotropicTwobodyEnergy;
use itertools::iproduct;
use serde::Serialize;
use std::fmt::Debug;

use super::ReferencePlatform;
use crate::{
    cell::BoundaryConditions, energy::EnergyTerm, Change, Group, GroupChange, GroupCollection,
    Info, Particle, SyncFrom,
};

/// Interface for nonbonded interactions.
///
/// # Todo
/// Move out of the reference platform module as it's (mostly) platform independent.
pub trait NonbondedInterface {
    fn platform(&self) -> &ReferencePlatform;

    /// Calculates the energy between two particles
    fn particle_with_particle(&self, particle1: &Particle, particle2: &Particle) -> f64;

    /// Matches all possible single group perturbations and returns the energy
    fn single_group_change(&self, group_index: usize, change: &GroupChange) -> f64 {
        match change {
            GroupChange::RigidBody => self.group_with_all(group_index),
            GroupChange::None => 0.0,
            _ => todo!("implement other group changes"),
        }
    }

    /// Calculates the energy between two groups
    fn group_to_group(&self, group1: &Group, group2: &Group) -> f64 {
        let particles1 = self.platform().particles[group1.iter_active()].iter();
        let particles2 = self.platform().particles[group2.iter_active()].iter();
        iproduct!(particles1, particles2)
            .map(|(i, j)| self.particle_with_particle(i, j))
            .sum()
    }

    /// Calculates the energy between a single group and all other groups
    fn group_with_all(&self, group_index: usize) -> f64 {
        let group = &self.platform().groups()[group_index];
        self.platform()
            .groups()
            .iter()
            .filter(|other_group| other_group.index() != group_index)
            .map(|other_group| self.group_to_group(group, other_group))
            .sum()
    }
    /// Calculates the full energy of the system by summing over
    /// all group-to-group interactions
    fn all_with_all(&self) -> f64 {
        let groups = &self.platform().groups();
        iproduct!(groups.iter(), groups.iter())
            .map(|(i, j)| self.group_to_group(i, j))
            .sum()
    }
}

// impl<T: NonbondedInterface> EnergyTerm for T {
//     fn energy_change(&self, change: &Change) -> f64 {
//         match change {
//             Change::Everything => self.all_with_all(),
//             Change::SingleGroup(group_index, group_change) => {
//                 self.single_group_change(*group_index, group_change)
//             }
//             Change::None => 0.0,
//             _ => todo!("implement other changes"),
//         }
//     }

//     fn update(&mut self, _change: &Change) -> anyhow::Result<()> {
//         Ok(())
//     }
// }

impl<T> NonbondedInterface for Nonbonded<'_, T>
where
    T: IsotropicTwobodyEnergy + 'static,
{
    fn platform(&self) -> &ReferencePlatform {
        self.platform
    }

    fn particle_with_particle(&self, particle1: &Particle, particle2: &Particle) -> f64 {
        let distance_squared = self
            .platform()
            .cell
            .distance_squared(&particle1.pos, &particle2.pos);
        self.pair_potentials[particle1.id][particle2.id].isotropic_twobody_energy(distance_squared)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Nonbonded<'a, T: IsotropicTwobodyEnergy> {
    /// Matrix of pair potentials base on particle ids
    pair_potentials: Vec<Vec<T>>,
    /// Reference to the platform
    #[serde(skip)]
    platform: &'a ReferencePlatform,
}

impl<T> EnergyTerm for Nonbonded<'static, T>
where
    T: IsotropicTwobodyEnergy + 'static + Clone,
{
    fn energy_change(&self, change: &Change) -> f64 {
        match change {
            Change::Everything => self.all_with_all(),
            Change::SingleGroup(group_index, group_change) => {
                self.single_group_change(*group_index, group_change)
            }
            Change::None => 0.0,
            _ => todo!("implement other changes"),
        }
    }

    fn update(&mut self, _change: &Change) -> anyhow::Result<()> {
        Ok(())
    }
}

impl<T> Nonbonded<'_, T>
where
    T: IsotropicTwobodyEnergy + 'static,
{
    pub fn new(platform: &'static ReferencePlatform) -> Self {
        // TODO: Here we should fill out the pair potential matrix by looping
        // over all particle id's.
        let pair_potentials = Vec::new();
        Self {
            pair_potentials,
            platform,
        }
    }
}

impl<T> Info for Nonbonded<'_, T>
where
    T: IsotropicTwobodyEnergy,
{
    fn citation(&self) -> Option<&'static str> {
        None
    }
}

impl<T> SyncFrom for Nonbonded<'static, T>
where
    T: IsotropicTwobodyEnergy + 'static + Clone,
{
    fn sync_from(&mut self, other: &dyn AsAny, change: &Change) -> anyhow::Result<()> {
        let other = other
            .downcast_ref::<Self>()
            .ok_or_else(|| anyhow::anyhow!("Could not downcast"))?;

        match change {
            Change::Everything => *self = other.clone(),
            _ => todo!(),
        }
        Ok(())
    }
}
