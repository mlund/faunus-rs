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

use anyhow::{bail, Ok};
use as_any::{AsAny, Downcast};
use interact::twobody::TwobodyEnergy;
use itertools::iproduct;
use std::fmt::Debug;

use crate::platform::reference::ReferencePlatform;
use crate::{
    cite::Citation, energy::EnergyTerm, Change, Group, GroupCollection, Particle, PointParticle,
    SyncFromAny,
};

#[derive(Debug, Clone)]
pub struct Nonbonded<'a, T: TwobodyEnergy> {
    /// Matrix of pair potentials base on particle ids
    pair_potentials: Vec<Vec<T>>,
    /// Reference to the platform
    platform: &'a ReferencePlatform,
}

impl<T> EnergyTerm for Nonbonded<'static, T>
where
    T: TwobodyEnergy + 'static,
{
    fn energy_change(&self, change: &Change) -> Option<f64> {
        let energy = match change {
            Change::Everything => self.all_with_all(),
            Change::SingleParticle(group_index, particle_index) => {
                self.particle_with_all(*group_index, *particle_index)
            }
            _ => todo!("implement other changes"),
        };
        Some(energy)
    }
}

impl<T> Nonbonded<'_, T>
where
    T: TwobodyEnergy + 'static,
{
    /// Calculates the energy between two particles
    #[inline]
    fn particle_with_particle(&self, particle1: &Particle, particle2: &Particle) -> f64 {
        let distance_squared = self
            .platform
            .cell
            .distance(particle1.pos(), particle2.pos())
            .powi(2); // TODO: use squared distance directly
        self.pair_potentials[particle1.id][particle2.id].twobody_energy(distance_squared)
    }

    /// Single particle with all remaining active particles
    fn particle_with_all(&self, group_index: usize, rel_index: usize) -> f64 {
        let groups = &self.platform.groups();
        let index = groups[group_index].absolute_index(rel_index).unwrap();
        groups
            .iter()
            .flat_map(|group| group.indices())
            .filter(|other| *other != index)
            .fold(0.0, |sum, other| {
                let particle1 = &self.platform.particles[index];
                let particle2 = &self.platform.particles[other];
                sum + self.particle_with_particle(particle1, particle2)
            })
    }

    /// Calculates the energy between two groups
    pub fn group_to_group(&self, group1: &Group, group2: &Group) -> f64 {
        let particles1 = self.platform.particles[group1.indices()].iter();
        let particles2 = self.platform.particles[group2.indices()].iter();
        iproduct!(particles1, particles2)
            .fold(0.0, |sum, (i, j)| sum + self.particle_with_particle(i, j))
    }

    /// Calculates the full energy of the system by summing over
    /// all group-to-group interactions
    pub fn all_with_all(&self) -> f64 {
        let groups = self.platform.groups();
        iproduct!(groups.iter(), groups.iter())
            .fold(0.0, |sum, (i, j)| sum + self.group_to_group(i, j))
    }
}

impl<T> Citation for Nonbonded<'_, T>
where
    T: TwobodyEnergy,
{
    fn citation(&self) -> Option<&'static str> {
        None
    }
}

impl<T> SyncFromAny for Nonbonded<'static, T>
where
    T: TwobodyEnergy + 'static,
{
    fn sync_from(&mut self, other: &dyn AsAny, change: &Change) -> anyhow::Result<()> {
        if let Some(other) = other.downcast_ref::<Self>() {
            match change {
                Change::Everything => {
                    *self = other.clone();
                }
                _ => todo!(),
            }
            return Ok(());
        }
        bail!("Could not downcast")
    }
}
