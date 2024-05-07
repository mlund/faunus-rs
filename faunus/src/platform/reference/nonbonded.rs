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
use std::rc::Rc;

use super::ReferencePlatform;
use crate::{
    cell::BoundaryConditions, energy::EnergyTerm, topology::TopologyLike, Change, Group,
    GroupChange, GroupCollection, Particle, PointParticle, SyncFrom,
};

/// Common interface for nonbonded interactions.
///
/// # Todo
/// This should ideally use a Context instead or ReferencePlatform to be fully platform independent.
trait NonbondedCommon {
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

impl<T: NonbondedCommon + SyncFrom + std::fmt::Debug + 'static> EnergyTerm for T {
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

/// Nonbonded interactions with pointer to pair potentials.
pub struct NonbondedDynamic {
    pair_potentials: Vec<Vec<Box<dyn IsotropicTwobodyEnergy>>>,
    platform: Rc<ReferencePlatform>,
}

impl NonbondedDynamic {
    /// Sets a default pair potential for all `AtomKind` pairs.
    pub fn with_default(
        platform: Rc<ReferencePlatform>,
        default_pot: impl IsotropicTwobodyEnergy + Clone + 'static,
    ) -> Self {
        let n = platform.topology.atoms().len();
        let make_row = || {
            (0..n)
                .map(|_| Box::new(default_pot.clone()) as Box<dyn IsotropicTwobodyEnergy>)
                .collect()
        };
        Self {
            pair_potentials: (0..n).map(|_| make_row()).collect(),
            platform,
        }
    }

    /// Sets a pair potential for a specific pair of `AtomKind`s.
    pub fn set_pair_potential(
        &mut self,
        pair: (usize, usize),
        pair_pot: impl IsotropicTwobodyEnergy + Clone + 'static,
    ) {
        self.pair_potentials[pair.0][pair.1] = Box::new(pair_pot.clone());
        self.pair_potentials[pair.1][pair.0] = Box::new(pair_pot);
    }
}

impl NonbondedCommon for NonbondedDynamic {
    fn platform(&self) -> &ReferencePlatform {
        &self.platform
    }

    #[inline]
    fn particle_with_particle(&self, particle1: &Particle, particle2: &Particle) -> f64 {
        let r2 = self
            .platform()
            .cell
            .distance_squared(&particle1.pos, &particle2.pos);
        self.pair_potentials[particle1.atom_id()][particle2.atom_id()].isotropic_twobody_energy(r2)
    }
}

///  Nonbonded interactions with a single, static pair potential for all `AtomKind` pairs.
#[derive(Clone, Serialize)]
pub struct NonbondedStatic<'a, T> {
    /// Matrix of pair potentials base on `AtomKind` id
    pair_potentials: Vec<Vec<T>>,
    /// Reference to the platform
    #[serde(skip)]
    platform: &'a ReferencePlatform,
}

impl<T> NonbondedStatic<'_, T>
where
    T: IsotropicTwobodyEnergy + Clone + 'static,
{
    pub fn with_default(platform: &'static ReferencePlatform, default_pot: T) -> Self {
        let n = platform.topology.atoms().len();
        let row = (0..n).map(|_| default_pot.clone()).collect();
        Self {
            pair_potentials: vec![row; n],
            platform,
        }
    }
}

impl<T> NonbondedCommon for NonbondedStatic<'_, T>
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
            .distance_squared(particle1.pos(), particle2.pos());
        self.pair_potentials[particle1.atom_id()][particle2.atom_id()]
            .isotropic_twobody_energy(distance_squared)
    }
}

impl<T> SyncFrom for NonbondedStatic<'static, T>
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
