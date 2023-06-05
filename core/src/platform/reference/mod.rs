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

use crate::{
    energy::EnergyTerm,
    group::{GroupCollection, GroupSize},
    Change, Group, Particle, SyncFrom,
};
use anyhow::{anyhow, Ok};

pub mod nonbonded;

/// This is the default platform running on the CPU. Particles are stored in
/// a single vector, and groups are stored in a separate vector. This mostly
/// follows the same layout as the original C++ Faunus code (version 2 and lower).
#[derive(Debug)]
pub struct ReferencePlatform {
    particles: Vec<Particle>,
    groups: Vec<Group>,
    _cell: crate::cell::Cuboid,
    _energies: Vec<Box<dyn EnergyTerm>>,
}

impl Clone for ReferencePlatform {
    fn clone(&self) -> Self {
        todo!("clone boxed energies")
    }
}

impl SyncFrom for ReferencePlatform {
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything => {
                *self = other.clone();
            }
            _ => todo!(),
        }
        Ok(())
    }
}

impl GroupCollection for ReferencePlatform {
    fn groups(&self) -> &[Group] {
        self.groups.as_ref()
    }

    fn get_particles(&self, group_index: usize) -> Vec<Particle> {
        let indices = self.groups[group_index].indices();
        self.particles[indices].to_vec()
    }

    fn add_group(&mut self, id: Option<usize>, particles: &[Particle]) -> anyhow::Result<&Group> {
        if particles.is_empty() {
            return Err(anyhow!("Cannot create empty group"));
        }
        let range = self.particles.len()..self.particles.len() + particles.len();
        self.particles.extend_from_slice(particles);
        self.groups.push(Group::new(self.groups.len(), id, range));
        Ok(self.groups.last().unwrap())
    }

    fn resize_group(&mut self, group_index: usize, status: GroupSize) -> anyhow::Result<()> {
        self.groups[group_index].resize(status)
    }

    fn set_particles<'a>(
        &mut self,
        group_index: usize,
        particles: impl Iterator<Item = &'a Particle>,
    ) -> anyhow::Result<()> {
        let group = &self.groups[group_index];
        self.set_indexed_particles(group_index, particles, 0..group.len())
    }

    fn set_indexed_particles<'a>(
        &mut self,
        group_index: usize,
        particles: impl Iterator<Item = &'a Particle>,
        relative_indices: impl Iterator<Item = usize>,
    ) -> anyhow::Result<()> {
        let group = &mut self.groups[group_index];
        let mut particles = particles.peekable();
        if particles.peek().is_none() {
            return Err(anyhow!("Cannot set empty particle list"));
        }
        for (count, (particle, index)) in particles.zip(relative_indices).enumerate() {
            if count > group.capacity() {
                return Err(anyhow!(
                    "Particle index {} exceeds group capacity {}",
                    count,
                    group.capacity()
                ));
            }
            self.particles[group.all_indices().start + index] = particle.clone();
        }
        Ok(())
    }

    fn get_indexed_particles(
        &self,
        group_index: usize,
        indices: impl Iterator<Item = usize> + Clone,
    ) -> Vec<Particle> {
        let group = &self.groups[group_index];
        if !indices.clone().all(|i| i < group.capacity()) {
            panic!("Particle index exceeds group capacity");
        }

        indices
            .map(|i_rel| i_rel + group.all_indices().start)
            .map(|i_abs| self.particles[i_abs].clone())
            .collect::<Vec<Particle>>()
    }
}

/// Group-wise collection of particles
///
/// Particles are grouped into groups, which are defined by a slice of particles.
/// Each group could be a rigid body, a molecule, etc.
/// The idea is to access the particle in a group-wise fashion, e.g. to update
/// the center of mass of a group, or to rotate a group of particles.
impl ReferencePlatform {
    pub fn new(cell: crate::cell::lumol::UnitCell) -> Self {
        Self {
            particles: Vec::new(),
            groups: Vec::new(),
            _cell: cell,
            _energies: Vec::new(),
        }
    }

    /// Get vector of indices to all other *active* particles in the system, excluding `range`
    fn _other_indices(&self, range: std::ops::Range<usize>) -> Vec<usize> {
        let no_overlap = |r: &std::ops::Range<usize>| {
            usize::max(r.start, range.start) > usize::min(r.end, range.end)
        };
        self.groups
            .iter()
            .map(|g| g.indices())
            .filter(no_overlap)
            .flatten()
            .collect()
    }
}
