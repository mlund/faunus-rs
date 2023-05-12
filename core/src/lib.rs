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

use crate::group::{Group, GroupCollection};
use anyhow::{anyhow, Ok};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

pub type Point = Vector3<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

pub mod cell;
pub mod cite;
pub mod group;
pub mod montecarlo;
pub mod time;
pub mod transform;

use crate::group::GroupChange;

trait PointParticle {
    /// Type of the particle identifier
    type Idtype;
    /// Type of the particle position
    type Positiontype;
    /// Identifier for the particle type
    fn id(&self) -> Self::Idtype;
    /// Get position
    fn pos(&self) -> &Self::Positiontype;
    /// Get mutable position
    fn pos_mut(&mut self) -> &mut Self::Positiontype;
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct Particle {
    id: usize,
    pos: Point,
}

impl PointParticle for Particle {
    type Idtype = usize;
    type Positiontype = Point;
    fn id(&self) -> Self::Idtype {
        self.id
    }
    fn pos(&self) -> &Self::Positiontype {
        &self.pos
    }
    fn pos_mut(&mut self) -> &mut Self::Positiontype {
        &mut self.pos
    }
}

pub struct Change {
    /// Whether the volume has changed
    pub volume_move: bool,

    /// If true, assume that everything has changed
    pub everything: bool,

    /// Partial list of groups that have changed
    pub groups: Vec<GroupChange>,
}

pub trait Context:
    GroupCollection + cell::SimulationCell + Clone + Default + std::fmt::Debug + Sized
{
    /// Get list of energies in the system
    fn energies(&self) {}
}

/// Collection of particles
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct DefaultContext {
    particles: ParticleVec,
    groups: Vec<Group>,
    // hamiltonian: Hamiltonian,
}

impl GroupCollection for DefaultContext {
    fn sync_from(&mut self, other_context: &Self, change: &Change) -> anyhow::Result<()> {
        if change.everything {
            self.particles = other_context.particles.clone();
            self.groups = other_context.groups.clone();
        } else {
            for group_change in change.groups.iter() {}
        }
        Ok(())
    }

    fn groups(&self) -> &[Group] {
        self.groups.as_ref()
    }

    fn group_particles(&self, group_index: usize) -> Vec<Particle> {
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

    fn set_particles<'a>(
        &mut self,
        group_index: usize,
        particles: impl Iterator<Item = &'a Particle>,
    ) -> anyhow::Result<()> {
        let group = &self.groups[group_index];
        self.set_particles_partial(group_index, particles, 0..group.len())
    }

    fn set_particles_partial<'a>(
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

    fn group_particles_partial(
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
impl DefaultContext {
    /// Get vector of indices to all other *active* particles in the system, excluding `range`
    pub fn other_indices(&self, range: std::ops::Range<usize>) -> Vec<usize> {
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
