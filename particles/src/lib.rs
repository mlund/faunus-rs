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

use anyhow::{anyhow, Ok};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

pub type Point = Vector3<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

pub mod cell;

trait PointParticle {
    /// Type of the particle identifier
    type Idtype;
    /// Type of the particle position
    type Positiontype;
    /// Identifier for the particle type
    fn id(&self) -> Self::Idtype;
    /// Get position
    fn pos(&self) -> &Self::Positiontype;
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
}

/// Activation status of a group of particles
#[derive(Serialize, Deserialize, Default, Copy, Clone, PartialEq, Debug)]
pub enum GroupSize {
    /// All particles are active and no more can be added
    #[default]
    Full,
    /// All particles are inactive and no more can be removed
    Empty,
    /// Some particles are active
    Partial(usize),
    /// Expand with `usize` particles
    Expand(usize),
    /// Shrink with `usize` particles
    Shrink(usize),
}

/// Description of a change to a single group of particles
///
/// Defines a change to a group of particles, e.g. a rigid body update,
/// adding or removing particles, etc. It is used in connection with Monte Carlo
/// moves to communicate an update to e.g. the Hamiltonian.
pub enum GroupChange {
    /// Rigid body update where *all* particles are e.g. rotated or translated with *no* internal energy change
    RigidBodyUpdate,
    /// Update a range of particle relative indices, assuming that the internal energy changes
    Update(Vec<usize>),
    /// Add `usize` particles at end
    Push(usize),
    /// Remove `usize` particles from end
    Pop(usize),
    /// The `id` of a set of particles has changed (relative indices)
    UpdateIdentity(Vec<usize>),
    /// Deactivate *all* particles
    Deactivate,
    /// Activate *all* particles
    Activate,
}

impl GroupChange {
    pub fn internal_change(&self) -> bool {
        !matches!(self, GroupChange::RigidBodyUpdate)
    }
}

#[derive(Serialize, Deserialize, Default, Debug, PartialEq, Clone)]
pub struct Group<'a> {
    /// Molecular type id (immutable)
    id: Option<usize>,
    /// Index in group vector (immutable)
    index: usize,
    /// Optional mass center
    mass_center: Option<Point>,
    /// Active number of active particles
    num_active: usize,
    /// Slice of particles matching `range` (active and inactive)
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    particles: &'a [Particle],
    /// Indices in main particle vector (active and inactive; immutable)
    range: std::ops::Range<usize>,
    /// Size status
    size_status: GroupSize,
}

impl<'a> Iterator for Group<'a> {
    type Item = &'a Particle;
    /// Iterate over active particles
    fn next(&mut self) -> Option<Self::Item> {
        self.particles.iter().take(self.num_active).next()
    }
}

impl<'a> Group<'a> {
    /// Resize group within its capacity.
    ///
    /// The group can e.g. be set to full, empty, or to a specific number of active particles.
    /// This is used in connection with Grand Canonical Monte Carlo moves to add or remove particles or
    /// molecules.
    /// If resizing to zero, the group is `Empty` and considered *inactive*. If resizing to the
    /// capacity, the group is `Full` and considered *active*. Otherwise, the group is `Partial`.
    /// It is also possible to `Expand` or `Shrink` the group by a given number of particles. This
    /// is useful when adding or removing particles in a Grand Canonical Monte Carlo move.
    ///
    /// An error is returned if the requested size is larger than the capacity, or if there are
    /// too few active particles to shrink the group.
    pub fn resize(&mut self, status: GroupSize) -> anyhow::Result<()> {
        self.size_status = status;
        self.num_active = match self.size_status {
            GroupSize::Full => self.capacity(),
            GroupSize::Empty => 0,
            GroupSize::Partial(n) => match n {
                0 => return self.resize(GroupSize::Empty),
                n if n == self.capacity() => return self.resize(GroupSize::Full),
                _ => {
                    if n > self.capacity() {
                        return Err(anyhow::anyhow!(
                            "Cannot set group size to {} (max {})",
                            n,
                            self.capacity()
                        ));
                    }
                    n
                }
            },
            GroupSize::Expand(n) => return self.resize(GroupSize::Partial(self.num_active + n)),
            GroupSize::Shrink(n) => {
                return self.resize(GroupSize::Partial(
                    usize::checked_sub(self.num_active, n)
                        .ok_or(anyhow::anyhow!("Cannot shrink group by {} particles", n))?,
                ));
            }
        };
        Ok(())
    }

    /// Get activation status
    pub fn status(&self) -> GroupSize {
        self.size_status
    }

    /// Get molecular id
    #[inline]
    pub fn id(&self) -> Option<usize> {
        self.id
    }
    /// Maximum number of particles (active plus inactive)
    pub fn capacity(&self) -> usize {
        self.range.len()
    }

    /// Number of active particles
    pub fn len(&self) -> usize {
        self.num_active
    }

    /// True if no active particles
    pub fn is_empty(&self) -> bool {
        self.num_active == 0
    }

    /// Range of active _indices_ in full particle vector
    pub fn indices(&self) -> std::ops::Range<usize> {
        std::ops::Range {
            start: self.range.start,
            end: self.range.start + self.num_active,
        }
    }

    /// Index of group in group vector
    pub fn index(&self) -> usize {
        self.index
    }

    /// Iterator to inactive particles
    pub fn inactive(&self) -> impl Iterator<Item = &'a Particle> {
        self.particles.iter().skip(self.num_active)
    }
    /// Center of mass of the group
    pub fn mass_center(&self) -> Option<&Point> {
        self.mass_center.as_ref()
    }

    pub fn set_particles(&mut self, particles: &'a [Particle]) {
        self.particles = particles;
    }
}

pub enum GroupId {
    /// Molecular type id
    Id(usize),
    /// No molecular type id
    None,
}

pub trait Context {
    /// List of all groups in the system
    fn groups(&self) -> &[Group];

    /// Set volume of system
    fn set_volume(&mut self, volume: f64) -> anyhow::Result<()>;

    /// Get volume of system
    fn volume(&self) -> f64;

    /// Add a group to the system
    fn add_group(&mut self, id: GroupId, particles: &[Particle]) -> anyhow::Result<&mut Group>;

    /// Get list of energies in the system
    fn energies(&self);
}

/// Collection of particles
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct ParticleCollection {
    particles: ParticleVec,
    groups: Vec<Group<'static>>,
    // hamiltonian: Hamiltonian,
}

/// Group-wise collection of particles
///
/// Particles are grouped into groups, which are defined by a slice of particles.
/// Each group could be a rigid body, a molecule, etc.
/// The idea is to access the particle in a group-wise fashion, e.g. to update
/// the center of mass of a group, or to rotate a group of particles.
impl ParticleCollection {
    /// Vector of *all* particles in the system (active and inactive)
    pub fn particles(&self) -> &[Particle] {
        self.particles.as_ref()
    }

    /// Vector of groups in the system
    pub fn groups(&self) -> &[Group] {
        self.groups.as_ref()
    }

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

    /// Append a new group, created from an external slice of particles
    ///
    /// - the particles are cloned and appended to the particle vector
    /// - the new group is inserted at the end of the group vector
    pub fn push_group(
        &'static mut self,
        id: Option<usize>,
        particles: &[Particle],
    ) -> anyhow::Result<&mut Group> {
        if particles.is_empty() {
            return Err(anyhow!("Cannot create empty group"));
        }
        let range = self.particles.len()..self.particles.len() + particles.len();
        self.particles.extend_from_slice(particles);
        self.groups.push(Group {
            id,
            index: self.groups.len(),
            mass_center: None,
            num_active: range.len(),
            particles: &self.particles[range.clone()],
            range,
            size_status: GroupSize::Full,
        });
        // Ensure that group slices are valid after the (potential) group vector resize
        self.groups.iter_mut().for_each(|g| {
            g.particles = &self.particles[g.range.clone()];
        });
        Ok(self.groups.last_mut().unwrap())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_group() {}
}
