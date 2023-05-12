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

//! Handling of groups of particles

use crate::Change;
use crate::Particle;
use anyhow::Ok;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

pub type Point = Vector3<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

#[derive(Serialize, Deserialize, Default, Debug, PartialEq, Clone)]
pub struct Group {
    /// Molecular type id (immutable)
    id: Option<usize>,
    /// Index of group in main group vector (immutable and unique)
    index: usize,
    /// Optional mass center
    mass_center: Option<Point>,
    /// Active number of active particles
    num_active: usize,
    /// Absolute indices in main particle vector (active and inactive; immutable and unique)
    range: std::ops::Range<usize>,
    /// Size status
    size_status: GroupSize,
}

/// Activation status of a group of particles
///
/// This can be used to set the number of active particles in a group. The group can e.g. be set to
/// full, empty, or to a specific number of active particles. This is used in connection with Grand
/// Canonical Monte Carlo moves to add or remove particles or molecules.
/// If resizing to zero, the group is `Empty` and considered *inactive*. If resizing to the
/// capacity, the group is `Full` and considered *active*. Otherwise, the group is `Partial`.
#[derive(Serialize, Deserialize, Default, Copy, Clone, PartialEq, Debug)]
pub enum GroupSize {
    /// All particles are active and no more can be added
    #[default]
    Full,
    /// All particles are inactive and no more can be removed
    Empty,
    /// Some (usize) particles are active
    Partial(usize),
    /// Expand with `usize` particles
    Expand(usize),
    /// Shrink with `usize` particles
    Shrink(usize),
}

impl Group {
    /// Create a new group
    pub fn new(index: usize, id: Option<usize>, range: std::ops::Range<usize>) -> Self {
        Self {
            id,
            index,
            range: range.clone(),
            num_active: range.len(),
            ..Default::default()
        }
    }
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

    /// Get size status of the groups which can be `Full`, `Empty`, or `Partial`.
    pub fn size(&self) -> GroupSize {
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

    /// Range of (absolute) active indices in main particle vector
    pub fn indices(&self) -> std::ops::Range<usize> {
        std::ops::Range {
            start: self.range.start,
            end: self.range.start + self.num_active,
        }
    }

    /// Range of all absolute indices in main particle vector
    pub fn all_indices(&self) -> std::ops::Range<usize> {
        self.range.clone()
    }

    /// Index of group in group vector
    pub fn index(&self) -> usize {
        self.index
    }

    /// Iterator to inactive indices (absolute indices in main particle vector)
    pub fn inactive(&self) -> impl Iterator<Item = usize> {
        self.range.clone().skip(self.num_active)
    }
    /// Center of mass of the group
    pub fn mass_center(&self) -> Option<&Point> {
        self.mass_center.as_ref()
    }
    /// Set mass center
    pub fn set_mass_center(&mut self, mass_center: Point) {
        self.mass_center = Some(mass_center);
    }
}

/// Trait for groups of particles
///
/// Each group has a unique index in the group vector, and a unique range of indices in the
/// particle vector.
pub trait GroupCollection {
    /// Add a group to the system based on an id and a set of particles given by an iterator.
    fn add_group(&mut self, id: Option<usize>, particles: &[Particle]) -> anyhow::Result<&Group>;

    /// List of all groups in the system
    fn groups(&self) -> &[Group];

    /// Get copy of active particles for a given group
    fn group_particles(&self, group_index: usize) -> Vec<Particle>;

    /// Get copy of particles in a subset of particles for a given group.
    /// The indices are indices in the group, not in the full particle vector.
    fn group_particles_partial(
        &self,
        group_index: usize,
        indices: impl Iterator<Item = usize> + Clone,
    ) -> Vec<Particle>;

    /// Set particles for a given group. Errors if there's a mismatch with the current group size.
    fn set_particles<'a>(
        &mut self,
        group_index: usize,
        particles: impl Iterator<Item = &'a Particle>,
    ) -> anyhow::Result<()>;

    /// Set a subset of particles for a given group. Errors if there's a mismatch with the current group size.
    /// The indices are in the group, not in the full particle vector, and may be in the full capacity range of the group.
    fn set_particles_partial<'a>(
        &mut self,
        group_index: usize,
        particles: impl Iterator<Item = &'a Particle>,
        indices: impl Iterator<Item = usize>,
    ) -> anyhow::Result<()>;

    /// Sync properties from another group collection, based on a `Change` object.
    /// After this operation, the two group collections should be identical.
    fn sync_from(&mut self, other_context: &Self, change: &Change) -> anyhow::Result<()>;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group() {
        // Test group creation
        let mut group = Group {
            id: Some(20),
            index: 2,
            mass_center: None,
            num_active: 6,
            range: 0..10,
            size_status: GroupSize::Partial(6),
        };

        assert_eq!(group.len(), 6);
        assert_eq!(group.capacity(), 10);
        assert_eq!(group.indices(), 0..6);
        assert_eq!(group.id(), Some(20));
        assert_eq!(group.size(), GroupSize::Partial(6));
        assert_eq!(group.index(), 2);
        assert!(group.mass_center().is_none());

        // Test expand group by 2 elements
        let result = group.resize(GroupSize::Expand(2));
        assert!(result.is_ok());
        assert_eq!(group.len(), 8);
        assert_eq!(group.indices(), 0..8);
        assert_eq!(group.size(), GroupSize::Partial(8));

        // Test shrink group by 3 elements
        let result = group.resize(GroupSize::Shrink(3));
        assert!(result.is_ok());
        assert_eq!(group.len(), 5);
        assert_eq!(group.indices(), 0..5);
        assert_eq!(group.size(), GroupSize::Partial(5));

        // Test fully activate group
        let result = group.resize(GroupSize::Full);
        assert!(result.is_ok());
        assert_eq!(group.len(), 10);
        assert_eq!(group.indices(), 0..10);
        assert_eq!(group.size(), GroupSize::Full);

        // Test fully deactivate group
        let result = group.resize(GroupSize::Empty);
        assert!(result.is_ok());
        assert_eq!(group.len(), 0);
        assert_eq!(group.indices(), 0..0);
        assert_eq!(group.size(), GroupSize::Empty);
        assert!(group.is_empty());

        // Test shrink group with too many particles (should fail)
        let result = group.resize(GroupSize::Shrink(1));
        assert!(result.is_err());

        // Test expand beyond capacity (should fail)
        let result = group.resize(GroupSize::Expand(group.capacity() + 1));
        assert!(result.is_err());

        // Partial resize to maximum capacity should set status to FULL
        let result = group.resize(GroupSize::Partial(group.capacity()));
        assert!(result.is_ok());
        assert_eq!(group.size(), GroupSize::Full);

        // Partial resize to zero should set status to EMPTY
        let result = group.resize(GroupSize::Partial(0));
        assert!(result.is_ok());
        assert_eq!(group.size(), GroupSize::Empty);
    }
}
