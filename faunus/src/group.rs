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

use crate::{change::Change, change::GroupChange, Particle, SyncFrom};
use anyhow::Ok;
use derive_getters::Getters;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub type Point = Vector3<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

/// Group of particles.
///
/// A group is a contiguous set of particles in a system. It has a unique index in a global list of
/// groups, and a unique range of indices in a global particle list. The group can be resized
/// within its capacity.
///
/// # Examples
///
/// Here an example of a group with 3 particles, starting at index 20 in the main particle vector.
/// ~~~
/// use faunus::group::*;
/// let mut group = Group::new(7, 0, 20..23);
/// assert_eq!(group.len(), 3);
/// assert_eq!(group.size(), GroupSize::Full);
///
/// // Resize active particles from 3 -> 2
/// group.resize(GroupSize::Shrink(1)).unwrap();
/// assert_eq!(group.len(), 2);
/// assert_eq!(group.capacity(), 3);
/// assert_eq!(group.size(), GroupSize::Partial(2));
///
/// let selection = group.select(&ParticleSelection::Inactive);
/// assert_eq!(selection.unwrap(), vec![22]);
/// ~~~

#[derive(Serialize, Deserialize, Default, Debug, PartialEq, Clone, Getters)]
pub struct Group {
    /// Index of the molecule kind forming the group (immutable).
    molecule: usize,
    /// Index of the group in the main group vector (immutable and unique).
    index: usize,
    /// Optional mass center
    mass_center: Option<Point>,
    /// Number of active particles
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
    /// Special size used to expand with `usize` particles
    Expand(usize),
    /// Special size used to shrink with `usize` particles
    Shrink(usize),
}

/// Enum for selecting a subset of particles in a group
#[derive(Clone, PartialEq, Debug)]
pub enum ParticleSelection {
    /// All particles
    All,
    /// Active particles
    Active,
    /// Inactive particles
    Inactive,
    /// Specific indices (relative indices)
    RelIndex(Vec<usize>),
    /// Specific indices (absolute indices)
    AbsIndex(Vec<usize>),
    /// Particles with given id
    ById(usize),
}

/// Enum for selecting a subset of groups
#[derive(Clone, PartialEq, Debug)]
pub enum GroupSelection {
    /// All groups in the system
    All,
    /// Select by size
    Size(GroupSize),
    /// Single group with given index
    Single(usize),
    /// Groups with given index
    Index(Vec<usize>),
    /// Groups with a given molecule kind
    ByMoleculeId(usize),
}

impl Group {
    /// Create a new group
    pub fn new(index: usize, molecule: usize, range: core::ops::Range<usize>) -> Self {
        Self {
            molecule,
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

    /// Absolute indices of active particles in main particle vector
    pub fn iter_active(&self) -> std::ops::Range<usize> {
        std::ops::Range {
            start: self.range.start,
            end: self.range.start + self.num_active,
        }
    }

    /// Select (subset) of indices in the group.
    ///
    /// Absolute indices in main particle vector are returned and are guaranteed to be within the group.
    pub fn select(&self, selection: &ParticleSelection) -> Result<Vec<usize>, anyhow::Error> {
        let to_abs = |i: usize| i + self.iter_all().start;
        let indices: Vec<usize> = match selection {
            crate::group::ParticleSelection::AbsIndex(indices) => indices.clone(),
            crate::group::ParticleSelection::RelIndex(indices_rel) => {
                indices_rel.iter().map(|i| to_abs(*i)).collect()
            }
            crate::group::ParticleSelection::All => return Ok(self.iter_all().collect()),
            crate::group::ParticleSelection::Active => return Ok(self.iter_active().collect()),
            crate::group::ParticleSelection::Inactive => return Ok(self.iter_inactive().collect()),
            crate::group::ParticleSelection::ById(_) => {
                anyhow::bail!("Not implemented: select particles by id")
            }
        };
        if indices.iter().all(|i| self.contains(*i)) {
            return Ok(indices);
        }
        anyhow::bail!(
            "Invalid indices {:?} for group with range {:?}",
            indices,
            self.range
        )
    }

    /// Check if given index is within the group
    pub fn contains(&self, index: usize) -> bool {
        index >= self.range.start && index < self.range.end
    }

    /// Converts a relative index to an absolute index with range check.
    /// If called with `0`, the beginning of the group in the main particle vector is returned.
    pub fn absolute_index(&self, index: usize) -> anyhow::Result<usize> {
        if index >= self.num_active {
            anyhow::bail!("Index {} out of range (max {})", index, self.num_active - 1)
        } else {
            Ok(self.range.start + index)
        }
    }

    /// Absolute indices of *all* particles in main particle vector, both active and inactive (immutable).
    /// This reflects the full capacity of the group and never changes over the lifetime of the group.
    pub fn iter_all(&self) -> std::ops::Range<usize> {
        self.range.clone()
    }

    /// Iterator to inactive indices (absolute indices in main particle vector)
    pub fn iter_inactive(&self) -> impl Iterator<Item = usize> {
        self.range.clone().skip(self.num_active)
    }

    /// Set mass center
    pub fn set_mass_center(&mut self, mass_center: Point) {
        self.mass_center = Some(mass_center);
    }
}

/// Interface for groups of particles
///
/// Each group has a unique index in a global list of groups, and a unique range of indices in a
/// global particle list.
pub trait GroupCollection: SyncFrom {
    /// Add a group to the system based on an molecule id and a set of particles given by an iterator.
    fn add_group(&mut self, molecule: usize, particles: &[Particle]) -> anyhow::Result<&mut Group>;

    /// Resizes a group to a given size.
    ///
    /// Errors if the requested size is larger than the capacity, or if there are
    /// too few active particles to shrink a group.
    fn resize_group(&mut self, group_index: usize, size: GroupSize) -> anyhow::Result<()>;

    /// All groups in the system
    ///
    /// The first group has index 0, the second group has index 1, etc.
    fn groups(&self) -> &[Group];

    /// Copy of i'th particle in the system
    fn particle(&self, index: usize) -> Particle;

    /// Get the number of particles in the system.
    fn n_particles(&self) -> usize;

    /// Find group indices based on a selection
    ///
    /// The selection can be used to select a subset of groups based on their index or id.
    /// If the selection is `All`, all groups are returned. If the selection is `Single(i)`, the
    /// group with index `i` is returned. If the selection is `Index(indices)`, the groups with
    /// indices in `indices` are returned. If the selection is `ById(id)`, the groups with the
    /// given id are returned.
    fn select(&self, selection: &GroupSelection) -> Vec<usize> {
        match selection {
            GroupSelection::Single(i) => vec![*i],
            GroupSelection::Index(indices) => indices.clone(),
            GroupSelection::Size(size) => self
                .groups()
                .iter()
                .enumerate()
                .filter(|(_i, g)| g.size() == *size)
                .map(|(i, _)| i)
                .collect(),
            GroupSelection::All => (0..self.groups().len()).collect(),
            GroupSelection::ByMoleculeId(_) => todo!("not implemented"),
        }
    }

    /// Extract copy of particles with given indices
    ///
    /// This can potentially be an expensive operation as it involves copying the particles
    /// from the underlying storage model.
    fn get_particles(&self, indices: impl IntoIterator<Item = usize>) -> Vec<Particle> {
        indices.into_iter().map(|i| self.particle(i)).collect()
    }

    /// Set particles for a given group.
    fn set_particles<'a>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        source: impl IntoIterator<Item = &'a Particle> + Clone,
    ) -> anyhow::Result<()>;

    /// Synchronize with a group in another context
    ///
    /// This is used to synchronize groups between different contexts after
    /// e.g. a Monte Carlo move.
    /// Errors if there's a mismatch in group index, id, or capacity.
    /// The following is synchronized:
    /// - Group size
    /// - Particle properties (position, id, etc.)
    fn sync_group_from(
        &mut self,
        group_index: usize,
        change: GroupChange,
        other: &impl GroupCollection,
    ) -> anyhow::Result<()> {
        let other_group = &other.groups()[group_index];
        let group = &self.groups()[group_index];
        if (other_group.molecule() != group.molecule())
            || (other_group.index() != group.index())
            || (other_group.capacity() != group.capacity())
        {
            anyhow::bail!("Group mismatch");
        }
        match change {
            GroupChange::PartialUpdate(indices) => {
                assert_eq!(group.size(), other_group.size());
                let indices = indices
                    .iter()
                    .map(|i| other_group.absolute_index(*i).unwrap());
                let particles = other.get_particles(indices.clone());
                self.set_particles(indices, particles.iter())?;
            }
            GroupChange::RigidBody => {
                self.sync_group_from(
                    group_index,
                    GroupChange::PartialUpdate((0..other_group.len()).collect()),
                    other,
                )?;
            }
            GroupChange::Resize(size) => match size {
                GroupSize::Full => {
                    assert_eq!(other_group.size(), GroupSize::Full);
                    self.resize_group(group_index, size)?;
                    self.sync_group_from(group_index, GroupChange::RigidBody, other)?
                }
                GroupSize::Empty => {
                    assert!(other_group.is_empty());
                    self.resize_group(group_index, size)?
                }
                GroupSize::Shrink(n) => {
                    assert_eq!(group.len() - n, other_group.len());
                    self.resize_group(group_index, size)?
                }
                GroupSize::Expand(n) => {
                    assert_eq!(group.len() + n, other_group.len());
                    // sync the extra n active indices in the other group
                    let indices = (other_group.len()..other_group.len() + n).collect::<Vec<_>>();
                    self.resize_group(group_index, size)?;
                    self.sync_group_from(group_index, GroupChange::PartialUpdate(indices), other)?
                }
                GroupSize::Partial(n) => {
                    let dn = group.len() as isize - n as isize;
                    let size = match dn.cmp(&0) {
                        Ordering::Greater => GroupSize::Expand(dn as usize),
                        Ordering::Less => GroupSize::Shrink(-dn as usize),
                        Ordering::Equal => return Ok(()),
                    };
                    self.sync_group_from(group_index, GroupChange::Resize(size), other)?;
                    todo!("is this the behavior we want?");
                }
            },
            _ => todo!("implement other group changes"),
        }
        Ok(())
    }

    /// Synchonize with another context
    ///
    /// This is used to synchronize groups between different contexts after
    /// e.g. an accepted Monte Carlo move that proposes a change to the system.
    fn sync_from_groupcollection(
        &mut self,
        change: &Change,
        other: &impl GroupCollection,
    ) -> anyhow::Result<()> {
        match change {
            Change::Everything => {
                for i in 0..self.groups().len() {
                    self.sync_group_from(i, GroupChange::RigidBody, other)?
                }
            }
            Change::SingleGroup(group_index, change) => {
                self.sync_group_from(*group_index, change.clone(), other)?
            }
            Change::Volume(_policy, _volumes) => {
                todo!("implement volume change")
            }
            Change::Groups(changes) => {
                for _change in changes {
                    todo!("implement group changes")
                }
            }
            _ => todo!("implement other changes"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group() {
        // Test group creation
        let mut group = Group {
            molecule: 20,
            index: 2,
            mass_center: None,
            num_active: 6,
            range: 0..10,
            size_status: GroupSize::Partial(6),
        };

        assert_eq!(group.len(), 6);
        assert_eq!(group.capacity(), 10);
        assert_eq!(group.iter_active(), 0..6);
        assert_eq!(group.molecule(), &20);
        assert_eq!(group.size(), GroupSize::Partial(6));
        assert_eq!(group.index(), &2);
        assert!(group.mass_center().is_none());

        // Test expand group by 2 elements
        let result = group.resize(GroupSize::Expand(2));
        assert!(result.is_ok());
        assert_eq!(group.len(), 8);
        assert_eq!(group.iter_active(), 0..8);
        assert_eq!(group.size(), GroupSize::Partial(8));

        // Test shrink group by 3 elements
        let result = group.resize(GroupSize::Shrink(3));
        assert!(result.is_ok());
        assert_eq!(group.len(), 5);
        assert_eq!(group.iter_active(), 0..5);
        assert_eq!(group.size(), GroupSize::Partial(5));

        // Test fully activate group
        let result = group.resize(GroupSize::Full);
        assert!(result.is_ok());
        assert_eq!(group.len(), 10);
        assert_eq!(group.iter_active(), 0..10);
        assert_eq!(group.size(), GroupSize::Full);

        // Test fully deactivate group
        let result = group.resize(GroupSize::Empty);
        assert!(result.is_ok());
        assert_eq!(group.len(), 0);
        assert_eq!(group.iter_active(), 0..0);
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

        // Relative selection
        let mut group = Group::new(7, 0, 20..23);
        assert_eq!(group.len(), 3);
        let indices = group
            .select(&ParticleSelection::RelIndex(vec![0, 1, 2]))
            .unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Absolute selection
        let indices = group
            .select(&ParticleSelection::AbsIndex(vec![20, 21, 22]))
            .unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Select all
        let indices = group.select(&ParticleSelection::All).unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Out of range selection
        assert!(group
            .select(&ParticleSelection::RelIndex(vec![1, 2, 3]))
            .is_err());

        // Test partial selection
        group.resize(GroupSize::Shrink(1)).unwrap();
        let indices = group.select(&ParticleSelection::Active).unwrap();
        assert_eq!(indices, vec![20, 21]);
        let indices = group.select(&ParticleSelection::Inactive).unwrap();
        assert_eq!(indices, vec![22]);
    }
}
