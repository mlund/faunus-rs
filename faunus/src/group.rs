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

use crate::{
    change::{Change, GroupChange},
    Context, Particle, SyncFrom,
};
use anyhow::Ok;
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
/// ~~~

#[derive(Default, Debug, PartialEq, Clone)]
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
    /// All particles.
    All,
    /// Active particles.
    Active,
    /// Inactive particles.
    Inactive,
    /// Specific indices (relative indices).
    RelIndex(Vec<usize>),
    /// Specific indices (absolute indices).
    AbsIndex(Vec<usize>),
    /// All active particles with given atom id.
    ById(usize),
}

/// Enum for selecting a subset of groups
#[derive(Clone, PartialEq, Debug, Default)]
pub enum GroupSelection {
    /// All groups in the system.
    #[default]
    All,
    /// Select by size.
    Size(GroupSize),
    /// Single group with given index.
    Single(usize),
    /// Groups with given index.
    Index(Vec<usize>),
    /// Groups with a given molecule kind.
    ByMoleculeId(usize),
    /// Groups with any of the given molecule kinds.
    ByMoleculeIds(Vec<usize>),
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

    /// Get the absolute index of the first particle in the group.
    pub fn start(&self) -> usize {
        self.range.start
    }

    /// Get size status of the groups which can be `Full`, `Empty`, or `Partial`.
    pub fn size(&self) -> GroupSize {
        self.size_status
    }

    /// Get the index of the group.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the molecule index of the group.
    pub fn molecule(&self) -> usize {
        self.molecule
    }

    /// Get the center of mass of the group.
    pub fn mass_center(&self) -> Option<&Point> {
        self.mass_center.as_ref()
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

    /// Check whether the particle with specified relative index is active.
    pub fn is_active(&self, rel_index: usize) -> bool {
        rel_index < self.num_active
    }

    /// Select (subset) of indices in the group.
    ///
    /// Absolute indices in main particle vector are returned and are guaranteed to be within the group.
    pub fn select(
        &self,
        selection: &ParticleSelection,
        context: &impl Context,
    ) -> Result<Vec<usize>, anyhow::Error> {
        let to_abs = |i: usize| i + self.iter_all().start;
        let indices: Vec<usize> = match selection {
            crate::group::ParticleSelection::AbsIndex(indices) => indices.clone(),
            crate::group::ParticleSelection::RelIndex(indices_rel) => {
                indices_rel.iter().map(|i| to_abs(*i)).collect()
            }
            crate::group::ParticleSelection::All => return Ok(self.iter_all().collect()),
            crate::group::ParticleSelection::Active => return Ok(self.iter_active().collect()),
            crate::group::ParticleSelection::Inactive => return Ok(self.iter_inactive().collect()),
            crate::group::ParticleSelection::ById(id) => {
                return Ok(self.select_by_id(context, self.iter_active(), *id))
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

    /// Select particle indices based on the indices of atom kinds.
    fn select_by_id(
        &self,
        context: &impl Context,
        absolute_indices: std::ops::Range<usize>,
        id: usize,
    ) -> Vec<usize> {
        let atom_indices = context.topology_ref().moleculekinds()[self.molecule].atom_indices();

        atom_indices
            .iter()
            .zip(absolute_indices)
            .filter_map(|(atom_kind_id, atom_index)| {
                if atom_kind_id == &id {
                    Some(atom_index)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if given index is within the group
    pub fn contains(&self, index: usize) -> bool {
        index >= self.range.start && index < self.range.end
    }

    /// Converts a relative index to an absolute index with range check.
    /// If called with `0`, the beginning of the group in the main particle vector is returned.
    /// Returns an error if the index points to an inactive particle.
    pub fn to_absolute_index(&self, index: usize) -> anyhow::Result<usize> {
        if index >= self.num_active {
            anyhow::bail!(
                "Index {} out of range ({} active particles)",
                index,
                self.num_active
            )
        } else {
            Ok(self.range.start + index)
        }
    }

    /// Converts an absolute index into a relative index with range check.
    /// Returns an error if the absolute index is not inside the group.
    pub fn to_relative_index(&self, index: usize) -> anyhow::Result<usize> {
        match self.range.clone().position(|i| i == index) {
            Some(relative) => Ok(relative),
            None => anyhow::bail!(
                "Absolute index {} not inside group (range {:?})",
                index,
                self.range
            ),
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

    /// All groups in the system.
    ///
    /// The first group has index 0, the second group has index 1, etc.
    fn groups(&self) -> &[Group];

    /// Copy of i'th particle in the system.
    fn particle(&self, index: usize) -> Particle;

    /// Get group lists of the system.
    fn group_lists(&self) -> &GroupLists;

    /// Get the number of particles in the system.
    fn num_particles(&self) -> usize {
        self.groups().iter().map(|group| group.capacity()).sum()
    }

    /// Get the number of activate particles in the system.
    fn num_active_particles(&self) -> usize {
        self.groups().iter().map(|group| group.len()).sum()
    }

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
                .filter_map(|(i, g)| if g.size() == *size { Some(i) } else { None })
                .collect(),
            GroupSelection::All => (0..self.groups().len()).collect(),
            GroupSelection::ByMoleculeId(i) => self
                .group_lists()
                .get_full_groups(*i)
                .iter()
                .cloned()
                .chain(self.group_lists().get_partial_groups(*i).iter().cloned())
                .collect::<Vec<usize>>(),
            GroupSelection::ByMoleculeIds(vec) => {
                let mut vector = vec
                    .iter()
                    .flat_map(|&id| self.select(&GroupSelection::ByMoleculeId(id)))
                    .collect::<Vec<usize>>();
                vector.sort();
                vector
            }
        }
    }

    /// Extract copy of particles with given indices
    ///
    /// This can potentially be an expensive operation as it involves copying the particles
    /// from the underlying storage model.
    fn get_particles(&self, indices: impl IntoIterator<Item = usize>) -> Vec<Particle>
    where
        Self: Sized,
    {
        indices.into_iter().map(|i| self.particle(i)).collect()
    }

    /// Extract copy of all particles in the system (both active and inactive).
    fn get_all_particles(&self) -> Vec<Particle> {
        (0..self.num_particles())
            .map(|i| self.particle(i))
            .collect()
    }

    /// Extract copy of active particles in the system.
    fn get_active_particles(&self) -> Vec<Particle> {
        self.groups()
            .iter()
            .flat_map(|group| group.iter_active())
            .map(|index| self.particle(index))
            .collect()
    }

    /// Set particles for a given group.
    fn set_particles<'a>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        source: impl IntoIterator<Item = &'a Particle> + Clone,
    ) -> anyhow::Result<()>
    where
        Self: Sized;

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
    ) -> anyhow::Result<()>
    where
        Self: Sized,
    {
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
                    .map(|i| other_group.to_absolute_index(*i).unwrap());
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
    ) -> anyhow::Result<()>
    where
        Self: Sized,
    {
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

/// Structure storing groups separated into three types:
/// - `full` - all of the atoms of the group are active
/// - `partial` - some of the atoms of the group are active, some are inactive
/// - `empty` - all of the atoms of the group are inactive
///
/// Length of each outer vector corresponds to the number of molecule kinds in the system.
/// Each inner vector then stores ids of groups corresponding to the specific molecule kind.
#[derive(Debug, PartialEq, Clone)]
pub struct GroupLists {
    full: Vec<Vec<usize>>,
    partial: Vec<Vec<usize>>,
    empty: Vec<Vec<usize>>,
}

impl GroupLists {
    /// Create and initialize a new GroupLists structure.
    ///
    /// ## Parameters
    /// `n_molecules` - the number of molecule kinds defined in the system
    pub(crate) fn new(n_molecules: usize) -> GroupLists {
        GroupLists {
            full: vec![Vec::new(); n_molecules],
            partial: vec![Vec::new(); n_molecules],
            empty: vec![Vec::new(); n_molecules],
        }
    }

    /// Add group to the GroupList. The group will be automatically assigned to the correct list.
    /// This method assumes that the group is NOT yet present in the GroupLists.
    ///
    /// ## Notes
    /// - The time complexity of this operation is O(1).
    pub(crate) fn add_group(&mut self, group: &Group) {
        let list = match group.size() {
            GroupSize::Full => &mut self.full,
            GroupSize::Partial(_) => &mut self.partial,
            GroupSize::Empty => &mut self.empty,
            _ => panic!("Unsupported GroupSize."),
        };

        GroupLists::add_to_list(list, group);
    }

    /// Update the position of the group in the GroupLists.
    ///
    /// ## Notes
    /// - The time complexity of this operation is O(n).
    /// - This operation always consists of searching for the group (O(n)).
    ///   If the position of the group must be updated, searching is followed by
    ///   removing the group from the original vector via `swap_remove` (O(1)) and by
    ///   adding the group to the correct vector (O(1)).
    pub(crate) fn update_group(&mut self, group: &Group) {
        match self.find_group(group) {
            Some((list, index, size)) => {
                // we can't use just `==` because GroupSize::Partial must match any GroupSize::Partial
                match (group.size(), size) {
                    (GroupSize::Empty, GroupSize::Empty) => (),
                    (GroupSize::Partial(_), GroupSize::Partial(_)) => (),
                    (GroupSize::Full, GroupSize::Full) => (),
                    // update is needed only if the current group size does not match the previous one
                    _ => {
                        list.swap_remove(index);
                        self.add_group(group);
                    }
                }
            }
            // group is not present in any list, add it
            None => self.add_group(group),
        }
    }

    /// Get all full groups with the given molecule ID.
    pub(crate) fn get_full_groups(&self, id: usize) -> &[usize] {
        &self.full[id]
    }

    /// Get all partial groups with the given molecule ID.
    pub(crate) fn get_partial_groups(&self, id: usize) -> &[usize] {
        &self.partial[id]
    }

    /// Get all empty groups with the given molecule ID.
    #[allow(dead_code)]
    pub(crate) fn get_empty_groups(&self, id: usize) -> &[usize] {
        &self.empty[id]
    }

    /// Returns indices of all groups matching given molecule id and size.
    ///
    /// The lookup complexity is O(1).
    pub fn find_molecules(&self, molecule_id: usize, size: GroupSize) -> Option<&[usize]> {
        let indices = match size {
            GroupSize::Full => self.full.get(molecule_id),
            GroupSize::Partial(_) => self.partial.get(molecule_id),
            GroupSize::Empty => self.empty.get(molecule_id),
            _ => panic!("Unsupported GroupSize."),
        };
        indices.map(|i| i.as_slice())
    }

    /// Find the group in GroupLists.
    ///
    /// Returns the inner vector in which the group is located,
    /// the index of the group in the vector, and
    /// the type of the vector as GroupSize enum.
    ///
    /// ## Notes
    /// - The time complexity of this operation is O(n), where `n` is the number of
    ///   groups with the same molecule kind as the searched group.
    fn find_group(&mut self, group: &Group) -> Option<(&mut Vec<usize>, usize, GroupSize)> {
        [&mut self.full, &mut self.partial, &mut self.empty]
            .into_iter()
            .zip([GroupSize::Full, GroupSize::Partial(1), GroupSize::Empty])
            .find_map(|(outer, size)| {
                let inner = outer
                    .get_mut(group.molecule())
                    .expect("Incorrectly initialized GroupLists structure.");

                inner
                    .iter()
                    .position(|&x| x == group.index())
                    .map(|pos| (inner, pos, size))
            })
    }

    /// Add group to target list.
    ///
    /// ## Notes
    /// - The time complexity of this operation is O(1).
    fn add_to_list(list: &mut [Vec<usize>], group: &Group) {
        list.get_mut(group.molecule())
            .expect("Incorrectly initialized GroupLists structure.")
            .push(group.index());
    }
}

#[cfg(test)]
mod tests {
    use crate::platform::reference::ReferencePlatform;

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

        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        assert_eq!(group.len(), 6);
        assert_eq!(group.capacity(), 10);
        assert_eq!(group.iter_active(), 0..6);
        assert_eq!(group.molecule(), 20);
        assert_eq!(group.size(), GroupSize::Partial(6));
        assert_eq!(group.index(), 2);
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
            .select(&ParticleSelection::RelIndex(vec![0, 1, 2]), &context)
            .unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Absolute selection
        let indices = group
            .select(&ParticleSelection::AbsIndex(vec![20, 21, 22]), &context)
            .unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Select all
        let indices = group.select(&ParticleSelection::All, &context).unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Out of range selection
        assert!(group
            .select(&ParticleSelection::RelIndex(vec![1, 2, 3]), &context)
            .is_err());

        // Test partial selection
        group.resize(GroupSize::Shrink(1)).unwrap();
        let indices = group.select(&ParticleSelection::Active, &context).unwrap();
        assert_eq!(indices, vec![20, 21]);
        let indices = group
            .select(&ParticleSelection::Inactive, &context)
            .unwrap();
        assert_eq!(indices, vec![22]);
    }

    #[test]
    fn test_group_select_by_id() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        let group = context.groups().get(1).unwrap();
        let expected0 = vec![7, 11, 12, 13];
        let expected1 = vec![8, 9, 10];

        assert_eq!(
            group.select(&ParticleSelection::ById(0), &context).unwrap(),
            expected0
        );

        assert_eq!(
            group.select(&ParticleSelection::ById(1), &context).unwrap(),
            expected1
        );

        let group = context.groups().get(45).unwrap();
        let expected_active: Vec<usize> = vec![];

        assert_eq!(
            group.select(&ParticleSelection::ById(2), &context).unwrap(),
            expected_active
        );
    }

    #[test]
    fn test_absolute_relative_indices() {
        let group = Group {
            molecule: 20,
            index: 2,
            mass_center: None,
            num_active: 6,
            range: 10..27,
            size_status: GroupSize::Partial(6),
        };

        assert_eq!(group.to_absolute_index(4).unwrap(), 14);
        assert_eq!(group.to_relative_index(21).unwrap(), 11);
    }

    #[test]
    fn test_group_lists() {
        let mut group_lists = GroupLists::new(3);

        assert_eq!(group_lists.full.len(), 3);
        assert_eq!(group_lists.partial.len(), 3);
        assert_eq!(group_lists.empty.len(), 3);

        let mut group1 = Group::new(0, 0, 3..8);
        let group2 = Group::new(1, 0, 8..13);
        let mut group3 = Group::new(2, 1, 13..20);

        group_lists.add_group(&group1);
        group_lists.add_group(&group2);
        group_lists.add_group(&group3);

        assert!(group_lists.full[0].contains(&0));
        assert!(group_lists.full[0].contains(&1));
        assert!(group_lists.full[1].contains(&2));

        group1.resize(GroupSize::Empty).unwrap();
        group3.resize(GroupSize::Partial(3)).unwrap();

        group_lists.update_group(&group1);
        group_lists.update_group(&group2);
        group_lists.update_group(&group3);

        assert!(!group_lists.full[0].contains(&0));
        assert!(group_lists.empty[0].contains(&0));
        assert!(group_lists.full[0].contains(&1));
        assert!(!group_lists.full[1].contains(&2));
        assert!(group_lists.partial[1].contains(&2));
    }

    #[test]
    fn test_group_selections() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        // All groups
        let expected = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        ];
        let selected = context.select(&GroupSelection::All);
        assert_eq!(selected, expected);

        // Full groups
        let expected = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
            66,
        ];
        let selected = context.select(&GroupSelection::Size(GroupSize::Full));
        assert_eq!(selected, expected);

        // Empty groups
        let expected = vec![
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
        ];
        let selected = context.select(&GroupSelection::Size(GroupSize::Empty));
        assert_eq!(selected, expected);

        // Single group with index
        let expected = vec![16];
        let selected = context.select(&GroupSelection::Single(16));
        assert_eq!(selected, expected);

        // Several groups with index
        let expected = vec![16, 19, 24, 35];
        let selected = context.select(&GroupSelection::Index(vec![16, 19, 24, 35]));
        assert_eq!(selected, expected);

        // With molecule ID
        let expected = vec![0, 1, 2, 60, 61];
        let selected = context.select(&GroupSelection::ByMoleculeId(0));
        assert_eq!(selected, expected);

        // With several molecule IDs
        let expected = context.select(&GroupSelection::Size(GroupSize::Full));
        let selected = context.select(&GroupSelection::ByMoleculeIds(vec![0, 1]));
        assert_eq!(selected, expected);
    }
}
