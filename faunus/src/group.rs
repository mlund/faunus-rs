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
    topology::{GroupKind, Topology},
    Point, UnitQuaternion,
};
use serde::{Deserialize, Serialize};

/// Group of particles.
///
/// A group is a contiguous set of particles in a system. It has a unique index in a global list of
/// groups, and a unique range of indices in a global particle list. The group can be resized
/// within its capacity.
///
/// A group of three particles starting at index 20 has `len() == 3` and `GroupSize::Full`;
/// shrinking it by one leaves `len() == 2` with the capacity still 3. See `resize_shrinks_a_group`.
#[derive(Debug, PartialEq, Clone)]
pub struct Group {
    /// Index of the molecule kind forming the group (immutable).
    molecule: MoleculeId,
    /// Index of the group in the main group vector (immutable and unique).
    index: usize,
    /// Mass center and bounding radius; `None` for groups that have no mass center.
    geometry: Option<GroupGeometry>,
    /// Number of active particles
    num_active: usize,
    /// Absolute indices in main particle vector (active and inactive; immutable and unique)
    range: std::ops::Range<usize>,
    /// Size status
    size_status: GroupSize,
    /// Rigid-body orientation, persisted in state files for restart consistency
    quaternion: UnitQuaternion,
}

impl Default for Group {
    fn default() -> Self {
        Self {
            molecule: MoleculeId::new(0),
            index: 0,
            geometry: None,
            num_active: 0,
            range: 0..0,
            size_status: GroupSize::default(),
            quaternion: UnitQuaternion::identity(),
        }
    }
}

/// Index into the global particle array.
///
/// Distinct from [`RelIndex`] on purpose: the two are freely interchangeable as `usize`, and mixing
/// them silently addresses the wrong particle. There is deliberately no `From<usize>` — an implicit
/// `.into()` at a call site would let a raw integer become whichever space the callee happens to
/// want, which is the mistake these types exist to prevent. Convert between the spaces through
/// [`Group::to_absolute`] / [`Group::to_relative`], which need the group and can fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AbsIndex(usize);

/// Index of a particle within its group, counted from the group's first particle.
///
/// See [`AbsIndex`] for why there is no `From<usize>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RelIndex(usize);

/// Index into the group array, i.e. which molecule.
///
/// Numerically indistinguishable from [`AbsIndex`] — group 1 is also a valid particle index — so it
/// is a separate type for the same reason. See [`AbsIndex`] for why there is no `From<usize>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GroupIndex(usize);

impl AbsIndex {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }
    /// The raw index, for addressing the particle arrays.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl GroupIndex {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }
    /// The raw index, for addressing the group array.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for GroupIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl RelIndex {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }
    /// The raw offset from the group's first particle.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for AbsIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::fmt::Display for RelIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Index of a molecule *kind*, into [`Topology::moleculekinds`](crate::topology::Topology::moleculekinds).
///
/// A molecule-kind id and a group index are both `usize` and both valid in the other's space, so
/// mixing them silently addresses the wrong molecule. It is therefore a separate type. See
/// [`AbsIndex`] for why there is no `From<usize>`.
///
/// `Default` (id 0) exists only so moves that carry the id in a `#[serde(skip)]` field can derive
/// it; the real id is resolved by name at move setup.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MoleculeId(usize);

/// Index of an atom *kind*, into [`Topology::atomkinds`](crate::topology::Topology::atomkinds).
///
/// Distinct from a particle index on purpose: `atom_kind` takes a particle index and returns an
/// `AtomKindId`, and `set_atom_kind` takes both. Keeping them separate types stops the two from
/// being transposed — a swap that would otherwise retype the wrong atom. See [`AbsIndex`] for why
/// there is no `From<usize>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AtomKindId(usize);

impl MoleculeId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }
    /// The raw index, for addressing the molecule-kind array.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl AtomKindId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }
    /// The raw index, for addressing the atom-kind array.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for MoleculeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::fmt::Display for AtomKindId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Geometric summary of a group's active particles, derived from their positions and masses.
///
/// The two quantities are computed together and are meaningless apart — the radius is measured
/// *from* the mass center — so they are stored, backed up, and restored as one value. A group
/// without a mass center (an atomic mega-group) has no geometry at all.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct GroupGeometry {
    /// Center of mass, respecting periodic boundaries.
    pub mass_center: Point,
    /// Max distance from the mass center to any active particle, for spatial culling.
    pub bounding_radius: f64,
}

/// Activation status of a group of particles
///
/// This can be used to set the number of active particles in a group. The group can e.g. be set to
/// full, empty, or to a specific number of active particles. This is used in connection with Grand
/// Canonical Monte Carlo moves to add or remove particles or molecules.
/// If resizing to zero, the group is `Empty` and considered *inactive*. If resizing to the
/// capacity, the group is `Full` and considered *active*. Otherwise, the group is `Partial`.
#[derive(Serialize, Deserialize, Default, Copy, Clone, PartialEq, Eq, Debug)]
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

impl GroupSize {
    /// Create a `GroupSize` from an active count and a group capacity.
    ///
    /// Normalizes to `Full`/`Empty` at the boundaries so that downstream code
    /// (e.g. `GroupLists`) classifies groups consistently.
    pub fn from_count(active: usize, capacity: usize) -> Self {
        if active == capacity {
            Self::Full
        } else if active == 0 {
            Self::Empty
        } else {
            Self::Partial(active)
        }
    }
}

/// Enum for selecting a subset of particles in a group
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ParticleSelection {
    /// All particles.
    All,
    /// Active particles.
    Active,
    /// Inactive particles.
    Inactive,
    /// Specific particles, addressed within the group.
    Relative(Vec<RelIndex>),
    /// Specific particles, addressed in the global particle array.
    Absolute(Vec<AbsIndex>),
    /// All active particles with given atom id.
    ById(AtomKindId),
}

/// Enum for selecting a subset of groups
#[derive(Clone, PartialEq, Eq, Debug, Default)]
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
    ByMoleculeId(MoleculeId),
    /// Groups with any of the given molecule kinds.
    ByMoleculeIds(Vec<MoleculeId>),
}

impl Group {
    /// Create a new group
    pub fn new(index: usize, molecule: MoleculeId, range: core::ops::Range<usize>) -> Self {
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
    ///
    /// # Errors
    /// Returns an error if the requested size exceeds capacity or if shrinking by more particles than active.
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
                        .ok_or_else(|| anyhow::anyhow!("Cannot shrink group by {} particles", n))?,
                ));
            }
        };
        Ok(())
    }

    /// Get the absolute index of the first particle in the group.
    pub const fn start(&self) -> usize {
        self.range.start
    }

    /// Get size status of the groups which can be `Full`, `Empty`, or `Partial`.
    pub const fn size(&self) -> GroupSize {
        self.size_status
    }

    /// Get the index of the group.
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Get the molecule kind of the group.
    pub const fn molecule(&self) -> MoleculeId {
        self.molecule
    }

    /// Get the center of mass of the group.
    pub const fn mass_center(&self) -> Option<&Point> {
        match &self.geometry {
            Some(geometry) => Some(&geometry.mass_center),
            None => None,
        }
    }

    /// Maximum number of particles (active plus inactive)
    pub fn capacity(&self) -> usize {
        self.range.len()
    }

    /// Number of active particles
    pub const fn len(&self) -> usize {
        self.num_active
    }

    /// True if no active particles
    pub const fn is_empty(&self) -> bool {
        self.num_active == 0
    }

    /// True if all particle slots are active
    pub fn is_full(&self) -> bool {
        self.num_active == self.range.len()
    }

    /// Absolute indices of active particles in main particle vector
    pub const fn iter_active(&self) -> std::ops::Range<usize> {
        std::ops::Range {
            start: self.range.start,
            end: self.range.start + self.num_active,
        }
    }

    /// Check whether the particle with specified relative index is active.
    pub const fn is_active(&self, rel_index: usize) -> bool {
        rel_index < self.num_active
    }

    /// Select (subset) of indices in the group.
    ///
    /// Absolute indices in main particle vector are returned and are guaranteed to be within the group.
    pub fn select(
        &self,
        selection: &ParticleSelection,
        topology: &Topology,
    ) -> anyhow::Result<Vec<usize>> {
        let to_abs = |i: RelIndex| i.get() + self.iter_all().start;
        let indices: Vec<usize> = match selection {
            ParticleSelection::Absolute(indices) => indices.iter().map(|i| i.get()).collect(),
            ParticleSelection::Relative(indices_rel) => {
                indices_rel.iter().map(|i| to_abs(*i)).collect()
            }
            ParticleSelection::All => return Ok(self.iter_all().collect()),
            ParticleSelection::Active => return Ok(self.iter_active().collect()),
            ParticleSelection::Inactive => return Ok(self.iter_inactive().collect()),
            ParticleSelection::ById(id) => {
                return Ok(self.select_by_id(topology, self.iter_active(), *id))
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
        topology: &Topology,
        absolute_indices: std::ops::Range<usize>,
        id: AtomKindId,
    ) -> Vec<usize> {
        let atom_indices = topology.moleculekind(self.molecule).atom_indices();

        atom_indices
            .iter()
            .zip(absolute_indices)
            .filter_map(|(atom_kind_id, atom_index)| {
                if *atom_kind_id == id.get() {
                    Some(atom_index)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if given index is within the group
    pub const fn contains(&self, index: usize) -> bool {
        index >= self.range.start && index < self.range.end
    }

    /// Converts a relative index to an absolute index with range check.
    /// If called with `0`, the beginning of the group in the main particle vector is returned.
    ///
    /// # Errors
    /// Returns an error if the index points to an inactive particle.
    #[must_use = "this returns a Result that should be handled"]
    pub fn to_absolute(&self, index: RelIndex) -> anyhow::Result<AbsIndex> {
        if index.get() >= self.num_active {
            anyhow::bail!(
                "Index {} out of range ({} active particles)",
                index,
                self.num_active
            )
        } else {
            Ok(AbsIndex::new(self.range.start + index.get()))
        }
    }

    /// Converts an absolute index into a relative index with range check.
    ///
    /// # Errors
    /// Returns an error if the absolute index is not inside the group.
    #[must_use = "this returns a Result that should be handled"]
    pub fn to_relative(&self, index: AbsIndex) -> anyhow::Result<RelIndex> {
        match self.range.clone().position(|i| i == index.get()) {
            Some(relative) => Ok(RelIndex::new(relative)),
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
    pub fn iter_inactive(&self) -> impl Iterator<Item = usize> + use<> {
        self.range.clone().skip(self.num_active)
    }

    /// Get bounding radius (max distance from mass center to any active particle).
    pub const fn bounding_radius(&self) -> Option<f64> {
        match &self.geometry {
            Some(geometry) => Some(geometry.bounding_radius),
            None => None,
        }
    }

    /// Mass center and bounding radius together, as they are always derived together.
    pub const fn geometry(&self) -> Option<GroupGeometry> {
        self.geometry
    }

    /// Set — or clear — the derived geometry.
    ///
    /// Taking an `Option` is what lets a rejected move restore a group that had no geometry
    /// before it ran.
    pub const fn set_geometry(&mut self, geometry: Option<GroupGeometry>) {
        self.geometry = geometry;
    }

    /// Rigid-body orientation quaternion (for MC↔LD state transfer).
    pub fn quaternion(&self) -> &UnitQuaternion {
        &self.quaternion
    }

    /// Set the orientation quaternion.
    pub fn set_quaternion(&mut self, q: UnitQuaternion) {
        self.quaternion = q;
    }

    /// Compose a rotation onto the current quaternion: q_new = rotation * q_current.
    pub fn rotate_by(&mut self, rotation: &UnitQuaternion) {
        self.quaternion = rotation * self.quaternion;
    }
}

/// Interface for groups of particles
///
/// Each group has a unique index in a global list of groups, and a unique range of indices in a
/// global particle list.
pub trait GroupCollection {
    /// All groups in the system.
    ///
    /// The first group has index 0, the second group has index 1, etc.
    fn groups(&self) -> &[Group];

    /// The group at `index`.
    fn group(&self, index: GroupIndex) -> &Group {
        &self.groups()[index.get()]
    }

    /// Position of the i'th particle in the system.
    fn position(&self, index: usize) -> Point;

    /// Atom kind index of the i'th particle.
    fn atom_kind(&self, index: usize) -> AtomKindId;

    /// Counter bumped whenever any particle's atom kind changes.
    ///
    /// Together with [`group_lists_generation`](Self::group_lists_generation) this is everything a
    /// selection's outcome can depend on.
    fn atom_kinds_generation(&self) -> u64;

    /// Index of the group owning particle `index`, if any.
    ///
    /// Groups occupy disjoint, ascending, contiguous particle ranges, so this is a binary search.
    fn group_of_particle(&self, index: usize) -> Option<usize> {
        let groups = self.groups();
        let found = groups.binary_search_by(|group| {
            if index < group.start() {
                std::cmp::Ordering::Greater
            } else if group.contains(index) {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Less
            }
        });
        found.ok().map(|position| groups[position].index())
    }

    /// Get group lists of the system.
    ///
    /// Prefer the delegating helpers (`find_molecules`, `find_atomic_group`, etc.)
    /// over calling this directly.
    fn group_lists(&self) -> &GroupLists;

    /// Returns indices of all groups matching given molecule kind and size.
    /// Empty when the kind has no group of that size.
    fn find_molecules(&self, molecule_id: MoleculeId, size: GroupSize) -> &[usize] {
        self.group_lists().find_molecules(molecule_id, size)
    }

    /// Find the single mega-group index for an atomic molecule kind.
    fn find_atomic_group(&self, molecule_id: MoleculeId) -> Option<usize> {
        self.group_lists().find_atomic_group(molecule_id)
    }

    /// Count groups matching given molecule kind and size.
    fn count_molecules(&self, molecule_id: MoleculeId, size: GroupSize) -> usize {
        self.group_lists().count_molecules(molecule_id, size)
    }

    /// Count non-empty groups (full + partial) for a molecule kind.
    fn count_nonempty(&self, molecule_id: MoleculeId) -> usize {
        self.group_lists().count_nonempty(molecule_id)
    }

    /// Count active particles for a molecule kind.
    ///
    /// For atomic/reservoir mega-groups, returns the number of active atoms in the group.
    /// For molecular groups, returns the number of non-empty groups.
    fn count_active(&self, molecule_id: MoleculeId, group_kind: GroupKind) -> usize {
        match group_kind {
            GroupKind::Atomic | GroupKind::Reservoir => self
                .find_atomic_group(molecule_id)
                .map(|gi| self.groups()[gi].len())
                .unwrap_or(0),
            GroupKind::Molecular => self.count_nonempty(molecule_id),
        }
    }

    /// Monotonically increasing counter, bumped when group lists change.
    fn group_lists_generation(&self) -> u64 {
        self.group_lists().generation()
    }

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
                .find_molecules(*i, GroupSize::Full)
                .iter()
                .chain(self.find_molecules(*i, GroupSize::Partial(0)))
                .copied()
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
}

/// Everything that changes a group collection.
///
/// Split from the read-only [`GroupCollection`] so that analyses, move proposals and energy terms —
/// which are bound to the read half — cannot reshape the system they are only meant to observe.
/// The callers are the framework itself: `transform`, `state`, `topology::block` and `backend`.
pub trait GroupCollectionMut: GroupCollection {
    /// Add a group to the system based on a molecule kind, positions, and atom kind indices.
    fn add_group(
        &mut self,
        molecule: MoleculeId,
        positions: &[Point],
        atom_ids: &[usize],
    ) -> anyhow::Result<&mut Group>;

    /// Write every particle's position and recompute every group's mass center and bounding
    /// radius.
    ///
    /// The bulk counterpart of [`place_group`](Self::place_group), for an integrator that has
    /// advanced the whole system at once. Orientations are *not* touched: a rigid-body integrator
    /// advances them itself and reports them through
    /// [`set_group_orientation`](Self::set_group_orientation), and it would be wrong to overwrite
    /// that with a fit to the coordinates it derived from them.
    fn set_all_positions(&mut self, positions: &[Point]) -> anyhow::Result<()>;

    /// Record the orientation a rigid-body integrator computed for a group.
    ///
    /// The one place an orientation is *supplied* rather than derived from coordinates, and it
    /// earns the exception: the integrator advanced the rigid body's rotational degrees of
    /// freedom itself, so its quaternion is the source of the coordinates it wrote, not a claim
    /// about them.
    fn set_group_orientation(&mut self, group_index: usize, orientation: crate::UnitQuaternion);

    /// Resizes a group to a given size.
    ///
    /// Errors if the requested size is larger than the capacity, or if there are
    /// too few active particles to shrink a group.
    fn resize_group(&mut self, group_index: usize, size: GroupSize) -> anyhow::Result<()>;

    /// Set atom kind index of the i'th particle, keeping derived state consistent.
    ///
    /// A kind change invalidates selections that match on atom identity, and — when the mass
    /// differs — the owning group's mass center and bounding radius. Implementations must
    /// maintain both, so that callers never have to remember to.
    fn set_atom_kind(&mut self, index: usize, atom_id: AtomKindId);

    /// Set atom kind index without refreshing the owning group's geometry.
    ///
    /// Only for bulk restores that recompute every group's geometry afterwards; a per-particle
    /// refresh would then be quadratic. Like [`set_atom_kind`](Self::set_atom_kind), it bumps the
    /// atom-kind generation when — and only when — the kind actually changes.
    fn set_atom_kind_unchecked(&mut self, index: usize, atom_id: AtomKindId);

    /// Swap all per-particle array fields (position, atom kind) between two particle indices.
    fn swap_particles(&mut self, i: usize, j: usize);

    /// Update only positions (not atom kinds) for the given indices.
    fn set_positions<'a>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        positions: impl IntoIterator<Item = &'a Point>,
    ) where
        Self: Sized;

    /// Write a whole group's coordinates and settle the derived state that follows from them.
    ///
    /// The mass center and bounding radius are recomputed from the coordinates just written, and
    /// so is the orientation unless the caller passes one it already knows (a Gibbs transfer
    /// carries the molecule's pose across from the other box; see the module docs on how an
    /// orientation is recovered otherwise). A caller replacing a group's coordinates wholesale
    /// therefore cannot leave it describing the molecule that used to occupy the slot — which it
    /// has no way to remember and every reason to forget.
    fn place_group(
        &mut self,
        group_index: usize,
        positions: &[Point],
        orientation: Option<crate::UnitQuaternion>,
    ) -> anyhow::Result<()>;
}

/// Structure storing groups separated into three types:
/// - `full` - all of the atoms of the group are active
/// - `partial` - some of the atoms of the group are active, some are inactive
/// - `empty` - all of the atoms of the group are inactive
///
/// Length of each outer vector corresponds to the number of molecule kinds in the system.
/// Each inner vector then stores ids of groups corresponding to the specific molecule kind.
#[derive(Debug, Clone)]
pub struct GroupLists {
    full: Vec<Vec<usize>>,
    partial: Vec<Vec<usize>>,
    empty: Vec<Vec<usize>>,
    /// Monotonically increasing counter, bumped when the lists change or a
    /// partial group's active count changes (see [`update_group`](Self::update_group)).
    generation: u64,
}

impl PartialEq for GroupLists {
    fn eq(&self, other: &Self) -> bool {
        self.full == other.full && self.partial == other.partial && self.empty == other.empty
    }
}

impl Eq for GroupLists {}

impl GroupLists {
    /// Create and initialize a new GroupLists structure.
    ///
    /// ## Parameters
    /// `n_molecules` - the number of molecule kinds defined in the system
    pub(crate) fn new(n_molecules: usize) -> Self {
        Self {
            full: vec![Vec::new(); n_molecules],
            partial: vec![Vec::new(); n_molecules],
            empty: vec![Vec::new(); n_molecules],
            generation: 0,
        }
    }

    /// Monotonically increasing counter, bumped when the group lists change or a
    /// partial group's active count changes within the partial range.
    /// Consumers can compare against a stored generation to detect staleness.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
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

        Self::add_to_list(list, group);
        self.generation += 1;
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
                    // Empty (0 active) and Full (capacity active) have a fixed active
                    // count, so remaining in the same category means nothing changed.
                    (GroupSize::Empty, GroupSize::Empty) | (GroupSize::Full, GroupSize::Full) => (),
                    // The group stays in the partial list, but its active count may have
                    // changed within the partial range (e.g. a Grand Canonical insert or
                    // delete). Bump the generation so generation-keyed consumers — notably
                    // atom-level `SelectionCache`s — re-resolve. See issue #34.
                    (GroupSize::Partial(_), GroupSize::Partial(_)) => self.generation += 1,
                    // size category changed: move the group to the correct list
                    _ => {
                        list.swap_remove(index);
                        // add_group already bumps generation
                        self.add_group(group);
                    }
                }
            }
            // group is not present in any list, add it
            None => self.add_group(group),
        }
    }

    /// Count groups matching given molecule kind and size.
    pub(crate) fn count_molecules(&self, molecule_id: MoleculeId, size: GroupSize) -> usize {
        self.find_molecules(molecule_id, size).len()
    }

    /// Count non-empty groups (full + partial) for a molecule kind.
    pub(crate) fn count_nonempty(&self, molecule_id: MoleculeId) -> usize {
        self.count_molecules(molecule_id, GroupSize::Full)
            + self.count_molecules(molecule_id, GroupSize::Partial(0))
    }

    /// Find the single mega-group index for an atomic molecule kind.
    ///
    /// Checks partial first since atomic mega-groups are typically partially filled.
    pub(crate) fn find_atomic_group(&self, molecule_id: MoleculeId) -> Option<usize> {
        self.find_molecules(molecule_id, GroupSize::Partial(0))
            .iter()
            .chain(self.find_molecules(molecule_id, GroupSize::Full))
            .chain(self.find_molecules(molecule_id, GroupSize::Empty))
            .copied()
            .next()
    }

    /// Returns indices of all groups matching given molecule id and size.
    ///
    /// Empty when the kind has no group of that size. This deliberately does *not* return an
    /// `Option`: the lists are pre-sized to one entry per molecule kind, so the only way to
    /// miss is an out-of-range id, which cannot arise from a topology lookup. An `Option`
    /// whose `Some` may still wrap an empty slice invites `is_some()` checks that are always
    /// true — which is exactly how the speciation startup validation went vacuous.
    ///
    /// The lookup complexity is O(1).
    pub(crate) fn find_molecules(&self, molecule_id: MoleculeId, size: GroupSize) -> &[usize] {
        let indices = match size {
            GroupSize::Full => self.full.get(molecule_id.get()),
            GroupSize::Partial(_) => self.partial.get(molecule_id.get()),
            GroupSize::Empty => self.empty.get(molecule_id.get()),
            _ => panic!("Unsupported GroupSize."),
        };
        indices.map_or(&[], |i| i.as_slice())
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
                    .get_mut(group.molecule().get())
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
        list.get_mut(group.molecule().get())
            .expect("Incorrectly initialized GroupLists structure.")
            .push(group.index());
    }
}

/// Pins today's behaviour of the two index spaces before they gain distinct types.
///
/// Relative indices count *active* particles from the group's start; absolute indices address the
/// global particle array. The two conversions are deliberately not inverses of each other over the
/// whole group, and the tests below record exactly where they part company.
#[cfg(test)]
mod index_space_characterization {
    use super::*;

    /// Six active particles at absolute 10..16, four inactive at 16..20.
    fn partially_active_group() -> Group {
        let mut group = Group::new(0, MoleculeId::new(0), 10..20);
        group.resize(GroupSize::Partial(6)).unwrap();
        group
    }

    #[test]
    fn relative_and_absolute_round_trip_over_active_particles() {
        let group = partially_active_group();
        for relative in (0..6).map(RelIndex::new) {
            let absolute = group.to_absolute(relative).unwrap();
            assert_eq!(absolute.get(), 10 + relative.get());
            assert_eq!(group.to_relative(absolute).unwrap(), relative);
        }
    }

    #[test]
    fn to_absolute_rejects_inactive_particles() {
        let group = partially_active_group();
        // Relative 6 exists in the capacity but is inactive.
        assert!(group.to_absolute(RelIndex::new(6)).is_err());
        assert!(group.to_absolute(RelIndex::new(10)).is_err());
    }

    /// The asymmetry: `to_relative_index` validates against the *full* range, so it happily maps an
    /// inactive particle to a relative index that `to_absolute_index` then refuses. A future
    /// `RelIndex` newtype must either preserve this or narrow it deliberately.
    #[test]
    fn to_relative_accepts_inactive_particles_that_to_absolute_rejects() {
        let group = partially_active_group();
        let inactive_absolute = AbsIndex::new(17);
        let relative = group.to_relative(inactive_absolute).unwrap();
        assert_eq!(relative.get(), 7);
        assert!(group.to_absolute(relative).is_err());
    }

    #[test]
    fn to_relative_rejects_indices_outside_the_group() {
        let group = partially_active_group();
        assert!(group.to_relative(AbsIndex::new(9)).is_err());
        assert!(group.to_relative(AbsIndex::new(20)).is_err());
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::backend::Backend;
    use crate::WithTopology;

    use super::*;

    /// Was the `Group` doc example, before the module became crate-private and rustdoc stopped
    /// compiling it.
    #[test]
    fn resize_shrinks_a_group() {
        let mut group = Group::new(7, MoleculeId::new(0), 20..23);
        assert_eq!(group.len(), 3);
        assert_eq!(group.size(), GroupSize::Full);

        group.resize(GroupSize::Shrink(1)).unwrap();
        assert_eq!(group.len(), 2);
        assert_eq!(group.capacity(), 3);
        assert_eq!(group.size(), GroupSize::Partial(2));
    }

    #[test]
    fn test_group() {
        // Test group creation
        let mut group = Group {
            molecule: MoleculeId::new(20),
            index: 2,
            geometry: None,
            num_active: 6,
            range: 0..10,
            size_status: GroupSize::Partial(6),
            ..Default::default()
        };

        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        assert_eq!(group.len(), 6);
        assert_eq!(group.capacity(), 10);
        assert_eq!(group.iter_active(), 0..6);
        assert_eq!(group.molecule(), MoleculeId::new(20));
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
        assert!(group.is_full());

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
        let mut group = Group::new(7, MoleculeId::new(0), 20..23);
        assert_eq!(group.len(), 3);
        let indices = group
            .select(
                &ParticleSelection::Relative(vec![
                    RelIndex::new(0),
                    RelIndex::new(1),
                    RelIndex::new(2),
                ]),
                context.topology_ref(),
            )
            .unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Absolute selection
        let indices = group
            .select(
                &ParticleSelection::Absolute(vec![
                    AbsIndex::new(20),
                    AbsIndex::new(21),
                    AbsIndex::new(22),
                ]),
                context.topology_ref(),
            )
            .unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Select all
        let indices = group
            .select(&ParticleSelection::All, context.topology_ref())
            .unwrap();
        assert_eq!(indices, vec![20, 21, 22]);

        // Out of range selection
        assert!(group
            .select(
                &ParticleSelection::Relative(vec![
                    RelIndex::new(1),
                    RelIndex::new(2),
                    RelIndex::new(3)
                ]),
                context.topology_ref()
            )
            .is_err());

        // Test partial selection
        group.resize(GroupSize::Shrink(1)).unwrap();
        let indices = group
            .select(&ParticleSelection::Active, context.topology_ref())
            .unwrap();
        assert_eq!(indices, vec![20, 21]);
        let indices = group
            .select(&ParticleSelection::Inactive, context.topology_ref())
            .unwrap();
        assert_eq!(indices, vec![22]);
    }

    #[test]
    fn test_group_select_by_id() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let group = context.groups().get(1).unwrap();
        let expected0 = vec![7, 11, 12, 13];
        let expected1 = vec![8, 9, 10];

        assert_eq!(
            group
                .select(
                    &ParticleSelection::ById(AtomKindId::new(0)),
                    context.topology_ref()
                )
                .unwrap(),
            expected0
        );

        assert_eq!(
            group
                .select(
                    &ParticleSelection::ById(AtomKindId::new(1)),
                    context.topology_ref()
                )
                .unwrap(),
            expected1
        );

        let group = context.groups().get(45).unwrap();
        let expected_active: Vec<usize> = vec![];

        assert_eq!(
            group
                .select(
                    &ParticleSelection::ById(AtomKindId::new(2)),
                    context.topology_ref()
                )
                .unwrap(),
            expected_active
        );
    }

    #[test]
    fn test_absolute_relative_indices() {
        let group = Group {
            molecule: MoleculeId::new(20),
            index: 2,
            geometry: None,
            num_active: 6,
            range: 10..27,
            size_status: GroupSize::Partial(6),
            ..Default::default()
        };

        assert_eq!(group.to_absolute(RelIndex::new(4)).unwrap().get(), 14);
        assert_eq!(group.to_relative(AbsIndex::new(21)).unwrap().get(), 11);
    }

    #[test]
    fn test_group_lists() {
        let mut group_lists = GroupLists::new(3);

        assert_eq!(group_lists.full.len(), 3);
        assert_eq!(group_lists.partial.len(), 3);
        assert_eq!(group_lists.empty.len(), 3);
        assert_eq!(group_lists.generation(), 0);

        let mut group1 = Group::new(0, MoleculeId::new(0), 3..8);
        let group2 = Group::new(1, MoleculeId::new(0), 8..13);
        let mut group3 = Group::new(2, MoleculeId::new(1), 13..20);

        group_lists.add_group(&group1);
        group_lists.add_group(&group2);
        group_lists.add_group(&group3);
        assert_eq!(group_lists.generation(), 3);

        assert!(group_lists.full[0].contains(&0));
        assert!(group_lists.full[0].contains(&1));
        assert!(group_lists.full[1].contains(&2));

        group1.resize(GroupSize::Empty).unwrap();
        group3.resize(GroupSize::Partial(3)).unwrap();

        let gen_before = group_lists.generation();
        group_lists.update_group(&group1); // Full→Empty: bumps generation
        group_lists.update_group(&group2); // Full→Full: no change
        group_lists.update_group(&group3); // Full→Partial: bumps generation
        assert_eq!(group_lists.generation(), gen_before + 2);

        assert!(!group_lists.full[0].contains(&0));
        assert!(group_lists.empty[0].contains(&0));
        assert!(group_lists.full[0].contains(&1));
        assert!(!group_lists.full[1].contains(&2));
        assert!(group_lists.partial[1].contains(&2));

        // count_molecules: mol 0 has 1 full (group2), 0 partial, 1 empty (group1)
        assert_eq!(
            group_lists.count_molecules(MoleculeId::new(0), GroupSize::Full),
            1
        );
        assert_eq!(
            group_lists.count_molecules(MoleculeId::new(0), GroupSize::Partial(0)),
            0
        );
        assert_eq!(
            group_lists.count_molecules(MoleculeId::new(0), GroupSize::Empty),
            1
        );
        // mol 1 has 0 full, 1 partial (group3), 0 empty
        assert_eq!(
            group_lists.count_molecules(MoleculeId::new(1), GroupSize::Full),
            0
        );
        assert_eq!(
            group_lists.count_molecules(MoleculeId::new(1), GroupSize::Partial(0)),
            1
        );

        // count_nonempty: mol 0 = 1 full + 0 partial = 1; mol 1 = 0 + 1 = 1
        assert_eq!(group_lists.count_nonempty(MoleculeId::new(0)), 1);
        assert_eq!(group_lists.count_nonempty(MoleculeId::new(1)), 1);

        // find_atomic_group: mol 0 has partial=[] then full=[1] then empty=[0] → first is 1
        assert_eq!(group_lists.find_atomic_group(MoleculeId::new(0)), Some(1));
        // mol 1 has partial=[2] → first is 2
        assert_eq!(group_lists.find_atomic_group(MoleculeId::new(1)), Some(2));
        // mol 2 has nothing
        assert_eq!(group_lists.find_atomic_group(MoleculeId::new(2)), None);
    }

    /// Active-count changes within the partial range must bump the generation so
    /// generation-keyed caches (e.g. atom-level `SelectionCache`) re-resolve in a
    /// Grand Canonical simulation. Regression test for issue #34.
    #[test]
    fn test_partial_resize_bumps_generation() {
        let mut group_lists = GroupLists::new(1);
        let mut group = Group::new(0, MoleculeId::new(0), 0..10); // created Full (10 active)
        group_lists.add_group(&group);

        group.resize(GroupSize::Partial(5)).unwrap();
        group_lists.update_group(&group); // Full→Partial: bumps
        let gen = group_lists.generation();

        // Shrink within the partial range: must bump.
        group.resize(GroupSize::Partial(3)).unwrap();
        group_lists.update_group(&group);
        assert_eq!(
            group_lists.generation(),
            gen + 1,
            "Partial→Partial active-count change must bump generation"
        );

        // Grow within the partial range: must bump again.
        group.resize(GroupSize::Partial(7)).unwrap();
        group_lists.update_group(&group);
        assert_eq!(group_lists.generation(), gen + 2);

        // Empty→Empty and Full→Full have fixed active counts and must NOT bump.
        let mut empty = Group::new(1, MoleculeId::new(0), 10..20);
        empty.resize(GroupSize::Empty).unwrap();
        group_lists.add_group(&empty); // adds to empty list (bumps once)
        let gen = group_lists.generation();
        empty.resize(GroupSize::Empty).unwrap();
        group_lists.update_group(&empty); // Empty→Empty: no bump
        assert_eq!(group_lists.generation(), gen);
    }

    /// Verify that GroupCollection trait methods delegate consistently to GroupLists.
    #[test]
    fn test_group_collection_trait_methods() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();
        let gl = context.group_lists();

        // Trait methods should return the same results as GroupLists methods
        for mol_id in (0..context.topology_ref().moleculekinds().len()).map(MoleculeId::new) {
            for size in [GroupSize::Full, GroupSize::Partial(0), GroupSize::Empty] {
                assert_eq!(
                    context.find_molecules(mol_id, size),
                    gl.find_molecules(mol_id, size)
                );
            }
            assert_eq!(
                context.find_atomic_group(mol_id),
                gl.find_atomic_group(mol_id)
            );
            assert_eq!(
                context.count_molecules(mol_id, GroupSize::Full),
                gl.count_molecules(mol_id, GroupSize::Full)
            );
            assert_eq!(context.count_nonempty(mol_id), gl.count_nonempty(mol_id));
        }
        assert_eq!(context.group_lists_generation(), gl.generation());
    }

    #[test]
    fn test_group_selections() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
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
        let selected = context.select(&GroupSelection::ByMoleculeId(MoleculeId::new(0)));
        assert_eq!(selected, expected);

        // With several molecule IDs
        let expected = context.select(&GroupSelection::Size(GroupSize::Full));
        let selected = context.select(&GroupSelection::ByMoleculeIds(vec![
            MoleculeId::new(0),
            MoleculeId::new(1),
        ]));
        assert_eq!(selected, expected);
    }

    #[test]
    fn quaternion_default_is_identity() {
        let group = Group::default();
        assert_eq!(*group.quaternion(), crate::UnitQuaternion::identity());
        let group = Group::new(0, MoleculeId::new(0), 0..5);
        assert_eq!(*group.quaternion(), crate::UnitQuaternion::identity());
    }

    #[test]
    fn quaternion_rotate_by() {
        let mut group = Group::new(0, MoleculeId::new(0), 0..3);
        let axis = nalgebra::UnitVector3::new_normalize(crate::Point::new(0.0, 0.0, 1.0));
        let q1 = crate::UnitQuaternion::from_axis_angle(&axis, 0.3);
        let q2 = crate::UnitQuaternion::from_axis_angle(&axis, 0.5);
        group.rotate_by(&q1);
        group.rotate_by(&q2);
        let expected = q2 * q1;
        assert!((group.quaternion().angle_to(&expected)).abs() < 1e-12);
    }

    #[test]
    fn quaternion_set_get() {
        let mut group = Group::new(0, MoleculeId::new(0), 0..3);
        let axis = nalgebra::UnitVector3::new_normalize(crate::Point::new(1.0, 0.0, 0.0));
        let q = crate::UnitQuaternion::from_axis_angle(&axis, 1.2);
        group.set_quaternion(q);
        assert_eq!(*group.quaternion(), q);
    }
}
