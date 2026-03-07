// Copyright 2023-2024 Mikael Lund
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

//! Implementation of the Nonbonded energy terms.

use interatomic::twobody::{
    ArcPotential, IsotropicTwobodyEnergy, NoInteraction, SplineConfig, SplinedPotential,
};
use ndarray::Array2;
use std::path::Path;
use std::sync::RwLock;

use crate::{
    cell::{PeriodicDirections, SimulationCell},
    energy::{builder::PairPotentialBuilder, EnergyTerm},
    topology::Topology,
    Change, Context, Group, GroupChange,
};

use super::{builder::HamiltonianBuilder, exclusions::ExclusionMatrix, EnergyChange};

/// Trait implemented by all Energy Terms dealing with nonbonded interactions.
pub(super) trait NonbondedTerm {
    /// Calculates the energy between two interacting particles given by absolute indices.
    fn particle_with_particle(&self, context: &impl Context, i: usize, j: usize) -> f64;

    /// Compute the energy of a particle interacting with particles of the specified group.
    /// Ensures self-avoidance, i.e. makes sure that the particle does not interact with itself.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group to calculate interactions with
    ///
    /// ## Example
    /// - Group 1 contains three active particles: A, B, C.
    /// - Calling this method with particle A and group 1 will return the sum of interactions A-B and A-C.
    #[inline(always)]
    fn particle_with_group(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        group
            .iter_active()
            .filter(|j| *j != i)
            .map(|j| self.particle_with_particle(context, i, j))
            .sum()
    }

    /// Compute the energy of a particle interacting with particles of the specified group.
    ///
    /// ## Warning
    /// **Does not ensure self-avoidance!**
    /// Do not use if particle `i` belongs to `group`. Instead, use `particle_with_group`.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group the particle interacts with
    ///
    /// ## Example
    /// - Group 1 contains active particles A, and B.
    ///   Group 2 contains active particles C, D, and E.
    /// - Calling this method with particle A and group 2 will return the sum of interactions
    ///   A-C, A-D, and A-E.
    /// - Calling this method with particle A and group 1 will return the sum of interactions
    ///   A-A, A-B. To get just A-B, use `particle_with_group`.
    #[inline(always)]
    fn particle_with_group_unchecked(
        &self,
        context: &impl Context,
        i: usize,
        group: &Group,
    ) -> f64 {
        group
            .iter_active()
            .map(|j| self.particle_with_particle(context, i, j))
            .sum()
    }

    /// Compute the energy of a particle interacting with particles of all other groups.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group the particle is part of
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with particle A will return the sum of interactions
    ///   A-C, A-D, A-E, and A-F.
    #[inline(always)]
    fn particle_with_other_groups(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        context
            .groups()
            .iter()
            .filter(|group_j| group_j.index() != group.index())
            .map(|group_j| self.particle_with_group(context, i, group_j))
            .sum()
    }

    /// Compute the energy of a particle interacting with all other particles.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group the particle is part of
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with particle A will return the sum of interactions
    ///   A-B, A-C, A-D, A-E, and A-F.
    #[inline(always)]
    fn particle_with_all(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        self.particle_with_other_groups(context, i, group)
            + self.particle_with_group(context, i, group)
    }

    /// Compute the energy of a group interacting with a different group.
    ///
    /// ## Warning
    /// **Does not ensure self-avoidance!**
    /// Do not use if `group1` and `group2` are the same group. Instead, use `group_with_itself`.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group1` - the first interacting group
    /// - `group2` - the second interacting group
    ///
    /// ## Example
    /// - Group 1 contains active particles A, and B.
    ///   Group 2 contains active particles C, D, and E.
    /// - Calling this method with group 1 and group 2 will return the sum of interactions
    ///   A-C, A-D, A-E, B-C, B-D, and B-E.
    #[inline(always)]
    fn group_with_group(&self, context: &impl Context, group1: &Group, group2: &Group) -> f64 {
        group1
            .iter_active()
            .map(|i| self.particle_with_group_unchecked(context, i, group2))
            .sum()
    }

    /// Compute the energy of a group interacting with itself.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group` - interacting group
    ///
    /// ## Example
    /// - Group contains active particles A, B, and C.
    /// - Calling this method will return the sum of interactions A-B, A-C, B-C.
    #[inline(always)]
    fn group_with_itself(&self, context: &impl Context, group: &Group) -> f64 {
        group
            .iter_active()
            .enumerate()
            .flat_map(|(i, p1)| {
                group
                    .iter_active()
                    .skip(i + 1)
                    .map(move |p2| self.particle_with_particle(context, p1, p2))
            })
            .sum()
    }

    /// Compute the energy of a single group interacting with all other groups (not itself).
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group` - interacting group
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with group 1 will return the sum of interactions
    ///   A-C, A-D, A-E, A-F, B-C, B-D, B-E and B-F.
    #[inline(always)]
    fn group_with_other_groups(&self, context: &impl Context, group: &Group) -> f64 {
        group
            .iter_active()
            .map(|i| self.particle_with_other_groups(context, i, group))
            .sum()
    }

    /// Compute the energy of a single group interacting with all particles
    /// (including particles of the group itself).
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group` - interacting group
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with group 1 will return the sum of interactions
    ///   A-B, A-C, A-D, A-E, A-F, B-C, B-D, B-E and B-F.
    #[inline(always)]
    #[allow(dead_code)]
    fn group_with_all(&self, context: &impl Context, group: &Group) -> f64 {
        self.group_with_other_groups(context, group) + self.group_with_itself(context, group)
    }

    /// Energy between two sets of particle indices with automatic deduplication.
    ///
    /// When both slices are identical, only unique pairs (i < j) are summed.
    /// Otherwise, all cross-pairs are included.
    fn indices_with_indices(
        &self,
        context: &impl Context,
        indices1: &[usize],
        indices2: &[usize],
    ) -> f64 {
        let same = indices1 == indices2;
        indices1
            .iter()
            .enumerate()
            .map(|(idx, &i)| {
                let others = if same { &indices2[idx + 1..] } else { indices2 };
                others
                    .iter()
                    .map(|&j| self.particle_with_particle(context, i, j))
                    .sum::<f64>()
            })
            .sum()
    }

    /// Compute the energy of all nonbonded interactions.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method will return the sum of interactions
    ///   A-B, A-C, A-D, A-E, A-F, B-C, B-D, B-E, B-F, C-D, C-E, C-F, D-E, D-F, E-F.
    #[inline(always)]
    fn total_nonbonded(&self, context: &impl Context) -> f64 {
        context
            .groups()
            .iter()
            .enumerate()
            .flat_map(|(i, group_i)| {
                context
                    .groups()
                    .iter()
                    .skip(i + 1)
                    .map(move |group_j| self.group_with_group(context, group_i, group_j))
                    .chain(std::iter::once(self.group_with_itself(context, group_i)))
            })
            .sum()
    }
}

/// Energy term for computing nonbonded interactions
/// using a matrix of `IsotropicTwobodyEnergy` trait objects.
///
/// The type parameter `P` determines the potential type:
/// - [`ArcPotential`] for dynamic dispatch (default)
/// - [`SplinedPotential`] for pre-tabulated spline evaluation
#[derive(Debug)]
pub struct NonbondedMatrix<P = ArcPotential> {
    /// Matrix of pair potentials based on atom type ids.
    potentials: Array2<P>,
    /// Matrix of excluded interactions.
    exclusions: ExclusionMatrix,
    /// Pairwise inter-group energy cache for O(1) old-energy lookup in MC moves.
    cache: RwLock<Option<GroupEnergyCache>>,
    /// Global interaction cutoff for bounding-sphere culling (None = no culling).
    cutoff: Option<f64>,
    /// Enable bounding-sphere culling of distant group pairs.
    use_bounding_spheres: bool,
}

impl<P: Clone> Clone for NonbondedMatrix<P> {
    fn clone(&self) -> Self {
        Self {
            potentials: self.potentials.clone(),
            exclusions: self.exclusions.clone(),
            cache: RwLock::new(self.cache.read().unwrap().clone()),
            cutoff: self.cutoff,
            use_bounding_spheres: self.use_bounding_spheres,
        }
    }
}

/// Type alias for the splined variant of [`NonbondedMatrix`].
///
/// This is a performance-optimized version that pre-tabulates all pair potentials
/// using cubic Hermite splines for O(1) energy evaluation.
///
/// # Notes
/// - Splined potentials trade memory for speed and are most beneficial for
///   complex potentials (coarse-grained models, combined potentials).
/// - The spline operates in r² space to avoid square root calculations.
/// - Energy evaluation uses Horner's method requiring only 4 FMA operations.
pub type NonbondedMatrixSplined = NonbondedMatrix<SplinedPotential>;

impl<P: IsotropicTwobodyEnergy> EnergyChange for NonbondedMatrix<P> {
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        // O(1) cache hit for rigid body moves (the MC hot path)
        if let Change::SingleGroup(gi, GroupChange::RigidBody) = change {
            if let Some(ref cache) = *self.cache.read().unwrap() {
                debug_assert_eq!(cache.n_groups, context.groups().len());
                return cache.group_energies[*gi];
            }
        }

        // SoA fast path: bypasses per-pair Context trait dispatch and Cell enum
        // matching by reading directly from contiguous f64 slices with inline
        // branchless PBC distance (~26% faster than scalar AoS path).
        if let (Some((x, y, z)), Some(atom_kinds)) =
            (context.positions_soa(), context.atom_kinds_u32())
        {
            let cell = context.cell();
            // Prefer cached PBC params (SoaPlatform); fall back to computing them (AosPlatform)
            let pbc = context
                .pbc_params()
                .or_else(|| PbcParams::try_from_cell(cell));
            let soa = SoaSlices {
                x,
                y,
                z,
                atom_kinds,
                pbc,
                cell,
                cell_list: context.cell_list(),
            };
            let groups = context.groups();
            return match change {
                Change::Everything | Change::Volume(_, _) => self.total_nonbonded_soa(&soa, groups),
                Change::SingleGroup(gi, GroupChange::RigidBody) => {
                    // Cache miss — lazy-initialize all pairwise energies
                    self.initialize_cache_soa(&soa, groups);
                    self.cache.read().unwrap().as_ref().unwrap().group_energies[*gi]
                }
                Change::SingleGroup(group_index, group_change) => {
                    self.single_group_change_soa(&soa, groups, *group_index, group_change)
                }
                Change::Groups(vec) => vec
                    .iter()
                    .map(|(group_index, group_change)| {
                        self.single_group_change_soa(&soa, groups, *group_index, group_change)
                    })
                    .sum(),
                Change::None => 0.0,
            };
        }
        // Scalar fallback for AoS platforms
        match change {
            Change::Everything | Change::Volume(_, _) => self.total_nonbonded(context),
            Change::SingleGroup(group_index, group_change) => {
                self.single_group_change(context, *group_index, group_change)
            }
            Change::Groups(vec) => vec
                .iter()
                .map(|(group, change)| self.single_group_change(context, *group, change))
                .sum(),
            Change::None => 0.0,
        }
    }
}

impl<P: IsotropicTwobodyEnergy> NonbondedTerm for NonbondedMatrix<P> {
    /// Calculates the energy between two particles given by indices.
    #[inline(always)]
    fn particle_with_particle(&self, context: &impl Context, i: usize, j: usize) -> f64 {
        let distance_squared = context.get_distance_squared(i, j);
        self.exclusions.get((i, j)) as f64
            * self
                .potentials
                .get((context.get_atomkind(i), context.get_atomkind(j)))
                .expect("Atom kinds should exist in the nonbonded matrix.")
                .isotropic_twobody_energy(distance_squared)
    }
}

impl<P> NonbondedMatrix<P> {
    /// Get square matrix of pair potentials for all atom type combinations.
    pub const fn get_potentials(&self) -> &Array2<P> {
        &self.potentials
    }

    /// Enable or disable bounding-sphere culling of distant group pairs.
    pub(crate) fn set_bounding_spheres(&mut self, enabled: bool) {
        self.use_bounding_spheres = enabled;
    }
}

impl<P: IsotropicTwobodyEnergy> NonbondedMatrix<P> {
    /// Matches all possible single group perturbations and returns the energy.
    fn single_group_change(
        &self,
        context: &impl Context,
        group_index: usize,
        change: &GroupChange,
    ) -> f64 {
        match change {
            GroupChange::RigidBody => {
                self.group_with_other_groups(context, &context.groups()[group_index])
            }
            GroupChange::Resize(_) | GroupChange::UpdateIdentity(_) => {
                // Unlike RigidBody, the active particle set changed so intra-group
                // interactions must also be recomputed
                let group = &context.groups()[group_index];
                self.group_with_other_groups(context, group)
                    + self.group_with_itself(context, group)
            }
            GroupChange::PartialUpdate(x) => {
                let group = &context.groups()[group_index];
                x.iter()
                    .filter_map(|&rel| group.to_absolute_index(rel).ok())
                    .map(|abs_i| self.particle_with_all(context, abs_i, group))
                    .sum()
            }
            GroupChange::None => 0.0,
        }
    }
}

impl NonbondedMatrix {
    /// Create from YAML file and a topology
    ///
    /// Can be used to generate a new `EnergyTerm` with:
    ///
    /// ```ignore
    /// let energy = EnergyTerm::From(NonbondedMatrix::from_file(...).unwrap());
    /// ```
    pub fn from_file(
        file: impl AsRef<Path>,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let builder = HamiltonianBuilder::from_file(file)?;
        Self::new(
            &builder.pairpot_builder.unwrap(),
            topology,
            medium,
            builder.combine_with_default,
        )
    }

    /// Create a new NonbondedReference structure wrapped in an EnergyTerm enum.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(
        pairpot_builder: &PairPotentialBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
        combine_with_default: bool,
    ) -> anyhow::Result<Self> {
        let atoms = topology.atomkinds();
        let n_atom_types = atoms.len();

        let mut potentials: Array2<ArcPotential> = Array2::from_elem(
            (n_atom_types, n_atom_types),
            ArcPotential::new(NoInteraction),
        );

        for i in 0..n_atom_types {
            for j in 0..n_atom_types {
                let interaction = pairpot_builder.get_interaction(
                    &atoms[i],
                    &atoms[j],
                    medium.clone(),
                    combine_with_default,
                )?;
                potentials[(i, j)] = ArcPotential(interaction.into());
            }
        }

        let exclusions = ExclusionMatrix::from_topology(topology);

        Ok(Self {
            potentials,
            exclusions,
            cache: RwLock::new(None),
            cutoff: None,
            use_bounding_spheres: true,
        })
    }

    /// Get mutable reference to pair potentials matrix.
    pub const fn get_potentials_mut(&mut self) -> &mut Array2<ArcPotential> {
        &mut self.potentials
    }
}

impl From<NonbondedMatrix> for EnergyTerm {
    fn from(nonbonded: NonbondedMatrix) -> Self {
        Self::NonbondedMatrix(nonbonded)
    }
}

impl NonbondedMatrix<SplinedPotential> {
    /// Create a splined nonbonded matrix from an existing [`NonbondedMatrix`].
    ///
    /// # Parameters
    /// - `nonbonded`: The source nonbonded matrix containing pair potentials to spline.
    /// - `cutoff`: The cutoff distance for all interactions.
    /// - `config`: Optional spline configuration. If `None`, uses default settings.
    ///
    /// # Example
    /// ```ignore
    /// let nonbonded = NonbondedMatrix::new(&builder, &topology, medium, false)?;
    /// let splined = NonbondedMatrixSplined::from_nonbonded(&nonbonded, 12.0, None);
    /// ```
    pub fn from_nonbonded(
        nonbonded: &NonbondedMatrix,
        cutoff: f64,
        config: Option<SplineConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();
        let source = nonbonded.get_potentials();
        let shape = source.raw_dim();

        let potentials = Array2::from_shape_fn(shape, |(i, j)| {
            let potential = source.get((i, j)).expect("Index should be valid");
            SplinedPotential::with_cutoff(potential, cutoff, config.clone())
        });

        Self {
            potentials,
            exclusions: nonbonded.exclusions.clone(),
            cache: RwLock::new(None),
            cutoff: Some(cutoff),
            use_bounding_spheres: true,
        }
    }
}

/// Precomputed parameters for branchless minimum image distance.
///
/// Unifies all orthorhombic cell types (Cuboid, Slit, Cylinder, Sphere, Endless)
/// into a single arithmetic path: `dx -= box * round(dx * inv_box)`.
/// Non-periodic directions use `(f64::MAX, 0.0)` so `round(dx * 0.0) = 0`
/// and the correction vanishes without branching.
#[derive(Clone, Copy, Debug)]
pub struct PbcParams {
    box_len: [f64; 3],
    inv_box_len: [f64; 3],
}

impl PbcParams {
    /// Build from a simulation cell. Returns `None` for HexagonalPrism
    /// which needs non-orthorhombic Wigner-Seitz nearest image reduction.
    pub(crate) fn try_from_cell(cell: &impl SimulationCell) -> Option<Self> {
        // HexagonalPrism is the only cell that requires orthorhombic expansion
        if cell.orthorhombic_expansion().is_some() {
            return None;
        }
        let pbc = cell.pbc();
        let periodic_xy = matches!(
            pbc,
            PeriodicDirections::PeriodicXYZ | PeriodicDirections::PeriodicXY
        );
        let periodic_z = matches!(
            pbc,
            PeriodicDirections::PeriodicXYZ | PeriodicDirections::PeriodicZ
        );

        let bb = cell.bounding_box();
        // Non-periodic directions get (MAX, 0.0): round(dx * 0.0) = 0, so no correction
        let make = |periodic: bool, len: f64| -> (f64, f64) {
            if periodic {
                (len, 1.0 / len)
            } else {
                (f64::MAX, 0.0)
            }
        };
        let (bx, by, bz) = bb.map_or((f64::MAX, f64::MAX, f64::MAX), |b| (b.x, b.y, b.z));
        let (lx, ix) = make(periodic_xy, bx);
        let (ly, iy) = make(periodic_xy, by);
        let (lz, iz) = make(periodic_z, bz);

        Some(Self {
            box_len: [lx, ly, lz],
            inv_box_len: [ix, iy, iz],
        })
    }

    /// Branchless minimum image distance squared between two points.
    #[inline(always)]
    fn distance_squared(&self, xi: f64, yi: f64, zi: f64, xj: f64, yj: f64, zj: f64) -> f64 {
        let mut dx = xi - xj;
        dx -= self.box_len[0] * (dx * self.inv_box_len[0]).round();
        let mut dy = yi - yj;
        dy -= self.box_len[1] * (dy * self.inv_box_len[1]).round();
        let mut dz = zi - zj;
        dz -= self.box_len[2] * (dz * self.inv_box_len[2]).round();
        dx.mul_add(dx, dy.mul_add(dy, dz * dz))
    }
}

/// Pairwise inter-group nonbonded energy cache.
///
/// Stores E(i,j) for all group pairs so that `group_energies[m]` (the total
/// inter-group energy of group m) can be returned in O(1) instead of O(N_groups).
/// On accept, symmetric delta propagation keeps all entries consistent in O(N_groups).
#[derive(Debug, Clone, Default)]
struct GroupEnergyCache {
    /// `pairwise[i * n + j]` = nonbonded energy between groups i and j
    pairwise: Vec<f64>,
    /// `group_energies[i]` = Σ_j pairwise[i * n + j]
    group_energies: Vec<f64>,
    n_groups: usize,
    // Backup buffers live inline so save_backup() reuses capacity instead of
    // allocating new Vecs on every MC step.
    backup_row: Vec<f64>,
    backup_group_energies: Vec<f64>,
    backup_group_index: usize,
    has_backup: bool,
}

impl GroupEnergyCache {
    fn save_backup(&mut self, group_index: usize) {
        let n = self.n_groups;
        let row_start = group_index * n;
        self.backup_row.clear();
        self.backup_row
            .extend_from_slice(&self.pairwise[row_start..row_start + n]);
        self.backup_group_energies.clear();
        self.backup_group_energies
            .extend_from_slice(&self.group_energies);
        self.backup_group_index = group_index;
        self.has_backup = true;
    }

    /// Restore both row and column of the moved group to keep the matrix symmetric.
    fn undo(&mut self) {
        if self.has_backup {
            let m = self.backup_group_index;
            let n = self.n_groups;
            for j in 0..n {
                self.pairwise[m * n + j] = self.backup_row[j];
                self.pairwise[j * n + m] = self.backup_row[j];
            }
            self.group_energies
                .copy_from_slice(&self.backup_group_energies);
            self.has_backup = false;
        }
    }

    fn discard_backup(&mut self) {
        self.has_backup = false;
    }
}

/// Conservative lower bound on group-group distance using bounding spheres.
///
/// Returns `None` if either group lacks a mass center or bounding radius,
/// meaning the caller should fall back to exact pair evaluation.
#[cfg(test)]
fn min_group_distance(gi: &Group, gj: &Group, cell: &impl SimulationCell) -> Option<f64> {
    let com_i = gi.mass_center()?;
    let com_j = gj.mass_center()?;
    let ri = gi.bounding_radius()?;
    let rj = gj.bounding_radius()?;
    let com_dist = cell.distance(com_i, com_j).norm();
    Some((com_dist - ri - rj).max(0.0))
}

/// Borrowed SoA arrays for batch nonbonded energy evaluation.
///
/// Holds `cell` for HexagonalPrism fallback; all other cell types
/// use precomputed `pbc` params for inline branchless distance.
struct SoaSlices<'a, C: SimulationCell> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    atom_kinds: &'a [u32],
    cell: &'a C,
    pbc: Option<PbcParams>,
    cell_list: Option<&'a crate::celllist::CellList>,
}

impl<'a, C: SimulationCell> SoaSlices<'a, C> {
    /// Minimum image squared distance from `(xi, yi, zi)` to particle `j`.
    ///
    /// # Safety
    /// `j` must be in bounds for `self.x`, `self.y`, `self.z`.
    #[inline(always)]
    unsafe fn distance_squared_to(&self, xi: f64, yi: f64, zi: f64, j: usize) -> f64 {
        if let Some(pbc) = &self.pbc {
            pbc.distance_squared(
                xi,
                yi,
                zi,
                *self.x.get_unchecked(j),
                *self.y.get_unchecked(j),
                *self.z.get_unchecked(j),
            )
        } else {
            // HexagonalPrism fallback: reconstruct Points for Wigner-Seitz reduction
            let pi = crate::Point::new(xi, yi, zi);
            let pj = crate::Point::new(
                *self.x.get_unchecked(j),
                *self.y.get_unchecked(j),
                *self.z.get_unchecked(j),
            );
            self.cell.distance_squared(&pi, &pj)
        }
    }
}

impl<P: IsotropicTwobodyEnergy> NonbondedMatrix<P> {
    /// Fast check if two groups are beyond interaction range using squared
    /// COM distance and precomputed (cutoff + R_i + R_j)². Uses branchless
    /// PBC distance when available, avoiding sqrt and trait dispatch.
    #[inline(always)]
    fn groups_beyond_cutoff(&self, gi: &Group, gj: &Group, pbc: Option<&PbcParams>) -> bool {
        if !self.use_bounding_spheres {
            return false;
        }
        let cutoff = match self.cutoff {
            Some(c) => c,
            None => return false,
        };
        let (com_i, com_j) = match (gi.mass_center(), gj.mass_center()) {
            (Some(a), Some(b)) => (a, b),
            _ => return false,
        };
        // (cutoff + R_i + R_j)² — no sqrt needed
        let ri = gi.bounding_radius().unwrap_or(0.0);
        let rj = gj.bounding_radius().unwrap_or(0.0);
        let threshold = cutoff + ri + rj;
        let threshold_sq = threshold * threshold;
        let dist_sq = match pbc {
            Some(pbc) => pbc.distance_squared(com_i.x, com_i.y, com_i.z, com_j.x, com_j.y, com_j.z),
            None => {
                // Non-orthorhombic fallback (rare)
                let d = com_i - com_j;
                d.x * d.x + d.y * d.y + d.z * d.z
            }
        };
        dist_sq > threshold_sq
    }

    /// Energy of particle `i` with targets, reading directly from SoA arrays.
    ///
    /// Uses unchecked indexing to eliminate ~6 bounds checks per pair in the inner
    /// loop — this is the single hottest function in MC sweeps. All indices are
    /// validated by `debug_assert!` so debug builds still catch OOB.
    ///
    /// # Safety invariants (upheld by callers)
    /// - `i` and all `targets` values are in `[0, soa.x.len())`
    /// - `atom_kinds[i]` values index within `self.potentials` dimensions
    #[inline]
    fn particle_energy_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        i: usize,
        targets: impl Iterator<Item = usize>,
    ) -> f64 {
        debug_assert!(i < soa.x.len());
        // SAFETY: i is a valid particle index from an active group range.
        let (xi, yi, zi, kind_i) = unsafe {
            (
                *soa.x.get_unchecked(i),
                *soa.y.get_unchecked(i),
                *soa.z.get_unchecked(i),
                *soa.atom_kinds.get_unchecked(i) as usize,
            )
        };
        // Row slice gives a contiguous &[u8] for sequential cache-line reads over j.
        let excl_row = self.exclusions.row(i);
        let mut energy = 0.0;
        for j in targets {
            debug_assert!(j < soa.x.len());
            // SAFETY: j is a valid particle index from an active group range;
            // atom kinds are topology-derived indices into the potentials matrix.
            // No j==i guard needed: callers use disjoint ranges, and the exclusion
            // matrix diagonal is 0 (self-exclusion) so it would be skipped anyway.
            unsafe {
                if *excl_row.get_unchecked(j) == 0 {
                    continue;
                }
                let rsq = soa.distance_squared_to(xi, yi, zi, j);
                let kind_j = *soa.atom_kinds.get_unchecked(j) as usize;
                energy += self
                    .potentials
                    .uget((kind_i, kind_j))
                    .isotropic_twobody_energy(rsq);
            }
        }
        energy
    }

    /// Intra-group energy (i < j pairs) via SoA arrays.
    #[inline]
    fn group_with_itself_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        group: &Group,
    ) -> f64 {
        let active = group.iter_active();
        active
            .clone()
            .map(|i| self.particle_energy_soa(soa, i, (i + 1)..active.end))
            .sum()
    }

    /// Total nonbonded energy via SoA arrays.
    fn total_nonbonded_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        groups: &[Group],
    ) -> f64 {
        groups
            .iter()
            .enumerate()
            .map(|(gi, group_i)| {
                let inter: f64 = groups
                    .iter()
                    .skip(gi + 1)
                    .filter(|group_j| {
                        !self.groups_beyond_cutoff(group_i, group_j, soa.pbc.as_ref())
                    })
                    .flat_map(|group_j| {
                        group_i
                            .iter_active()
                            .map(move |i| self.particle_energy_soa(soa, i, group_j.iter_active()))
                    })
                    .sum();
                inter + self.group_with_itself_soa(soa, group_i)
            })
            .sum()
    }

    /// Inter-group energy of one particle via SoA arrays.
    #[inline]
    fn inter_group_energy_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        groups: &[Group],
        i: usize,
        own_group: &Group,
    ) -> f64 {
        groups
            .iter()
            .filter(|gj| gj.index() != own_group.index())
            .filter(|gj| !self.groups_beyond_cutoff(own_group, gj, soa.pbc.as_ref()))
            .map(|gj| self.particle_energy_soa(soa, i, gj.iter_active()))
            .sum()
    }

    /// Single group change via pre-extracted SoA arrays.
    fn single_group_change_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        groups: &[Group],
        group_index: usize,
        change: &GroupChange,
    ) -> f64 {
        let group = &groups[group_index];
        match change {
            GroupChange::RigidBody => self.inter_group_energy_all_soa(soa, groups, group),
            GroupChange::Resize(_) | GroupChange::UpdateIdentity(_) => {
                self.inter_group_energy_all_soa(soa, groups, group)
                    + self.group_with_itself_soa(soa, group)
            }
            GroupChange::PartialUpdate(indices) => indices
                .iter()
                .map(|&rel_idx| {
                    group.to_absolute_index(rel_idx).map_or(0.0, |abs_i| {
                        self.particle_energy_all_soa(soa, groups, abs_i, group)
                    })
                })
                .sum(),
            GroupChange::None => 0.0,
        }
    }

    /// Inter-group energy of an entire group with all other groups.
    ///
    /// With a cell list, iterates spatial neighbors per atom and excludes
    /// own-group contributions. Falls back to bounding-sphere-filtered
    /// group iteration otherwise.
    fn inter_group_energy_all_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        groups: &[Group],
        group: &Group,
    ) -> f64 {
        if let Some(cl) = soa.cell_list {
            let own_range = group.iter_active();
            return group
                .iter_active()
                .map(|i| {
                    self.particle_energy_soa(
                        soa,
                        i,
                        cl.neighbors(i)
                            .filter(move |&j| j < own_range.start || j >= own_range.end),
                    )
                })
                .sum();
        }
        group
            .iter_active()
            .map(|i| self.inter_group_energy_soa(soa, groups, i, group))
            .sum()
    }

    /// Total energy of particle `abs_i` with all other particles.
    ///
    /// When a cell list is available, iterates only over spatial neighbors
    /// (O(neighbors) instead of O(N)). Falls back to group-based iteration otherwise.
    #[inline]
    fn particle_energy_all_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        groups: &[Group],
        abs_i: usize,
        own_group: &Group,
    ) -> f64 {
        if let Some(cl) = soa.cell_list {
            return self.particle_energy_soa(
                soa,
                abs_i,
                cl.neighbors(abs_i).filter(move |&j| j != abs_i),
            );
        }
        // Fallback: inter-group + intra-group
        self.inter_group_energy_soa(soa, groups, abs_i, own_group)
            + self.particle_energy_soa(soa, abs_i, own_group.iter_active())
    }

    /// Inter-group nonbonded energy between two specific groups via SoA arrays.
    #[inline]
    fn group_pair_energy_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        group_i: &Group,
        group_j: &Group,
    ) -> f64 {
        if self.groups_beyond_cutoff(group_i, group_j, soa.pbc.as_ref()) {
            return 0.0;
        }
        group_i
            .iter_active()
            .map(|i| self.particle_energy_soa(soa, i, group_j.iter_active()))
            .sum()
    }

    /// Compute all pairwise inter-group energies and populate the cache.
    fn initialize_cache_soa(&self, soa: &SoaSlices<'_, impl SimulationCell>, groups: &[Group]) {
        let n = groups.len();
        let mut pairwise = vec![0.0; n * n];
        let mut group_energies = vec![0.0; n];

        for (gi, group_i) in groups.iter().enumerate() {
            debug_assert!(group_i.index() == gi);
            for group_j in groups.iter().skip(gi + 1) {
                let gj = group_j.index();
                let e = self.group_pair_energy_soa(soa, group_i, group_j);
                pairwise[gi * n + gj] = e;
                pairwise[gj * n + gi] = e;
                group_energies[gi] += e;
                group_energies[gj] += e;
            }
        }

        *self.cache.write().unwrap() = Some(GroupEnergyCache {
            pairwise,
            group_energies,
            n_groups: n,
            ..Default::default()
        });
    }
}

impl<P: IsotropicTwobodyEnergy> NonbondedMatrix<P> {
    /// Backup cache state before a move. Invalidates for topology-changing moves.
    pub(super) fn save_backup(&mut self, change: &Change) {
        match change {
            Change::SingleGroup(gi, GroupChange::RigidBody | GroupChange::PartialUpdate(_)) => {
                if let Some(c) = self.cache.get_mut().unwrap().as_mut() {
                    c.save_backup(*gi);
                }
            }
            // Topology-changing moves (resize, insert, identity swap, etc.)
            // invalidate the entire N×N pairwise matrix.
            _ => {
                *self.cache.get_mut().unwrap() = None;
            }
        }
    }

    /// Recompute the moved group's cache row with new positions (accept/reject agnostic).
    pub(super) fn update_cache(&mut self, context: &impl Context, change: &Change) {
        let group_index = match change {
            Change::SingleGroup(gi, GroupChange::RigidBody | GroupChange::PartialUpdate(_)) => gi,
            _ => return,
        };

        // take() avoids a borrow conflict: we need &mut cache while calling
        // group_pair_energy_soa(&self), and the cache lives inside self.
        let mut cache_opt = self.cache.get_mut().unwrap().take();
        if let Some(ref mut cache) = cache_opt {
            if let (Some((x, y, z)), Some(atom_kinds)) =
                (context.positions_soa(), context.atom_kinds_u32())
            {
                let cell = context.cell();
                let pbc = context
                    .pbc_params()
                    .or_else(|| PbcParams::try_from_cell(cell));
                let soa = SoaSlices {
                    x,
                    y,
                    z,
                    atom_kinds,
                    pbc,
                    cell,
                    cell_list: None,
                };
                let groups = context.groups();
                let n = cache.n_groups;
                let m = *group_index;

                for gj in groups.iter() {
                    let j = gj.index();
                    if j == m {
                        continue;
                    }
                    let new_pair = self.group_pair_energy_soa(&soa, &groups[m], gj);
                    let old_pair = cache.pairwise[m * n + j];
                    let delta = new_pair - old_pair;
                    cache.pairwise[m * n + j] = new_pair;
                    cache.pairwise[j * n + m] = new_pair;
                    cache.group_energies[j] += delta;
                }
                // Resum from the updated row to avoid accumulating floating-point drift.
                cache.group_energies[m] = (0..n).map(|j| cache.pairwise[m * n + j]).sum();
            }
        }
        *self.cache.get_mut().unwrap() = cache_opt;
    }

    /// Restore cache from backup (MC reject path).
    pub(super) fn undo(&mut self) {
        if let Some(c) = self.cache.get_mut().unwrap().as_mut() {
            c.undo();
        }
    }

    /// Drop cache backup (MC accept path).
    pub(super) fn discard_backup(&mut self) {
        if let Some(c) = self.cache.get_mut().unwrap().as_mut() {
            c.discard_backup();
        }
    }

    /// Invalidate the pairwise energy cache (e.g. after Langevin dynamics).
    /// The cache will be lazily rebuilt on the next rigid-body energy evaluation.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn invalidate_cache(&mut self) {
        *self.cache.get_mut().unwrap() = None;
    }
}

impl From<NonbondedMatrixSplined> for EnergyTerm {
    fn from(nonbonded: NonbondedMatrixSplined) -> Self {
        Self::NonbondedMatrixSplined(nonbonded)
    }
}

impl From<&NonbondedMatrix> for NonbondedMatrixSplined {
    /// Create a splined nonbonded matrix from a reference to [`NonbondedMatrix`]
    /// using default spline configuration and a cutoff of 15.0 Å.
    ///
    /// For custom cutoff or configuration, use [`NonbondedMatrixSplined::from_nonbonded`] instead.
    fn from(nonbonded: &NonbondedMatrix) -> Self {
        Self::from_nonbonded(nonbonded, 15.0, None)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, sync::Arc};

    use float_cmp::assert_approx_eq;

    use crate::{
        cell::{Cell, Cuboid},
        energy::{builder::HamiltonianBuilder, Hamiltonian},
        group::{GroupCollection, GroupSize},
        montecarlo::NewOld,
        platform::aos::AosPlatform,
        topology::Topology,
    };

    use super::*;

    /// Compare behavior of two `IsotropicTwobodyEnergy` trait objects.
    fn assert_behavior(obj1: &dyn IsotropicTwobodyEnergy, obj2: &dyn IsotropicTwobodyEnergy) {
        let testing_distances = [0.00201, 0.7, 12.3, 12457.6];

        for &dist in testing_distances.iter() {
            assert_approx_eq!(
                f64,
                obj1.isotropic_twobody_energy(dist),
                obj2.isotropic_twobody_energy(dist)
            );
        }
    }

    #[test]
    fn test_nonbonded_matrix_new() {
        let file = "tests/files/topology_pass.yaml";
        let topology = Topology::from_file(file).unwrap();
        let pairpot_builder = HamiltonianBuilder::from_file(file)
            .unwrap()
            .pairpot_builder
            .unwrap();
        let medium: Option<interatomic::coulomb::Medium> =
            serde_yaml::from_reader(std::fs::File::open(file).unwrap())
                .ok()
                .and_then(|s: serde_yaml::Value| {
                    let medium = s.get("system")?.get("medium")?;
                    serde_yaml::from_value(medium.clone()).ok()
                });

        let nonbonded = NonbondedMatrix::new(&pairpot_builder, &topology, medium, false).unwrap();

        assert_eq!(
            nonbonded.potentials.len(),
            topology.atomkinds().len() * topology.atomkinds().len()
        );

        for i in 0..topology.atomkinds().len() {
            for j in (i + 1)..topology.atomkinds().len() {
                assert_behavior(
                    nonbonded.potentials.get((i, j)).unwrap(),
                    nonbonded.potentials.get((j, i)).unwrap(),
                );
            }
        }

        // O, C with anything: default interaction
        let o_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "O")
            .unwrap();
        let c_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "C")
            .unwrap();

        let default = nonbonded.potentials.get((o_index, o_index)).unwrap();

        for i in [o_index, c_index] {
            for j in 0..topology.atomkinds().len() {
                assert_behavior(nonbonded.potentials.get((i, j)).unwrap(), default);
            }
        }

        // X interacts slightly differently with charged atoms because it is itself charged
        let x_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "X")
            .unwrap();
        let ow_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "OW")
            .unwrap();

        for i in 0..topology.atomkinds().len() {
            if i == x_index || i == ow_index {
                continue;
            }

            assert_behavior(nonbonded.potentials.get((x_index, i)).unwrap(), default);
        }
    }

    /// Assert particle-particle interaction energy.
    fn assert_part_part(
        system: &impl Context,
        nonbonded: &NonbondedMatrix,
        i: usize,
        j: usize,
        expected: f64,
    ) {
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_particle(system, i, j),
            expected
        );
    }

    /// Get nonbonded matrix for testing.
    fn get_test_matrix() -> (AosPlatform, NonbondedMatrix) {
        let file = "tests/files/nonbonded_interactions.yaml";
        let topology = Topology::from_file(file).unwrap();
        let builder = HamiltonianBuilder::from_file(file)
            .unwrap()
            .pairpot_builder
            .unwrap();

        let medium = interatomic::coulomb::Medium::new(
            298.15,
            interatomic::coulomb::permittivity::Permittivity::Vacuum,
            None,
        );

        let nonbonded = NonbondedMatrix::new(&builder, &topology, Some(medium), false).unwrap();

        let mut rng = rand::thread_rng();
        let system = AosPlatform::from_raw_parts(
            Arc::new(topology),
            Cell::Cuboid(Cuboid::cubic(20.0)),
            RefCell::new(Hamiltonian::from(vec![nonbonded.clone().into()])),
            None,
            &mut rng,
        )
        .unwrap();

        (system, nonbonded)
    }

    #[test]
    fn test_nonbonded_matrix_particle_particle() {
        let (system, nonbonded) = get_test_matrix();

        // intramolecular

        let intramolecular_a1b_energy = -0.356652949245542;
        for (i, j) in [(0, 1), (3, 4), (6, 7)] {
            assert_part_part(&system, &nonbonded, i, j, intramolecular_a1b_energy);
            assert_part_part(&system, &nonbonded, j, i, intramolecular_a1b_energy);
        }

        let intramolecular_a1a2_energy = 0.0;
        for (i, j) in [(0, 2), (3, 5), (6, 8)] {
            assert_part_part(&system, &nonbonded, i, j, intramolecular_a1a2_energy);
            assert_part_part(&system, &nonbonded, j, i, intramolecular_a1a2_energy);
        }

        let intramolecular_a2b_energy = -0.000233230711693257;
        for (i, j) in [(1, 2), (4, 5), (7, 8)] {
            assert_part_part(&system, &nonbonded, i, j, intramolecular_a2b_energy);
            assert_part_part(&system, &nonbonded, j, i, intramolecular_a2b_energy);
        }

        // intermolecular

        let intermolecular_a1a1_energy = [
            401.06633566678175,
            401.06633566678175,
            -0.000090421636081691,
        ];
        for ((i, j), energy) in [(0, 3), (0, 6), (3, 6)]
            .into_iter()
            .zip(intermolecular_a1a1_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a1b_energy = [
            -0.000508026822504991,
            -0.356652949245542,
            -0.000508026822504991,
            -2.3703647517146784e-5,
            -0.356652949245542,
            -0.000508026822504991,
        ];
        for ((i, j), energy) in [(0, 4), (0, 7), (3, 7), (4, 6), (1, 3), (1, 6)]
            .into_iter()
            .zip(intermolecular_a1b_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a1a2_energy = [
            -6.406572630990959e-6,
            310.66787793413096,
            491.1915281349669,
            -1.2499998437500003e-6,
            310.66787793413096,
            -6.406572630990959e-6,
        ];
        for ((i, j), energy) in [(0, 5), (0, 8), (3, 8), (5, 6), (2, 3), (2, 6)]
            .into_iter()
            .zip(intermolecular_a1a2_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_bb_energy =
            [-0.713305898491084, -0.713305898491084, -0.01156737611454047];
        for ((i, j), energy) in [(1, 4), (1, 7), (4, 7)]
            .into_iter()
            .zip(intermolecular_bb_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a2b_energy = [
            -1.748899941931173e-5,
            -0.0075075032697152,
            -0.0075075032697152,
            -2.6853740564936314e-6,
            -0.0075075032697152,
            -1.748899941931173e-5,
        ];
        for ((i, j), energy) in [(1, 5), (1, 8), (4, 8), (7, 5), (4, 2), (7, 2)]
            .into_iter()
            .zip(intermolecular_a2b_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a2a2_energy = [
            401.06633566678175,
            401.06633566678175,
            -9.042163608169031e-5,
        ];
        for ((i, j), energy) in [(2, 5), (2, 8), (5, 8)]
            .into_iter()
            .zip(intermolecular_a2a2_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_self_group() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 3, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 0, 1);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 0, &system.groups()[0]),
            expected
        );

        let expected = nonbonded.particle_with_particle(&system, 3, 4)
            + nonbonded.particle_with_particle(&system, 4, 5);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 4, &system.groups()[1]),
            expected
        )
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_group() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 1, 3)
            + nonbonded.particle_with_particle(&system, 1, 4)
            + nonbonded.particle_with_particle(&system, 1, 5);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 1, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 0, &system.groups()[2]),
            expected
        );

        let expected = nonbonded.particle_with_particle(&system, 5, 0)
            + nonbonded.particle_with_particle(&system, 5, 1);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 5, &system.groups()[0]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_other_groups() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_group(&system, 0, &system.groups()[1])
            + nonbonded.particle_with_group(&system, 0, &system.groups()[2]);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_other_groups(&system, 0, &system.groups()[0]),
            expected
        );

        let expected = nonbonded.particle_with_group(&system, 3, &system.groups()[0])
            + nonbonded.particle_with_group(&system, 3, &system.groups()[2]);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_other_groups(&system, 3, &system.groups()[1]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_all() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 1, 0)
            + nonbonded.particle_with_particle(&system, 1, 3)
            + nonbonded.particle_with_particle(&system, 1, 4)
            + nonbonded.particle_with_particle(&system, 1, 5);

        assert_approx_eq!(
            f64,
            nonbonded.particle_with_all(&system, 1, &system.groups()[0]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_group() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_group(&system, 0, &system.groups()[1])
            + nonbonded.particle_with_group(&system, 1, &system.groups()[1]);

        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1]),
            expected
        );
        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[1], &system.groups()[0]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[2]),
            expected
        );
        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[2], &system.groups()[0]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_itself() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 0, 1);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_itself(&system, &system.groups()[0]),
            expected
        );

        let expected = nonbonded.particle_with_particle(&system, 3, 4)
            + nonbonded.particle_with_particle(&system, 3, 5)
            + nonbonded.particle_with_particle(&system, 4, 5);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_itself(&system, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_itself(&system, &system.groups()[2]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_other_groups() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected =
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1]);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_other_groups(&system, &system.groups()[0]),
            expected
        );
        assert_approx_eq!(
            f64,
            nonbonded.group_with_other_groups(&system, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_other_groups(&system, &system.groups()[2]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_all() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected =
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1])
                + nonbonded.group_with_itself(&system, &system.groups()[0]);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_all(&system, &system.groups()[0]),
            expected
        );

        let expected =
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1])
                + nonbonded.group_with_itself(&system, &system.groups()[1]);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_all(&system, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_all(&system, &system.groups()[2]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_total_nonbonded() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let interactions = [
            (0, 1),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ];

        let expected = interactions
            .into_iter()
            .map(|(i, j)| nonbonded.particle_with_particle(&system, i, j))
            .sum();
        assert_approx_eq!(f64, nonbonded.total_nonbonded(&system), expected);
    }

    #[test]
    fn test_nonbonded_matrix_energy() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        // no change
        let change = Change::None;
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), 0.0);

        // change everything
        let change = Change::Everything;
        let expected = nonbonded.total_nonbonded(&system);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change volume
        let change = Change::Volume(
            crate::cell::VolumeScalePolicy::Isotropic,
            NewOld {
                old: 104.0,
                new: 108.0,
            },
        );
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // multiple groups with no change
        let change = Change::Groups(vec![(0, GroupChange::None), (1, GroupChange::None)]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), 0.0);

        // change single rigid group
        let change = Change::SingleGroup(1, GroupChange::RigidBody);
        let expected = nonbonded.group_with_other_groups(&system, &system.groups()[1]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change multiple rigid groups
        let change = Change::Groups(vec![
            (0, GroupChange::RigidBody),
            (1, GroupChange::RigidBody),
        ]);
        let expected = nonbonded.group_with_other_groups(&system, &system.groups()[0])
            + nonbonded.group_with_other_groups(&system, &system.groups()[1]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change several particles within a single group
        let change = Change::SingleGroup(1, GroupChange::PartialUpdate(vec![0, 1]));
        let expected = nonbonded.particle_with_all(&system, 3, &system.groups()[1])
            + nonbonded.particle_with_all(&system, 4, &system.groups()[1]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change several particles in multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![1])),
            (1, GroupChange::PartialUpdate(vec![0, 1])),
        ]);
        let expected = nonbonded.particle_with_all(&system, 3, &system.groups()[1])
            + nonbonded.particle_with_all(&system, 4, &system.groups()[1])
            + nonbonded.particle_with_all(&system, 1, &system.groups()[0]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change several particles in multiple groups, some of which are inactive
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![1, 2])),
            (1, GroupChange::PartialUpdate(vec![0, 1])),
            (2, GroupChange::PartialUpdate(vec![0])),
        ]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);
    }

    // ====== NonbondedMatrixSplined tests ======

    /// Get splined nonbonded matrix for testing.
    fn get_test_splined_matrix() -> (AosPlatform, NonbondedMatrix, NonbondedMatrixSplined) {
        let (system, nonbonded) = get_test_matrix();
        let cutoff = 15.0; // Use a cutoff that covers all test distances
        let splined = NonbondedMatrixSplined::from_nonbonded(&nonbonded, cutoff, None);
        (system, nonbonded, splined)
    }

    #[test]
    fn test_nonbonded_matrix_splined_new() {
        let (_, nonbonded, splined) = get_test_splined_matrix();

        // Check that the splined matrix has the same dimensions as the original
        assert_eq!(
            splined.get_potentials().raw_dim(),
            nonbonded.get_potentials().raw_dim()
        );
    }

    #[test]
    fn test_nonbonded_matrix_splined_particle_particle() {
        let (system, nonbonded, splined) = get_test_splined_matrix();

        // Use tolerance since splines are approximations
        let relative_tolerance = 2e-3; // 0.2% relative error for larger values
        let absolute_tolerance = 1e-5; // Absolute tolerance for very small values

        // Test some representative pairs
        let test_pairs = [(0, 1), (0, 3), (1, 4), (3, 4), (0, 5), (1, 5)];

        for (i, j) in test_pairs {
            let analytical = nonbonded.particle_with_particle(&system, i, j);
            let splined_energy = splined.particle_with_particle(&system, i, j);
            let abs_diff = (analytical - splined_energy).abs();

            // For very small energies, check absolute difference
            // For larger energies, check relative difference
            if analytical.abs() < 1e-4 {
                assert!(
                    abs_diff < absolute_tolerance,
                    "Pair ({}, {}): analytical={}, splined={}, abs_diff={}",
                    i,
                    j,
                    analytical,
                    splined_energy,
                    abs_diff
                );
            } else {
                let relative_error = abs_diff / analytical.abs();
                assert!(
                    relative_error < relative_tolerance,
                    "Pair ({}, {}): analytical={}, splined={}, relative_error={}",
                    i,
                    j,
                    analytical,
                    splined_energy,
                    relative_error
                );
            }
        }
    }

    #[test]
    fn test_nonbonded_matrix_splined_total_nonbonded() {
        let (mut system, nonbonded, splined) = get_test_splined_matrix();

        // Deactivate some particles like in the original test
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let analytical_total = nonbonded.total_nonbonded(&system);
        let splined_total = splined.total_nonbonded(&system);

        // Allow for some tolerance due to spline approximation
        let tolerance = 1e-3;
        let relative_error = ((analytical_total - splined_total) / analytical_total).abs();

        assert!(
            relative_error < tolerance,
            "Total energy: analytical={}, splined={}, relative_error={}",
            analytical_total,
            splined_total,
            relative_error
        );
    }

    #[test]
    fn test_nonbonded_matrix_splined_energy_changes() {
        let (mut system, nonbonded, splined) = get_test_splined_matrix();

        // Deactivate some particles
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let tolerance = 1e-3;

        // Test Change::None
        let change = Change::None;
        assert_approx_eq!(f64, splined.energy(&system, &change), 0.0);

        // Test Change::Everything
        let change = Change::Everything;
        let analytical = nonbonded.energy(&system, &change);
        let splined_energy = splined.energy(&system, &change);
        let relative_error = ((analytical - splined_energy) / analytical).abs();
        assert!(
            relative_error < tolerance,
            "Change::Everything: analytical={}, splined={}, relative_error={}",
            analytical,
            splined_energy,
            relative_error
        );

        // Test single rigid group change
        let change = Change::SingleGroup(1, GroupChange::RigidBody);
        let analytical = nonbonded.energy(&system, &change);
        let splined_energy = splined.energy(&system, &change);
        let relative_error = ((analytical - splined_energy) / analytical).abs();
        assert!(
            relative_error < tolerance,
            "SingleGroup RigidBody: analytical={}, splined={}, relative_error={}",
            analytical,
            splined_energy,
            relative_error
        );

        // Test partial update
        let change = Change::SingleGroup(1, GroupChange::PartialUpdate(vec![0, 1]));
        let analytical = nonbonded.energy(&system, &change);
        let splined_energy = splined.energy(&system, &change);
        let relative_error = ((analytical - splined_energy) / analytical).abs();
        assert!(
            relative_error < tolerance,
            "SingleGroup PartialUpdate: analytical={}, splined={}, relative_error={}",
            analytical,
            splined_energy,
            relative_error
        );
    }

    #[test]
    fn test_min_group_distance() {
        use crate::cell::Cuboid;
        let cell = Cuboid::cubic(20.0);

        let mut g1 = Group::new(0, 0, 0..3);
        g1.set_mass_center(crate::Point::new(0.0, 0.0, 0.0));
        g1.set_bounding_radius(1.0);

        let mut g2 = Group::new(1, 0, 3..6);
        g2.set_mass_center(crate::Point::new(5.0, 0.0, 0.0));
        g2.set_bounding_radius(1.5);

        // COM distance = 5.0, sum of radii = 2.5, min distance = 2.5
        let d = min_group_distance(&g1, &g2, &cell).unwrap();
        assert_approx_eq!(f64, d, 2.5);

        // Overlapping spheres → 0
        let mut g3 = Group::new(2, 0, 6..9);
        g3.set_mass_center(crate::Point::new(1.0, 0.0, 0.0));
        g3.set_bounding_radius(2.0);
        let d = min_group_distance(&g1, &g3, &cell).unwrap();
        assert_approx_eq!(f64, d, 0.0);

        // PBC wrapping: groups near opposite edges
        let mut g4 = Group::new(3, 0, 9..12);
        g4.set_mass_center(crate::Point::new(9.0, 0.0, 0.0));
        g4.set_bounding_radius(0.5);
        // COM distance via PBC = 20 - 9 = 11, but PBC gives min image = 9
        // Actually: g1 at 0, g4 at 9 → distance via PBC in [-10,10] box: 9.0
        let d = min_group_distance(&g1, &g4, &cell).unwrap();
        assert_approx_eq!(f64, d, 9.0 - 1.0 - 0.5);

        // Missing bounding radius → None
        let g5 = Group::new(4, 0, 12..15);
        assert!(min_group_distance(&g1, &g5, &cell).is_none());
    }

    #[test]
    fn test_nonbonded_matrix_splined_with_config() {
        let (system, nonbonded, _) = get_test_splined_matrix();

        // Test with high accuracy config
        let config = SplineConfig::high_accuracy();
        let splined_high = NonbondedMatrixSplined::from_nonbonded(&nonbonded, 15.0, Some(config));

        // Test with fast config
        let config = SplineConfig::fast();
        let splined_fast = NonbondedMatrixSplined::from_nonbonded(&nonbonded, 15.0, Some(config));

        // Both should produce reasonable energies
        let energy_high = splined_high.total_nonbonded(&system);
        let energy_fast = splined_fast.total_nonbonded(&system);
        let analytical = nonbonded.total_nonbonded(&system);

        // High accuracy should be closer to analytical
        let error_high = ((analytical - energy_high) / analytical).abs();
        let error_fast = ((analytical - energy_fast) / analytical).abs();

        // Both should be within reasonable bounds
        assert!(
            error_high < 1e-3,
            "High accuracy error too large: {}",
            error_high
        );
        assert!(
            error_fast < 1e-2,
            "Fast config error too large: {}",
            error_fast
        );

        // High accuracy should generally be better (or at least not significantly worse)
        // Note: this isn't always guaranteed but should hold for most cases
        assert!(
            error_high <= error_fast * 1.1,
            "High accuracy ({}) should be better than fast ({})",
            error_high,
            error_fast
        );
    }
}
