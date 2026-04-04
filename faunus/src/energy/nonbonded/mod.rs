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

pub(crate) mod cache;
#[cfg(test)]
mod tests;

use super::pairpot::PairPot;
use cache::GroupEnergyCache;
use interatomic::twobody::{IsotropicTwobodyEnergy, SplineConfig, SplinedPotential};
use ndarray::Array2;
use std::path::Path;
use std::sync::RwLock;

use crate::{
    cell::{PbcParams, SimulationCell},
    energy::{builder::PairPotentialBuilder, EnergyTerm},
    topology::Topology,
    Change, Context, Group, GroupChange,
};

use super::{builder::HamiltonianBuilder, exclusions::ExclusionMatrix, EnergyChange};

/// Sort a molecule-type pair into canonical `[min, max]` order for symmetric lookup.
#[inline(always)]
const fn canonical_mol_pair(a: usize, b: usize) -> [usize; 2] {
    if a <= b {
        [a, b]
    } else {
        [b, a]
    }
}

/// Energy term for computing nonbonded interactions
/// using a matrix of `IsotropicTwobodyEnergy` trait objects.
///
/// The type parameter `P` determines the potential type:
/// - `ArcPotential` for dynamic dispatch (default)
/// - `SplinedPotential` for pre-tabulated spline evaluation
///
/// Entire molecule-type pairs can be excluded via `exclude_molecule_pair`,
/// skipping all inter-group interactions between those molecule kinds.
/// This is used automatically when [`super::tabulated::TabulatedEnergy`] handles the same pairs.
#[derive(Debug)]
pub struct NonbondedMatrix<P = PairPot> {
    /// Matrix of pair potentials based on atom type ids.
    pub(super) potentials: Array2<P>,
    /// Matrix of excluded interactions.
    pub(super) exclusions: ExclusionMatrix,
    /// Pairwise inter-group energy cache for O(1) old-energy lookup in MC moves.
    cache: RwLock<Option<GroupEnergyCache>>,
    /// Global interaction cutoff for bounding-sphere culling (None = no culling).
    cutoff: Option<f64>,
    /// Enable bounding-sphere culling of distant group pairs.
    use_bounding_spheres: bool,
    /// Molecule-type pairs excluded from nonbonded evaluation (e.g. handled by TabulatedEnergy).
    /// Each entry is a sorted `[mol_a, mol_b]` pair of molecule kind indices.
    molecule_pair_exclusions: Vec<[usize; 2]>,
}

impl<P: Clone> Clone for NonbondedMatrix<P> {
    fn clone(&self) -> Self {
        Self {
            potentials: self.potentials.clone(),
            exclusions: self.exclusions.clone(),
            cache: RwLock::new(self.cache.read().unwrap().clone()),
            cutoff: self.cutoff,
            use_bounding_spheres: self.use_bounding_spheres,
            molecule_pair_exclusions: self.molecule_pair_exclusions.clone(),
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

// ─── Reference (scalar) implementations ──────────────────────────────────────
//
// These methods use the Context trait for per-pair distance/atom-kind lookups.
// They serve as ground-truth reference for tests and for `indices_with_indices`
// (called from EnergyTerm). The production hot path uses the SoA methods below.

impl<P: IsotropicTwobodyEnergy> NonbondedMatrix<P> {
    /// Energy between two particles given by absolute indices.
    #[inline(always)]
    pub(super) fn particle_with_particle(&self, context: &impl Context, i: usize, j: usize) -> f64 {
        let distance_squared = context.get_distance_squared(i, j);
        self.exclusions.get((i, j)) as f64
            * self
                .potentials
                .get((context.atom_kind(i), context.atom_kind(j)))
                .expect("Atom kinds should exist in the nonbonded matrix.")
                .isotropic_twobody_energy(distance_squared)
    }

    /// Energy of particle `i` with all other particles in `group` (self-avoiding).
    #[inline(always)]
    #[allow(dead_code)]
    fn particle_with_group(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        group
            .iter_active()
            .filter(|j| *j != i)
            .map(|j| self.particle_with_particle(context, i, j))
            .sum()
    }

    /// Energy of particle `i` with all other groups.
    #[inline(always)]
    #[allow(dead_code)]
    fn particle_with_other_groups(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        context
            .groups()
            .iter()
            .filter(|group_j| group_j.index() != group.index())
            .map(|group_j| self.particle_with_group(context, i, group_j))
            .sum()
    }

    /// Energy of particle `i` with all other particles.
    #[inline(always)]
    #[allow(dead_code)]
    fn particle_with_all(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        self.particle_with_other_groups(context, i, group)
            + self.particle_with_group(context, i, group)
    }

    /// Energy between two groups (no self-avoidance check — groups must differ).
    #[inline(always)]
    #[allow(dead_code)]
    fn group_with_group(&self, context: &impl Context, group1: &Group, group2: &Group) -> f64 {
        group1
            .iter_active()
            .flat_map(|i| {
                group2
                    .iter_active()
                    .map(move |j| self.particle_with_particle(context, i, j))
            })
            .sum()
    }

    /// Intra-group energy (unique pairs only).
    #[inline(always)]
    #[allow(dead_code)]
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

    /// Energy of a group with all other groups.
    #[inline(always)]
    #[allow(dead_code)]
    fn group_with_other_groups(&self, context: &impl Context, group: &Group) -> f64 {
        group
            .iter_active()
            .map(|i| self.particle_with_other_groups(context, i, group))
            .sum()
    }

    /// Energy of a group with all particles (inter + intra).
    #[inline(always)]
    #[allow(dead_code)]
    fn group_with_all(&self, context: &impl Context, group: &Group) -> f64 {
        self.group_with_other_groups(context, group) + self.group_with_itself(context, group)
    }

    /// Energy between two sets of particle indices with automatic deduplication.
    ///
    /// When both slices are identical, only unique pairs (i < j) are summed.
    /// Otherwise, all cross-pairs are included.
    pub(super) fn indices_with_indices(
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

    /// Total nonbonded energy (reference implementation for tests).
    #[allow(dead_code)]
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

// ─── SoA (Structure-of-Arrays) hot path ──────────────────────────────────────

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

/// Build SoA slices from a Context, extracting all arrays.
fn soa_from_context<'a>(context: &'a impl Context) -> SoaSlices<'a, impl SimulationCell + 'a> {
    let (x, y, z) = context.positions_soa();
    let atom_kinds = context.atom_kinds_u32();
    let cell = context.cell();
    let pbc = context
        .pbc_params()
        .or_else(|| PbcParams::try_from_cell(cell));
    SoaSlices {
        x,
        y,
        z,
        atom_kinds,
        pbc,
        cell,
        cell_list: context.cell_list(),
    }
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

    /// Minimum image displacement vector from `(xi, yi, zi)` to particle `j`.
    ///
    /// # Safety
    /// `j` must be in bounds for `self.x`, `self.y`, `self.z`.
    #[inline(always)]
    unsafe fn distance_vector_to(&self, xi: f64, yi: f64, zi: f64, j: usize) -> [f64; 3] {
        if let Some(pbc) = &self.pbc {
            pbc.distance_vector(
                xi,
                yi,
                zi,
                *self.x.get_unchecked(j),
                *self.y.get_unchecked(j),
                *self.z.get_unchecked(j),
            )
        } else {
            // cell.distance(pi, pj) = pi - pj; negate to get i→j like PBC path
            let pi = crate::Point::new(xi, yi, zi);
            let pj = crate::Point::new(
                *self.x.get_unchecked(j),
                *self.y.get_unchecked(j),
                *self.z.get_unchecked(j),
            );
            let d = self.cell.distance(&pj, &pi);
            [d.x, d.y, d.z]
        }
    }
}

/// Extract `(xi, yi, zi, kind_i)` for particle `i` from SoA arrays.
///
/// # Safety
/// `i` must be in bounds for all SoA arrays.
#[inline(always)]
unsafe fn read_particle(
    soa: &SoaSlices<'_, impl SimulationCell>,
    i: usize,
) -> (f64, f64, f64, usize) {
    (
        *soa.x.get_unchecked(i),
        *soa.y.get_unchecked(i),
        *soa.z.get_unchecked(i),
        *soa.atom_kinds.get_unchecked(i) as usize,
    )
}

impl<P: IsotropicTwobodyEnergy> EnergyChange for NonbondedMatrix<P> {
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        // O(1) cache hit for rigid body moves (the MC hot path)
        if let Change::SingleGroup(gi, GroupChange::RigidBody) = change {
            if let Some(ref cache) = *self.cache.read().unwrap() {
                debug_assert_eq!(cache.n_groups, context.groups().len());
                return cache.group_energies[*gi];
            }
        }

        let soa = soa_from_context(context);
        let groups = context.groups();
        match change {
            Change::Everything | Change::Volume(_, _) => self.total_nonbonded_soa(&soa, groups),
            Change::SingleGroup(gi, GroupChange::RigidBody) => {
                // Cache miss — lazy-initialize all pairwise energies
                self.initialize_cache_soa(&soa, groups);
                self.cache.read().unwrap().as_ref().unwrap().group_energies[*gi]
            }
            Change::SingleGroup(group_index, group_change) => {
                self.single_group_change_soa(&soa, groups, *group_index, group_change)
            }
            Change::Groups(vec) => self.multi_group_change_soa(&soa, groups, vec),
            Change::None => 0.0,
        }
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

    /// Get the list of excluded molecule-type pairs.
    #[must_use]
    pub(crate) fn molecule_pair_exclusions(&self) -> &[[usize; 2]] {
        &self.molecule_pair_exclusions
    }

    /// Whether any molecule-type pairs are excluded.
    #[inline(always)]
    pub(crate) fn has_molecule_pair_exclusions(&self) -> bool {
        !self.molecule_pair_exclusions.is_empty()
    }

    /// Exclude a molecule-type pair from nonbonded evaluation.
    ///
    /// All inter-group interactions between groups of these two molecule kinds
    /// will be skipped. Use when the pair is handled by another energy term
    /// (e.g. [`TabulatedEnergy`]).
    pub(crate) fn exclude_molecule_pair(&mut self, mol_a: usize, mol_b: usize) {
        let pair = canonical_mol_pair(mol_a, mol_b);
        if !self.molecule_pair_exclusions.contains(&pair) {
            self.molecule_pair_exclusions.push(pair);
        }
    }

    /// Check if a molecule-type pair is excluded from nonbonded evaluation.
    #[inline(always)]
    fn is_molecule_pair_excluded(&self, mol_a: usize, mol_b: usize) -> bool {
        // Fast path: most simulations have no molecule-pair exclusions
        if !self.has_molecule_pair_exclusions() {
            return false;
        }
        self.molecule_pair_exclusions
            .contains(&canonical_mol_pair(mol_a, mol_b))
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
        builder.validate(topology.atomkinds())?;
        Self::new(&builder.pairpot_builder.unwrap(), topology, medium)
    }

    /// Create a new NonbondedMatrix using enum-dispatched [`PairPot`] potentials.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(
        pairpot_builder: &PairPotentialBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let atoms = topology.atomkinds();
        let n_atom_types = atoms.len();

        let mut potentials: Array2<PairPot> =
            Array2::from_elem((n_atom_types, n_atom_types), PairPot::default());

        for i in 0..n_atom_types {
            for j in 0..n_atom_types {
                potentials[(i, j)] =
                    pairpot_builder.get_pair_pot(&atoms[i], &atoms[j], medium.clone())?;
            }
        }

        let exclusions = ExclusionMatrix::from_topology(topology);

        Ok(Self {
            potentials,
            exclusions,
            cache: RwLock::new(None),
            cutoff: None,
            use_bounding_spheres: true,
            molecule_pair_exclusions: Vec::new(),
        })
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
    /// let nonbonded = NonbondedMatrix::new(&builder, &topology, medium)?;
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

        // Warn if all potentials are negligible at half the cutoff, indicating
        // grid points are wasted on a long flat tail (risk of spline ringing).
        let half_rsq = (cutoff * 0.5) * (cutoff * 0.5);
        let max_energy_at_half = source
            .iter()
            .map(|p| p.isotropic_twobody_energy(half_rsq).abs())
            .fold(0.0f64, f64::max);
        if max_energy_at_half < 1e-6 {
            log::warn!(
                "All pair potentials are < 1e-6 kJ/mol at r = {:.1} Å (half the spline cutoff). \
                 Consider reducing the spline cutoff to avoid wasting grid resolution on a flat tail.",
                cutoff * 0.5
            );
        }

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
            molecule_pair_exclusions: nonbonded.molecule_pair_exclusions.clone(),
        }
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

// ─── SoA energy and force evaluation ─────────────────────────────────────────

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
        let (xi, yi, zi, kind_i) = unsafe { read_particle(soa, i) };
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
                        !self.is_molecule_pair_excluded(group_i.molecule(), group_j.molecule())
                            && !self.groups_beyond_cutoff(group_i, group_j, soa.pbc.as_ref())
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
            .filter(|gj| !self.is_molecule_pair_excluded(own_group.molecule(), gj.molecule()))
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
            GroupChange::RigidBody | GroupChange::ResizeExcludeIntra(_) => {
                self.inter_group_energy_all_soa(soa, groups, group)
            }
            GroupChange::Resize(_) | GroupChange::UpdateIdentity(_) => {
                self.inter_group_energy_all_soa(soa, groups, group)
                    + self.group_with_itself_soa(soa, group)
            }
            // ResizePartial reuses the PartialUpdate path for O(N) per affected atom
            GroupChange::PartialUpdate(indices) | GroupChange::ResizePartial(_, indices) => indices
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

    /// Energy for multiple simultaneous group changes (SoA), avoiding double-counted cross-terms.
    fn multi_group_change_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        groups: &[Group],
        changes: &[(usize, GroupChange)],
    ) -> f64 {
        // Typically 2–4 entries; stack array avoids heap allocation on the hot path
        let mut changed_indices = [0usize; 8];
        let mut n_changed = 0;
        for (gi, gc) in changes {
            if gc.is_whole_group() {
                debug_assert!(
                    n_changed < changed_indices.len(),
                    "too many whole-group changes"
                );
                changed_indices[n_changed] = *gi;
                n_changed += 1;
            }
        }
        let changed_indices = &changed_indices[..n_changed];

        let mut energy = 0.0;
        for (gi, gc) in changes {
            let group = &groups[*gi];
            if gc.is_whole_group() {
                energy += groups
                    .iter()
                    .filter(|gj| {
                        gj.index() != group.index() && !changed_indices.contains(&gj.index())
                    })
                    .map(|gj| self.group_pair_energy_soa(soa, group, gj))
                    .sum::<f64>();
                if gc.internal_change() {
                    energy += self.group_with_itself_soa(soa, group);
                }
            } else {
                energy += self.single_group_change_soa(soa, groups, *gi, gc);
            }
        }
        // Cross-terms between changed groups, counted once
        for (i, (gi, _)) in changes.iter().enumerate() {
            if !changed_indices.contains(gi) {
                continue;
            }
            for (gj, _) in &changes[i + 1..] {
                if !changed_indices.contains(gj) {
                    continue;
                }
                energy += self.group_pair_energy_soa(soa, &groups[*gi], &groups[*gj]);
            }
        }
        energy
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
        // Cell list iterates spatial neighbors without group ownership info,
        // so it cannot skip excluded molecule-type pairs. Fall back to
        // group-based iteration where the exclusion check is applied per pair.
        if !self.has_molecule_pair_exclusions() {
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
        // Cell list lacks group ownership, so cannot enforce molecule-pair exclusions
        if !self.has_molecule_pair_exclusions() {
            if let Some(cl) = soa.cell_list {
                return self.particle_energy_soa(
                    soa,
                    abs_i,
                    cl.neighbors(abs_i).filter(move |&j| j != abs_i),
                );
            }
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
        if self.is_molecule_pair_excluded(group_i.molecule(), group_j.molecule()) {
            return 0.0;
        }
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

        *self.cache.write().unwrap() = Some(GroupEnergyCache::new(pairwise, group_energies, n));
    }

    /// Compute per-atom nonbonded forces for all active particles using SoA arrays.
    ///
    /// For each unique pair (i, j), evaluates `-dU/d(r²)` from the pair potential
    /// and accumulates the force vector on both atoms (Newton's third law).
    fn accumulate_forces_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        groups: &[Group],
        forces: &mut [[f64; 3]],
    ) {
        for (gi, group_i) in groups.iter().enumerate() {
            // Inter-group pairs (gi < gj)
            for group_j in groups.iter().skip(gi + 1) {
                if self.is_molecule_pair_excluded(group_i.molecule(), group_j.molecule()) {
                    continue;
                }
                if self.groups_beyond_cutoff(group_i, group_j, soa.pbc.as_ref()) {
                    continue;
                }
                for i in group_i.iter_active() {
                    self.accumulate_particle_forces_soa(soa, i, group_j.iter_active(), forces);
                }
            }
            // Intra-group pairs (i < j)
            let active = group_i.iter_active();
            for i in active.clone() {
                self.accumulate_particle_forces_soa(soa, i, (i + 1)..active.end, forces);
            }
        }
    }

    /// Accumulate forces on atom `i` from interactions with `targets`, applying Newton's third law.
    ///
    /// # Safety invariants (upheld by callers)
    /// - `i` and all `targets` values are in bounds for SoA arrays and `forces`
    #[inline]
    fn accumulate_particle_forces_soa(
        &self,
        soa: &SoaSlices<'_, impl SimulationCell>,
        i: usize,
        targets: impl Iterator<Item = usize>,
        forces: &mut [[f64; 3]],
    ) {
        debug_assert!(i < soa.x.len());
        // SAFETY: i is a valid particle index from an active group range.
        let (xi, yi, zi, kind_i) = unsafe { read_particle(soa, i) };
        let excl_row = self.exclusions.row(i);

        for j in targets {
            debug_assert!(j < soa.x.len());
            unsafe {
                if *excl_row.get_unchecked(j) == 0 {
                    continue;
                }
                let dr = soa.distance_vector_to(xi, yi, zi, j);
                let rsq = dr[0].mul_add(dr[0], dr[1].mul_add(dr[1], dr[2] * dr[2]));
                let kind_j = *soa.atom_kinds.get_unchecked(j) as usize;
                let f_mag = self
                    .potentials
                    .uget((kind_i, kind_j))
                    .isotropic_twobody_force(rsq);
                if f_mag == 0.0 {
                    continue;
                }
                // f_mag = -dU/d(r²); force on i: F_i = -2·f_mag·dr (dr points i→j)
                let scale = -2.0 * f_mag;
                let fx = scale * dr[0];
                let fy = scale * dr[1];
                let fz = scale * dr[2];
                forces[i][0] += fx;
                forces[i][1] += fy;
                forces[i][2] += fz;
                forces[j][0] -= fx;
                forces[j][1] -= fy;
                forces[j][2] -= fz;
            }
        }
    }

    /// Compute all nonbonded forces on active particles.
    ///
    /// Returns a dense array indexed by absolute particle index.
    /// Inactive particles have zero force.
    pub(super) fn forces(&self, context: &impl Context) -> Vec<crate::Point> {
        let n = context
            .groups()
            .iter()
            .map(|g| g.start() + g.capacity())
            .max()
            .unwrap_or(0);
        let mut forces = vec![[0.0f64; 3]; n];

        let soa = soa_from_context(context);
        self.accumulate_forces_soa(&soa, context.groups(), &mut forces);

        forces
            .into_iter()
            .map(|[x, y, z]| crate::Point::new(x, y, z))
            .collect()
    }

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
            // Bypass cell list so all group-pair interactions are fully recomputed
            let mut soa = soa_from_context(context);
            soa.cell_list = None;
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
