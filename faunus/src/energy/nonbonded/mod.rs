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

#[cfg(test)]
mod tests;

use super::group_pair_cache::{row_from_pairs, GroupPairCache, GroupPairs};
use super::pairpot::PairPot;
use super::stateful::{derived_energy, StatefulEnergy};
use crate::ObserveContext;
use interatomic::twobody::{IsotropicTwobodyEnergy, SplineConfig, SplinedPotential};
use std::path::Path;

use crate::{
    cell::{PbcParams, SimulationCell},
    energy::{builder::PairPotentialBuilder, EnergyTerm},
    group::{AbsIndex, MoleculeId},
    topology::Topology,
    Change, Group, GroupChange,
};

use super::{
    builder::HamiltonianBuilder, exclusions::ExclusionMatrix, square_matrix::SquareMatrix,
    EnergyChange,
};

/// Sort a molecule-type pair into canonical `[min, max]` order for symmetric lookup.
#[inline(always)]
const fn canonical_mol_pair(a: MoleculeId, b: MoleculeId) -> [MoleculeId; 2] {
    if a.get() <= b.get() {
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
#[derive(Debug, Clone)]
pub struct NonbondedMatrix<P = PairPot> {
    /// Matrix of pair potentials based on atom type ids.
    pub(super) potentials: SquareMatrix<P>,
    /// Matrix of excluded interactions.
    pub(super) exclusions: ExclusionMatrix,
    /// Pairwise inter-group energy cache for O(1) old-energy lookup in MC moves.
    cache: GroupPairCache,
    /// Global interaction cutoff for bounding-sphere culling (None = no culling).
    cutoff: Option<f64>,
    /// Enable bounding-sphere culling of distant group pairs.
    use_bounding_spheres: bool,
    /// Molecule-type pairs excluded from nonbonded evaluation (e.g. handled by TabulatedEnergy).
    /// Each entry is a sorted `[mol_a, mol_b]` pair of molecule kind indices.
    molecule_pair_exclusions: Vec<[MoleculeId; 2]>,
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
// These methods use the ObserveContext trait for per-pair distance/atom-kind lookups.
// They serve as ground-truth reference for tests and for `indices_with_indices`
// (called from EnergyTerm). The production hot path uses the batch-array methods below.

impl<P: IsotropicTwobodyEnergy> NonbondedMatrix<P> {
    /// Energy between two particles given by absolute indices.
    #[inline(always)]
    pub(super) fn particle_with_particle(
        &self,
        context: &impl ObserveContext,
        i: usize,
        j: usize,
    ) -> f64 {
        let distance_squared = context.get_distance_squared(i, j);
        self.exclusions.get((i, j)) as f64
            * self
                .potentials
                .get((context.atom_kind(i).get(), context.atom_kind(j).get()))
                .expect("Atom kinds should exist in the nonbonded matrix.")
                .isotropic_twobody_energy(distance_squared)
    }

    /// Energy of particle `i` with all other particles in `group` (self-avoiding).
    #[inline(always)]
    #[allow(dead_code)]
    fn particle_with_group_reference(
        &self,
        context: &impl ObserveContext,
        i: usize,
        group: &Group,
    ) -> f64 {
        group
            .iter_active()
            .filter(|j| *j != i)
            .map(|j| self.particle_with_particle(context, i, j))
            .sum()
    }

    /// Energy of particle `i` with all other groups.
    #[inline(always)]
    #[allow(dead_code)]
    fn particle_with_other_groups_reference(
        &self,
        context: &impl ObserveContext,
        i: usize,
        group: &Group,
    ) -> f64 {
        context
            .groups()
            .iter()
            .filter(|group_j| group_j.index() != group.index())
            .map(|group_j| self.particle_with_group_reference(context, i, group_j))
            .sum()
    }

    /// Energy of particle `i` with all other particles.
    #[inline(always)]
    #[allow(dead_code)]
    fn particle_with_all_reference(
        &self,
        context: &impl ObserveContext,
        i: usize,
        group: &Group,
    ) -> f64 {
        self.particle_with_other_groups_reference(context, i, group)
            + self.particle_with_group_reference(context, i, group)
    }

    /// Energy between two groups (no self-avoidance check — groups must differ).
    #[inline(always)]
    #[allow(dead_code)]
    fn group_with_group_reference(
        &self,
        context: &impl ObserveContext,
        group1: &Group,
        group2: &Group,
    ) -> f64 {
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
    fn group_with_itself_reference(&self, context: &impl ObserveContext, group: &Group) -> f64 {
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
    fn group_with_other_groups_reference(
        &self,
        context: &impl ObserveContext,
        group: &Group,
    ) -> f64 {
        group
            .iter_active()
            .map(|i| self.particle_with_other_groups_reference(context, i, group))
            .sum()
    }

    /// Energy of a group with all particles (inter + intra).
    #[inline(always)]
    #[allow(dead_code)]
    fn group_with_all_reference(&self, context: &impl ObserveContext, group: &Group) -> f64 {
        self.group_with_other_groups_reference(context, group)
            + self.group_with_itself_reference(context, group)
    }

    /// Energy between two sets of particle indices with automatic deduplication.
    ///
    /// When both slices are identical, only unique pairs (i < j) are summed.
    /// Otherwise, all cross-pairs are included.
    pub(super) fn indices_with_indices(
        &self,
        context: &impl ObserveContext,
        indices1: &[AbsIndex],
        indices2: &[AbsIndex],
    ) -> f64 {
        let same = indices1 == indices2;
        indices1
            .iter()
            .enumerate()
            .map(|(idx, &i)| {
                let others = if same { &indices2[idx + 1..] } else { indices2 };
                others
                    .iter()
                    .map(|&j| self.particle_with_particle(context, i.get(), j.get()))
                    .sum::<f64>()
            })
            .sum()
    }

    /// Total nonbonded energy (reference implementation for tests).
    #[allow(dead_code)]
    fn total_nonbonded_reference(&self, context: &impl ObserveContext) -> f64 {
        context
            .groups()
            .iter()
            .enumerate()
            .flat_map(|(i, group_i)| {
                context
                    .groups()
                    .iter()
                    .skip(i + 1)
                    .map(move |group_j| self.group_with_group_reference(context, group_i, group_j))
                    .chain(std::iter::once(
                        self.group_with_itself_reference(context, group_i),
                    ))
            })
            .sum()
    }
}

// ─── Batch-array hot path ────────────────────────────────────────────────────

/// Borrowed particle arrays for batch nonbonded energy evaluation.
///
/// Holds `cell` for HexagonalPrism fallback; all other cell types
/// use precomputed `pbc` params for inline branchless distance.
struct ParticleArrays<'a, C: SimulationCell> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    atom_kinds: &'a [u32],
    cell: &'a C,
    pbc: Option<PbcParams>,
    cell_list: Option<&'a crate::celllist::CellList>,
}

/// Build borrowed particle arrays from an observable context.
fn particle_arrays_from_context<'a>(
    context: &'a impl ObserveContext,
) -> ParticleArrays<'a, impl SimulationCell + 'a> {
    let (x, y, z) = context.positions();
    let atom_kinds = context.atom_kinds_u32();
    let cell = context.cell();
    let pbc = context
        .pbc_params()
        .or_else(|| PbcParams::try_from_cell(cell));
    ParticleArrays {
        x,
        y,
        z,
        atom_kinds,
        pbc,
        cell,
        cell_list: context.cell_list(),
    }
}

struct CachedPairs<'a, P, S: SimulationCell, C> {
    matrix: &'a NonbondedMatrix<P>,
    arrays: ParticleArrays<'a, S>,
    context: &'a C,
}

impl<P: IsotropicTwobodyEnergy, S: SimulationCell, C: ObserveContext> GroupPairs
    for CachedPairs<'_, P, S, C>
{
    fn n_groups(&self) -> usize {
        self.context.groups().len()
    }

    fn pair(&self, i: usize, j: usize) -> f64 {
        let groups = self.context.groups();
        self.matrix
            .group_pair_energy(&self.arrays, &groups[i], &groups[j])
    }

    /// A pair that bounding spheres can't cull would cost O(|m|·|j|) directly. A cell-list sweep costs
    /// O(|m|·neighbours), and pairs it doesn't visit are beyond the cutoff. Culling beats a sweep when
    /// it applies to both groups.
    fn row(&self, m: usize, first: usize, out: &mut [f64]) {
        let Some(cell_list) = self.arrays.cell_list else {
            return row_from_pairs(self, m, first, out);
        };
        let groups = self.context.groups();
        let own = &groups[m];
        let own_cullable = self.matrix.cullable(own);
        let mut swept = Vec::new();
        for j in (first..groups.len()).filter(|&j| j != m) {
            if own_cullable && self.matrix.cullable(&groups[j]) {
                out[j] = self.pair(m, j);
            } else {
                out[j] = 0.0;
                if !self
                    .matrix
                    .is_molecule_pair_excluded(own.molecule(), groups[j].molecule())
                {
                    swept.push((groups[j].iter_active(), j));
                }
            }
        }
        // A cullable group has few partners to sweep, but a sweep walks every neighbour regardless.
        // Direct evaluation wins when it visits no more pairs.
        if own_cullable && !swept.is_empty() {
            let direct = own.iter_active().len()
                * swept.iter().map(|(active, _)| active.len()).sum::<usize>();
            let walked: usize = own
                .iter_active()
                .map(|i| cell_list.neighbors(i).count())
                .sum();
            if direct <= walked {
                for (_, j) in swept.drain(..) {
                    out[j] = self.pair(m, j);
                }
            }
        }
        self.sweep(cell_list, m, &swept, out);
    }
}

impl<P: IsotropicTwobodyEnergy, S: SimulationCell, C: ObserveContext> CachedPairs<'_, P, S, C> {
    /// Add to `out[g]` the pairs of group `m` with each `(active range, g)` in `partners`, ascending.
    ///
    /// The ranges are copied out of `Group`, a large struct, because this search runs per neighbour.
    fn sweep(
        &self,
        cell_list: &crate::celllist::CellList,
        m: usize,
        partners: &[(std::ops::Range<usize>, usize)],
        out: &mut [f64],
    ) {
        if partners.is_empty() {
            return;
        }
        let own = &self.context.groups()[m];
        for i in own.iter_active() {
            let probe = self.matrix.probe(&self.arrays, i);
            // An atomic group's neighbours are largely its own, so skip the search for those.
            for j in cell_list.neighbors(i).filter(|&j| !own.contains(j)) {
                // Groups occupy ascending particle ranges, so the owner can only be the last partner
                // starting at or before j. Other neighbours belong to no partner.
                let candidates = partners.partition_point(|(active, _)| active.start <= j);
                let owner = partners[..candidates].last();
                if let Some((_, g)) = owner.filter(|(active, _)| active.contains(&j)) {
                    out[*g] += probe.energy_with(j);
                }
            }
        }
    }
}

/// One particle of a pair loop, read once rather than per partner.
///
/// A sweep picks its partners one at a time. Going through [`NonbondedMatrix::particle_energy`] for
/// that adds a call site that can cost it its inlining, and with it every loop's speed.
struct PairProbe<'a, P, S: SimulationCell> {
    potentials: &'a SquareMatrix<P>,
    arrays: &'a ParticleArrays<'a, S>,
    x: f64,
    y: f64,
    z: f64,
    kind: usize,
    exclusions: &'a [u8],
}

impl<P: IsotropicTwobodyEnergy, S: SimulationCell> PairProbe<'_, P, S> {
    #[inline(always)]
    fn energy_with(&self, j: usize) -> f64 {
        debug_assert!(j < self.arrays.x.len());
        // SAFETY: j is a valid particle index from an active group range;
        // atom kinds are topology-derived indices into the potentials matrix.
        // No j==i guard needed: callers use disjoint ranges, and the exclusion
        // matrix diagonal is 0 (self-exclusion) so it would be skipped anyway.
        unsafe {
            if *self.exclusions.get_unchecked(j) == 0 {
                return 0.0;
            }
            let rsq = self.arrays.distance_squared_to(self.x, self.y, self.z, j);
            let kind_j = *self.arrays.atom_kinds.get_unchecked(j) as usize;
            self.potentials
                .uget((self.kind, kind_j))
                .isotropic_twobody_energy(rsq)
        }
    }
}

impl<'a, C: SimulationCell> ParticleArrays<'a, C> {
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

/// Extract `(xi, yi, zi, kind_i)` for particle `i` from particle arrays.
///
/// # Safety
/// `i` must be in bounds for all particle arrays.
#[inline(always)]
unsafe fn read_particle(
    arrays: &ParticleArrays<'_, impl SimulationCell>,
    i: usize,
) -> (f64, f64, f64, usize) {
    (
        *arrays.x.get_unchecked(i),
        *arrays.y.get_unchecked(i),
        *arrays.z.get_unchecked(i),
        *arrays.atom_kinds.get_unchecked(i) as usize,
    )
}

impl<P: IsotropicTwobodyEnergy> EnergyChange for NonbondedMatrix<P> {
    fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        derived_energy(self, context, change)
    }
}

impl<P: IsotropicTwobodyEnergy> StatefulEnergy for NonbondedMatrix<P> {
    fn total_energy(&self, context: &impl ObserveContext) -> f64 {
        let arrays = particle_arrays_from_context(context);
        self.total_nonbonded(&arrays, context.groups())
    }

    fn partial_energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        // The MC hot path: O(1) unless the cache needs building or has stale rows
        if let Change::SingleGroup(gi, GroupChange::RigidBody) = change {
            return self.cache.group_energy(*gi, &self.cached_pairs(context));
        }

        let arrays = particle_arrays_from_context(context);
        let groups = context.groups();
        match change {
            Change::Everything | Change::Volume(_, _) => self.total_nonbonded(&arrays, groups),
            Change::SingleGroup(group_index, group_change) => {
                self.single_group_change(&arrays, groups, *group_index, group_change)
            }
            Change::Groups(vec) => self.multi_group_change(&arrays, groups, vec),
            Change::None => 0.0,
        }
    }

    fn refresh(&mut self, context: &impl ObserveContext, change: &Change) -> anyhow::Result<()> {
        self.cache.refresh(change, &self.cached_pairs(context));
        Ok(())
    }

    fn save_backup(&mut self, context: &impl ObserveContext, change: &Change) {
        self.cache.save_backup(change, &self.cached_pairs(context));
    }

    fn undo(&mut self) {
        self.cache.undo();
    }

    fn discard_backup(&mut self) {
        self.cache.discard_backup();
    }
}

impl<P> NonbondedMatrix<P> {
    /// Get square matrix of pair potentials for all atom type combinations.
    pub const fn get_potentials(&self) -> &SquareMatrix<P> {
        &self.potentials
    }

    /// Enable or disable bounding-sphere culling of distant group pairs.
    pub(crate) fn set_bounding_spheres(&mut self, enabled: bool) {
        self.use_bounding_spheres = enabled;
    }

    /// Set the global interaction cutoff used for group-to-group culling.
    ///
    /// Culling only fires when both this cutoff is set and bounding spheres are
    /// enabled. To stay exact, the cutoff must be ≥ [`Self::required_cull_cutoff`],
    /// since a culled group pair skips *all* of its atom-atom interactions.
    pub(crate) fn set_cutoff(&mut self, cutoff: f64) {
        self.cutoff = Some(cutoff);
    }

    /// Get the list of excluded molecule-type pairs.
    #[must_use]
    pub(crate) fn molecule_pair_exclusions(&self) -> &[[MoleculeId; 2]] {
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
    pub(crate) fn exclude_molecule_pair(&mut self, mol_a: MoleculeId, mol_b: MoleculeId) {
        let pair = canonical_mol_pair(mol_a, mol_b);
        if !self.molecule_pair_exclusions.contains(&pair) {
            self.molecule_pair_exclusions.push(pair);
        }
    }

    /// Check if a molecule-type pair is excluded from nonbonded evaluation.
    #[inline(always)]
    fn is_molecule_pair_excluded(&self, mol_a: MoleculeId, mol_b: MoleculeId) -> bool {
        // Fast path: most simulations have no molecule-pair exclusions
        if !self.has_molecule_pair_exclusions() {
            return false;
        }
        self.molecule_pair_exclusions
            .contains(&canonical_mol_pair(mol_a, mol_b))
    }
}

impl NonbondedMatrix {
    /// Smallest group-to-group cutoff that keeps culling exact: the longest
    /// interaction range over every atom-type pair. Infinite if any pair
    /// potential is unbounded (e.g. bare Lennard-Jones or un-cutoff Coulomb).
    pub(crate) fn required_cull_cutoff(&self) -> f64 {
        self.potentials
            .iter()
            .map(PairPot::max_cutoff)
            .fold(0.0, f64::max)
    }

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
        Self::new(builder.nonbonded().unwrap(), topology, medium)
    }

    /// Create from a YAML string and a topology (no filesystem access).
    pub fn from_str(
        yaml: &str,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let builder = HamiltonianBuilder::from_str(yaml)?;
        builder.validate(topology.atomkinds())?;
        Self::new(builder.nonbonded().unwrap(), topology, medium)
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

        let mut potentials = SquareMatrix::from_element(n_atom_types, PairPot::default());

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
            cache: GroupPairCache::default(),
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

        let potentials = SquareMatrix::from_fn(source.order(), |i, j| {
            let potential = source.get((i, j)).expect("Index should be valid");
            SplinedPotential::with_cutoff(potential, cutoff, config.clone())
        });

        Self {
            potentials,
            exclusions: nonbonded.exclusions.clone(),
            cache: GroupPairCache::default(),
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

// ─── Batch-array energy and force evaluation ─────────────────────────────────

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
            // Without minimum-image support (non-orthorhombic cells) a plain
            // COM difference can exceed the true wrapped distance, which would
            // cull genuinely interacting pairs. Skip culling to stay exact.
            None => return false,
        };
        dist_sq > threshold_sq
    }

    /// Energy of particle `i` with targets, reading directly from particle arrays.
    ///
    /// Uses unchecked indexing to eliminate ~6 bounds checks per pair in the inner
    /// loop — this is the single hottest function in MC sweeps. All indices are
    /// validated by `debug_assert!` so debug builds still catch OOB.
    ///
    /// # Safety invariants (upheld by callers)
    /// - `i` and all `targets` values are in `[0, arrays.x.len())`
    /// - `atom_kinds[i]` values index within `self.potentials` dimensions
    #[inline]
    fn particle_energy(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        i: usize,
        targets: impl Iterator<Item = usize>,
    ) -> f64 {
        let probe = self.probe(arrays, i);
        let mut energy = 0.0;
        for j in targets {
            energy += probe.energy_with(j);
        }
        energy
    }

    /// Particle `i`, ready for a pair loop. The index invariants of [`Self::particle_energy`] apply.
    #[inline(always)]
    fn probe<'a, S: SimulationCell>(
        &'a self,
        arrays: &'a ParticleArrays<'a, S>,
        i: usize,
    ) -> PairProbe<'a, P, S> {
        debug_assert!(i < arrays.x.len());
        // SAFETY: i is a valid particle index from an active group range.
        let (x, y, z, kind) = unsafe { read_particle(arrays, i) };
        PairProbe {
            potentials: &self.potentials,
            arrays,
            x,
            y,
            z,
            kind,
            // Row slice gives a contiguous &[u8] for sequential cache-line reads over j.
            exclusions: self.exclusions.row(i),
        }
    }

    /// Intra-group energy (i < j pairs) via particle arrays.
    #[inline]
    fn group_with_itself(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        group: &Group,
    ) -> f64 {
        let active = group.iter_active();
        active
            .clone()
            .map(|i| self.particle_energy(arrays, i, (i + 1)..active.end))
            .sum()
    }

    /// Total nonbonded energy via particle arrays.
    fn total_nonbonded(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
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
                            && !self.groups_beyond_cutoff(group_i, group_j, arrays.pbc.as_ref())
                    })
                    .flat_map(|group_j| {
                        group_i
                            .iter_active()
                            .map(move |i| self.particle_energy(arrays, i, group_j.iter_active()))
                    })
                    .sum();
                inter + self.group_with_itself(arrays, group_i)
            })
            .sum()
    }

    /// Inter-group energy of one particle via particle arrays.
    #[inline]
    fn inter_group_energy(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        groups: &[Group],
        i: usize,
        own_group: &Group,
    ) -> f64 {
        groups
            .iter()
            .filter(|gj| gj.index() != own_group.index())
            .filter(|gj| !self.is_molecule_pair_excluded(own_group.molecule(), gj.molecule()))
            .filter(|gj| !self.groups_beyond_cutoff(own_group, gj, arrays.pbc.as_ref()))
            .map(|gj| self.particle_energy(arrays, i, gj.iter_active()))
            .sum()
    }

    /// Single group change via pre-extracted particle arrays.
    fn single_group_change(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        groups: &[Group],
        group_index: usize,
        change: &GroupChange,
    ) -> f64 {
        let group = &groups[group_index];
        match change {
            GroupChange::RigidBody | GroupChange::ResizeExcludeIntra(_) => {
                self.inter_group_energy_all(arrays, groups, group)
            }
            GroupChange::Resize(_) | GroupChange::UpdateIdentity(_) => {
                self.inter_group_energy_all(arrays, groups, group)
                    + self.group_with_itself(arrays, group)
            }
            // ResizePartial reuses the PartialUpdate path for O(N) per affected atom
            GroupChange::PartialUpdate(indices) | GroupChange::ResizePartial(_, indices) => indices
                .iter()
                .map(|&rel_idx| {
                    group.to_absolute(rel_idx).map_or(0.0, |abs_i| {
                        self.particle_energy_all(arrays, groups, abs_i.get(), group)
                    })
                })
                .sum(),
            // See `GroupChange::AtomicShrink` doc for the swap-and-pop rationale.
            GroupChange::AtomicShrink { rels, n_old } => {
                if group.len() != *n_old {
                    return 0.0;
                }
                let removed: Vec<usize> = rels.iter().map(|r| group.start() + r.get()).collect();
                let mut energy: f64 = removed
                    .iter()
                    .map(|&abs_i| self.particle_energy_all(arrays, groups, abs_i, group))
                    .sum();
                // Each pair *among* the removed atoms was counted by both of its members
                // above; drop one copy so the pair is removed once, not twice.
                for (k, &abs_i) in removed.iter().enumerate() {
                    energy -= self.particle_energy(arrays, abs_i, removed[..k].iter().copied());
                }
                energy
            }
            GroupChange::None => 0.0,
        }
    }

    /// Energy for multiple simultaneous group changes, counting every
    /// pair of changed groups exactly once regardless of change kind.
    ///
    /// The energy framework calls this in the old and new states and
    /// subtracts; we just need to return a value `Q` such that
    /// `Q_new - Q_old = ΔU`. The simplest such `Q` for multi-change moves is
    ///
    ///   Σ_{changed gi}  per-change-contrib(gi)
    /// + Σ_{i<j, both changed}  group_pair_energy(gi, gj)
    ///
    /// The per-change contribution mirrors `single_group_change` but
    /// only sums against *unchanged* groups (cross-terms between changed
    /// groups are handled exactly once in the second sum). The naive
    /// implementation — letting `single_group_change` iterate all
    /// other groups — double-counts the cross-term between two partial
    /// changes (the original bug).
    fn multi_group_change(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        groups: &[Group],
        changes: &[(usize, GroupChange)],
    ) -> f64 {
        // Whether a group carries a non-None change. Tested directly against `changes` — small for
        // speciation (2–4) and unbounded for a cluster move — so the hot path allocates nothing
        // regardless of cluster size (`changes` is a slice reference, hence `Copy` into the closure).
        let is_changed = move |idx: usize| {
            changes
                .iter()
                .any(|(gi, gc)| *gi == idx && !matches!(gc, GroupChange::None))
        };

        let mut energy = 0.0;

        // Phase 1: each change vs. unchanged groups, plus own-group intra.
        for (gi, gc) in changes {
            let group = &groups[*gi];
            let unchanged_groups = || {
                groups
                    .iter()
                    .filter(move |gk| gk.index() != *gi && !is_changed(gk.index()))
            };
            match gc {
                GroupChange::None => continue,
                gc if gc.is_whole_group() => {
                    energy += unchanged_groups()
                        .map(|gk| self.group_pair_energy(arrays, group, gk))
                        .sum::<f64>();
                    if gc.internal_change() {
                        energy += self.group_with_itself(arrays, group);
                    }
                }
                GroupChange::PartialUpdate(rel) | GroupChange::ResizePartial(_, rel) => {
                    // Resolve affected absolute indices once; we reuse this list
                    // to deduplicate intra pairs when multiple affected atoms
                    // share the same group. A `Vec` (not a fixed buffer) so a group with
                    // arbitrarily many affected atoms cannot panic — see the single-group path.
                    let affected_abs: Vec<usize> = rel
                        .iter()
                        .filter_map(|&r| group.to_absolute(r).ok().map(|abs| abs.get()))
                        .collect();
                    let affected_abs = affected_abs.as_slice();

                    // Cross with each valid unchanged group: outer-loop the
                    // group filters (per-pair exclusion / cutoff), inner-loop
                    // the affected atoms — the filters depend only on the
                    // group pair, not on which affected atom we're at.
                    for gk in unchanged_groups() {
                        if self.is_molecule_pair_excluded(group.molecule(), gk.molecule()) {
                            continue;
                        }
                        if self.groups_beyond_cutoff(group, gk, arrays.pbc.as_ref()) {
                            continue;
                        }
                        for &abs_i in affected_abs {
                            energy += self.particle_energy(arrays, abs_i, gk.iter_active());
                        }
                    }

                    // Intra: pair each `abs_i` with every active atom in its
                    // own group *except* affected atoms processed earlier, so
                    // each affected-affected pair within the group is counted
                    // exactly once. The exclusion-matrix diagonal handles the
                    // self pair.
                    for (k, &abs_i) in affected_abs.iter().enumerate() {
                        let earlier = &affected_abs[..k];
                        energy += self.particle_energy(
                            arrays,
                            abs_i,
                            group.iter_active().filter(|j| !earlier.contains(j)),
                        );
                    }
                }
                GroupChange::AtomicShrink { rels, n_old } => {
                    // See `GroupChange::AtomicShrink` doc for the swap-and-pop rationale.
                    if group.len() != *n_old {
                        continue;
                    }
                    let removed: Vec<usize> =
                        rels.iter().map(|r| group.start() + r.get()).collect();
                    let removed = removed.as_slice();

                    for gk in unchanged_groups() {
                        if self.is_molecule_pair_excluded(group.molecule(), gk.molecule()) {
                            continue;
                        }
                        if self.groups_beyond_cutoff(group, gk, arrays.pbc.as_ref()) {
                            continue;
                        }
                        for &abs_i in removed {
                            energy += self.particle_energy(arrays, abs_i, gk.iter_active());
                        }
                    }

                    // Intra: skip atoms removed earlier in this loop so each
                    // removed–removed pair inside the mega-group is counted once.
                    for (k, &abs_i) in removed.iter().enumerate() {
                        let earlier = &removed[..k];
                        energy += self.particle_energy(
                            arrays,
                            abs_i,
                            group.iter_active().filter(|j| !earlier.contains(j)),
                        );
                    }
                }
                GroupChange::RigidBody
                | GroupChange::UpdateIdentity(_)
                | GroupChange::Resize(_)
                | GroupChange::ResizeExcludeIntra(_) => {
                    unreachable!("non-internal GroupChange variants are handled by other paths")
                }
            }
        }

        // Phase 2: cross-term between each unique pair of changed groups.
        // `group_pair_energy` already applies exclusion + cutoff filters.
        //
        // A group index must appear at most once in `changes`: several changes to one
        // group have to be coalesced into a single entry (see `GroupChange::AtomicShrink`),
        // otherwise the loop below would pair that group with *itself* and count its
        // internal energy as a cross-term.
        debug_assert!(
            {
                let mut seen: Vec<usize> = changes
                    .iter()
                    .filter(|(_, gc)| !matches!(gc, GroupChange::None))
                    .map(|(gi, _)| *gi)
                    .collect();
                seen.sort_unstable();
                let len = seen.len();
                seen.dedup();
                seen.len() == len
            },
            "multi_group_change: duplicate group index in changes"
        );
        for (i, (gi, gci)) in changes.iter().enumerate() {
            if matches!(gci, GroupChange::None) {
                continue;
            }
            for (gj, gcj) in &changes[i + 1..] {
                if matches!(gcj, GroupChange::None) || gi == gj {
                    continue;
                }
                energy += self.group_pair_energy(arrays, &groups[*gi], &groups[*gj]);
            }
        }

        energy
    }

    /// Inter-group energy of an entire group with all other groups.
    ///
    /// With a cell list, iterates spatial neighbors per atom and excludes
    /// own-group contributions. Falls back to bounding-sphere-filtered
    /// group iteration otherwise.
    fn inter_group_energy_all(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        groups: &[Group],
        group: &Group,
    ) -> f64 {
        // Cell list iterates spatial neighbors without group ownership info,
        // so it cannot skip excluded molecule-type pairs. Fall back to
        // group-based iteration where the exclusion check is applied per pair.
        if !self.has_molecule_pair_exclusions() {
            if let Some(cl) = arrays.cell_list {
                let own_range = group.iter_active();
                return group
                    .iter_active()
                    .map(|i| {
                        self.particle_energy(
                            arrays,
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
            .map(|i| self.inter_group_energy(arrays, groups, i, group))
            .sum()
    }

    /// Total energy of particle `abs_i` with all other particles.
    ///
    /// When a cell list is available, iterates only over spatial neighbors
    /// (O(neighbors) instead of O(N)). Falls back to group-based iteration otherwise.
    #[inline]
    fn particle_energy_all(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        groups: &[Group],
        abs_i: usize,
        own_group: &Group,
    ) -> f64 {
        // Cell list lacks group ownership, so cannot enforce molecule-pair exclusions
        if !self.has_molecule_pair_exclusions() {
            if let Some(cl) = arrays.cell_list {
                return self.particle_energy(
                    arrays,
                    abs_i,
                    cl.neighbors(abs_i).filter(move |&j| j != abs_i),
                );
            }
        }
        // Fallback: inter-group + intra-group
        self.inter_group_energy(arrays, groups, abs_i, own_group)
            + self.particle_energy(arrays, abs_i, own_group.iter_active())
    }

    /// Inter-group nonbonded energy between two specific groups via particle arrays.
    #[inline]
    fn group_pair_energy(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        group_i: &Group,
        group_j: &Group,
    ) -> f64 {
        if self.is_molecule_pair_excluded(group_i.molecule(), group_j.molecule()) {
            return 0.0;
        }
        if self.groups_beyond_cutoff(group_i, group_j, arrays.pbc.as_ref()) {
            return 0.0;
        }
        group_i
            .iter_active()
            .map(|i| self.particle_energy(arrays, i, group_j.iter_active()))
            .sum()
    }

    /// Whether bounding spheres can rule out `group`'s pairs, for which it needs a mass center.
    fn cullable(&self, group: &Group) -> bool {
        self.use_bounding_spheres && group.mass_center().is_some()
    }

    fn cached_pairs<'a>(&'a self, context: &'a impl ObserveContext) -> impl GroupPairs + 'a {
        CachedPairs {
            matrix: self,
            arrays: particle_arrays_from_context(context),
            context,
        }
    }

    /// Compute per-atom nonbonded forces for all active particles using particle arrays.
    ///
    /// For each unique pair (i, j), evaluates `-dU/d(r²)` from the pair potential
    /// and accumulates the force vector on both atoms (Newton's third law).
    fn accumulate_forces(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        groups: &[Group],
        forces: &mut [[f64; 3]],
    ) {
        for (gi, group_i) in groups.iter().enumerate() {
            // Inter-group pairs (gi < gj)
            for group_j in groups.iter().skip(gi + 1) {
                if self.is_molecule_pair_excluded(group_i.molecule(), group_j.molecule()) {
                    continue;
                }
                if self.groups_beyond_cutoff(group_i, group_j, arrays.pbc.as_ref()) {
                    continue;
                }
                for i in group_i.iter_active() {
                    self.accumulate_particle_forces(arrays, i, group_j.iter_active(), forces);
                }
            }
            // Intra-group pairs (i < j)
            let active = group_i.iter_active();
            for i in active.clone() {
                self.accumulate_particle_forces(arrays, i, (i + 1)..active.end, forces);
            }
        }
    }

    /// Accumulate forces on atom `i` from interactions with `targets`, applying Newton's third law.
    ///
    /// # Safety invariants (upheld by callers)
    /// - `i` and all `targets` values are in bounds for particle arrays and `forces`
    #[inline]
    fn accumulate_particle_forces(
        &self,
        arrays: &ParticleArrays<'_, impl SimulationCell>,
        i: usize,
        targets: impl Iterator<Item = usize>,
        forces: &mut [[f64; 3]],
    ) {
        debug_assert!(i < arrays.x.len());
        // SAFETY: i is a valid particle index from an active group range.
        let (xi, yi, zi, kind_i) = unsafe { read_particle(arrays, i) };
        let excl_row = self.exclusions.row(i);

        for j in targets {
            debug_assert!(j < arrays.x.len());
            unsafe {
                if *excl_row.get_unchecked(j) == 0 {
                    continue;
                }
                let dr = arrays.distance_vector_to(xi, yi, zi, j);
                let rsq = dr[0].mul_add(dr[0], dr[1].mul_add(dr[1], dr[2] * dr[2]));
                let kind_j = *arrays.atom_kinds.get_unchecked(j) as usize;
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
    pub(super) fn forces(&self, context: &impl ObserveContext) -> Vec<crate::Point> {
        let n = context
            .groups()
            .iter()
            .map(|g| g.start() + g.capacity())
            .max()
            .unwrap_or(0);
        let mut forces = vec![[0.0f64; 3]; n];

        let arrays = particle_arrays_from_context(context);
        self.accumulate_forces(&arrays, context.groups(), &mut forces);

        forces
            .into_iter()
            .map(|[x, y, z]| crate::Point::new(x, y, z))
            .collect()
    }
}
