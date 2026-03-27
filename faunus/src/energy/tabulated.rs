//! Tabulated rigid-body energy from pre-computed icosphere tables.
//!
//! Supports both 6D molecule-molecule tables ([`icotable::Table6DFlat`],
//! [`icotable::Table6DAdaptive`]) and 3D molecule-atom tables
//! ([`icotable::Table3DAdaptive`]) produced by Duello.

use super::nonbonded::cache::GroupEnergyCache;
use crate::cell::BoundaryConditions;
use crate::{Change, Context, GroupChange};
use anyhow::Context as _;
use icotable::{f16, PointGroup, Table3DAdaptive, Table6DAdaptive, Table6DFlat, Vector3};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Wrapper supporting flat 6D (legacy), adaptive 6D, and adaptive 3D table formats.
#[derive(Clone, Debug)]
enum TableKind {
    Flat6D(Table6DFlat<f16>),
    Adaptive6D(Table6DAdaptive<f32>),
    Adaptive3D(Table3DAdaptive<f32>),
}

/// Dispatch a method or field access across all `TableKind` variants.
macro_rules! dispatch_table {
    ($self:expr, |$t:ident| $body:expr) => {
        match $self {
            TableKind::Flat6D($t) => $body,
            TableKind::Adaptive6D($t) => $body,
            TableKind::Adaptive3D($t) => $body,
        }
    };
}

impl TableKind {
    /// Load a 6D table file, trying adaptive first since it's the current Duello output format;
    /// flat is legacy and only attempted as fallback.
    fn load_6d(path: &std::path::Path) -> anyhow::Result<Self> {
        match Table6DAdaptive::<f32>::load(path) {
            Ok(t) => Ok(Self::Adaptive6D(t)),
            Err(_) => Ok(Self::Flat6D(Table6DFlat::<f16>::load(path)?)),
        }
    }

    /// Load a 3D adaptive table file.
    fn load_3d(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self::Adaptive3D(Table3DAdaptive::<f32>::load(path)?))
    }

    fn is_3d(&self) -> bool {
        matches!(self, Self::Adaptive3D(_))
    }

    fn rmin(&self) -> f64 {
        dispatch_table!(self, |t| t.rmin)
    }

    fn rmax(&self) -> f64 {
        dispatch_table!(self, |t| t.rmax)
    }

    fn tail_energy(&self, r: f64) -> f64 {
        dispatch_table!(self, |t| t.tail_energy(r))
    }

    fn validate_metadata(&self) -> anyhow::Result<()> {
        dispatch_table!(self, |t| t.validate_metadata())
    }

    /// 6D Boltzmann-weighted lookup. Panics if called on a 3D table.
    fn lookup_boltzmann_6d(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
        inv_thermal_energy: f64,
    ) -> f64 {
        match self {
            Self::Flat6D(t) => t.lookup_boltzmann(r, omega, dir_a, dir_b, inv_thermal_energy),
            Self::Adaptive6D(t) => t.lookup_boltzmann(r, omega, dir_a, dir_b, inv_thermal_energy),
            Self::Adaptive3D(_) => unreachable!("6D lookup on 3D table"),
        }
    }

    /// 3D Boltzmann-weighted lookup. Panics if called on a 6D table.
    fn lookup_boltzmann_3d(&self, r: f64, dir: &Vector3, inv_thermal_energy: f64) -> f64 {
        match self {
            Self::Adaptive3D(t) => t.lookup_boltzmann(r, dir, inv_thermal_energy),
            _ => unreachable!("3D lookup on 6D table"),
        }
    }

    fn metadata(&self) -> Option<&icotable::TableMetadata> {
        dispatch_table!(self, |t| t.metadata.as_ref())
    }

    /// Temperature (K) used when generating the table, if stored in metadata.
    fn generation_temperature(&self) -> Option<f64> {
        self.metadata().and_then(|m| m.temperature)
    }

    fn point_group(&self) -> &PointGroup {
        self.metadata()
            .map_or(&PointGroup::Asymmetric, |m| &m.point_group)
    }

    /// Summary string for logging.
    fn summary(&self) -> String {
        match self {
            Self::Flat6D(t) => format!(
                "{} R-bins, {} omega-bins, {} vertices (flat 6D)",
                t.n_r, t.n_omega, t.n_vertices
            ),
            Self::Adaptive6D(t) => format!(
                "{} R-bins, {} omega-bins, {} levels (adaptive 6D)",
                t.n_r,
                t.n_omega,
                t.levels.len()
            ),
            Self::Adaptive3D(t) => {
                format!("{} R-bins, {} levels (adaptive 3D)", t.n_r, t.levels.len())
            }
        }
    }
}

/// A single molecule-pair table entry.
#[derive(Clone)]
pub(crate) struct Entry {
    mol_id_a: usize,
    mol_id_b: usize,
    /// Shared via `Arc` so cloning (e.g. VirtualTranslate) is O(1).
    table: Arc<TableKind>,
    /// Pre-symmetrized table, user opted into single lookup, or 3D table (inherently asymmetric).
    skip_swap_averaging: bool,
}

/// Tabulated molecule-molecule and molecule-atom energy term.
///
/// Uses pre-computed energy tables for rigid-body pairs:
/// - **6D tables**: (R, ω, θ₁φ₁, θ₂φ₂) for two rigid molecules
/// - **3D tables**: (R, θ, φ) for a rigid molecule and an atomic group
///
/// Covered molecule-type pairs are automatically excluded from the nonbonded
/// energy term to prevent double-counting.
///
/// Angular interpolation uses Boltzmann-weighted barycentric averaging to
/// avoid Jensen's inequality bias at repulsive contacts.
///
/// The cache uses `RwLock` for lazy initialization from `energy(&self)`;
/// mutating methods (`undo`, `save_backup`, etc.) use `get_mut()` since
/// they receive `&mut self` from the `dispatch_stateful!` macro.
pub struct TabulatedEnergy {
    entries: Vec<Entry>,
    /// Inverse thermal energy 1/kT in mol/kJ for Boltzmann-weighted interpolation.
    inv_thermal_energy: f64,
    cache: RwLock<Option<GroupEnergyCache>>,
    /// Tracks |exp(-βU_fwd) - exp(-βU_rev)| for 6D self-interaction lookups.
    /// Only accumulated when double-lookup is active.
    /// `Cell` because `pair_energy` takes `&self` (needed for `RwLock` lazy cache init).
    swap_stats: Cell<(usize, f64, f64)>,
}

/// Manual Clone because `RwLock` doesn't derive it; we snapshot the current cache state.
impl Clone for TabulatedEnergy {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            inv_thermal_energy: self.inv_thermal_energy,
            cache: RwLock::new(self.cache.read().expect("cache lock poisoned").clone()),
            swap_stats: self.swap_stats.clone(),
        }
    }
}

impl std::fmt::Debug for TabulatedEnergy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TabulatedEnergy")
            .field("n_entries", &self.entries.len())
            .finish()
    }
}

/// Builder entry for a 6D rigid molecule-molecule table.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tabulated6DEntryBuilder {
    pub molecules: [String; 2],
    pub file: PathBuf,
    /// Skip the second (swapped) lookup for this homo-dimer table.
    /// Halves per-pair cost at the expense of a small energy drift
    /// from interpolation asymmetry.
    #[serde(default)]
    pub single_lookup: bool,
}

/// Builder entry for a 3D rigid molecule-atom table.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tabulated3DEntryBuilder {
    /// `[rigid_molecule, atom_molecule]` — order matters: the first molecule
    /// must be the rigid body whose orientation is used for the lookup.
    pub molecules: [String; 2],
    pub file: PathBuf,
}

/// Deserializable builder for 6D table entries.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(transparent)]
pub struct Tabulated6DBuilder(pub Vec<Tabulated6DEntryBuilder>);

/// Deserializable builder for 3D table entries.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(transparent)]
pub struct Tabulated3DBuilder(pub Vec<Tabulated3DEntryBuilder>);

/// Resolve a molecule name to its type index.
fn resolve_molecule(
    topology: &crate::topology::Topology,
    name: &str,
    label: &str,
) -> anyhow::Result<usize> {
    topology
        .moleculekinds()
        .iter()
        .position(|m| m.name() == name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Molecule '{}' in {} energy term does not exist",
                name,
                label
            )
        })
}

/// Adaptive tables classify repulsive slabs at the generation temperature. Running at a
/// lower temperature makes repulsive contacts more significant, so the classification
/// may discard angular detail that matters for correct Metropolis acceptance.
fn warn_temperature_mismatch(table: &TableKind, file: &std::path::Path, inv_thermal_energy: f64) {
    if let Some(table_temp) = table.generation_temperature() {
        let sim_temp = 1.0 / (inv_thermal_energy * crate::R_IN_KJ_PER_MOL);
        if sim_temp < table_temp * 0.95 {
            log::warn!(
                "Table '{}' was generated at {:.1} K but simulation runs at {:.1} K. \
                 Repulsive slab classification may be too aggressive at lower temperature.",
                file.display(),
                table_temp,
                sim_temp,
            );
        }
    }
}

impl Tabulated6DBuilder {
    pub(crate) fn build_entries(
        &self,
        topology: &crate::topology::Topology,
        inv_thermal_energy: f64,
    ) -> anyhow::Result<Vec<Entry>> {
        self.0
            .iter()
            .map(|eb| {
                let mol_id_a = resolve_molecule(topology, &eb.molecules[0], "tabulated6d")?;
                let mol_id_b = resolve_molecule(topology, &eb.molecules[1], "tabulated6d")?;
                let table = TableKind::load_6d(&eb.file)
                    .with_context(|| format!("Failed to load 6D table '{}'", eb.file.display()))?;
                table
                    .validate_metadata()
                    .with_context(|| format!("Invalid table '{}'", eb.file.display()))?;
                warn_temperature_mismatch(&table, &eb.file, inv_thermal_energy);
                let sym_info = match table.point_group() {
                    PointGroup::Exchange => ", exchange-symmetric",
                    PointGroup::Asymmetric => "",
                };
                log::info!(
                    "Loaded 6D table for ({}, {}): {}{}",
                    &eb.molecules[0],
                    &eb.molecules[1],
                    table.summary(),
                    sym_info,
                );
                let skip_swap_averaging =
                    eb.single_lookup || *table.point_group() == PointGroup::Exchange;
                if eb.single_lookup && mol_id_a == mol_id_b {
                    log::info!(
                        "  Single-lookup enabled for ({}, {}): ~2x faster, small energy drift expected",
                        &eb.molecules[0], &eb.molecules[1],
                    );
                }
                Ok(Entry {
                    mol_id_a,
                    mol_id_b,
                    skip_swap_averaging,
                    table: Arc::new(table),
                })
            })
            .collect()
    }
}

impl Tabulated3DBuilder {
    pub(crate) fn build_entries(
        &self,
        topology: &crate::topology::Topology,
        inv_thermal_energy: f64,
    ) -> anyhow::Result<Vec<Entry>> {
        self.0
            .iter()
            .map(|eb| {
                let mol_id_a = resolve_molecule(topology, &eb.molecules[0], "tabulated3d")?;
                let mol_id_b = resolve_molecule(topology, &eb.molecules[1], "tabulated3d")?;
                let table = TableKind::load_3d(&eb.file)
                    .with_context(|| format!("Failed to load 3D table '{}'", eb.file.display()))?;
                table
                    .validate_metadata()
                    .with_context(|| format!("Invalid table '{}'", eb.file.display()))?;
                warn_temperature_mismatch(&table, &eb.file, inv_thermal_energy);
                log::info!(
                    "Loaded 3D table for ({}, {}): {}",
                    &eb.molecules[0],
                    &eb.molecules[1],
                    table.summary(),
                );
                Ok(Entry {
                    mol_id_a,
                    mol_id_b,
                    skip_swap_averaging: true, // inherently asymmetric
                    table: Arc::new(table),
                })
            })
            .collect()
    }
}

impl TabulatedEnergy {
    /// Build from combined 6D and 3D entry lists.
    pub(crate) fn new(entries: Vec<Entry>, inv_thermal_energy: f64) -> Self {
        log::info!(
            "TabulatedEnergy: {} entries, inv_thermal_energy = {:.6} mol/kJ (kT = {:.4} kJ/mol)",
            entries.len(),
            inv_thermal_energy,
            1.0 / inv_thermal_energy
        );
        Self {
            entries,
            inv_thermal_energy,
            cache: RwLock::new(None),
            swap_stats: Cell::new((0, 0.0, 0.0)),
        }
    }

    /// Molecule-type index pairs covered by this term.
    pub(crate) fn molecule_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.entries.iter().map(|e| (e.mol_id_a, e.mol_id_b))
    }

    /// Find the table entry for a pair of molecule IDs.
    /// Order-independent: returns `swapped=true` when the caller's pair is reversed
    /// relative to the table's `(mol_id_a, mol_id_b)` ordering.
    fn find_entry(&self, mol_a: usize, mol_b: usize) -> Option<(&Entry, bool)> {
        self.entries.iter().find_map(|e| {
            if e.mol_id_a == mol_a && e.mol_id_b == mol_b {
                Some((e, false))
            } else if e.mol_id_a == mol_b && e.mol_id_b == mol_a {
                Some((e, true))
            } else {
                None
            }
        })
    }

    /// Energy between two groups via table lookup.
    ///
    /// For 6D self-interaction tables (mol_a == mol_b), the lookup is by default
    /// averaged over both A/B perspectives because `inverse_orient` maps
    /// the swapped pair to different angular grid points.
    ///
    /// For 3D tables, only one perspective is used (molecule→atom direction
    /// in the molecule's body frame).
    fn pair_energy(&self, context: &impl Context, gi: usize, gj: usize) -> f64 {
        let groups = context.groups();
        let ga = &groups[gi];
        let gb = &groups[gj];

        let Some((entry, swapped)) = self.find_entry(ga.molecule(), gb.molecule()) else {
            return 0.0;
        };

        let (Some(com_a), Some(com_b)) = (ga.mass_center(), gb.mass_center()) else {
            return 0.0;
        };

        let sep = context.cell().distance(com_a, com_b);
        let r = sep.norm();
        if r > entry.table.rmax() {
            return entry.table.tail_energy(r);
        }
        if r < entry.table.rmin() {
            return f64::INFINITY;
        }

        if entry.table.is_3d() {
            return self.pair_energy_3d(entry, swapped, &sep, r, ga, gb);
        }
        self.pair_energy_6d(entry, swapped, &sep, r, ga, gb)
    }

    /// 3D lookup: rigid molecule orientation + separation direction.
    fn pair_energy_3d(
        &self,
        entry: &Entry,
        swapped: bool,
        sep: &crate::Point,
        r: f64,
        ga: &crate::group::Group,
        gb: &crate::group::Group,
    ) -> f64 {
        // mol_id_a is always the rigid molecule; when swapped, flip the
        // separation so it points molecule→atom in the correct direction.
        let (rigid_group, sign) = if swapped { (gb, -1.0) } else { (ga, 1.0) };
        let oriented_sep = sep * sign;
        // Body-frame direction aligns with the fixed reference frame of the Duello scan.
        let dir = rigid_group
            .quaternion()
            .inverse_transform_vector(&(oriented_sep / r));
        entry
            .table
            .lookup_boltzmann_3d(r, &dir, self.inv_thermal_energy)
    }

    /// 6D lookup: two rigid molecule orientations + separation.
    fn pair_energy_6d(
        &self,
        entry: &Entry,
        swapped: bool,
        sep: &crate::Point,
        r: f64,
        ga: &crate::group::Group,
        gb: &crate::group::Group,
    ) -> f64 {
        // Reorder quaternions and negate separation so that `inverse_orient`
        // always sees the table's canonical (mol_a, mol_b) ordering.
        let (q_a, q_b, oriented_sep) = if swapped {
            (gb.quaternion(), ga.quaternion(), -sep)
        } else {
            (ga.quaternion(), gb.quaternion(), *sep)
        };

        let (_r, omega, dir_a, dir_b) = icotable::inverse_orient(&oriented_sep, q_a, q_b);
        let e_forward =
            entry
                .table
                .lookup_boltzmann_6d(r, omega, &dir_a, &dir_b, self.inv_thermal_energy);

        // Hetero-dimer, pre-symmetrized, or user opted into single lookup
        if entry.mol_id_a != entry.mol_id_b || entry.skip_swap_averaging {
            return e_forward;
        }

        // Self-interaction: average both perspectives to restore exchange
        // symmetry broken by interpolation on different angular grid points.
        let (_r, omega2, dir_a2, dir_b2) = icotable::inverse_orient(&(-oriented_sep), q_b, q_a);
        let e_reverse =
            entry
                .table
                .lookup_boltzmann_6d(r, omega2, &dir_a2, &dir_b2, self.inv_thermal_energy);

        // Track asymmetry in Boltzmann space: repulsive states (exp≈0)
        // contribute negligibly, isolating the interpolation error at
        // physically accessible configurations.
        let bf = (-self.inv_thermal_energy * e_forward).exp();
        let br = (-self.inv_thermal_energy * e_reverse).exp();
        if bf.is_finite() && br.is_finite() {
            self.welford_update((bf - br).abs());
        }

        0.5 * (e_forward + e_reverse)
    }

    fn total_energy(&self, context: &impl Context) -> f64 {
        let n = context.groups().len();
        let mut sum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                sum += self.pair_energy(context, i, j);
            }
        }
        sum
    }

    /// Lazily initialize cache on first RigidBody query.
    /// Uses `RwLock` (not `get_mut`) because `energy()` only has `&self`.
    fn ensure_cache(&self, context: &impl Context) {
        let needs_init = self.cache.read().expect("cache lock poisoned").is_none();
        if needs_init {
            let n = context.groups().len();
            let mut pairwise = vec![0.0; n * n];
            let mut group_energies = vec![0.0; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let e = self.pair_energy(context, i, j);
                    pairwise[i * n + j] = e;
                    pairwise[j * n + i] = e;
                    group_energies[i] += e;
                    group_energies[j] += e;
                }
            }
            *self.cache.write().expect("cache lock poisoned") =
                Some(GroupEnergyCache::new(pairwise, group_energies, n));
        }
    }

    pub(crate) fn update_cache(&mut self, context: &impl Context, change: &Change) {
        let &Change::SingleGroup(gi, GroupChange::RigidBody) = change else {
            return;
        };
        // Take cache out to avoid borrow conflict with pair_energy(&self)
        let Some(mut cache) = self.cache.get_mut().expect("cache lock poisoned").take() else {
            return;
        };
        let n = cache.n_groups;
        for j in 0..n {
            if j == gi {
                continue;
            }
            let new_e = self.pair_energy(context, gi, j);
            let delta = new_e - cache.pairwise[gi * n + j];
            cache.pairwise[gi * n + j] = new_e;
            cache.pairwise[j * n + gi] = new_e;
            cache.group_energies[gi] += delta;
            cache.group_energies[j] += delta;
        }
        *self.cache.get_mut().expect("cache lock poisoned") = Some(cache);
    }

    pub(crate) fn save_backup(&mut self, change: &Change) {
        match change {
            Change::SingleGroup(gi, GroupChange::RigidBody) => {
                if let Some(c) = self.cache.get_mut().expect("cache lock poisoned").as_mut() {
                    c.save_backup(*gi);
                }
            }
            // Topology-changing moves invalidate the N×N pairwise matrix.
            _ => {
                *self.cache.get_mut().expect("cache lock poisoned") = None;
            }
        }
    }

    pub(crate) fn undo(&mut self) {
        if let Some(c) = self.cache.get_mut().expect("cache lock poisoned").as_mut() {
            c.undo();
        }
    }

    pub(crate) fn discard_backup(&mut self) {
        if let Some(c) = self.cache.get_mut().expect("cache lock poisoned").as_mut() {
            c.discard_backup();
        }
    }

    /// Welford online update of (count, mean, M2) statistics.
    fn welford_update(&self, value: f64) {
        let (n, mean, m2) = self.swap_stats.get();
        let n = n + 1;
        let delta = value - mean;
        let new_mean = mean + delta / n as f64;
        let new_m2 = m2 + delta * (value - new_mean);
        self.swap_stats.set((n, new_mean, new_m2));
    }

    pub(crate) fn to_yaml(&self) -> serde_yml::Value {
        let (n, mean, m2) = self.swap_stats.get();
        let stddev = if n > 1 { (m2 / n as f64).sqrt() } else { 0.0 };
        let mut map = serde_yml::Mapping::new();
        map.insert("boltzmann_swap_asymmetry_samples".into(), (n as u64).into());
        map.insert("boltzmann_swap_asymmetry_mean".into(), mean.into());
        map.insert("boltzmann_swap_asymmetry_stddev".into(), stddev.into());
        serde_yml::Value::Mapping(map)
    }
}

impl super::EnergyChange for TabulatedEnergy {
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::None => 0.0,
            Change::Everything | Change::Volume(_, _) => self.total_energy(context),
            Change::SingleGroup(gi, GroupChange::RigidBody) => {
                self.ensure_cache(context);
                self.cache
                    .read()
                    .expect("cache lock poisoned")
                    .as_ref()
                    .unwrap()
                    .group_energies[*gi]
            }
            Change::SingleGroup(gi, _) => {
                let n = context.groups().len();
                (0..n)
                    .filter(|j| j != gi)
                    .map(|j| self.pair_energy(context, *gi, j))
                    .sum()
            }
            Change::Groups(group_changes) => {
                let n = context.groups().len();
                let changed: Vec<usize> = group_changes.iter().map(|(gi, _)| *gi).collect();
                let mut sum = 0.0;
                for &gi in &changed {
                    for j in 0..n {
                        if !changed.contains(&j) && j != gi {
                            sum += self.pair_energy(context, gi, j);
                        }
                    }
                }
                for (idx_a, &gi) in changed.iter().enumerate() {
                    for &gj in &changed[idx_a + 1..] {
                        sum += self.pair_energy(context, gi, gj);
                    }
                }
                sum
            }
        }
    }
}
