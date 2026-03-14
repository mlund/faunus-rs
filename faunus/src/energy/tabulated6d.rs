//! Tabulated 6D rigid molecule-molecule energy from pre-computed icosphere tables.
//!
//! Loads binary [`icotable::Table6DFlat`] files produced by Duello and provides
//! O(1) cached energy lookups for rigid-body MC moves.

use super::nonbonded::cache::GroupEnergyCache;
use crate::cell::BoundaryConditions;
use crate::{Change, Context, GroupChange};
use icotable::{f16, Table6DAdaptive, Table6DFlat, Vector3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Wrapper supporting both flat (legacy) and adaptive table formats.
/// Adaptive tables use per-slab resolution tiers (repulsive/scalar/nearest/interpolated)
/// to reduce storage and lookup cost, especially at short range (overlap) and long range
/// (smooth angular surface).
#[derive(Clone, Debug)]
enum TableKind {
    Flat(Table6DFlat<f16>),
    Adaptive(Table6DAdaptive<f32>),
}

impl TableKind {
    /// Load a table file, trying adaptive format first, then flat.
    /// Both formats use bincode serialization, so the wrong format will fail
    /// deserialization cleanly and we fall back to the other.
    fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        match Table6DAdaptive::<f32>::load(path) {
            Ok(t) => Ok(Self::Adaptive(t)),
            Err(_) => Ok(Self::Flat(Table6DFlat::<f16>::load(path)?)),
        }
    }

    fn rmin(&self) -> f64 {
        match self {
            Self::Flat(t) => t.rmin,
            Self::Adaptive(t) => t.rmin,
        }
    }

    fn rmax(&self) -> f64 {
        match self {
            Self::Flat(t) => t.rmax,
            Self::Adaptive(t) => t.rmax,
        }
    }

    fn tail_energy(&self, r: f64) -> f64 {
        match self {
            Self::Flat(t) => t.tail_energy(r),
            Self::Adaptive(t) => t.tail_energy(r),
        }
    }

    fn validate_metadata(&self) -> anyhow::Result<()> {
        match self {
            Self::Flat(t) => t.validate_metadata(),
            Self::Adaptive(t) => t.validate_metadata(),
        }
    }

    fn lookup_boltzmann(
        &self,
        r: f64,
        omega: f64,
        dir_a: &Vector3,
        dir_b: &Vector3,
        beta: f64,
    ) -> f64 {
        match self {
            Self::Flat(t) => t.lookup_boltzmann(r, omega, dir_a, dir_b, beta),
            Self::Adaptive(t) => t.lookup_boltzmann(r, omega, dir_a, dir_b, beta),
        }
    }

    /// Temperature (K) used when generating the table, if stored in metadata.
    fn generation_temperature(&self) -> Option<f64> {
        match self {
            Self::Flat(t) => t.metadata.as_ref().and_then(|m| m.temperature),
            Self::Adaptive(t) => t.metadata.as_ref().and_then(|m| m.temperature),
        }
    }

    /// Summary string for logging.
    fn summary(&self) -> String {
        match self {
            Self::Flat(t) => format!(
                "{} R-bins, {} omega-bins, {} vertices (flat)",
                t.n_r, t.n_omega, t.n_vertices
            ),
            Self::Adaptive(t) => format!(
                "{} R-bins, {} omega-bins, {} levels (adaptive)",
                t.n_r,
                t.n_omega,
                t.levels.len()
            ),
        }
    }
}

/// A single molecule-pair table entry.
#[derive(Clone)]
struct Entry {
    mol_id_a: usize,
    mol_id_b: usize,
    /// Shared via `Arc` so cloning (e.g. VirtualTranslate) is O(1).
    table: Arc<TableKind>,
}

/// Tabulated 6D molecule-molecule energy term.
///
/// Uses pre-computed energy tables over (R, ω, θ₁φ₁, θ₂φ₂) for rigid-body
/// pairs. Each entry maps a pair of molecule types to a binary table file.
/// Covered molecule-type pairs are automatically excluded from the nonbonded
/// energy term to prevent double-counting.
///
/// Angular interpolation uses Boltzmann-weighted barycentric averaging to
/// avoid Jensen's inequality bias at repulsive contacts.
///
/// The cache uses `RwLock` for lazy initialization from `energy(&self)`;
/// mutating methods (`undo`, `save_backup`, etc.) use `get_mut()` since
/// they receive `&mut self` from the `dispatch_stateful!` macro.
pub struct Tabulated6D {
    entries: Vec<Entry>,
    /// Inverse thermal energy 1/kT in mol/kJ for Boltzmann-weighted interpolation.
    beta: f64,
    cache: RwLock<Option<GroupEnergyCache>>,
}

impl Clone for Tabulated6D {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            beta: self.beta,
            cache: RwLock::new(self.cache.read().expect("cache lock poisoned").clone()),
        }
    }
}

impl std::fmt::Debug for Tabulated6D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tabulated6D")
            .field("n_entries", &self.entries.len())
            .finish()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tabulated6DEntryBuilder {
    pub molecules: [String; 2],
    pub file: PathBuf,
}

/// Deserializable builder for the full Tabulated6D term.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(transparent)]
pub struct Tabulated6DBuilder(pub Vec<Tabulated6DEntryBuilder>);

impl Tabulated6DBuilder {
    pub fn build(&self, context: &impl Context, beta: f64) -> anyhow::Result<Tabulated6D> {
        let topology = context.topology();
        let entries = self
            .0
            .iter()
            .map(|eb| {
                let resolve = |name: &str| {
                    topology
                        .moleculekinds()
                        .iter()
                        .position(|m| m.name() == name)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Molecule '{}' in tabulated6d energy term does not exist",
                                name
                            )
                        })
                };
                let mol_id_a = resolve(&eb.molecules[0])?;
                let mol_id_b = resolve(&eb.molecules[1])?;
                let table = TableKind::load(&eb.file).map_err(|e| {
                    anyhow::anyhow!("Failed to load table '{}': {}", eb.file.display(), e)
                })?;
                table
                    .validate_metadata()
                    .map_err(|e| anyhow::anyhow!("Invalid table '{}': {}", eb.file.display(), e))?;
                // Adaptive tables classify repulsive slabs using the generation
                // temperature. At lower simulation temperatures, energy differences
                // matter more and the repulsive/non-repulsive boundary becomes
                // more critical for correct Metropolis acceptance.
                if let Some(table_temp) = table.generation_temperature() {
                    let sim_temp = 1.0 / (beta * physical_constants::MOLAR_GAS_CONSTANT * 1e-3);
                    if sim_temp < table_temp * 0.95 {
                        log::warn!(
                            "Table '{}' was generated at {:.1} K but simulation runs at {:.1} K. \
                             Repulsive slab classification may be too aggressive at lower temperature.",
                            eb.file.display(),
                            table_temp,
                            sim_temp,
                        );
                    }
                }
                log::info!(
                    "Loaded 6D table for ({}, {}): {}",
                    &eb.molecules[0],
                    &eb.molecules[1],
                    table.summary(),
                );
                Ok(Entry {
                    mol_id_a,
                    mol_id_b,
                    table: Arc::new(table),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        log::info!(
            "Tabulated6D: beta = {:.6} mol/kJ (kT = {:.4} kJ/mol)",
            beta,
            1.0 / beta
        );
        Ok(Tabulated6D {
            entries,
            beta,
            cache: RwLock::new(None),
        })
    }
}

impl Tabulated6D {
    /// Molecule-type index pairs covered by this term.
    pub(crate) fn molecule_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.entries.iter().map(|e| (e.mol_id_a, e.mol_id_b))
    }

    /// Find the table entry for a pair of molecule IDs (order-independent).
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

    /// Energy between two groups via 6D table lookup.
    fn pair_energy(&self, context: &impl Context, gi: usize, gj: usize) -> f64 {
        let groups = context.groups();
        let ga = &groups[gi];
        let gb = &groups[gj];

        let (entry, swapped) = match self.find_entry(ga.molecule(), gb.molecule()) {
            Some(pair) => pair,
            None => return 0.0,
        };

        let (com_a, com_b) = match (ga.mass_center(), gb.mass_center()) {
            (Some(a), Some(b)) => (a, b),
            _ => return 0.0,
        };

        let sep = context.cell().distance(com_a, com_b);
        let r = sep.norm();
        if r > entry.table.rmax() {
            return entry.table.tail_energy(r);
        }
        if r < entry.table.rmin() {
            return f64::INFINITY;
        }

        // Swap quaternions and negate separation to match table's (mol_a, mol_b) ordering
        let (q_a, q_b, oriented_sep) = if swapped {
            (gb.quaternion(), ga.quaternion(), -sep)
        } else {
            (ga.quaternion(), gb.quaternion(), sep)
        };

        let (_r_val, omega, dir_a, dir_b) = icotable::inverse_orient(&oriented_sep, q_a, q_b);
        entry
            .table
            .lookup_boltzmann(r, omega, &dir_a, &dir_b, self.beta)
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
        let gi = match change {
            Change::SingleGroup(gi, GroupChange::RigidBody) => *gi,
            _ => return,
        };
        // Take cache out to avoid borrow conflict with pair_energy(&self)
        let mut cache = match self.cache.get_mut().expect("cache lock poisoned").take() {
            Some(c) => c,
            None => return,
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
}

impl super::EnergyChange for Tabulated6D {
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
