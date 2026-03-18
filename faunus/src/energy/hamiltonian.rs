use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    builder::HamiltonianBuilder,
    external_pressure::ExternalPressure,
    nonbonded::{NonbondedMatrix, NonbondedMatrixSplined},
    CellOverlap, EnergyTerm,
};
use crate::{topology::Topology, Change, Context};
use interatomic::coulomb::{DebyeLength, Temperature};
use std::cell::Cell;
use std::path::Path;
use std::time::{Duration, Instant};

/// Trait implemented by structures that can compute
/// and return an energy relevant to some change in the system.
pub trait EnergyChange {
    /// Compute the energy associated with some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64;
}

/// Collection of energy terms.
///
/// The Hamiltonian is a collection of energy terms,
/// that itself implements the `EnergyTerm` trait for summing them up.
#[derive(Debug, Clone, Default)]
pub struct Hamiltonian {
    energy_terms: Vec<EnergyTerm>,
    /// Accumulated wall-clock time spent in each energy term's `energy()` call.
    energy_timers: Vec<Cell<Duration>>,
    /// Accumulated wall-clock time spent in each energy term's `update()` call.
    update_timers: Vec<Cell<Duration>>,
}

impl Hamiltonian {
    /// Create a Hamiltonian from the provided HamiltonianBuilder and topology.
    pub(crate) fn new(
        builder: &HamiltonianBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let mut hamiltonian = Self::default();
        let temperature = medium.as_ref().map(|m| m.temperature());

        hamiltonian.push(CellOverlap.into());

        // If Ewald is configured, inject real-space pair potential into nonbonded
        // defaults *before* building (and potentially splining) the pair matrix.
        let mut pairpot_builder;
        let pairpot_ref = if let Some(ewald) = &builder.ewald {
            let debye_length = medium.as_ref().and_then(|m| m.debye_length());
            let real_space = interatomic::coulomb::pairwise::RealSpaceEwald::new(
                ewald.cutoff,
                ewald.accuracy,
                debye_length,
            );
            pairpot_builder = builder.pairpot_builder.clone().unwrap_or_default();
            pairpot_builder.push_default(super::builder::PairInteraction::CoulombRealSpaceEwald(
                real_space,
            ));
            log::info!(
                "Ewald: injected real-space pair potential (cutoff={}, accuracy={})",
                ewald.cutoff,
                ewald.accuracy
            );
            Some(&pairpot_builder)
        } else {
            builder.pairpot_builder.as_ref()
        };

        if let Some(nonbonded_matrix) = pairpot_ref {
            hamiltonian.push(Self::build_nonbonded_term(
                nonbonded_matrix,
                builder,
                topology,
                medium.clone(),
            )?);

            // Splined nonbonded skips excluded pairs entirely (SR + Coulomb).
            // For titration/alchemical moves, we need the Coulomb part back.
            let any_keep = topology
                .moleculekinds()
                .iter()
                .any(|m| m.keep_excluded_coulomb());
            if any_keep && nonbonded_matrix.has_coulomb() {
                let term = super::excluded_coulomb::ExcludedCoulomb::new(
                    nonbonded_matrix,
                    topology,
                    medium.clone(),
                    builder.combine_with_default,
                )?;
                hamiltonian.push(term.into());
                log::info!("Added excluded-pair Coulomb correction");
            }
        }

        if topology
            .moleculekinds()
            .iter()
            .any(|m| m.has_bonded_potentials())
        {
            hamiltonian.push(IntramolecularBonded::default().into());
        }

        // IntermolecularBonded term should only be added if it is actually needed
        if !topology.intermolecular().is_empty() {
            hamiltonian.push(IntermolecularBonded::new(topology));
        }

        if let Some(sasa_builder) = &builder.sasa {
            hamiltonian.push(sasa_builder.build()?.into());
        }

        if let Some(pressure) = &builder.pressure {
            let temperature = temperature.ok_or_else(|| {
                anyhow::anyhow!("Medium with temperature required for isobaric energy term")
            })?;
            let rt = crate::R_IN_KJ_PER_MOL * temperature;
            hamiltonian.push(ExternalPressure::new(pressure, rt).into());
        }

        Ok(hamiltonian)
    }

    /// Build a nonbonded energy term (plain or splined) from a pair potential builder.
    fn build_nonbonded_term(
        pairpot_builder: &super::builder::PairPotentialBuilder,
        builder: &HamiltonianBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<EnergyTerm> {
        let nonbonded = NonbondedMatrix::new(
            pairpot_builder,
            topology,
            medium,
            builder.combine_with_default,
        )?;
        if let Some(spline_opts) = &builder.spline {
            let config = spline_opts.to_spline_config();
            let mut splined = NonbondedMatrixSplined::from_nonbonded(
                &nonbonded,
                spline_opts.cutoff,
                Some(config),
            );
            splined.set_bounding_spheres(spline_opts.bounding_spheres);
            log::info!(
                "Using splined nonbonded potentials (cutoff={}, n_points={})",
                spline_opts.cutoff,
                spline_opts.n_points
            );
            Ok(splined.into())
        } else {
            Ok(nonbonded.into())
        }
    }

    /// Create a Hamiltonian from a YAML file and topology.
    pub fn from_file(
        filename: impl AsRef<Path>,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let builder = HamiltonianBuilder::from_file(filename)?;
        Self::new(&builder, topology, medium)
    }

    /// Appends an energy term to the back of the Hamiltonian.
    pub(crate) fn push(&mut self, term: EnergyTerm) {
        self.energy_terms.push(term);
        self.energy_timers.push(Cell::new(Duration::ZERO));
        self.update_timers.push(Cell::new(Duration::ZERO));
    }

    /// Inserts an energy term at the front for early rejection of infinite energy.
    pub(crate) fn push_front(&mut self, term: EnergyTerm) {
        self.energy_terms.insert(0, term);
        self.energy_timers.insert(0, Cell::new(Duration::ZERO));
        self.update_timers.insert(0, Cell::new(Duration::ZERO));
    }

    /// Removes the first energy term. Returns `None` if empty.
    pub(crate) fn pop_front(&mut self) -> Option<EnergyTerm> {
        if self.energy_terms.is_empty() {
            return None;
        }
        self.energy_timers.remove(0);
        self.update_timers.remove(0);
        Some(self.energy_terms.remove(0))
    }

    /// Access the individual energy terms.
    pub fn energy_terms(&self) -> &[EnergyTerm] {
        &self.energy_terms
    }

    /// Replace the nonbonded energy term with a rebuilt version using an updated real-space scheme.
    ///
    /// Called after Ewald optimization changes α, requiring a new pair matrix and splines.
    pub(crate) fn rebuild_nonbonded(
        &mut self,
        builder: &HamiltonianBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
        real_space: interatomic::coulomb::pairwise::RealSpaceEwald,
    ) -> anyhow::Result<()> {
        let mut pairpot_builder = builder.pairpot_builder.clone().unwrap_or_default();
        pairpot_builder.push_default(super::builder::PairInteraction::CoulombRealSpaceEwald(
            real_space,
        ));

        let new_term = Self::build_nonbonded_term(&pairpot_builder, builder, topology, medium)?;

        for term in &mut self.energy_terms {
            // Ewald rebuild replaces the entire nonbonded term; carry over
            // molecule-pair exclusions so tabulated6d pairs stay excluded.
            if let Some(old) = term.molecule_pair_exclusions() {
                let old = old.to_vec();
                *term = new_term;
                for [a, b] in old {
                    term.exclude_molecule_pair(a, b);
                }
                log::info!("Rebuilt nonbonded interactions with optimized Ewald real-space scheme");
                return Ok(());
            }
        }
        Ok(())
    }

    /// Exclude a molecule-type pair from the nonbonded energy term.
    ///
    /// All inter-group interactions between groups of these two molecule kinds
    /// will be skipped by the nonbonded term. Use when the pair is handled by
    /// another energy term (e.g. [`Tabulated6D`]).
    pub(crate) fn exclude_nonbonded_molecule_pair(&mut self, mol_a: usize, mol_b: usize) {
        self.energy_terms
            .iter_mut()
            .for_each(|t| t.exclude_molecule_pair(mol_a, mol_b));
    }

    /// Invalidate all nonbonded energy caches.
    /// Call after bulk position updates (e.g. Langevin dynamics) that bypass
    /// the normal MC backup/update path.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn invalidate_caches(&mut self) {
        self.energy_terms
            .iter_mut()
            .for_each(|t| t.invalidate_cache());
    }

    /// Compute each term's energy, returning `(name, energy)` pairs.
    pub fn per_term_energies(
        &self,
        context: &impl Context,
        change: &Change,
    ) -> Vec<(&'static str, f64)> {
        self.energy_terms
            .iter()
            .filter_map(|term| {
                let name = crate::Info::short_name(term)?;
                Some((name, term.energy(context, change)))
            })
            .collect()
    }

    /// Compute per-atom forces from all energy terms.
    ///
    /// Returns a dense vector indexed by absolute particle index, with contributions
    /// from all force-providing terms summed together.
    pub fn forces(&self, context: &impl Context) -> Vec<crate::Point> {
        let mut result: Option<Vec<crate::Point>> = None;
        for term in &self.energy_terms {
            let term_forces = term.forces(context);
            if term_forces.is_empty() {
                continue;
            }
            match &mut result {
                None => result = Some(term_forces),
                Some(acc) => {
                    // Extend if a later term has more particles
                    if term_forces.len() > acc.len() {
                        acc.resize(term_forces.len(), crate::Point::zeros());
                    }
                    for (a, f) in acc.iter_mut().zip(term_forces.iter()) {
                        *a += f;
                    }
                }
            }
        }
        result.unwrap_or_default()
    }

    /// Update internal state due to a change in the system.
    ///
    /// After a system change, the internal state of the energy terms may need to be updated.
    /// For example, in Ewald summation, the reciprocal space energy needs to be updated.
    pub(crate) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        for (term, timer) in self.energy_terms.iter_mut().zip(self.update_timers.iter()) {
            let start = Instant::now();
            term.update(context, change)?;
            timer.set(timer.get() + start.elapsed());
        }
        Ok(())
    }

    /// Save backups of all energy terms for later undo.
    ///
    /// Call before applying a move so that terms like Ewald can snapshot
    /// positions of affected particles in their pre-move state.
    pub(crate) fn save_backups(&mut self, change: &Change, context: &impl Context) {
        self.energy_terms
            .iter_mut()
            .for_each(|term| term.save_backup(change, context));
    }

    /// Update with backup for later undo on MC reject.
    pub(crate) fn update_with_backup(
        &mut self,
        context: &impl Context,
        change: &Change,
    ) -> anyhow::Result<()> {
        self.save_backups(change, context);
        self.update(context, change)
    }

    /// Restore all energy terms from their internal backups.
    pub(crate) fn undo(&mut self) {
        self.energy_terms.iter_mut().for_each(|term| term.undo());
    }

    /// Drop all energy term backups.
    pub(crate) fn discard_backup(&mut self) {
        self.energy_terms
            .iter_mut()
            .for_each(|term| term.discard_backup());
    }

    /// Per-term information as a YAML mapping (term name → info).
    ///
    /// Only includes terms that provide information via `EnergyTerm::to_yaml()`.
    pub fn info_to_yaml(&self) -> serde_yaml::Value {
        let map: serde_yaml::Mapping = self
            .energy_terms
            .iter()
            .filter_map(|term| {
                let name = crate::Info::short_name(term)?;
                let info = term.to_yaml()?;
                Some((serde_yaml::Value::String(name.to_string()), info))
            })
            .collect();
        serde_yaml::Value::Mapping(map)
    }

    /// Per-term energy timing as a YAML-serializable map (term name → percentage of total).
    ///
    /// Reports both `energy()` and `update()` time per term.
    pub fn timing_to_yaml(&self) -> serde_yaml::Value {
        let total: f64 = self
            .energy_timers
            .iter()
            .chain(self.update_timers.iter())
            .map(|t| t.get().as_secs_f64())
            .sum();
        let map: serde_yaml::Mapping = self
            .energy_terms
            .iter()
            .zip(self.energy_timers.iter())
            .zip(self.update_timers.iter())
            .filter_map(|((term, e_timer), u_timer)| {
                let name = crate::Info::short_name(term)?;
                let to_pct = |secs: f64| {
                    if total > 0.0 {
                        (secs / total * 10000.0).round() / 100.0
                    } else {
                        0.0
                    }
                };
                let e_pct = to_pct(e_timer.get().as_secs_f64());
                let u_pct = to_pct(u_timer.get().as_secs_f64());
                let combined = e_pct + u_pct;
                if combined == 0.0 {
                    return None;
                }
                // Show "energy + update" breakdown only when update is significant
                let label = if u_pct >= 0.01 {
                    format!("{combined} (energy: {e_pct}, update: {u_pct})")
                } else {
                    format!("{combined}")
                };
                Some((
                    serde_yaml::Value::String(name.to_string()),
                    serde_yaml::Value::String(label),
                ))
            })
            .collect();
        serde_yaml::Value::Mapping(map)
    }
    /// Add energy terms that require a live Context (particles already placed).
    ///
    /// Must be called after `Hamiltonian::new()` and particle insertion.
    pub(crate) fn finalize(
        &mut self,
        builder: &HamiltonianBuilder,
        context: &impl Context,
        medium: Option<&interatomic::coulomb::Medium>,
    ) -> anyhow::Result<()> {
        if let Some(constrain_builders) = &builder.constrain {
            for cb in constrain_builders {
                self.push(cb.build(context)?.into());
            }
        }
        if let Some(ext_builders) = &builder.customexternal {
            for eb in ext_builders {
                self.push(eb.build()?.into());
            }
        }

        let require_thermal_energy = |term: &str| -> anyhow::Result<f64> {
            let m = medium.ok_or_else(|| {
                anyhow::anyhow!("Medium with temperature required for {term} energy term")
            })?;
            Ok(crate::R_IN_KJ_PER_MOL * m.temperature())
        };

        if let Some(pm_builder) = &builder.polymer_depletion {
            let thermal_energy = require_thermal_energy("polymer_depletion")?;
            self.push(pm_builder.build(context, thermal_energy)?.into());
        }
        if let Some(tab_builder) = &builder.tabulated6d {
            let thermal_energy = require_thermal_energy("tabulated6d")?;
            let tab = tab_builder.build(context, 1.0 / thermal_energy)?;
            for (mol_a, mol_b) in tab.molecule_pairs() {
                self.exclude_nonbonded_molecule_pair(mol_a, mol_b);
                log::info!(
                    "Excluded molecule pair ({}, {}) from nonbonded (handled by tabulated6d)",
                    mol_a,
                    mol_b
                );
            }
            self.push(tab.into());
        }
        if let Some(ewald_builder) = &builder.ewald {
            let medium = medium
                .ok_or_else(|| anyhow::anyhow!("Ewald requires a medium with permittivity"))?;
            let initial_alpha = {
                let debye_length = medium.debye_length();
                interatomic::coulomb::pairwise::RealSpaceEwald::new(
                    ewald_builder.cutoff,
                    ewald_builder.accuracy,
                    debye_length,
                )
                .alpha()
            };
            let ewald = super::EwaldReciprocalEnergy::new(ewald_builder, context, medium)?;
            if ewald.alpha() != initial_alpha {
                self.rebuild_nonbonded(
                    builder,
                    context.topology_ref(),
                    Some(medium.clone()),
                    ewald.real_space_scheme(),
                )?;
            }
            self.push(ewald.into());
        }
        Ok(())
    }
}

impl<T: Into<Vec<EnergyTerm>>> From<T> for Hamiltonian {
    fn from(energy_terms: T) -> Self {
        let terms: Vec<EnergyTerm> = energy_terms.into();
        let n = terms.len();
        Self {
            energy_terms: terms,
            energy_timers: vec![Cell::new(Duration::ZERO); n],
            update_timers: vec![Cell::new(Duration::ZERO); n],
        }
    }
}

impl EnergyChange for Hamiltonian {
    /// Compute the energy of the Hamiltonian associated with a change in the system.
    /// The energy is returned in the units of kJ/mol.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        let mut sum: f64 = 0.0;
        for (term, timer) in self.energy_terms.iter().zip(self.energy_timers.iter()) {
            let start = Instant::now();
            let energy = term.energy(context, change);
            timer.set(timer.get() + start.elapsed());
            if energy.is_finite() {
                sum += energy;
            } else {
                return energy;
            }
        }
        sum
    }
}
