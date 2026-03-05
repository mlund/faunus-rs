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
            let nonbonded = NonbondedMatrix::new(
                nonbonded_matrix,
                topology,
                medium,
                builder.combine_with_default,
            )?;

            // Use splined potentials if configured
            if let Some(spline_opts) = &builder.spline {
                let config = spline_opts.to_spline_config();
                let splined = NonbondedMatrixSplined::from_nonbonded(
                    &nonbonded,
                    spline_opts.cutoff,
                    Some(config),
                );
                log::info!(
                    "Using splined nonbonded potentials (cutoff={}, n_points={})",
                    spline_opts.cutoff,
                    spline_opts.n_points
                );
                hamiltonian.push(splined.into());
            } else {
                hamiltonian.push(nonbonded.into());
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
            let thermal_energy = ExternalPressure::thermal_energy_from_temperature(temperature);
            hamiltonian.push(ExternalPressure::new(pressure, thermal_energy).into());
        }

        Ok(hamiltonian)
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
    }

    /// Access the individual energy terms.
    pub fn energy_terms(&self) -> &[EnergyTerm] {
        &self.energy_terms
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

    /// Update internal state due to a change in the system.
    ///
    /// After a system change, the internal state of the energy terms may need to be updated.
    /// For example, in Ewald summation, the reciprocal space energy needs to be updated.
    pub(crate) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        self.energy_terms
            .iter_mut()
            .try_for_each(|term| term.update(context, change))
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

    /// Per-term energy timing as a YAML-serializable map (term name → percentage of total).
    pub fn timing_to_yaml(&self) -> serde_yaml::Value {
        let total: f64 = self
            .energy_timers
            .iter()
            .map(|t| t.get().as_secs_f64())
            .sum();
        let map: serde_yaml::Mapping = self
            .energy_terms
            .iter()
            .zip(self.energy_timers.iter())
            .filter_map(|(term, timer)| {
                let name = crate::Info::short_name(term)?;
                let pct = if total > 0.0 {
                    (timer.get().as_secs_f64() / total * 10000.0).round() / 100.0
                } else {
                    0.0
                };
                Some((
                    serde_yaml::Value::String(name.to_string()),
                    serde_yaml::Value::Number(serde_yaml::Number::from(pct)),
                ))
            })
            .collect();
        serde_yaml::Value::Mapping(map)
    }
}

impl<T: Into<Vec<EnergyTerm>>> From<T> for Hamiltonian {
    fn from(energy_terms: T) -> Self {
        let terms: Vec<EnergyTerm> = energy_terms.into();
        let timers = vec![Cell::new(Duration::ZERO); terms.len()];
        Self {
            energy_terms: terms,
            energy_timers: timers,
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
