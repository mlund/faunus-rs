use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    builder::HamiltonianBuilder,
    external_pressure::ExternalPressure,
    nonbonded::{NonbondedMatrix, NonbondedMatrixSplined},
    CellOverlap, EnergyTerm,
};
use crate::{topology::Topology, Change, Context};
use interatomic::coulomb::Temperature;
use std::path::Path;

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
}

impl Hamiltonian {
    /// Synchronize cached state from another Hamiltonian after an MC accept/reject step.
    pub(crate) fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()> {
        for (term, other_term) in self.energy_terms.iter_mut().zip(other.energy_terms.iter()) {
            term.sync_from(other_term, change)?;
        }
        Ok(())
    }

    /// Create a Hamiltonian from the provided HamiltonianBuilder and topology.
    pub(crate) fn new(
        builder: &HamiltonianBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let mut hamiltonian = Self::default();
        let temperature = medium.as_ref().map(|m| m.temperature());

        hamiltonian.push(CellOverlap.into());

        if let Some(nonbonded_matrix) = &builder.pairpot_builder {
            let nonbonded = NonbondedMatrix::new(nonbonded_matrix, topology, medium)?;

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

        hamiltonian.push(IntramolecularBonded::default().into());

        // IntermolecularBonded term should only be added if it is actually needed
        if !topology.intermolecular().is_empty() {
            hamiltonian.push(IntermolecularBonded::new(topology));
        }

        if let Some(sasa_builder) = &builder.sasa {
            hamiltonian.push(sasa_builder.build()?.into());
        }

        if let Some(pressure) = &builder.isobaric {
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

    /// Update with backup for later undo on MC reject.
    pub(crate) fn update_with_backup(
        &mut self,
        context: &impl Context,
        change: &Change,
    ) -> anyhow::Result<()> {
        self.energy_terms
            .iter_mut()
            .for_each(|term| term.save_backup(change));
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
}

impl<T: Into<Vec<EnergyTerm>>> From<T> for Hamiltonian {
    fn from(energy_terms: T) -> Self {
        Self {
            energy_terms: energy_terms.into(),
        }
    }
}

impl EnergyChange for Hamiltonian {
    /// Compute the energy of the Hamiltonian associated with a change in the system.
    /// The energy is returned in the units of kJ/mol.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        let mut sum: f64 = 0.0;
        let energies = self
            .energy_terms
            .iter()
            .map(|term| term.energy(context, change));

        for energy in energies {
            if energy.is_finite() {
                sum += energy;
            } else {
                return energy; // infinite or NaN
            }
        }
        sum
    }
}
