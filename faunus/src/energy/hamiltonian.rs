use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    builder::HamiltonianBuilder,
    nonbonded::NonbondedMatrix,
    sasa::SasaEnergy,
};
use crate::{topology::Topology, Change, Context, SyncFrom};
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

impl SyncFrom for Hamiltonian {
    /// Synchronize the Hamiltonian from other Hamiltonian.
    fn sync_from(&mut self, other: &Hamiltonian, change: &Change) -> anyhow::Result<()> {
        for (term, other_term) in self.energy_terms.iter_mut().zip(other.energy_terms.iter()) {
            term.sync_from(other_term, change)?;
        }

        Ok(())
    }
}

impl Hamiltonian {
    /// Create a Hamiltonian from the provided HamiltonianBuilder and topology.
    pub(crate) fn new(builder: &HamiltonianBuilder, topology: &Topology) -> anyhow::Result<Self> {
        let nonbonded = NonbondedMatrix::new(&builder.nonbonded, topology)?;
        let intramolecular_bonded = IntramolecularBonded::new();

        let mut hamiltonian =
            Hamiltonian::from_energy_terms(vec![nonbonded, intramolecular_bonded]);

        // IntermolecularBonded term should only be added if it is actually needed
        if !topology.intermolecular().is_empty() {
            hamiltonian.add_energy_term(IntermolecularBonded::new(topology));
        }

        Ok(hamiltonian)
    }

    /// Create a Hamiltonian from the provided energy terms.
    pub(crate) fn from_energy_terms(terms: Vec<EnergyTerm>) -> Self {
        Hamiltonian {
            energy_terms: terms,
        }
    }

    /// Add an energy term into the Hamiltonian.
    pub(crate) fn add_energy_term(&mut self, term: EnergyTerm) {
        self.energy_terms.push(term);
    }

    /// Update internal state due to a change in the system.
    ///
    /// After a system change, the internal state of the energy terms may need to be updated.
    /// For example, in Ewald summation, the reciprocal space energy needs to be updated.
    pub(crate) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        self.energy_terms
            .iter_mut()
            .try_for_each(|term| term.update(context, change))?;

        Ok(())
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

#[derive(Debug, Clone)]
pub enum EnergyTerm {
    /// Non-bonded interactions between particles.
    NonbondedMatrix(NonbondedMatrix),
    /// Intramolecular bonded interactions.
    IntramolecularBonded(IntramolecularBonded),
    /// Intermolecular bonded interactions.
    IntermolecularBonded(IntermolecularBonded),
    /// Solvent accessible surface area energy.
    SasaEnergy(SasaEnergy),
}

impl EnergyTerm {
    /// Create an EnergyTerm for NonbondedMatrix by reading an input file.
    /// The input file must contain topology of the System and an `energy` section.
    pub fn nonbonded_from_file(filename: impl AsRef<Path> + Clone) -> anyhow::Result<Self> {
        let hamiltonian_builder = HamiltonianBuilder::from_file(filename.clone())?;
        let topology = Topology::from_file(filename)?;

        NonbondedMatrix::new(&hamiltonian_builder.nonbonded, &topology)
    }

    /// Update internal state due to a change in the system.
    fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match self {
            EnergyTerm::NonbondedMatrix(_) | EnergyTerm::IntramolecularBonded(_) => Ok(()),
            EnergyTerm::IntermolecularBonded(x) => x.update(context, change),
            EnergyTerm::SasaEnergy(x) => x.update(context, change),
        }
    }
}

impl EnergyChange for EnergyTerm {
    /// Compute the energy of the EnergyTerm relevant to the change in the system.
    /// The energy is returned in the units of kJ/mol.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match self {
            Self::NonbondedMatrix(x) => x.energy(context, change),
            Self::IntramolecularBonded(x) => x.energy(context, change),
            Self::IntermolecularBonded(x) => x.energy(context, change),
            Self::SasaEnergy(x) => x.energy(context, change),
        }
    }
}

impl SyncFrom for EnergyTerm {
    /// Synchronize the EnergyTerm from other EnergyTerm
    fn sync_from(&mut self, other: &EnergyTerm, change: &Change) -> anyhow::Result<()> {
        use EnergyTerm::*;
        match (self, other) {
            (NonbondedMatrix(x), NonbondedMatrix(y)) => x.sync_from(y, change)?,
            (IntramolecularBonded(_), IntramolecularBonded(_)) => (),
            (IntermolecularBonded(x), IntermolecularBonded(y)) => x.sync_from(y, change)?,
            (SasaEnergy(x), SasaEnergy(y)) => x.sync_from(y, change)?,
            _ => anyhow::bail!("Cannot sync incompatible energy terms."),
        }
        Ok(())
    }
}