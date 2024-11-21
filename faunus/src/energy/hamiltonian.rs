use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    builder::HamiltonianBuilder,
    nonbonded::NonbondedMatrix,
    EnergyTerm,
};
use crate::{topology::Topology, Change, Context, SyncFrom};

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
        let nonbonded = NonbondedMatrix::new(&builder.pairpot_builder, topology)?;
        let intramolecular_bonded = IntramolecularBonded::make_energy();

        let mut hamiltonian = Hamiltonian::from(vec![nonbonded.into(), intramolecular_bonded]);

        // IntermolecularBonded term should only be added if it is actually needed
        if !topology.intermolecular().is_empty() {
            hamiltonian.add_energy_term(IntermolecularBonded::new(topology));
        }

        Ok(hamiltonian)
    }

    // /// Create a Hamiltonian from the provided energy terms.
    // pub(crate) fn from_energy_terms(terms: Vec<EnergyTerm>) -> Self {
    //     Hamiltonian {
    //         energy_terms: terms,
    //     }
    // }

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

impl From<Vec<EnergyTerm>> for Hamiltonian {
    fn from(energy_terms: Vec<EnergyTerm>) -> Self {
        Self { energy_terms }
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
