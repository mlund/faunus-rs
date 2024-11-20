use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    builder::HamiltonianBuilder,
    nonbonded::NonbondedMatrix,
    sasa::SasaEnergy,
    EnergyChange,
};
use crate::{topology::Topology, Change, Context, SyncFrom};
use std::path::Path;

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

        NonbondedMatrix::make_energy(&hamiltonian_builder.nonbonded, &topology)
    }

    /// Update internal state due to a change in the system.
    pub(crate) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
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
