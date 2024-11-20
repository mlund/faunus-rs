use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    nonbonded::NonbondedMatrix,
    sasa::SasaEnergy,
    EnergyChange,
};
use crate::{Change, Context, SyncFrom};

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
    /// Update internal state due to a change in the system.
    pub fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
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
