use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    constrain::Constrain,
    external_pressure::ExternalPressure,
    nonbonded::{NonbondedMatrix, NonbondedMatrixSplined},
    sasa::SasaEnergy,
    CellOverlap, EnergyChange,
};
use crate::{Change, Context};

#[derive(Debug, Clone)]
pub enum EnergyTerm {
    /// Non-bonded interactions between particles.
    NonbondedMatrix(NonbondedMatrix),
    /// Non-bonded interactions using splined pair potentials.
    NonbondedMatrixSplined(NonbondedMatrixSplined),
    /// Intramolecular bonded interactions.
    IntramolecularBonded(IntramolecularBonded),
    /// Intermolecular bonded interactions.
    IntermolecularBonded(IntermolecularBonded),
    /// Solvent accessible surface area energy.
    SasaEnergy(SasaEnergy),
    /// Cell overlap energy.
    CellOverlap(CellOverlap),
    /// Collective variable constraint.
    Constrain(Constrain),
    /// External pressure (NPT ensemble).
    ExternalPressure(ExternalPressure),
}

impl EnergyTerm {
    /// Update internal state due to a change in the system.
    pub fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match self {
            Self::NonbondedMatrix(_)
            | Self::NonbondedMatrixSplined(_)
            | Self::IntramolecularBonded(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_) => Ok(()),
            Self::IntermolecularBonded(x) => x.update(context, change),
            Self::SasaEnergy(x) => x.update(context, change),
        }
    }
}

impl EnergyChange for EnergyTerm {
    /// Compute the energy of the EnergyTerm relevant to the change in the system.
    /// The energy is returned in the units of kJ/mol.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match self {
            Self::NonbondedMatrix(x) => x.energy(context, change),
            Self::NonbondedMatrixSplined(x) => x.energy(context, change),
            Self::IntramolecularBonded(x) => x.energy(context, change),
            Self::IntermolecularBonded(x) => x.energy(context, change),
            Self::SasaEnergy(x) => x.energy(context, change),
            Self::CellOverlap(x) => x.energy(context, change),
            Self::Constrain(x) => x.energy(context, change),
            Self::ExternalPressure(x) => x.energy(context, change),
        }
    }
}

impl EnergyTerm {
    /// Synchronize cached state from another energy term after an MC accept/reject step.
    pub(crate) fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()> {
        use EnergyTerm::*;
        match (self, other) {
            (NonbondedMatrix(x), NonbondedMatrix(y)) => x.sync_from(y, change),
            (NonbondedMatrixSplined(x), NonbondedMatrixSplined(y)) => x.sync_from(y, change),
            (IntramolecularBonded(_), IntramolecularBonded(_)) => Ok(()),
            (IntermolecularBonded(x), IntermolecularBonded(y)) => x.sync_from(y, change),
            (SasaEnergy(x), SasaEnergy(y)) => x.sync_from(y, change),
            (CellOverlap(_), CellOverlap(_)) => Ok(()),
            (Constrain(_), Constrain(_)) => Ok(()),
            (ExternalPressure(_), ExternalPressure(_)) => Ok(()),
            _ => anyhow::bail!("Cannot sync incompatible energy terms."),
        }
    }
}

impl From<SasaEnergy> for EnergyTerm {
    fn from(sasa: SasaEnergy) -> Self {
        Self::SasaEnergy(sasa)
    }
}

impl From<Constrain> for EnergyTerm {
    fn from(constrain: Constrain) -> Self {
        Self::Constrain(constrain)
    }
}
