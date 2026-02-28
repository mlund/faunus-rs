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

/// Dispatch a no-arg method to stateful energy terms; stateless terms are no-ops.
/// Explicit variant listing ensures new variants trigger a compile error.
macro_rules! dispatch_stateful {
    ($self:expr, $method:ident) => {
        match $self {
            EnergyTerm::IntermolecularBonded(x) => x.$method(),
            EnergyTerm::SasaEnergy(x) => x.$method(),
            EnergyTerm::NonbondedMatrix(_)
            | EnergyTerm::NonbondedMatrixSplined(_)
            | EnergyTerm::IntramolecularBonded(_)
            | EnergyTerm::CellOverlap(_)
            | EnergyTerm::Constrain(_)
            | EnergyTerm::ExternalPressure(_) => {}
        }
    };
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

    /// Save internal state for later undo. Stateless terms are no-ops.
    pub(crate) fn save_backup(&mut self, change: &Change) {
        match self {
            Self::IntermolecularBonded(x) => x.save_backup(change),
            Self::SasaEnergy(x) => x.save_backup(),
            Self::NonbondedMatrix(_)
            | Self::NonbondedMatrixSplined(_)
            | Self::IntramolecularBonded(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_) => {}
        }
    }

    /// Restore from internal backup (reject path).
    pub fn undo(&mut self) {
        dispatch_stateful!(self, undo);
    }

    /// Drop internal backup (accept path).
    pub fn discard_backup(&mut self) {
        dispatch_stateful!(self, discard_backup);
    }

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
