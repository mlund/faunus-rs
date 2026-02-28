use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    constrain::Constrain,
    custom_external::CustomExternal,
    external_pressure::ExternalPressure,
    nonbonded::{NonbondedMatrix, NonbondedMatrixSplined, NonbondedTerm},
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
    /// Custom external potential from math expression.
    CustomExternal(CustomExternal),
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
            | EnergyTerm::ExternalPressure(_)
            | EnergyTerm::CustomExternal(_) => {}
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
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_) => Ok(()),
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
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_) => {}
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

    /// Nonbonded energy between two sets of atom indices; `None` for non-nonbonded terms.
    pub fn nonbonded_energy_between_atoms(
        &self,
        context: &impl Context,
        atoms1: &[usize],
        atoms2: &[usize],
    ) -> Option<f64> {
        match self {
            Self::NonbondedMatrix(nb) => Some(nb.indices_with_indices(context, atoms1, atoms2)),
            Self::NonbondedMatrixSplined(nb) => {
                Some(nb.indices_with_indices(context, atoms1, atoms2))
            }
            _ => None,
        }
    }
}

impl crate::Info for EnergyTerm {
    fn short_name(&self) -> Option<&'static str> {
        Some(match self {
            Self::NonbondedMatrix(_) | Self::NonbondedMatrixSplined(_) => "nonbonded",
            Self::IntramolecularBonded(_) => "intramolecular",
            Self::IntermolecularBonded(_) => "intermolecular",
            Self::SasaEnergy(_) => "sasa",
            Self::CellOverlap(_) => "celloverlap",
            Self::Constrain(_) => "constrain",
            Self::ExternalPressure(_) => "externalpressure",
            Self::CustomExternal(_) => "customexternal",
        })
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
            Self::CustomExternal(x) => x.energy(context, change),
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
