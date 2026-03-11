use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    constrain::Constrain,
    custom_external::CustomExternal,
    ewald::EwaldReciprocalEnergy,
    excluded_coulomb::ExcludedCoulomb,
    external_pressure::ExternalPressure,
    nonbonded::{NonbondedMatrix, NonbondedMatrixSplined},
    polymer_depletion::PolymerDepletion,
    sasa::SasaEnergy,
    tabulated6d::Tabulated6D,
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
    /// Ewald reciprocal-space electrostatic energy.
    EwaldReciprocal(Box<EwaldReciprocalEnergy>),
    /// Polymer depletion many-body interaction.
    PolymerDepletion(PolymerDepletion),
    /// Coulomb correction for excluded (bonded) pairs.
    ExcludedCoulomb(ExcludedCoulomb),
    /// Tabulated 6D rigid molecule-molecule energy.
    Tabulated6D(Tabulated6D),
}

/// Dispatch a no-arg method to stateful energy terms; stateless terms are no-ops.
/// Explicit variant listing ensures new variants trigger a compile error.
macro_rules! dispatch_stateful {
    ($self:expr, $method:ident) => {
        match $self {
            EnergyTerm::IntermolecularBonded(x) => x.$method(),
            EnergyTerm::SasaEnergy(x) => x.$method(),
            EnergyTerm::NonbondedMatrix(x) => x.$method(),
            EnergyTerm::NonbondedMatrixSplined(x) => x.$method(),
            EnergyTerm::EwaldReciprocal(x) => x.$method(),
            EnergyTerm::PolymerDepletion(x) => x.$method(),
            EnergyTerm::Tabulated6D(x) => x.$method(),
            EnergyTerm::IntramolecularBonded(_)
            | EnergyTerm::CellOverlap(_)
            | EnergyTerm::Constrain(_)
            | EnergyTerm::ExternalPressure(_)
            | EnergyTerm::CustomExternal(_)
            | EnergyTerm::ExcludedCoulomb(_) => {}
        }
    };
}

impl EnergyTerm {
    /// Update internal state due to a change in the system.
    pub fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match self {
            Self::NonbondedMatrix(x) => {
                x.update_cache(context, change);
                Ok(())
            }
            Self::NonbondedMatrixSplined(x) => {
                x.update_cache(context, change);
                Ok(())
            }
            Self::IntermolecularBonded(x) => x.update(context, change),
            Self::SasaEnergy(x) => x.update(context, change),
            Self::EwaldReciprocal(x) => x.update(context, change),
            Self::PolymerDepletion(x) => x.update(context, change),
            Self::Tabulated6D(x) => {
                x.update_cache(context, change); // mirrors NonbondedMatrix pattern
                Ok(())
            }
            Self::IntramolecularBonded(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_)
            | Self::ExcludedCoulomb(_) => Ok(()),
        }
    }

    /// Save internal state for later undo. Stateless terms are no-ops.
    ///
    /// Context is passed so that terms like Ewald can snapshot positions
    /// of affected particles before the move is applied.
    pub(crate) fn save_backup(&mut self, change: &Change, context: &impl Context) {
        match self {
            Self::IntermolecularBonded(x) => x.save_backup(change),
            Self::SasaEnergy(x) => x.save_backup(),
            Self::PolymerDepletion(x) => x.save_backup(),
            Self::NonbondedMatrix(x) => x.save_backup(change),
            Self::NonbondedMatrixSplined(x) => x.save_backup(change),
            Self::EwaldReciprocal(x) => x.save_backup(change, context),
            Self::Tabulated6D(x) => x.save_backup(change),
            Self::IntramolecularBonded(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_)
            | Self::ExcludedCoulomb(_) => {}
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

    /// Invalidate any internal energy caches (e.g. after Langevin dynamics
    /// has moved all molecules, making the pairwise cache stale).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn invalidate_cache(&mut self) {
        match self {
            Self::NonbondedMatrix(x) => x.invalidate_cache(),
            Self::NonbondedMatrixSplined(x) => x.invalidate_cache(),
            _ => {}
        }
    }

    /// Optional per-term information as YAML, for output reporting.
    pub fn to_yaml(&self) -> Option<serde_yaml::Value> {
        match self {
            Self::PolymerDepletion(x) => Some(x.to_yaml()),
            Self::EwaldReciprocal(x) => Some(x.to_yaml()),
            Self::ExternalPressure(x) => Some(x.to_yaml()),
            Self::CustomExternal(x) => Some(x.to_yaml()),
            Self::SasaEnergy(x) => Some(x.to_yaml()),
            Self::NonbondedMatrix(_)
            | Self::NonbondedMatrixSplined(_)
            | Self::IntramolecularBonded(_)
            | Self::IntermolecularBonded(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExcludedCoulomb(_)
            | Self::Tabulated6D(_) => None,
        }
    }

    /// Compute per-atom forces contributed by this term.
    ///
    /// Returns a dense vector indexed by absolute particle index.
    /// Terms that do not contribute forces return an empty vector.
    pub(crate) fn forces(&self, context: &impl Context) -> Vec<crate::Point> {
        match self {
            Self::NonbondedMatrix(x) => x.forces(context),
            Self::NonbondedMatrixSplined(x) => x.forces(context),
            Self::IntramolecularBonded(_)
            | Self::IntermolecularBonded(_)
            | Self::SasaEnergy(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_)
            | Self::EwaldReciprocal(_)
            | Self::PolymerDepletion(_)
            | Self::ExcludedCoulomb(_)
            | Self::Tabulated6D(_) => Vec::new(),
        }
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
            Self::EwaldReciprocal(_) => "ewald_reciprocal",
            Self::PolymerDepletion(_) => "polymer_depletion",
            Self::ExcludedCoulomb(_) => "excluded_coulomb",
            Self::Tabulated6D(_) => "tabulated6d",
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
            Self::EwaldReciprocal(x) => x.energy(context, change),
            Self::PolymerDepletion(x) => x.energy(context, change),
            Self::ExcludedCoulomb(x) => x.energy(context, change),
            Self::Tabulated6D(x) => x.energy(context, change),
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

impl From<PolymerDepletion> for EnergyTerm {
    fn from(pm: PolymerDepletion) -> Self {
        Self::PolymerDepletion(pm)
    }
}

impl From<Tabulated6D> for EnergyTerm {
    fn from(t: Tabulated6D) -> Self {
        Self::Tabulated6D(t)
    }
}
