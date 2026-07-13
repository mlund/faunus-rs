use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    constrain::Constrain,
    contact_tessellation::ContactTessellationEnergy,
    custom_external::CustomExternal,
    custom_pair::CustomPair,
    ewald::EwaldReciprocalEnergy,
    excluded_coulomb::ExcludedCoulomb,
    external_pressure::ExternalPressure,
    nonbonded::{NonbondedMatrix, NonbondedMatrixSplined},
    penalty::Penalty,
    polymer_depletion::PolymerDepletion,
    sasa::SasaEnergy,
    stateful::StatefulEnergy,
    tabulated::TabulatedEnergy,
    CellOverlap, EnergyChange,
};
use crate::Change;
use crate::ObserveContext;

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
    SasaEnergy(Box<SasaEnergy>),
    /// Cell overlap energy.
    CellOverlap(CellOverlap),
    /// Collective variable constraint.
    Constrain(Constrain),
    /// External pressure (NPT ensemble).
    ExternalPressure(ExternalPressure),
    /// Custom external potential from math expression.
    CustomExternal(CustomExternal),
    /// User-defined energy/force between two rigid-body centers of mass.
    CustomPair(CustomPair),
    /// Ewald reciprocal-space electrostatic energy.
    EwaldReciprocal(Box<EwaldReciprocalEnergy>),
    /// Polymer depletion many-body interaction.
    PolymerDepletion(PolymerDepletion),
    /// Coulomb correction for excluded (bonded) pairs.
    ExcludedCoulomb(ExcludedCoulomb),
    /// Tabulated rigid-body energy (6D molecule-molecule and 3D molecule-atom).
    Tabulated(TabulatedEnergy),
    /// Flat-histogram bias penalty on collective variable(s).
    Penalty(Penalty),
    /// Contact tessellation energy between rigid bodies.
    ContactTessellation(ContactTessellationEnergy),
}

/// Dispatch a unit-returning [`StatefulEnergy`] method to the stateful terms; stateless
/// terms are no-ops. Explicit variant listing ensures a new variant triggers a compile error.
macro_rules! dispatch_stateful {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            EnergyTerm::IntermolecularBonded(x) => x.$method($($arg),*),
            EnergyTerm::SasaEnergy(x) => x.$method($($arg),*),
            EnergyTerm::ContactTessellation(x) => x.$method($($arg),*),
            EnergyTerm::NonbondedMatrix(x) => x.$method($($arg),*),
            EnergyTerm::NonbondedMatrixSplined(x) => x.$method($($arg),*),
            EnergyTerm::EwaldReciprocal(x) => x.$method($($arg),*),
            EnergyTerm::PolymerDepletion(x) => x.$method($($arg),*),
            EnergyTerm::Tabulated(x) => x.$method($($arg),*),
            EnergyTerm::IntramolecularBonded(_)
            | EnergyTerm::CellOverlap(_)
            | EnergyTerm::Constrain(_)
            | EnergyTerm::ExternalPressure(_)
            | EnergyTerm::CustomExternal(_)
            | EnergyTerm::CustomPair(_)
            | EnergyTerm::ExcludedCoulomb(_)
            | EnergyTerm::Penalty(_) => {}
        }
    };
}

/// Dispatch a method that every variant forwards identically, passing `$args`.
/// Unlike [`dispatch_stateful`] there is no stateless subset: the compiler
/// rejects a new variant until it is added here.
macro_rules! dispatch_all {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            EnergyTerm::NonbondedMatrix(x) => x.$method($($arg),*),
            EnergyTerm::NonbondedMatrixSplined(x) => x.$method($($arg),*),
            EnergyTerm::IntramolecularBonded(x) => x.$method($($arg),*),
            EnergyTerm::IntermolecularBonded(x) => x.$method($($arg),*),
            EnergyTerm::SasaEnergy(x) => x.$method($($arg),*),
            EnergyTerm::CellOverlap(x) => x.$method($($arg),*),
            EnergyTerm::Constrain(x) => x.$method($($arg),*),
            EnergyTerm::ExternalPressure(x) => x.$method($($arg),*),
            EnergyTerm::CustomExternal(x) => x.$method($($arg),*),
            EnergyTerm::CustomPair(x) => x.$method($($arg),*),
            EnergyTerm::EwaldReciprocal(x) => x.$method($($arg),*),
            EnergyTerm::PolymerDepletion(x) => x.$method($($arg),*),
            EnergyTerm::ExcludedCoulomb(x) => x.$method($($arg),*),
            EnergyTerm::Tabulated(x) => x.$method($($arg),*),
            EnergyTerm::Penalty(x) => x.$method($($arg),*),
            EnergyTerm::ContactTessellation(x) => x.$method($($arg),*),
        }
    };
}

/// Generate `From<Concrete> for EnergyTerm` wrappers. Prefix an entry with
/// `boxed` when the variant stores a `Box`.
macro_rules! impl_energyterm_from {
    (boxed $variant:ident($ty:ty), $($rest:tt)*) => {
        impl From<$ty> for EnergyTerm {
            fn from(value: $ty) -> Self {
                Self::$variant(Box::new(value))
            }
        }
        impl_energyterm_from!($($rest)*);
    };
    ($variant:ident($ty:ty), $($rest:tt)*) => {
        impl From<$ty> for EnergyTerm {
            fn from(value: $ty) -> Self {
                Self::$variant(value)
            }
        }
        impl_energyterm_from!($($rest)*);
    };
    () => {};
}

impl_energyterm_from! {
    boxed SasaEnergy(SasaEnergy),
    boxed EwaldReciprocal(EwaldReciprocalEnergy),
    CellOverlap(CellOverlap),
    Constrain(Constrain),
    ExternalPressure(ExternalPressure),
    CustomExternal(CustomExternal),
    CustomPair(CustomPair),
    PolymerDepletion(PolymerDepletion),
    ExcludedCoulomb(ExcludedCoulomb),
    Tabulated(TabulatedEnergy),
    Penalty(Penalty),
    ContactTessellation(ContactTessellationEnergy),
}

impl EnergyTerm {
    /// Update internal state due to a change in the system.
    pub(crate) fn update(
        &mut self,
        context: &impl ObserveContext,
        change: &Change,
    ) -> anyhow::Result<()> {
        match self {
            Self::NonbondedMatrix(x) => x.refresh(context, change),
            Self::NonbondedMatrixSplined(x) => x.refresh(context, change),
            Self::IntermolecularBonded(x) => x.refresh(context, change),
            Self::SasaEnergy(x) => x.refresh(context, change),
            Self::EwaldReciprocal(x) => x.refresh(context, change),
            Self::PolymerDepletion(x) => x.refresh(context, change),
            Self::Tabulated(x) => x.refresh(context, change),
            Self::ContactTessellation(x) => x.refresh(context, change),
            Self::IntramolecularBonded(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_)
            | Self::CustomPair(_)
            | Self::ExcludedCoulomb(_)
            | Self::Penalty(_) => Ok(()),
        }
    }

    /// Save internal state for later undo. Stateless terms are no-ops.
    ///
    /// The context is passed so that terms like Ewald can snapshot positions
    /// of affected particles before the move is applied.
    pub(crate) fn save_backup(&mut self, change: &Change, context: &impl ObserveContext) {
        dispatch_stateful!(self, save_backup, context, change);
    }

    /// Restore from internal backup (reject path).
    pub fn undo(&mut self) {
        dispatch_stateful!(self, undo);
    }

    /// Drop internal backup (accept path).
    pub fn discard_backup(&mut self) {
        dispatch_stateful!(self, discard_backup);
    }

    /// Exclude a molecule-type pair from the nonbonded energy term.
    pub(crate) fn exclude_molecule_pair(
        &mut self,
        mol_a: crate::group::MoleculeId,
        mol_b: crate::group::MoleculeId,
    ) {
        // Only the nonbonded matrix terms carry molecule-pair exclusions; every other term
        // correctly ignores this, so a new variant defaulting to a no-op is intended.
        #[allow(clippy::wildcard_enum_match_arm)]
        match self {
            Self::NonbondedMatrix(x) => x.exclude_molecule_pair(mol_a, mol_b),
            Self::NonbondedMatrixSplined(x) => x.exclude_molecule_pair(mol_a, mol_b),
            _ => {}
        }
    }

    /// Get molecule-type pairs excluded from nonbonded, if applicable.
    #[must_use]
    pub(crate) fn molecule_pair_exclusions(&self) -> Option<&[[crate::group::MoleculeId; 2]]> {
        #[allow(clippy::wildcard_enum_match_arm)] // only nonbonded terms expose exclusions
        match self {
            Self::NonbondedMatrix(x) => Some(x.molecule_pair_exclusions()),
            Self::NonbondedMatrixSplined(x) => Some(x.molecule_pair_exclusions()),
            _ => None,
        }
    }

    /// Invalidate any internal energy caches (e.g. after Langevin dynamics
    /// has moved all molecules, making the pairwise cache stale).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn invalidate_cache(&mut self) {
        #[allow(clippy::wildcard_enum_match_arm)]
        // only nonbonded terms hold an invalidatable cache
        match self {
            Self::NonbondedMatrix(x) => x.invalidate_cache(),
            Self::NonbondedMatrixSplined(x) => x.invalidate_cache(),
            _ => {}
        }
    }

    /// Optional per-term information as YAML, for output reporting.
    pub fn to_yaml(&self) -> Option<serde_yml::Value> {
        match self {
            Self::PolymerDepletion(x) => Some(x.to_yaml()),
            Self::EwaldReciprocal(x) => Some(x.to_yaml()),
            Self::ExternalPressure(x) => Some(x.to_yaml()),
            Self::CustomExternal(x) => Some(x.to_yaml()),
            Self::CustomPair(x) => Some(x.to_yaml()),
            Self::SasaEnergy(x) => Some(x.to_yaml()),
            Self::ContactTessellation(x) => Some(x.to_yaml()),
            Self::NonbondedMatrix(_)
            | Self::NonbondedMatrixSplined(_)
            | Self::IntramolecularBonded(_)
            | Self::IntermolecularBonded(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExcludedCoulomb(_)
            | Self::Penalty(_) => None,
            Self::Tabulated(x) => Some(x.to_yaml()),
        }
    }

    /// Compute per-atom forces contributed by this term.
    ///
    /// Returns a dense vector indexed by absolute particle index.
    /// Terms that do not contribute forces return an empty vector.
    pub(crate) fn forces(&self, context: &impl ObserveContext) -> Vec<crate::Point> {
        match self {
            Self::NonbondedMatrix(x) => x.forces(context),
            Self::NonbondedMatrixSplined(x) => x.forces(context),
            Self::CustomPair(x) => x.forces(context),
            Self::IntramolecularBonded(_)
            | Self::IntermolecularBonded(_)
            | Self::SasaEnergy(_)
            | Self::ContactTessellation(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_)
            | Self::EwaldReciprocal(_)
            | Self::PolymerDepletion(_)
            | Self::ExcludedCoulomb(_)
            | Self::Tabulated(_)
            | Self::Penalty(_) => Vec::new(),
        }
    }

    /// True when the GPU Langevin pipeline computes this term's forces on-device,
    /// so the host must not recompute and double-count them in the overlay path.
    /// Bonded variants are listed even though their CPU `forces()` arms are empty
    /// today — they are computed inside the GPU loop.
    pub(super) fn handled_by_gpu_ld(&self) -> bool {
        matches!(
            self,
            Self::NonbondedMatrix(_)
                | Self::NonbondedMatrixSplined(_)
                | Self::IntramolecularBonded(_)
                | Self::IntermolecularBonded(_)
        )
    }

    /// True for variants whose `forces()` arm returns a non-empty vector.
    /// Used by the GPU LD pre-flight to decide whether to plumb an overlay
    /// callback. Keep in sync with the `forces()` match arms above.
    pub(super) fn contributes_force(&self) -> bool {
        matches!(
            self,
            Self::NonbondedMatrix(_) | Self::NonbondedMatrixSplined(_) | Self::CustomPair(_)
        )
    }

    /// Nonbonded energy between two sets of atom indices; `None` for non-nonbonded terms.
    pub(crate) fn nonbonded_energy_between_atoms(
        &self,
        context: &impl ObserveContext,
        atoms1: &[crate::group::AbsIndex],
        atoms2: &[crate::group::AbsIndex],
    ) -> Option<f64> {
        #[allow(clippy::wildcard_enum_match_arm)] // only nonbonded terms score index-pair energies
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
            Self::CellOverlap(_) => "cell_overlap",
            Self::Constrain(_) => "constrain",
            Self::ExternalPressure(_) => "external_pressure",
            Self::CustomExternal(_) => "custom_external",
            Self::CustomPair(_) => "custom_pair",
            Self::EwaldReciprocal(_) => "ewald_reciprocal",
            Self::PolymerDepletion(_) => "polymer_depletion",
            Self::ExcludedCoulomb(_) => "excluded_coulomb",
            Self::Tabulated(_) => "tabulated",
            Self::Penalty(_) => "penalty",
            Self::ContactTessellation(_) => "contact_tessellation",
        })
    }
}

impl EnergyChange for EnergyTerm {
    /// Compute the energy of the EnergyTerm relevant to the change in the system.
    /// The energy is returned in the units of kJ/mol.
    fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        dispatch_all!(self, energy, context, change)
    }
}
