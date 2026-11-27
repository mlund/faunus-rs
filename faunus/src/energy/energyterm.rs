use super::{
    akesson::Akesson,
    bonded::{self, IntermolecularBonded, IntramolecularBonded},
    constrain::Constrain,
    contact_tessellation::ContactTessellationEnergy,
    coulomb_self::CoulombSelfEnergy,
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
use crate::topology::DegreesOfFreedom;
use crate::Change;
use crate::ObserveContext;

/// Source of an energy term's forces under Langevin dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LdForces {
    /// Computed by the GPU pipeline.
    Device,
    /// Computed on the host each step and added as a COM force/torque overlay.
    Host,
    /// Energy cannot change during LD, so the force is zero.
    Constant,
    /// No forces; LD would sample a different Hamiltonian than MC.
    Unsupported,
}

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
    /// Mean field of the charge outside a slit's minimum-image box.
    Akesson(Akesson),
    /// Self energy of each charge under a truncated pair Coulomb scheme.
    CoulombSelf(CoulombSelfEnergy),
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
            | EnergyTerm::Akesson(_)
            | EnergyTerm::CoulombSelf(_)
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
            EnergyTerm::Akesson(x) => x.$method($($arg),*),
            EnergyTerm::CoulombSelf(x) => x.$method($($arg),*),
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
    Akesson(Akesson),
    CoulombSelf(CoulombSelfEnergy),
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
            | Self::Akesson(_)
            | Self::CoulombSelf(_)
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

    /// Optional per-term information as YAML, for output reporting.
    pub fn to_yaml(&self) -> Option<yaml_serde::Value> {
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
            | Self::Akesson(_)
            | Self::CoulombSelf(_)
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
            Self::EwaldReciprocal(x) => x.forces(context),
            Self::IntramolecularBonded(_)
            | Self::IntermolecularBonded(_)
            | Self::SasaEnergy(_)
            | Self::ContactTessellation(_)
            | Self::CellOverlap(_)
            | Self::Constrain(_)
            | Self::ExternalPressure(_)
            | Self::CustomExternal(_)
            | Self::Akesson(_)
            | Self::CoulombSelf(_)
            | Self::PolymerDepletion(_)
            | Self::ExcludedCoulomb(_)
            | Self::Tabulated(_)
            | Self::Penalty(_) => Vec::new(),
        }
    }

    /// Where Langevin dynamics gets this term's forces. `on_device` is true when the
    /// GPU computes pair and bonded forces, i.e. a splined nonbonded term exists;
    /// otherwise LD sums `forces()` on the host. No wildcard, so a new variant
    /// cannot reach LD without a decision. The Hamiltonian refuses `Host` when
    /// any molecule is flexible.
    pub(super) fn ld_forces(&self, context: &impl ObserveContext, on_device: bool) -> LdForces {
        let topology = context.topology_ref();
        // Inserted molecule kinds whose internal coordinates LD changes
        let flexible = || {
            context
                .groups()
                .iter()
                .map(|g| topology.moleculekind(g.molecule()))
                .filter(|m| m.degrees_of_freedom() == DegreesOfFreedom::Free)
        };
        // Bonded `forces()` arms are empty, so only the GPU kernels supply them
        let bonded = |supported: bool| {
            if on_device && supported {
                LdForces::Device
            } else {
                LdForces::Unsupported
            }
        };
        match self {
            Self::NonbondedMatrixSplined(_) => LdForces::Device,
            // Only reachable on the host path: the GPU has no unsplined pair kernel
            Self::NonbondedMatrix(_) | Self::CustomPair(_) => LdForces::Host,
            Self::IntramolecularBonded(_) => {
                if !flexible().any(|m| m.has_bonded_potentials()) {
                    LdForces::Constant
                } else {
                    bonded(flexible().all(|m| {
                        bonded::device_bonded_forces(m.bonds(), m.torsions(), m.dihedrals())
                    }))
                }
            }
            Self::IntermolecularBonded(_) => {
                let inter = topology.intermolecular();
                bonded(bonded::device_bonded_forces(
                    inter.bonds(),
                    inter.torsions(),
                    inter.dihedrals(),
                ))
            }
            // LD runs at fixed volume in a periodic cube, so neither can change
            Self::ExternalPressure(_) | Self::CellOverlap(_) => LdForces::Constant,
            // LD changes neither charges nor particle counts
            Self::CoulombSelf(_) => LdForces::Constant,
            // Excluded pairs only move apart in flexible molecules
            Self::ExcludedCoulomb(_) => {
                if flexible().any(|m| m.keep_excluded_coulomb() && !m.exclusions().is_empty()) {
                    LdForces::Unsupported
                } else {
                    LdForces::Constant
                }
            }
            Self::EwaldReciprocal(x) => x.ld_forces(on_device),
            Self::SasaEnergy(_)
            | Self::ContactTessellation(_)
            | Self::Constrain(_)
            | Self::CustomExternal(_)
            | Self::Akesson(_)
            | Self::PolymerDepletion(_)
            | Self::Tabulated(_)
            | Self::Penalty(_) => LdForces::Unsupported,
        }
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
            Self::Akesson(_) => "akesson",
            Self::CoulombSelf(_) => "coulomb_self",
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
