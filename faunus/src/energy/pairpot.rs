// Copyright 2023-2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Statically-dispatched pair potential with enum-based inlining.
//!
//! [`PairPot`] wraps short-range and electrostatic components as small enums
//! so the compiler can inline potential evaluation in the inner loop
//! while avoiding monomorphization bloat from generic type parameters.

use interatomic::twobody::{
    ArcPotential, AshbaughHatch, HardSphere, IonIon, IsotropicTwobodyEnergy, KimHummer,
    LennardJones, WeeksChandlerAndersen,
};
use interatomic::Cutoff;

/// Short-range (non-electrostatic) pair potential variants.
#[derive(Clone, Debug)]
pub(crate) enum ShortRange {
    None,
    LennardJones(LennardJones),
    Wca(WeeksChandlerAndersen),
    AshbaughHatch(AshbaughHatch),
    KimHummer(KimHummer),
    HardSphere(HardSphere),
    /// Fallback for unrecognized or composite short-range potentials.
    Dynamic(ArcPotential),
}

/// Electrostatic pair potential variants.
#[derive(Clone, Debug)]
pub(crate) enum Coulomb {
    None,
    Plain(IonIon<interatomic::coulomb::pairwise::Plain>),
    ReactionField(IonIon<interatomic::coulomb::pairwise::ReactionField>),
    Ewald(IonIon<interatomic::coulomb::pairwise::EwaldTruncated>),
    RealSpaceEwald(IonIon<interatomic::coulomb::pairwise::RealSpaceEwald>),
    Fanourgakis(IonIon<interatomic::coulomb::pairwise::Fanourgakis>),
    /// Fallback for unrecognized or composite electrostatic potentials.
    Dynamic(ArcPotential),
}

/// Pair potential with enum-based dispatch for the inner loop.
///
/// Combines a short-range and an electrostatic component, each classified
/// into small enums for branch-predicted inline evaluation. Unrecognized
/// types fall back to dynamic dispatch via [`ArcPotential`].
#[derive(Clone, Debug)]
pub struct PairPot {
    short_range: ShortRange,
    coulomb: Coulomb,
}

impl Default for PairPot {
    fn default() -> Self {
        Self {
            short_range: ShortRange::None,
            coulomb: Coulomb::None,
        }
    }
}

impl PairPot {
    pub(crate) fn from_parts(short_range: ShortRange, coulomb: Coulomb) -> Self {
        Self {
            short_range,
            coulomb,
        }
    }
}

// ─── Dispatch helpers ─────────────────────────────────────────────────────────

macro_rules! dispatch_short_range {
    ($self:expr, $method:ident, $none_val:expr, $($arg:expr),*) => {
        match &$self {
            ShortRange::None => $none_val,
            ShortRange::LennardJones(p) => p.$method($($arg),*),
            ShortRange::Wca(p) => p.$method($($arg),*),
            ShortRange::AshbaughHatch(p) => p.$method($($arg),*),
            ShortRange::KimHummer(p) => p.$method($($arg),*),
            ShortRange::HardSphere(p) => p.$method($($arg),*),
            ShortRange::Dynamic(p) => p.$method($($arg),*),
        }
    };
}

macro_rules! dispatch_coulomb {
    ($self:expr, $method:ident, $none_val:expr, $($arg:expr),*) => {
        match &$self {
            Coulomb::None => $none_val,
            Coulomb::Plain(p) => p.$method($($arg),*),
            Coulomb::ReactionField(p) => p.$method($($arg),*),
            Coulomb::Ewald(p) => p.$method($($arg),*),
            Coulomb::RealSpaceEwald(p) => p.$method($($arg),*),
            Coulomb::Fanourgakis(p) => p.$method($($arg),*),
            Coulomb::Dynamic(p) => p.$method($($arg),*),
        }
    };
}

// ─── Cutoff ───────────────────────────────────────────────────────────────────

impl Cutoff for ShortRange {
    fn cutoff(&self) -> f64 {
        dispatch_short_range!(self, cutoff, f64::INFINITY,)
    }
    fn lower_cutoff(&self) -> f64 {
        dispatch_short_range!(self, lower_cutoff, 0.0,)
    }
}

impl Cutoff for Coulomb {
    fn cutoff(&self) -> f64 {
        dispatch_coulomb!(self, cutoff, f64::INFINITY,)
    }
    fn lower_cutoff(&self) -> f64 {
        dispatch_coulomb!(self, lower_cutoff, 0.0,)
    }
}

impl Cutoff for PairPot {
    fn cutoff(&self) -> f64 {
        self.short_range.cutoff().min(self.coulomb.cutoff())
    }

    /// Matches `Combined` behavior: use the larger lower cutoff.
    fn lower_cutoff(&self) -> f64 {
        self.short_range
            .lower_cutoff()
            .max(self.coulomb.lower_cutoff())
    }
}

// ─── IsotropicTwobodyEnergy ───────────────────────────────────────────────────

impl IsotropicTwobodyEnergy for ShortRange {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        dispatch_short_range!(self, isotropic_twobody_energy, 0.0, distance_squared)
    }

    #[inline(always)]
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        dispatch_short_range!(self, isotropic_twobody_force, 0.0, distance_squared)
    }
}

impl IsotropicTwobodyEnergy for PairPot {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        dispatch_short_range!(
            self.short_range,
            isotropic_twobody_energy,
            0.0,
            distance_squared
        ) + dispatch_coulomb!(
            self.coulomb,
            isotropic_twobody_energy,
            0.0,
            distance_squared
        )
    }

    #[inline(always)]
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        dispatch_short_range!(
            self.short_range,
            isotropic_twobody_force,
            0.0,
            distance_squared
        ) + dispatch_coulomb!(self.coulomb, isotropic_twobody_force, 0.0, distance_squared)
    }
}
