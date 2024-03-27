// Copyright 2023 Mikael Lund
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

//! ## Twobody interactions
//!
//! Module for describing exactly two particle interacting with each other.
//!
//! - Hard-sphere overlap
//! - Harmonic potential
//! - Powerlaw potentials
//!   - Mie
//!   - Lennard-Jones
//!   - Weeks-Chandler-Andersen

use crate::Info;
use serde::{Deserialize, Serialize};

mod electrostatic;
mod hardsphere;
mod harmonic;
mod mie;
pub use self::electrostatic::IonIon;
pub use self::hardsphere::HardSphere;
pub use self::harmonic::Harmonic;
pub use self::mie::{LennardJones, Mie, WeeksChandlerAndersen};

/// Potential energy between a pair of isotropic particles
///
/// This uses the `typetag` crate to allow for dynamic dispatch
/// and requires that implementations are tagged with `#[typetag::serialize]`.
#[typetag::serialize(tag = "type")]
pub trait TwobodyEnergy: crate::Info + std::fmt::Debug {
    /// Interaction energy between a pair of isotropic particles
    fn twobody_energy(&self, distance_squared: f64) -> f64;
}

/// Combine twobody energies
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Combined<T, U>(T, U);

impl<T: TwobodyEnergy, U: TwobodyEnergy> Combined<T, U> {
    pub fn new(t: T, u: U) -> Self {
        Self(t, u)
    }
}

#[typetag::serialize]
impl<T: TwobodyEnergy + Serialize, U: TwobodyEnergy + Serialize> TwobodyEnergy for Combined<T, U> {
    #[inline]
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
        self.0.twobody_energy(distance_squared) + self.1.twobody_energy(distance_squared)
    }
}

impl<T: TwobodyEnergy, U: TwobodyEnergy> Info for Combined<T, U> {
    fn citation(&self) -> Option<&'static str> {
        todo!("Implement citation for Combined");
    }
}

/// Enum with all two-body variants.
///
/// Use for serialization and deserialization of two-body interactions in
/// e.g. user input.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TwobodyKind {
    HardSphere(HardSphere),
    Harmonic(Harmonic),
    #[serde(rename = "lj")]
    LennardJones(LennardJones),
    #[serde(rename = "wca")]
    WeeksChandlerAndersen(WeeksChandlerAndersen),
}

// Test TwobodyKind for serialization
#[test]
fn test_twobodykind_serialize() {
    let hardsphere = TwobodyKind::HardSphere(HardSphere::new(1.0));
    assert_eq!(
        serde_json::to_string(&hardsphere).unwrap(),
        "{\"hardsphere\":{\"σ\":1.0}}"
    );

    let harmonic = TwobodyKind::Harmonic(Harmonic::new(1.0, 0.5));
    assert_eq!(
        serde_json::to_string(&harmonic).unwrap(),
        "{\"harmonic\":{\"r₀\":1.0,\"k\":0.5}}"
    );

    let lj = TwobodyKind::LennardJones(LennardJones::new(0.1, 2.5));
    assert_eq!(
        serde_json::to_string(&lj).unwrap(),
        "{\"lj\":{\"ε\":0.1,\"σ\":2.5}}"
    );

    let lennard_jones = LennardJones::new(0.1, 2.5);
    let wca = TwobodyKind::WeeksChandlerAndersen(WeeksChandlerAndersen::new(lennard_jones));
    assert_eq!(
        serde_json::to_string(&wca).unwrap(),
        "{\"wca\":{\"ε\":0.1,\"σ\":2.5}}"
    );
}
