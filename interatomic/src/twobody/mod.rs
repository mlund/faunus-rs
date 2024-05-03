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
//! Module for describing exactly two particles interacting with each other.

pub use crate::Vector3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

mod hardsphere;
mod harmonic;
mod mie;
mod multipole;
pub use self::hardsphere::HardSphere;
pub use self::harmonic::Harmonic;
pub use self::mie::{LennardJones, Mie, WeeksChandlerAndersen};
pub use self::multipole::{IonIon, IonIonPlain, IonIonYukawa};

/// Relative orientation between a pair of anisotropic particles
/// # Todo
/// Unfinished and still not desided how to implement
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RelativeOrientation {
    /// Distance between the two particles
    pub distance: Vector3,
    pub orientation: Vector3,
}

/// Potential energy between a pair of anisotropic particles
pub trait AnisotropicTwobodyEnergy: Debug {
    /// Interaction energy between a pair of anisotropic particles, 𝑈(𝒓).
    fn anisotropic_twobody_energy(&self, orientation: &RelativeOrientation) -> f64;

    /// Force magnitude due to an anisotropic interaction potential, 𝐹(𝒓) = -𝞩𝑈(𝒓)
    fn anisotropic_twobody_force(&self, _: &RelativeOrientation) -> Vector3 {
        todo!()
    }
}

/// Potential energy between a pair of isotropic particles, 𝑈(𝑟)
pub trait IsotropicTwobodyEnergy: Debug + AnisotropicTwobodyEnergy {
    /// Interaction energy between a pair of isotropic particles.
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64;

    /// Force magnitude due to an isotropic interaction potential, 𝐹(𝑟) = -∇𝑈(𝑟)
    ///
    /// The default implementation uses a central difference to calculate the force
    /// and should be overridden with the exact analytical expression for better speed
    /// and accuracy.
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        const EPS: f64 = 1e-6;
        let delta_u = self.isotropic_twobody_energy(distance_squared + EPS)
            - self.isotropic_twobody_energy(distance_squared - EPS);
        -delta_u / (2.0 * EPS)
    }
}

/// All isotropic potentials implement the anisotropic trait
impl<T: IsotropicTwobodyEnergy> AnisotropicTwobodyEnergy for T {
    fn anisotropic_twobody_energy(&self, orientation: &RelativeOrientation) -> f64 {
        self.isotropic_twobody_energy(orientation.distance.norm_squared())
    }
    fn anisotropic_twobody_force(&self, orientation: &RelativeOrientation) -> Vector3 {
        let r_squared = orientation.distance.norm_squared();
        let r_hat = orientation.distance / r_squared.sqrt();
        self.isotropic_twobody_force(r_squared) * r_hat
    }
}

/// Combine two twobody energy schemes
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Combined<T, U>(T, U);

impl<T: IsotropicTwobodyEnergy, U: IsotropicTwobodyEnergy> Combined<T, U> {
    pub fn new(t: T, u: U) -> Self {
        Self(t, u)
    }
}

impl<T: IsotropicTwobodyEnergy, U: IsotropicTwobodyEnergy> IsotropicTwobodyEnergy
    for Combined<T, U>
{
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        self.0.isotropic_twobody_energy(distance_squared)
            + self.1.isotropic_twobody_energy(distance_squared)
    }
}

/// Plain Coulomb potential combined with Lennard-Jones
pub type CoulombLennardJones<'a> = Combined<IonIon<'a, coulomb::pairwise::Plain>, LennardJones>;

/// Yukawa potential combined with Lennard-Jones
pub type YukawaLennardJones<'a> = Combined<IonIon<'a, coulomb::pairwise::Yukawa>, LennardJones>;

// test Combined
#[test]
pub fn test_combined() {
    use approx::assert_relative_eq;
    let r2 = 0.5;
    let lj = LennardJones::new(0.5, 1.0);
    let harmonic = Harmonic::new(0.0, 10.0);
    let u_lj = lj.isotropic_twobody_energy(r2);
    let u_harmonic = harmonic.isotropic_twobody_energy(r2);
    let combined = Combined::new(lj, harmonic);
    assert_relative_eq!(combined.isotropic_twobody_energy(r2), u_lj + u_harmonic);
}

/*
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
*/
