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
//! Module for describing twobody interactions.
//!
//! - Hard-sphere overlap
//! - Harmonic potential
//! - Powerlaw potentials
//!   - Mie
//!   - Lennard-Jones
//!   - Weeks-Chandler-Andersen

use crate::{sqrt_serialize, square_deserialize, Info};
use serde::{Deserialize, Serialize};

mod mie;
pub use self::mie::{LennardJones, Mie, WeeksChandlerAndersen};

/// Potential energy between a pair of particles
///
/// This uses the `typetag` crate to allow for dynamic dispatch
/// and requires that implementations are tagged with `#[typetag::serialize]`.
#[typetag::serialize(tag = "type")]
pub trait TwobodyEnergy: crate::Info + std::fmt::Debug {
    /// Interaction energy between a pair of isotropic particles
    fn twobody_energy(&self, distance_squared: f64) -> f64;
}

/// Helper struct to combine twobody energies
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

/// Enum to describe two-body interaction variants.
/// Use for serialization and deserialization of two-body interactions in user input.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
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
    let harmonic = TwobodyKind::Harmonic(Harmonic::new(1.0, 0.5));
    let serialized = serde_json::to_string(&harmonic).unwrap();
    assert_eq!(serialized, "{\"harmonic\":{\"r₀\":1.0,\"k\":0.5}}");

    let lennard_jones = LennardJones::new(0.1, 2.5);
    let weekschandlerandersen = TwobodyKind::WeeksChandlerAndersen(WeeksChandlerAndersen::new(lennard_jones));
    let serialized = serde_json::to_string(&weekschandlerandersen).unwrap();
    assert_eq!(serialized, "{\"wca\":{\"ε\":0.1,\"σ\":2.5}}");
}

/// Hardsphere potential
///
/// More information [here](http://www.sklogwiki.org/SklogWiki/index.php/Hard_sphere_model).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct HardSphere {
    /// Minimum distance
    #[serde(
        rename = "σ",
        serialize_with = "sqrt_serialize",
        deserialize_with = "square_deserialize"
    )]
    min_distance_squared: f64,
}

impl HardSphere {
    pub fn new(min_distance: f64) -> Self {
        Self {
            min_distance_squared: min_distance.powi(2),
        }
    }
}

#[typetag::serialize]
impl TwobodyEnergy for HardSphere {
    #[inline]
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared < self.min_distance_squared {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

impl Info for HardSphere {
    fn short_name(&self) -> Option<&'static str> {
        Some("hardsphere")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("https://en.wikipedia.org/wiki/Hard_spheres")
    }
}

/// Harmonic potential
///
/// More information [here](https://en.wikipedia.org/wiki/Harmonic_oscillator).
/// # Examples
/// ~~~
/// use interact::twobody::{Harmonic, TwobodyEnergy};
/// let harmonic = Harmonic::new(1.0, 0.5);
/// let distance: f64 = 2.0;
/// assert_eq!(harmonic.twobody_energy(distance.powi(2)), 0.25);
/// ~~~
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Harmonic {
    #[serde(rename = "r₀")]
    eq_distance: f64,
    #[serde(rename = "k")]
    spring_constant: f64,
}

impl Harmonic {
    pub fn new(eq_distance: f64, spring_constant: f64) -> Self {
        Self {
            eq_distance,
            spring_constant,
        }
    }
}

impl Info for Harmonic {
    fn short_name(&self) -> Option<&'static str> {
        Some("harmonic")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Harmonic potential")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("https://en.wikipedia.org/wiki/Harmonic_oscillator")
    }
}

#[typetag::serialize]
impl TwobodyEnergy for Harmonic {
    #[inline]
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
        0.5 * self.spring_constant * (distance_squared.sqrt() - self.eq_distance).powi(2)
    }
}
