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

//! Library for describing twobody interactions.
//!
//! ## Twobody interactions
//!
//! - Hard-sphere overlap
//! - Harmonic potential
//! - Powerlaw potentials
//!   - Mie
//!   - Lennard-Jones
//!   - Weeks-Chandler-Andersen

use crate::{sqrt_serialize, square_deserialize};
use serde::{Deserialize, Serialize};

mod mie;
pub use self::mie::{LennardJones, Mie, WeeksChandlerAndersen};

/// Potential energy between a pair of particles
pub trait TwobodyEnergy {
    /// Interaction energy between a pair of isotropic particles
    fn twobody_energy(&self, distance_squared: f64) -> f64;
}

/// Helper struct to combine twobody energies
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Combined<T, U>(T, U);

impl<T: TwobodyEnergy, U: TwobodyEnergy> TwobodyEnergy for Combined<T, U> {
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
        self.0.twobody_energy(distance_squared) + self.1.twobody_energy(distance_squared)
    }
}

/// # Hardsphere potential
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
