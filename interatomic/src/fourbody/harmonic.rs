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

//! Implementation of the harmonic dihedral.

use super::FourbodyAngleEnergy;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Harmonic dihedral potential.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct HarmonicDihedral {
    #[cfg_attr(feature = "serde", serde(rename = "aeq"))]
    eq_angle: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    spring_constant: f64,
}

impl HarmonicDihedral {
    pub const fn new(eq_angle: f64, spring_constant: f64) -> Self {
        Self {
            eq_angle,
            spring_constant,
        }
    }
}

impl FourbodyAngleEnergy for HarmonicDihedral {
    #[inline(always)]
    fn fourbody_angle_energy(&self, angle: f64) -> f64 {
        0.5 * self.spring_constant * (angle - self.eq_angle).powi(2)
    }
}
