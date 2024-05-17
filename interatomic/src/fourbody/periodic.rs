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

//! Implementation of the periodic dihedral.

use super::FourbodyAngleEnergy;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Periodic dihedral potential.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct PeriodicDihedral {
    #[cfg_attr(feature = "serde", serde(rename = "phi"))]
    phase_angle: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    spring_constant: f64,
    #[cfg_attr(feature = "serde", serde(rename = "n"))]
    periodicity: f64,
}

impl PeriodicDihedral {
    pub fn new(phase_angle: f64, spring_constant: f64, periodicity: f64) -> Self {
        Self {
            phase_angle,
            spring_constant,
            periodicity,
        }
    }
}

impl FourbodyAngleEnergy for PeriodicDihedral {
    #[inline(always)]
    fn fourbody_angle_energy(&self, _angle: f64) -> f64 {
        todo!("Periodic dihedral is not yet implemented.")
    }
}
