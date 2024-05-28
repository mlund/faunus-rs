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

//! Implementation of the three-body cosine based potential as used in GROMOS-96.

use super::ThreebodyAngleEnergy;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Cosine based torsion potential.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct CosineTorsion {
    #[cfg_attr(feature = "serde", serde(rename = "aeq"))]
    eq_angle: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    spring_constant: f64,
}

impl CosineTorsion {
    pub fn new(eq_angle: f64, spring_constant: f64) -> Self {
        Self {
            eq_angle,
            spring_constant,
        }
    }
}

impl ThreebodyAngleEnergy for CosineTorsion {
    #[inline(always)]
    fn threebody_angle_energy(&self, _angle: f64) -> f64 {
        todo!("Cosine torsion is not yet implemented.")
    }
}
