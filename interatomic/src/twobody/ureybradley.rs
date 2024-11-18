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

//! Implementation of the Urey-Bradley potential.

use super::IsotropicTwobodyEnergy;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Harmonic Urey-Bradley potential.
/// See <https://manual.gromacs.org/documentation/current/reference-manual/functions/bonded-interactions.html#urey-bradley-potential>
/// for more information.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct UreyBradley {
    #[cfg_attr(feature = "serde", serde(rename = "req"))]
    eq_distance: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    spring_constant: f64,
}

impl UreyBradley {
    pub const fn new(eq_distance: f64, spring_constant: f64) -> Self {
        Self {
            eq_distance,
            spring_constant,
        }
    }
}

impl IsotropicTwobodyEnergy for UreyBradley {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, _distance_squared: f64) -> f64 {
        todo!("Urey-Bradley potential is not yet implemented");
    }
}
