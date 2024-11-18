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

use super::IsotropicTwobodyEnergy;
#[cfg(feature = "serde")]
use crate::{sqrt_serialize, square_deserialize};
use crate::{CombinationRule, Cutoff};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Hard sphere potential
///
/// The hard sphere potential is infinite if the distance is smaller than the minimum distance, and zero otherwise.
/// More information [here](http://www.sklogwiki.org/SklogWiki/index.php/Hard_sphere_model).
///
/// # Examples
/// ~~~
/// use interatomic::twobody::{HardSphere, IsotropicTwobodyEnergy};
/// let hardsphere = HardSphere::new(1.0);
/// let distance: f64 = 0.9; // smaller than the minimum distance
/// assert!(hardsphere.isotropic_twobody_energy(distance.powi(2)).is_infinite());
/// let distance: f64 = 1.1; // greater than the minimum distance
/// assert_eq!(hardsphere.isotropic_twobody_energy(distance.powi(2)), 0.0);
/// ~~~
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct HardSphere {
    /// Minimum distance
    #[cfg_attr(
        feature = "serde",
        serde(
            rename = "sigma",
            alias = "σ",
            serialize_with = "sqrt_serialize",
            deserialize_with = "square_deserialize"
        )
    )]
    min_distance_squared: f64,
}

impl HardSphere {
    /// Create by giving the minimum distance where if smaller, the energy is infinite or zero otherwise
    pub const fn new(min_distance: f64) -> Self {
        Self {
            min_distance_squared: min_distance * min_distance,
        }
    }

    pub fn from_combination_rule(rule: CombinationRule, sigmas: (f64, f64)) -> Self {
        let sigma = rule.mix_sigmas(sigmas);
        Self::new(sigma)
    }
}

impl IsotropicTwobodyEnergy for HardSphere {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared < self.min_distance_squared {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

impl Cutoff for HardSphere {
    fn cutoff(&self) -> f64 {
        self.cutoff_squared().sqrt()
    }
    fn cutoff_squared(&self) -> f64 {
        self.min_distance_squared
    }
}
