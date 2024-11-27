// Copyright 2024 Mikael Lund
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

use crate::twobody::AshbaughHatch;
use crate::twobody::IsotropicTwobodyEnergy;
use crate::Cutoff;
#[cfg(feature = "serde")]
use crate::{divide4_serialize, multiply4_deserialize, sqrt_serialize, square_deserialize};
use std::fmt;
use std::fmt::{Display, Formatter};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Lennard-Jones potential
///
/// $$ u(r) = 4\epsilon_{ij} \left [\left (\frac{\sigma_{ij}}{r}\right )^{12} - \left (\frac{\sigma_{ij}}{r}\right )^6 \right ]$$
///
/// Originally by J. E. Lennard-Jones, see
/// [doi:10/cqhgm7](https://dx.doi.org/10/cqhgm7) or
/// [Wikipedia](https://en.wikipedia.org/wiki/Lennard-Jones_potential).
///
/// # Examples:
/// ~~~
/// use interatomic::twobody::*;
/// let (epsilon, sigma) = (1.5, 2.0);
/// let lj = LennardJones::new(epsilon, sigma);
/// let (r_min, u_min) = (f64::powf(2.0, 1.0 / 6.0) * sigma, -epsilon);
/// assert_eq!(lj.isotropic_twobody_energy( r_min.powi(2) ), u_min);
/// ~~~
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct LennardJones {
    /// Four times epsilon, 4ε
    #[cfg_attr(
        feature = "serde",
        serde(
            rename = "epsilon",
            alias = "eps",
            alias = "ε",
            serialize_with = "divide4_serialize",
            deserialize_with = "multiply4_deserialize"
        )
    )]
    pub(crate) four_times_epsilon: f64,
    /// Squared diameter, σ²
    #[cfg_attr(
        feature = "serde",
        serde(
            rename = "sigma",
            alias = "σ",
            serialize_with = "sqrt_serialize",
            deserialize_with = "square_deserialize"
        )
    )]
    pub(crate) sigma_squared: f64,
}

impl LennardJones {
    pub const fn new(epsilon: f64, sigma: f64) -> Self {
        Self {
            four_times_epsilon: 4.0 * epsilon,
            sigma_squared: sigma * sigma,
        }
    }

    /// Construct from AB form, u = A/r¹² - B/r⁶
    pub fn from_ab(a: f64, b: f64) -> Self {
        Self {
            four_times_epsilon: b * b / a,
            sigma_squared: (a / b).cbrt(),
        }
    }

    /// Get epsilon parameter
    #[inline(always)]
    pub const fn get_epsilon(&self) -> f64 {
        self.four_times_epsilon * 0.25
    }

    /// Get sigma parameter
    pub fn get_sigma(&self) -> f64 {
        self.sigma_squared.sqrt()
    }
}

impl Cutoff for LennardJones {
    fn cutoff(&self) -> f64 {
        f64::INFINITY
    }
    fn cutoff_squared(&self) -> f64 {
        f64::INFINITY
    }
}

impl IsotropicTwobodyEnergy for LennardJones {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, squared_distance: f64) -> f64 {
        let x = self.sigma_squared / squared_distance; // σ²/r²
        let x = x * x * x; // σ⁶/r⁶
        self.four_times_epsilon * (x * x - x)
    }
}

impl From<AshbaughHatch> for LennardJones {
    fn from(ah: AshbaughHatch) -> Self {
        ah.lennard_jones
    }
}

impl Display for LennardJones {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "Lennard-Jones: ε = {:.3}, σ = {:.3}",
            self.get_epsilon(),
            self.get_sigma()
        )
    }
}
