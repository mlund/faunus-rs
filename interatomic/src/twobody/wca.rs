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

use crate::twobody::IsotropicTwobodyEnergy;
use crate::{twobody::LennardJones, Cutoff};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Weeks-Chandler-Andersen potential
///
/// This is a Lennard-Jones type potential, cut and shifted to zero:
///
/// $$u(r) = 4 \epsilon \left [ (\sigma_{ij}/r)^{12} - (\sigma_{ij}/r)^6 + \frac{1}{4} \right ]$$
///
/// for $r < r_{cut} = 2^{1/6} \sigma_{ij}$; zero otherwise.
///
/// Effectively, this provides soft repulsion without any attraction.
/// More information, see <https://doi.org/ct4kh9>.
///
/// # Examples:
/// ~~~
/// use interatomic::twobody::*;
/// let (epsilon, sigma) = (1.5, 2.0);
/// let wca = WeeksChandlerAndersen::new(epsilon, sigma);
/// assert_eq!(wca.isotropic_twobody_energy(sigma * sigma), epsilon);
/// ~~~
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct WeeksChandlerAndersen {
    #[cfg_attr(feature = "serde", serde(flatten))]
    lennard_jones: LennardJones,
}

impl WeeksChandlerAndersen {
    const ONEFOURTH: f64 = 0.25;
    pub(crate) const TWOTOTWOSIXTH: f64 = 1.2599210498948732; // f64::powf(2.0, 2.0/6.0)

    /// New from epsilon and sigma
    pub const fn new(epsilon: f64, sigma: f64) -> Self {
        Self {
            lennard_jones: LennardJones::new(epsilon, sigma),
        }
    }
}

impl Cutoff for WeeksChandlerAndersen {
    #[inline(always)]
    fn cutoff_squared(&self) -> f64 {
        self.lennard_jones.sigma_squared * WeeksChandlerAndersen::TWOTOTWOSIXTH
    }
    #[inline(always)]
    fn cutoff(&self) -> f64 {
        self.cutoff_squared().sqrt()
    }
}

/// Conversion from Lennard-Jones
impl From<LennardJones> for WeeksChandlerAndersen {
    fn from(lennard_jones: LennardJones) -> Self {
        Self { lennard_jones }
    }
}

impl IsotropicTwobodyEnergy for WeeksChandlerAndersen {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared > self.cutoff_squared() {
            return 0.0;
        }
        let x6 = (self.lennard_jones.sigma_squared / distance_squared).powi(3); // (σ/r)^6
        self.lennard_jones.four_times_epsilon * (x6 * x6 - x6 + WeeksChandlerAndersen::ONEFOURTH)
    }
}
