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
use crate::{
    twobody::{LennardJones, WeeksChandlerAndersen},
    Cutoff,
};
use std::fmt;
use std::fmt::{Display, Formatter};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Truncated and shifted Ashbaugh-Hatch
///
/// More information, see <https://doi.org/10.1021/ja802124e>.
///
/// # Examples:
/// ~~~
/// use interatomic::twobody::*;
/// // For λ=1.0 and cutoff = 2^(1/6)σ, we recover WCA:
/// let (epsilon, sigma, lambda) = (1.5, 2.0, 1.0);
/// let cutoff = 2.0_f64.powf(1.0/6.0) * sigma;
/// let lj = LennardJones::new(epsilon, sigma);
/// let ah = AshbaughHatch::new(lj.clone(), lambda, cutoff);
/// let wca = WeeksChandlerAndersen::from(lj);
/// let r2 = sigma * sigma;
/// assert_eq!(ah.isotropic_twobody_energy(r2), wca.isotropic_twobody_energy(r2));
/// ~~~
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct AshbaughHatch {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub(crate) lennard_jones: LennardJones,
    /// Dimensionless scaling factor, λ in the interval [0, 1]
    #[cfg_attr(feature = "serde", serde(alias = "λ", default))]
    lambda: f64,
    /// Spherical cutoff distance
    #[cfg_attr(feature = "serde", serde(alias = "rc", default))]
    cutoff: f64,
}

impl AshbaughHatch {
    pub const fn new(lennard_jones: LennardJones, lambda: f64, cutoff: f64) -> Self {
        Self {
            lennard_jones,
            lambda,
            cutoff,
        }
    }
}

impl Cutoff for AshbaughHatch {
    #[inline(always)]
    fn cutoff_squared(&self) -> f64 {
        self.cutoff * self.cutoff
    }
    #[inline(always)]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl IsotropicTwobodyEnergy for AshbaughHatch {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared > self.cutoff_squared() {
            return 0.0;
        }

        let lj = self
            .lennard_jones
            .isotropic_twobody_energy(distance_squared);

        let lj_rc = self
            .lennard_jones
            .isotropic_twobody_energy(self.cutoff_squared());

        if distance_squared
            <= self.lennard_jones.sigma_squared * WeeksChandlerAndersen::TWOTOTWOSIXTH
        {
            lj - self.lambda * lj_rc + self.lennard_jones.get_epsilon() * (1.0 - self.lambda)
        } else {
            self.lambda * (lj - lj_rc)
        }
    }
}

impl Display for AshbaughHatch {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "Ashbaugh-Hatch with λ = {:.3}, cutoff = {:.3}, ε = {:.3}, σ = {:.3}",
            self.lambda,
            self.cutoff,
            self.lennard_jones.get_epsilon(),
            self.lennard_jones.get_sigma()
        )
    }
}
