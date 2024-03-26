// Copyright 2023 Björn Stenqvist and Mikael Lund
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

use super::{Energy, Field, Force, Potential, ShortRangeFunction};
use crate::erfc_x;
#[cfg(test)]
use approx::assert_relative_eq;
use serde::{Deserialize, Serialize};

impl Potential for RealSpaceEwald {}
impl Field for RealSpaceEwald {}
impl Energy for RealSpaceEwald {}
impl Force for RealSpaceEwald {}

/// # Scheme for real-space Ewald interactionss
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RealSpaceEwald {
    /// Real space cutoff distance
    cutoff: f64,
    alpha: f64,
    /// alpha * cutoff
    eta: f64,
    /// kappa * cutoff (zero if no salt)
    zeta: f64,
}

impl RealSpaceEwald {
    /// Square root of pi
    const SQRT_PI: f64 = 1.7724538509055159;
    /// Construct a new Ewald scheme with given cutoff and alpha.
    pub fn new(cutoff: f64, alpha: f64, debye_length: Option<f64>) -> Self {
        let kappa = 1.0 / debye_length.unwrap_or(f64::INFINITY);
        Self {
            cutoff,
            alpha,
            eta: alpha * cutoff,
            zeta: cutoff * kappa,
        }
    }
}

impl crate::Info for RealSpaceEwald {
    fn citation(&self) -> Option<&'static str> {
        Some("doi:10.1002/andp.19213690304")
    }
    fn short_name(&self) -> Option<&'static str> {
        Some("ewald")
    }
}

impl crate::Cutoff for RealSpaceEwald {
    #[inline]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl ShortRangeFunction for RealSpaceEwald {
    /// The inverse Debye length if salt is present, otherwise `None`.
    #[inline]
    fn kappa(&self) -> Option<f64> {
        if self.zeta > 0.0 {
            Some(self.zeta / self.cutoff)
        } else {
            None
        }
    }
    #[inline]
    fn short_range_f0(&self, q: f64) -> f64 {
        0.5 * (erfc_x(self.eta * q + self.zeta / (2.0 * self.eta)) * f64::exp(2.0 * self.zeta * q)
            + erfc_x(self.eta * q - self.zeta / (2.0 * self.eta)))
    }

    fn short_range_f1(&self, q: f64) -> f64 {
        let exp_c = f64::exp(-(self.eta * q - self.zeta / (2.0 * self.eta)).powi(2));
        let erfc_c = erfc_x(self.eta * q + self.zeta / (2.0 * self.eta));
        -2.0 * self.eta / Self::SQRT_PI * exp_c + self.zeta * erfc_c * f64::exp(2.0 * self.zeta * q)
    }

    fn short_range_f2(&self, q: f64) -> f64 {
        let exp_c = f64::exp(-(self.eta * q - self.zeta / (2.0 * self.eta)).powi(2));
        let erfc_c = erfc_x(self.eta * q + self.zeta / (2.0 * self.eta));
        4.0 * self.eta.powi(2) / Self::SQRT_PI * (self.eta * q - self.zeta / self.eta) * exp_c
            + 2.0 * self.zeta.powi(2) * erfc_c * f64::exp(2.0 * self.zeta * q)
    }

    fn short_range_f3(&self, q: f64) -> f64 {
        let exp_c = f64::exp(-(self.eta * q - self.zeta / (2.0 * self.eta)).powi(2));
        let erfc_c = erfc_x(self.eta * q + self.zeta / (2.0 * self.eta));
        4.0 * self.eta.powi(3) / Self::SQRT_PI
            * (1.0
                - 2.0
                    * (self.eta * q - self.zeta / self.eta)
                    * (self.eta * q - self.zeta / (2.0 * self.eta))
                - self.zeta.powi(2) / self.eta.powi(2))
            * exp_c
            + 4.0 * self.zeta.powi(3) * erfc_c * f64::exp(2.0 * self.zeta * q)
    }
}

#[test]
fn test_ewald() {
    // Test short-ranged function
    let pot = RealSpaceEwald::new(29.0, 0.1, None);
    let eps = 1e-8;
    assert_relative_eq!(pot.short_range_f0(0.5), 0.04030484067840161, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.39971358519150996, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 3.36159125, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -21.54779992186245, epsilon = eps);

    let pot = RealSpaceEwald::new(29.0, 0.1, Some(23.0));
    let eps = 1e-6;
    assert_relative_eq!(pot.kappa().unwrap(), 1.0 / 23.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f0(0.5), 0.07306333588, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.63444119, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 4.423133599, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -19.85937171, epsilon = eps);
}
