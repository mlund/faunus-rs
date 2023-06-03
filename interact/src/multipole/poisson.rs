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

//! # Poisson scheme that cover cutoff based methods like Wolf, Fanougakis etc.

use super::{Energy, Field, Force, Potential, SplitingFunction};
use num::integer::binomial;
use serde::{Deserialize, Serialize};

impl<const C: i32, const D: i32> Potential for Poisson<C, D> {}
impl<const C: i32, const D: i32> Field for Poisson<C, D> {}
impl<const C: i32, const D: i32> Energy for Poisson<C, D> {}
impl<const C: i32, const D: i32> Force for Poisson<C, D> {}

/// Poisson scheme with and without specified Debye-length
///
/// A general scheme which, depending on two parameters `C` and `D`, can model several different pair-potentials.
/// The short-ranged function is given by:
///
/// S(q) = (1 - q~)^(D + 1) * sum_{c = 0}^{C - 1} ((C - c) / C) * (D - 1 + c choose c) * q^c
///
/// where `C` is the number of cancelled derivatives at origin -2 (starting from the second derivative),
/// and `D` is the number of cancelled derivatives at the cut-off (starting from the zeroth derivative).
///
/// For infinite Debye-length, the following holds:
///
/// | Type          | C   | D   | Reference / Comment
/// |---------------|-----|-----|---------------------
/// | `plain`       | 1   | -1  | Plain Coulomb
/// | `wolf`        | 1   | 0   | Undamped Wolf, doi:10.1063/1.478738
/// | `fennell`     | 1   | 1   | Levitt/undamped Fennell, doi:10/fp959p or 10/bqgmv2
/// | `kale`        | 1   | 2   | Kale, doi:10/csh8bg
/// | `mccann`      | 1   | 3   | McCann, doi:10.1021/ct300961
/// | `fukuda`      | 2   | 1   | Undamped Fukuda, doi:10.1063/1.3582791
/// | `markland`    | 2   | 2   | Markland, doi:10.1016/j.cplett.2008.09.019
/// | `stenqvist`   | 3   | 3   | Stenqvist, doi:10/c5fr
/// | `fanourgakis` | 4   | 3   | Fanourgakis, doi:10.1063/1.3216520
///
/// More info:
/// - http://dx.doi.org/10.1088/1367-2630/ab1ec1
///
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Poisson<const C: i32, const D: i32> {
    cutoff: f64,
    debye_length: f64,
    has_dipolar_selfenergy: bool,
    #[serde(skip)]
    reduced_kappa: f64,
    #[serde(skip)]
    use_yukawa_screening: bool,
    #[serde(skip)]
    reduced_kappa_squared: f64,
    #[serde(skip)]
    yukawa_denom: f64,
    #[serde(skip)]
    binom_cdc: f64,
}

/// Plain coulomb potential with S(q) = 1
pub type Plain = Poisson<1, -1>;
/// Undamped Fennel, see `doi:10/fp959p` or `10/bqgmv2`
pub type UndampedFennel = Poisson<1, 1>;
/// Undamped Wolf, see `doi:10.1063/1.478738`
pub type UndampedWolf = Poisson<1, 0>;
/// Kale, see `doi:10/csh8bg`
pub type Kale = Poisson<1, 2>;
/// McCann, see `doi:10.1021/ct300961`
pub type McCann = Poisson<1, 3>;
/// Undamped Fukuda, see `doi:10.1063/1.3582791`
pub type UndampedFukuda = Poisson<2, 1>;
/// Markland, see `doi:10.1016/j.cplett.2008.09.019`
pub type Markland = Poisson<2, 2>;
/// Stenqvist, see `doi:10/c5fr`
pub type Stenqvist = Poisson<3, 3>;
/// Fanourgakis, see `doi:10.1063/1.3216520`
pub type Fanourgakis = Poisson<4, 3>;

impl<const C: i32, const D: i32> Poisson<C, D> {
    pub fn new(cutoff: f64, debye_length: f64) -> Self {
        if C < 1 {
            panic!("`C` must be larger than zero");
        }
        if D < -1 && D != -C {
            panic!("If `D` is less than negative one, then it has to equal negative `C`");
        }
        if D == 0 && C != 1 {
            panic!("If `D` is zero, then `C` has to equal one ");
        }
        let mut has_dipolar_selfenergy = true;
        if C < 2 {
            has_dipolar_selfenergy = false;
        }
        let mut reduced_kappa = 0.0;
        let mut use_yukawa_screening = false;
        let mut reduced_kappa_squared = 0.0;
        let mut yukawa_denom = 0.0;
        let mut binom_cdc = 0.0;

        if !debye_length.is_infinite() {
            reduced_kappa = cutoff / debye_length;
            if reduced_kappa.abs() > 1e-6 {
                use_yukawa_screening = true;
                reduced_kappa_squared = reduced_kappa * reduced_kappa;
                yukawa_denom = 1.0 / (1.0 - (2.0 * reduced_kappa).exp());
                let _a1 = -f64::from(C + D) / f64::from(C);
                binom_cdc = f64::from(binomial(C + D, C) * D);
            }
        }
        if D != -C {
            binom_cdc = f64::from(binomial(C + D, C) * D);
        }

        Poisson {
            cutoff,
            debye_length,
            has_dipolar_selfenergy,
            reduced_kappa,
            use_yukawa_screening,
            reduced_kappa_squared,
            yukawa_denom,
            binom_cdc,
        }
    }
}

impl<const C: i32, const D: i32> crate::Info for Poisson<C, D> {
    fn short_name(&self) -> Option<&'static str> {
        Some("poisson")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("doi:10.1088/1367-2630/ab1ec1")
    }
}

impl<const C: i32, const D: i32> crate::Cutoff for Poisson<C, D> {
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl<const C: i32, const D: i32> SplitingFunction for Poisson<C, D> {
    fn kappa(&self) -> Option<f64> {
        None
    }
    fn short_range_function(&self, q: f64) -> f64 {
        if D == -C {
            return 1.0;
        }
        let mut qp = q;

        if self.use_yukawa_screening {
            qp = (1.0 - (2.0 * self.reduced_kappa * q).exp()) * self.yukawa_denom;
        }

        if D == 0 && C == 1 {
            return 1.0 - qp;
        }

        let sum: f64 = (0..C)
            .map(|c| {
                f64::from(num::integer::binomial(D - 1 + c, c)) * f64::from(C - c) / f64::from(C)
                    * qp.powi(c)
            })
            .sum();
        (1.0 - qp).powi(D + 1) * sum
    }
    fn short_range_function_derivative(&self, q: f64) -> f64 {
        if D == -C {
            return 0.0;
        }
        if D == 0 && C == 1 {
            return 0.0;
        }
        let mut qp = q;
        let mut dqpdq = 1.0;
        if self.use_yukawa_screening {
            let exp2kq = (2.0 * self.reduced_kappa * q).exp();
            qp = (1.0 - exp2kq) * self.yukawa_denom;
            dqpdq = -2.0 * self.reduced_kappa * exp2kq * self.yukawa_denom;
        }
        let mut tmp1 = 1.0;
        let mut tmp2 = 0.0;
        for c in 1..C {
            let factor = (binomial(D - 1 + c, c) * (C - c)) as f64 / C as f64;
            tmp1 += factor * qp.powi(c);
            tmp2 += factor * c as f64 * qp.powi(c - 1);
        }
        let dsdqp = -(D + 1) as f64 * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
        dsdqp * dqpdq
    }

    fn short_range_function_second_derivative(&self, q: f64) -> f64 {
        if D == -C {
            return 0.0;
        }
        if D == 0 && C == 1 {
            return 0.0;
        }
        let mut qp = q;
        let mut dqpdq = 1.0;
        let mut d2qpdq2 = 0.0;
        let mut dsdqp = 0.0;
        if self.use_yukawa_screening {
            qp = (1.0 - (2.0 * self.reduced_kappa * q).exp()) * self.yukawa_denom;
            dqpdq = -2.0
                * self.reduced_kappa
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d2qpdq2 = -4.0
                * self.reduced_kappa_squared
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            let mut tmp1 = 1.0;
            let mut tmp2 = 0.0;
            for i in 1..C {
                let b = binomial(D - 1 + i, i) as f64 * (C - i) as f64;
                tmp1 += b / C as f64 * qp.powi(i);
                tmp2 += b * i as f64 / C as f64 * qp.powi(i - 1);
            }
            dsdqp = -(D + 1) as f64 * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
        }
        let d2sdqp2 = self.binom_cdc * (1.0 - qp).powi(D - 1) * qp.powi(C - 1);
        d2sdqp2 * dqpdq * dqpdq + dsdqp * d2qpdq2
    }

    fn short_range_function_third_derivative(&self, q: f64) -> f64 {
        if D == -C {
            return 0.0;
        }
        if D == 0 && C == 1 {
            return 0.0;
        }
        let mut qp = q;
        let mut dqpdq = 1.0;
        let mut d2qpdq2 = 0.0;
        let mut d3qpdq3 = 0.0;
        let mut d2sdqp2 = 0.0;
        let mut dsdqp = 0.0;
        if self.use_yukawa_screening {
            qp = (1.0 - (2.0 * self.reduced_kappa * q).exp()) * self.yukawa_denom;
            dqpdq = -2.0
                * self.reduced_kappa
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d2qpdq2 = -4.0
                * self.reduced_kappa_squared
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d3qpdq3 = -8.0
                * self.reduced_kappa_squared
                * self.reduced_kappa
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d2sdqp2 = self.binom_cdc * (1.0 - qp).powi(D - 1) * qp.powi(C - 1);
            let mut tmp1 = 1.0;
            let mut tmp2 = 0.0;
            for c in 1..C {
                tmp1 += binomial(D - 1 + c, c) as f64 * (C - c) as f64 / C as f64 * qp.powi(c);
                tmp2 += binomial(D - 1 + c, c) as f64 * (C - c) as f64 / C as f64
                    * c as f64
                    * qp.powi(c - 1);
            }
            dsdqp = -(D + 1) as f64 * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
        }
        let d3sdqp3 = self.binom_cdc
            * (1.0 - qp).powi(D - 2)
            * qp.powi(C - 2)
            * ((2.0 - C as f64 - D as f64) * qp + C as f64 - 1.0);
        d3sdqp3 * dqpdq * dqpdq * dqpdq + 3.0 * d2sdqp2 * dqpdq * d2qpdq2 + dsdqp * d3qpdq3
    }
}

#[test]
fn test_poisson() {
    let pot = Stenqvist::new(29.0, f64::INFINITY);
    let eps = 1e-9; // Set epsilon for approximate equality

    // Test Stenqvist short-range function
    approx::assert_relative_eq!(pot.short_range_function(0.5), 0.15625, epsilon = eps);
    approx::assert_relative_eq!(
        pot.short_range_function_derivative(0.5),
        -1.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_second_derivative(0.5),
        3.75,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_third_derivative(0.5),
        0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_third_derivative(0.6),
        -5.76,
        epsilon = eps
    );
    approx::assert_relative_eq!(pot.short_range_function(1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_function_derivative(1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(
        pot.short_range_function_second_derivative(1.0),
        0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_third_derivative(1.0),
        0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(pot.short_range_function(0.0), 1.0, epsilon = eps);
    approx::assert_relative_eq!(
        pot.short_range_function_derivative(0.0),
        -2.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_second_derivative(0.0),
        0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_third_derivative(0.0),
        0.0,
        epsilon = eps
    );

    // Test Fanougarkis short-range function
    let pot = Fanourgakis::new(29.0, f64::INFINITY);
    approx::assert_relative_eq!(pot.short_range_function(0.5), 0.19921875, epsilon = eps);
    approx::assert_relative_eq!(
        pot.short_range_function_derivative(0.5),
        -1.1484375,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_second_derivative(0.5),
        3.28125,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.short_range_function_third_derivative(0.5),
        6.5625,
        epsilon = eps
    );

    // Test
}
