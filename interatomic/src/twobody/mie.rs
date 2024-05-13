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

use crate::twobody::IsotropicTwobodyEnergy;
#[cfg(feature = "serde")]
use crate::{divide4_serialize, multiply4_deserialize, sqrt_serialize, square_deserialize};
use crate::{CombinationRule, Cutoff};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Mie potential
///
/// This is a generalization of the Lennard-Jones potential due to G. Mie,
/// ["Zur kinetischen Theorie der einatomigen Körper"](https://doi.org/10.1002/andp.19033160802).
/// The energy is
/// $$ u(r) = ε C \left [\left (\frac{σ}{r}\right )^n - \left (\frac{σ}{r}\right )^m \right ]$$
/// where $C = \frac{n}{n-m} \cdot \left (\frac{n}{m}\right )^{\frac{m}{n-m}}$ and $n > m$.
/// The Lennard-Jones potential is recovered for $n = 12$ and $m = 6$.
///
/// # Examples:
/// ~~~
/// use interatomic::twobody::*;
/// let (epsilon, sigma, r2) = (1.5, 2.0, 2.5);
/// let mie = Mie::<12, 6>::new(epsilon, sigma);
/// let lj = LennardJones::new(epsilon, sigma);
/// assert_eq!(mie.isotropic_twobody_energy(r2), lj.isotropic_twobody_energy(r2));
/// ~~~

#[derive(Clone, Debug, PartialEq, Copy)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Mie<const N: u32, const M: u32> {
    /// Interaction strength, ε
    #[cfg_attr(feature = "serde", serde(rename = "eps"))]
    epsilon: f64,
    /// Diameter, σ
    #[cfg_attr(feature = "serde", serde(rename = "sigma"))]
    sigma: f64,
}

impl<const N: u32, const M: u32> Mie<N, M> {
    const C: f64 = (N / (N - M) * (N / M).pow(M / (N - M))) as f64;

    /// Compile-time optimization if N and M are divisible by 2
    const OPTIMIZE: bool = (N % 2 == 0) && (M % 2 == 0);
    const N_OVER_M: i32 = (N / M) as i32;
    const M_HALF: i32 = (M / 2) as i32;

    pub fn new(epsilon: f64, sigma: f64) -> Self {
        assert!(M > 0);
        assert!(N > M);
        Self { epsilon, sigma }
    }
    /// Construct from a combination rule
    pub fn from_combination_rule(
        rule: CombinationRule,
        epsilons: (f64, f64),
        sigmas: (f64, f64),
    ) -> Self {
        let (epsilon, sigma) = rule.mix(epsilons, sigmas);
        Self::new(epsilon, sigma)
    }
}

impl<const N: u32, const M: u32> IsotropicTwobodyEnergy for Mie<N, M> {
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        if Mie::<N, M>::OPTIMIZE {
            let mth_power = (self.sigma * self.sigma / distance_squared).powi(Mie::<N, M>::M_HALF); // (σ/r)^m
            return Mie::<N, M>::C
                * self.epsilon
                * (mth_power.powi(Mie::<N, M>::N_OVER_M) - mth_power);
        }
        let s_over_r = self.sigma / distance_squared.sqrt(); // (σ/r)
        Mie::<N, M>::C * self.epsilon * (s_over_r.powi(N as i32) - s_over_r.powi(M as i32))
    }
}

impl<const N: u32, const M: u32> Cutoff for Mie<N, M> {
    fn cutoff(&self) -> f64 {
        f64::INFINITY
    }
    fn cutoff_squared(&self) -> f64 {
        f64::INFINITY
    }
}

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
#[derive(Debug, Clone, PartialEq, Default, Copy)]
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
            rename = "eps",
            serialize_with = "divide4_serialize",
            deserialize_with = "multiply4_deserialize"
        )
    )]
    four_times_epsilon: f64,
    /// Squared diameter, σ²
    #[cfg_attr(
        feature = "serde",
        serde(
            rename = "sigma",
            serialize_with = "sqrt_serialize",
            deserialize_with = "square_deserialize"
        )
    )]
    sigma_squared: f64,
}

impl LennardJones {
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        Self {
            four_times_epsilon: 4.0 * epsilon,
            sigma_squared: sigma.powi(2),
        }
    }
    /// Construct using arbitrary combination rule.
    pub fn from_combination_rule(
        rule: CombinationRule,
        epsilons: (f64, f64),
        sigmas: (f64, f64),
    ) -> Self {
        let (epsilon, sigma) = rule.mix(epsilons, sigmas);
        Self::new(epsilon, sigma)
    }
    /// Construct from AB form, u = A/r¹² - B/r⁶
    pub fn from_ab(a: f64, b: f64) -> Self {
        Self {
            four_times_epsilon: b * b / a,
            sigma_squared: (a / b).cbrt(),
        }
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
    #[inline]
    fn isotropic_twobody_energy(&self, squared_distance: f64) -> f64 {
        let x = self.sigma_squared / squared_distance; // σ²/r²
        let x = x * x * x; // σ⁶/r⁶
        self.four_times_epsilon * (x * x - x)
    }
}

/// Weeks-Chandler-Andersen potential
///
/// This is a Lennard-Jones type potential, cut and shifted to zero:
///
/// $$u(r) = 4 \epsilon \left [ (\sigma_{ij}/r)^{12} - (\sigma_{ij}/r)^6 + \frac{1}{4} \right ]$$
///
/// for $r < r_{cut} = 2^{1/6} \sigma_{ij}$; zero otherwise.
///
/// Effectively, this provides soft repulsion without any attraction.
/// More information, see <https://dx.doi.org/doi.org/ct4kh9>.
#[derive(Debug, Clone, PartialEq, Default, Copy)]
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
    const TWOTOTWOSIXTH: f64 = 1.2599210498948732; // f64::powf(2.0, 2.0/6.0)
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        Self {
            lennard_jones: LennardJones::new(epsilon, sigma),
        }
    }

    /// Construct from combination rule
    pub fn from_combination_rule(
        rule: CombinationRule,
        epsilons: (f64, f64),
        sigmas: (f64, f64),
    ) -> Self {
        let (epsilon, sigma) = rule.mix(epsilons, sigmas);
        Self::new(epsilon, sigma)
    }
}

impl Cutoff for WeeksChandlerAndersen {
    #[inline]
    fn cutoff_squared(&self) -> f64 {
        self.lennard_jones.sigma_squared * WeeksChandlerAndersen::TWOTOTWOSIXTH
    }
    #[inline]
    fn cutoff(&self) -> f64 {
        self.cutoff_squared().sqrt()
    }
}

impl IsotropicTwobodyEnergy for WeeksChandlerAndersen {
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared > self.cutoff_squared() {
            return 0.0;
        }
        let x6 = (self.lennard_jones.sigma_squared / distance_squared).powi(3); // (s/r)^6
        self.lennard_jones.four_times_epsilon * (x6 * x6 - x6 + WeeksChandlerAndersen::ONEFOURTH)
    }
}
