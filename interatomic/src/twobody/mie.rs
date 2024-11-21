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
use crate::Cutoff;

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
    #[cfg_attr(feature = "serde", serde(alias = "eps", alias = "ε"))]
    epsilon: f64,
    /// Diameter, σ
    #[cfg_attr(feature = "serde", serde(alias = "σ"))]
    sigma: f64,
}

impl<const N: u32, const M: u32> Mie<N, M> {
    const C: f64 = (N / (N - M) * (N / M).pow(M / (N - M))) as f64;

    /// Compile-time optimization if N and M are divisible by 2
    const OPTIMIZE: bool = (N % 2 == 0) && (M % 2 == 0);
    const N_OVER_M: i32 = (N / M) as i32;
    const M_HALF: i32 = (M / 2) as i32;

    pub const fn new(epsilon: f64, sigma: f64) -> Self {
        assert!(M > 0);
        assert!(N > M);
        Self { epsilon, sigma }
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
