//! Lennard-Jones like potentials
//!
//! This includes the:
//! - orignal 12-6 potential,
//! - generalized Mie n-m potential,
//! - cut and shifted LJ, i.e. the Weeks-Chandler-Andersen potential
use crate::{
    arithmetic_mean, divide4_serialize, geometric_mean, multiply4_deserialize, sqrt_serialize,
    square_deserialize, Cutoff, TwobodyEnergy,
};
use serde::{Deserialize, Serialize};

/// # Mie potential
///
/// This is a generalization of the Lennard-Jones potential.
pub struct Mie<const N: u32, const M: u32> {
    /// Interaction strength, ε
    epsilon: f64,
    /// Diameter, σ
    sigma: f64,
}

impl<const N: u32, const M: u32> Mie<N, M> {
    const C: f64 = (N / (N - M) * (N / M).pow(M / (N - M))) as f64;

    /// Optimize if N and M are divisible by 2
    pub const OPTIMIZE: bool = (N % 2 == 0) && (M % 2 == 0);
    const N_OVER_M: i32 = (N / M) as i32;
    const M_HALF: i32 = (M / 2) as i32;

    pub fn new(epsilon: f64, sigma: f64) -> Self {
        assert!(M > 0);
        assert!(N > M);
        Self { epsilon, sigma }
    }
}

impl<const N: u32, const M: u32> TwobodyEnergy for Mie<N, M> {
    #[inline]
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
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

/// # Lennard-Jones potential
///
/// Originally by John Edward Lennard-Jones, see
/// [doi:10/cqhgm7](https://dx.doi.org/10/cqhgm7) or
/// [Wikipedia](https://en.wikipedia.org/wiki/Lennard-Jones_potential).
///
/// ## Examples:
/// ~~~
/// use interact::lj::LennardJones;
/// use interact::TwobodyEnergy;
/// let epsilon = 1.5;
/// let sigma = 2.0;
/// let lj = LennardJones::new(epsilon, sigma);
/// let r_min = f64::powf(2.0, 1.0 / 6.0) * sigma;
/// let u_min = -epsilon;
/// assert_eq!(lj.twobody_energy( r_min.powi(2) ), u_min);
/// ~~~
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Default)]
#[serde(default)]
pub struct LennardJones {
    /// Four times epsilon, 4ε
    #[serde(
        rename = "ε",
        serialize_with = "divide4_serialize",
        deserialize_with = "multiply4_deserialize"
    )]
    four_times_epsilon: f64,
    /// Squared diameter, σ²
    #[serde(
        rename = "σ",
        serialize_with = "sqrt_serialize",
        deserialize_with = "square_deserialize"
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
    /// Construct using the Lorentz-Berthelot mixing rule
    pub fn lorentz_berthelot(epsilons: (f64, f64), sigmas: (f64, f64)) -> Self {
        LennardJones::new(geometric_mean(epsilons), arithmetic_mean(sigmas))
    }
    /// Construct from AB form, u = A/r¹² - B/r⁶
    pub fn from_ab(a: f64, b: f64) -> Self {
        Self {
            four_times_epsilon: b * b / a,
            sigma_squared: (a / b).cbrt(),
        }
    }
}

impl TwobodyEnergy for LennardJones {
    #[inline]
    fn twobody_energy(&self, squared_distance: f64) -> f64 {
        let x = self.sigma_squared / squared_distance; // σ²/r²
        let x = x * x * x; // σ⁶/r⁶
        self.four_times_epsilon * (x * x - x)
    }
    fn cite(&self) -> Option<&'static str> {
        Some("doi:10/cqhgm7")
    }
}

/// # Weeks-Chandler-Andersen potential
///
/// This is a Lennard-Jones type potential, cut and shifted to zero at r_cut = 2^(1/6)σ.
/// More information [here](https://dx.doi.org/doi.org/ct4kh9).
pub struct WeeksChandlerAndersen {
    lennard_jones: LennardJones,
}

impl WeeksChandlerAndersen {
    const ONEFOURTH: f64 = 0.25;
    const TWOTOTWOSIXTH: f64 = 1.2599210498948732; // f64::powf(2.0, 2.0/6.0)
    pub fn new(lennard_jones: LennardJones) -> Self {
        Self { lennard_jones }
    }
}

impl Cutoff for WeeksChandlerAndersen {
    fn cutoff_squared(&self) -> f64 {
        self.lennard_jones.sigma_squared * WeeksChandlerAndersen::TWOTOTWOSIXTH
    }
}

impl TwobodyEnergy for WeeksChandlerAndersen {
    #[inline]
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared > self.cutoff_squared() {
            return 0.0;
        }
        let x = (self.lennard_jones.sigma_squared / distance_squared).powi(3); // (s/r)^6
        self.lennard_jones.four_times_epsilon * (x * x - x + WeeksChandlerAndersen::ONEFOURTH)
    }
    fn cite(&self) -> Option<&'static str> {
        Some("doi.org/ct4kh9")
    }
}
