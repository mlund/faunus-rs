use super::IsotropicTwobodyEnergy;
use crate::Info;
use serde::{Deserialize, Serialize};

/// Harmonic potential
///
/// More information [here](https://en.wikipedia.org/wiki/Harmonic_oscillator).
/// # Examples
/// ~~~
/// use interact::twobody::{Harmonic, IsotropicTwobodyEnergy};
/// let harmonic = Harmonic::new(1.0, 0.5);
/// let distance: f64 = 2.0;
/// assert_eq!(harmonic.isotropic_twobody_energy(distance.powi(2)), 0.25);
/// ~~~
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Harmonic {
    #[serde(rename = "r₀")]
    eq_distance: f64,
    #[serde(rename = "k")]
    spring_constant: f64,
}

impl Harmonic {
    pub fn new(eq_distance: f64, spring_constant: f64) -> Self {
        Self {
            eq_distance,
            spring_constant,
        }
    }
}

impl Info for Harmonic {
    fn short_name(&self) -> Option<&'static str> {
        Some("harmonic")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Harmonic potential")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("https://en.wikipedia.org/wiki/Harmonic_oscillator")
    }
}

impl IsotropicTwobodyEnergy for Harmonic {
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        0.5 * self.spring_constant * (distance_squared.sqrt() - self.eq_distance).powi(2)
    }
}