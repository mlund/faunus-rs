//! Library for describing interactions
//!
//! ## Twobody interactions
//!
//! - Lennard-Jones, Mie, Weeks-Chandler-Andersen
//! - Hard-sphere overlap
//! - Harmonic potential

pub mod lj;
pub mod qpochhammer;
pub mod spline;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Potential energy between a pair of particles
pub trait TwobodyEnergy {
    /// Interaction energy between a pair of isotropic particles (kJ/mol)
    fn twobody_energy(&self, distance_squared: f64) -> f64;
    /// Litterature reference, preferably a Digital Object Identifier in the form "doi:..."
    fn cite(&self) -> Option<&'static str> {
        None
    }
}

/// Defines a cutoff distance
pub trait Cutoff {
    /// Squared cutoff distance
    fn cutoff_squared(&self) -> f64;

    /// Cutoff distance
    fn cutoff(&self) -> f64 {
        self.cutoff_squared().sqrt()
    }
}

/// Rule for combining two numbers
pub trait CombinationRule {
    /// Take a pair of epsilons and sigmas and return combined (epsilon, sigma)
    fn mix(epsilons: (f64, f64), sigmas: (f64, f64)) -> (f64, f64);
    fn get_epsilon(&self) -> f64;
    fn get_sigma(&self) -> f64;
}

/// See https://en.wikipedia.org/wiki/Pythagorean_means
fn geometric_mean(values: (f64, f64)) -> f64 {
    f64::sqrt(values.0 * values.1)
}

/// See https://en.wikipedia.org/wiki/Pythagorean_means
fn arithmetic_mean(values: (f64, f64)) -> f64 {
    0.5 * (values.0 + values.1)
}

/// See https://en.wikipedia.org/wiki/Pythagorean_means
fn _harmonic_mean(values: (f64, f64)) -> f64 {
    2.0 * values.0 * values.1 / (values.0 + values.1)
}

/// Transform x^2 --> x when serializing
fn sqrt_serialize<S>(x: &f64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_f64(x.sqrt())
}

/// Transform x --> x^2 when deserializing
fn square_deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(f64::deserialize(deserializer)?.powi(2))
}

/// Transform x --> x/4 when serializing
fn divide4_serialize<S>(x: &f64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_f64(x / 4.0)
}

/// Transform x --> 4x when deserializing
fn multiply4_deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(f64::deserialize(deserializer)? * 4.0)
}

/// # Hardsphere potential
///
/// More information [here](http://www.sklogwiki.org/SklogWiki/index.php/Hard_sphere_model).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct HardSphere {
    /// Minimum distance
    #[serde(
        rename = "σ",
        serialize_with = "sqrt_serialize",
        deserialize_with = "square_deserialize"
    )]
    min_distance_squared: f64,
}

impl HardSphere {
    pub fn new(min_distance: f64) -> Self {
        Self {
            min_distance_squared: min_distance.powi(2),
        }
    }
}

impl TwobodyEnergy for HardSphere {
    #[inline]
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared < self.min_distance_squared {
            f64::INFINITY
        } else {
            0.0
        }
    }
}
