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

extern crate approx;
extern crate typetag;

/// A point in 3D space
pub type Point = nalgebra::Vector3<f64>;
/// A stack-allocated 3x3 square matrix
pub type Matrix3 = nalgebra::Matrix3<f64>;
use serde::{Deserialize, Deserializer, Serializer};

pub mod multipole;
mod qpochhammer;
pub mod spline;
pub mod twobody;

/// Defines information about a concept, like a short name, citation, url etc.
pub trait Info {
    /// Returns a short name for the concept. Use `_` for spaces and avoid weird characters.
    /// This is typically used as keywords in user input and output, e.g. in JSON files.
    fn short_name(&self) -> Option<&'static str> {
        None
    }
    /// Returns a long name for the concept. Spaces are allowed.
    fn long_name(&self) -> Option<&'static str> {
        None
    }

    /// Returns a citation string which should be a
    /// 1. Digital Object Identifier (DOI) in the format `doi:...` (preferred)
    /// 2. URL in the format `https://...`
    fn citation(&self) -> Option<&'static str> {
        None
    }
    /// Tries to extract a URL from the citation string
    fn url(&self) -> Option<String> {
        if let Some(c) = self.citation() {
            if let Some(doi) = c.strip_prefix("doi:") {
                Some(format!("https://doi.org/{}", doi))
            } else if c.starts_with("https://") || c.starts_with("http://") {
                Some(c.to_string())
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Defines a cutoff distance
pub trait Cutoff {
    /// Squared cutoff distance
    fn cutoff_squared(&self) -> f64 {
        self.cutoff().powi(2)
    }

    /// Cutoff distance
    fn cutoff(&self) -> f64;
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

/// Approximation of erfc-function
///
/// # Arguments
/// * `x` - Value for which erfc should be calculated
///
/// # Details
/// Reference for this approximation is found in Abramowitz and Stegun,
/// Handbook of mathematical functions, eq. 7.1.26
///
/// erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)e^{-x^2} + epsilon(x)
/// t = 1 / (1 + px)
/// |epsilon(x)| <= 1.5 * 10^-7
///
/// # Warning
/// Needs modification if x < 0
#[inline]
fn erfc_x(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * f64::exp(-x * x)
}
