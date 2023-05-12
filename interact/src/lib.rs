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

use serde::{Deserialize, Deserializer, Serializer};

pub mod qpochhammer;
pub mod spline;
pub mod twobody;

/// Defines a citation which can be used to reference the source of a model
pub trait Citation {
    /// Returns a citation string which should be a
    /// 1. Digital Object Identifier (DOI) in the format `doi:...` (preferred)
    /// 2. URL in the format `https://...`
    fn citation(&self) -> Option<&'static str> {
        None
    }
    /// Tries to extract a URL from the citation string
    fn url(&self) -> Option<String> {
        if self.citation()?.starts_with("doi:") {
            Some(format!(
                "https://doi.org/{}",
                &self.citation().unwrap()[4..]
            ))
        } else if self.citation()?.starts_with("https://")
            || self.citation()?.starts_with("http://")
        {
            Some(self.citation().unwrap().to_string())
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
