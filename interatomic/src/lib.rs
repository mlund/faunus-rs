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

//! # Interatomic
//!
//! A library for calculating interatomic interactions
//! such as van der Waals, electrostatics, and other two-body or many-body potentials.
//!

#[cfg(test)]
extern crate approx;
extern crate typetag;

/// A point in 3D space
pub type Vector3 = nalgebra::Vector3<f64>;
/// A stack-allocated 3x3 square matrix
pub type Matrix3 = nalgebra::Matrix3<f64>;
use num::{Float, NumCast};
use serde::{Deserialize, Deserializer, Serializer};

pub mod electrostatic;
mod qpochhammer;
pub mod spline;
pub mod twobody;

use physical_constants::{
    AVOGADRO_CONSTANT, ELEMENTARY_CHARGE, MOLAR_GAS_CONSTANT, VACUUM_ELECTRIC_PERMITTIVITY,
};
use std::f64::consts::PI;

/// Electrostatic prefactor, e²/4πε₀ × 10⁷ × NA (Å × kJ / mol).
///
/// Can be used to calculate e.g. the interaction energy bewteen two
/// point charges in kJ/mol:
///
/// Examples:
/// ```
/// use interatomic::ELECTRIC_PREFACTOR;
/// let z1 = 1.0;                    // unit-less charge number
/// let z2 = -1.0;                   // unit-less charge number
/// let r = 7.0;                     // separation in angstrom
/// let rel_dielectric_const = 80.0; // relative dielectric constant
/// let energy = ELECTRIC_PREFACTOR * z1 * z2 / (rel_dielectric_const * r);
/// assert_eq!(energy, -2.48099031507825); // in kJ/mol
pub const ELECTRIC_PREFACTOR: f64 =
    ELEMENTARY_CHARGE * ELEMENTARY_CHARGE * 1.0e10 * AVOGADRO_CONSTANT * 1e-3
        / (4.0 * PI * VACUUM_ELECTRIC_PERMITTIVITY);

/// Bjerrum length in vacuum at 298.15 K, e²/4πε₀kT (Å).
///
/// Examples:
/// ```
/// use interatomic::BJERRUM_LEN_VACUUM_298K;
/// let relative_dielectric_const = 80.0;
/// assert_eq!(BJERRUM_LEN_VACUUM_298K / relative_dielectric_const, 7.0057415269733);
/// ```
pub const BJERRUM_LEN_VACUUM_298K: f64 = ELECTRIC_PREFACTOR / (MOLAR_GAS_CONSTANT * 1e-3 * 298.15);

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
        match self.citation() {
            Some(c) => match c.strip_prefix("doi:") {
                Some(doi) => Some(format!("https://doi.org/{}", doi)),
                _ if c.starts_with("https://") || c.starts_with("http://") => Some(c.to_string()),
                _ => None,
            },
            None => None,
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

/// Defines an optional Debye screening length for electrostatic interactions
pub trait DebyeLength {
    /// Optional Debye length
    fn debye_length(&self) -> Option<f64>;

    /// Optional inverse Debye screening length
    fn kappa(&self) -> Option<f64> {
        self.debye_length().map(f64::recip)
    }
}

/// Defines the Bjerrum length, lB = e²/4πεkT commonly used in electrostatics (ångström).
pub trait BjerrumLength {
    fn bjerrum_length(&self) -> f64;
}

/// Combination rules for mixing epsilon and sigma values
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CombinationRule {
    /// The Lotentz-Berthelot combination rule (geometric mean on epsilon, arithmetic mean on sigma)
    LorentzBerthelot,
    /// The Fender-Halsey combination rule (harmonic mean on epsilon, arithmetic mean on sigma)
    FenderHalsey,
}

impl CombinationRule {
    /// Combines epsilon and sigma pairs using the selected combination rule
    pub fn mix(&self, epsilons: (f64, f64), sigmas: (f64, f64)) -> (f64, f64) {
        let epsilon = self.mix_epsilons(epsilons);
        let sigma = self.mix_sigmas(sigmas);
        (epsilon, sigma)
    }

    /// Combine epsilon values using the selected combination rule
    pub fn mix_epsilons(&self, epsilons: (f64, f64)) -> f64 {
        match self {
            Self::LorentzBerthelot => geometric_mean(epsilons),
            Self::FenderHalsey => harmonic_mean(epsilons),
        }
    }

    /// Combine sigma values using the selected combination rule
    pub fn mix_sigmas(&self, sigmas: (f64, f64)) -> f64 {
        match self {
            Self::LorentzBerthelot => arithmetic_mean(sigmas),
            Self::FenderHalsey => arithmetic_mean(sigmas),
        }
    }
}

/// See Pythagorean means on [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_means)
fn geometric_mean<T: Float>(values: (T, T)) -> T {
    T::sqrt(values.0 * values.1)
}

/// See Pythagorean means on [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_means)
fn arithmetic_mean<T: Float>(values: (T, T)) -> T {
    (values.0 + values.1) * NumCast::from(0.5).unwrap()
}

/// See Pythagorean means on [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_means)
fn harmonic_mean<T: Float>(values: (T, T)) -> T {
    values.0 * values.1 / (values.0 + values.1) * NumCast::from(2.0).unwrap()
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
