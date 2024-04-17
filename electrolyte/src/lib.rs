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

//! # Support for handling electrolyte solutions
//!
//! This module provides support for calculating properties of electrolyte solutions
//! such as the Debye length, ionic strength, and Bjerrum length.
//! It also has a module for empirical models of relative permittivity as a function
//! of temperature.

use anyhow::Result;
use physical_constants::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, VACUUM_ELECTRIC_PERMITTIVITY,
};
use std::f64::consts::PI;

mod permittivity;
pub use permittivity::{PermittivityNR, RelativePermittivity};
mod salt;
pub use salt::Salt;
mod medium;
pub use medium::Medium;

pub trait Temperature {
    /// Get the temperature in Kelvin
    fn temperature(&self) -> f64;
    /// Set the temperature in Kelvin
    fn set_temperature(&mut self, temperature: f64) -> Result<()>;
}

/// Trait for objects that has an ionic strength
pub trait IonicStrength {
    /// Get the ionic strength in mol/l
    fn ionic_strength(&self) -> f64;
    /// Try to set the ionic strength in mol/l
    fn set_ionic_strength(&mut self, _ionic_strength: f64) -> Result<()> {
        Err(anyhow::anyhow!(
            "Setting the ionic strength is not implemented"
        ))
    }
}

/// Trait for objects where a Debye length can be calculated
pub trait DebyeLength: IonicStrength + RelativePermittivity + Temperature {
    /// # Debye length in angstrom or `None` if the ionic strength is zero.
    ///
    /// May perform expensive operations so avoid use in speed critical code,
    /// such as inside tight interaction loops.
    fn debye_length(&self) -> Option<f64> {
        let temperature = self.temperature();
        let permittivity = self.permittivity(temperature).unwrap();
        let ionic_strength = self.ionic_strength();
        if ionic_strength > 0.0 {
            Some(debye_length(temperature, permittivity, ionic_strength))
        } else {
            None
        }
    }
    /// Inverse Debye length in 1/angstrom or `None`` if the ionic strength is zero.
    ///
    /// May perform expensive operations so avoid use in speed critical code,
    /// such as inside tight interaction loops.
    fn kappa(&self) -> Option<f64> {
        self.debye_length().map(f64::recip)
    }
}

/// Calculates the Bjerrum length, lB = e²/4πεkT commonly used in electrostatics (ångström).
///
/// More information at <https://en.wikipedia.org/wiki/Bjerrum_length>.
///
/// # Examples
/// ~~~
/// use electrolyte::bjerrum_length;
/// let lB = bjerrum_length(293.0, 80.0); // angstroms
/// assert_eq!(lB, 7.1288799871283);
/// ~~~
pub fn bjerrum_length(kelvin: f64, relative_dielectric_const: f64) -> f64 {
    const ANGSTROM_PER_METER: f64 = 1e10;
    ELEMENTARY_CHARGE.powi(2) * ANGSTROM_PER_METER
        / (4.0
            * PI
            * relative_dielectric_const
            * VACUUM_ELECTRIC_PERMITTIVITY
            * BOLTZMANN_CONSTANT
            * kelvin)
}

/// Calculates the Debye length in angstrom, λD = sqrt(8πlB·I·N·V)⁻¹, where I is the ionic strength in molar units (mol/l).
///
/// # Examples
/// ~~~
/// use electrolyte::debye_length;
/// let molarity = 0.03;                              // mol/l
/// let lambda = debye_length(293.0, 80.0, molarity); // angstroms
/// assert_eq!(lambda, 17.576538097378368);
/// ~~~
pub fn debye_length(kelvin: f64, relative_dielectric_const: f64, ionic_strength: f64) -> f64 {
    const LITER_PER_ANGSTROM3: f64 = 1e-27;
    (8.0 * PI
        * bjerrum_length(kelvin, relative_dielectric_const)
        * ionic_strength
        * AVOGADRO_CONSTANT
        * LITER_PER_ANGSTROM3)
        .sqrt()
        .recip()
}
