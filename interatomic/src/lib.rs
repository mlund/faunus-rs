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

/// A point in 3D space
pub type Vector3 = nalgebra::Vector3<f64>;
/// A stack-allocated 3x3 square matrix
pub type Matrix3 = nalgebra::Matrix3<f64>;

mod combination_rule;
pub use crate::combination_rule::CombinationRule;
pub mod fourbody;
mod qpochhammer;
pub mod spline;
pub mod threebody;
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
