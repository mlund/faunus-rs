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
pub type Vector3 = coulomb::Vector3;
/// A stack-allocated 3x3 square matrix
pub type Matrix3 = coulomb::Matrix3;

mod combination_rule;
pub use self::combination_rule::CombinationRule;
#[cfg(feature = "serde")]
use self::combination_rule::{
    divide4_serialize, multiply4_deserialize, sqrt_serialize, square_deserialize,
};

pub mod fourbody;
pub mod spline;
pub mod threebody;
pub mod twobody;
pub use coulomb::Cutoff;

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
pub use coulomb::TO_CHEMISTRY_UNIT as ELECTRIC_PREFACTOR;
