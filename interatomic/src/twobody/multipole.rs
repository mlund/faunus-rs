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

//! # Electric multipole interactions
//!
//! This module provides tools for calculating the two-body interaction energy between
//! electric multipole moments, such as monopoles, dipoles, quadrupoles etc.

use crate::twobody::IsotropicTwobodyEnergy;
use coulomb::pairwise::MultipoleEnergy;
#[cfg(feature = "serde")]
use serde::Serialize;

/// Monopole-monopole interaction energy
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct IonIon<T: MultipoleEnergy> {
    /// Charge number product of the two particles, z₁ × z₂
    #[cfg_attr(feature = "serde", serde(rename = "z₁z₂", alias = "z1z2"))]
    charge_product: f64,
    /// Potential energy function to use.
    #[cfg_attr(feature = "serde", serde(skip))]
    scheme: T,
}

impl<T: MultipoleEnergy> IonIon<T> {
    /// Create a new ion-ion interaction
    pub const fn new(charge_product: f64, scheme: T) -> Self {
        Self {
            charge_product,
            scheme,
        }
    }
}

impl<T: MultipoleEnergy + std::fmt::Debug + Clone + PartialEq + Send + Sync> IsotropicTwobodyEnergy
    for IonIon<T>
{
    /// Calculate the isotropic twobody energy (kJ/mol)
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        coulomb::TO_CHEMISTRY_UNIT / 80.0
            * self
                .scheme
                .ion_ion_energy(self.charge_product, 1.0, distance_squared.sqrt())
    }
}

/// Alias for ion-ion with Yukawa
pub type IonIonYukawa<'a> = IonIon<coulomb::pairwise::Yukawa>;

/// Alias for ion-ion with a plain Coulomb potential that can be screened
pub type IonIonPlain<'a> = IonIon<coulomb::pairwise::Plain>;

// Test ion-ion energy
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use coulomb::pairwise::Plain;

    #[test]
    fn test_ion_ion() {
        let r: f64 = 7.0;
        let cutoff = f64::INFINITY;
        let scheme = Plain::new(cutoff, None);
        let ionion = IonIon::new(1.0, scheme);
        let unscreened_energy = ionion.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(unscreened_energy, 2.48099031507825);
        let debye_length = 30.0;
        let scheme = Plain::new(cutoff, Some(debye_length));
        let ionion = IonIon::new(1.0, scheme);
        let screened_energy = ionion.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(
            screened_energy,
            unscreened_energy * (-r / debye_length).exp()
        );
    }
}
