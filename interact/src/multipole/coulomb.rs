// Copyright 2023 Björn Stenqvist and Mikael Lund
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

//! # Plain Coulomb potential where _S(q)_ = 1

use super::{Energy, Field, Force, Potential, SplitingFunction};
#[cfg(test)]
use crate::{Matrix3, Point};
#[cfg(test)]
use approx::assert_relative_eq;
use serde::{Deserialize, Serialize};

impl Potential for Coulomb {}
impl Field for Coulomb {}
impl Energy for Coulomb {}
impl Force for Coulomb {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Coulomb {
    cutoff: f64,
}

impl Coulomb {
    pub fn new(cutoff: f64) -> Self {
        Self { cutoff }
    }
}

impl crate::Info for Coulomb {
    fn short_name(&self) -> Option<&'static str> {
        Some("coulomb")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Coulomb potential")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("Premier mémoire sur l’électricité et le magnétisme by Charles-Augustin de Coulomb")
    }
}

impl crate::Cutoff for Coulomb {
    #[inline]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl SplitingFunction for Coulomb {
    #[inline]
    fn kappa(&self) -> Option<f64> {
        None
    }
    #[inline]
    fn short_range_function(&self, _q: f64) -> f64 {
        1.0
    }
    #[inline]
    fn short_range_function_derivative(&self, _q: f64) -> f64 {
        0.0
    }
    #[inline]
    fn short_range_function_second_derivative(&self, _q: f64) -> f64 {
        0.0
    }
    #[inline]
    fn short_range_function_third_derivative(&self, _q: f64) -> f64 {
        0.0
    }
}

#[test]
fn test_coulomb() {
    let cutoff: f64 = 29.0; // cutoff distance
    let z1 = 2.0; // charge
    let z2 = 3.0; // charge
    let mu1 = Point::new(19.0, 7.0, 11.0); // dipole moment
    let mu2 = Point::new(13.0, 17.0, 5.0); // dipole moment
    let quad1 = Matrix3::new(3.0, 7.0, 8.0, 5.0, 9.0, 6.0, 2.0, 1.0, 4.0); // quadrupole moment
    let _quad2 = Matrix3::zeros(); // quadrupole moment
    let r = Point::new(23.0, 0.0, 0.0); // distance vector
    let rq = Point::new(
        5.75 * (6.0f64).sqrt(),
        5.75 * (2.0f64).sqrt(),
        11.5 * (2.0f64).sqrt(),
    ); // distance vector for quadrupole check
    let rh = Point::new(1.0, 0.0, 0.0); // normalized distance vector

    let pot = Coulomb::new(cutoff);
    let eps = 1e-9;

    // Test short-ranged function
    assert_eq!(pot.short_range_function(0.5), 1.0);
    assert_eq!(pot.short_range_function_derivative(0.5), 0.0);
    assert_eq!(pot.short_range_function_second_derivative(0.5), 0.0);
    assert_eq!(pot.short_range_function_third_derivative(0.5), 0.0);

    // Test potentials
    assert_eq!(pot.ion_potential(z1, cutoff + 1.0), 0.0);
    assert_relative_eq!(
        pot.ion_potential(z1, r.norm()),
        0.08695652173913043,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &((cutoff + 1.0) * rh)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &r),
        0.035916824196597356,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.quadrupole_potential(&quad1, &rq),
        0.00093632817,
        epsilon = eps
    );

    // Test fields
    assert_relative_eq!(
        pot.ion_field(z1, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );
    let ion_field = pot.ion_field(z1, &r);
    assert_relative_eq!(ion_field[0], 0.003780718336, epsilon = eps);
    assert_relative_eq!(ion_field.norm(), 0.003780718336, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_field(&mu1, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );
    let dip_field = pot.dipole_field(&mu1, &r);
    assert_relative_eq!(dip_field[0], 0.003123202104, epsilon = eps);
    assert_relative_eq!(dip_field[1], -0.0005753267034, epsilon = eps);
    assert_relative_eq!(dip_field[2], -0.0009040848196, epsilon = eps);
    let quad_field = pot.quadrupole_field(&quad1, &r);
    assert_relative_eq!(quad_field[0], -0.00003752130674, epsilon = eps);
    assert_relative_eq!(quad_field[1], -0.00006432224013, epsilon = eps);
    assert_relative_eq!(quad_field[2], -0.00005360186677, epsilon = eps);

    // Test energies
    approx::assert_relative_eq!(pot.ion_ion_energy(z1, z2, cutoff + 1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(
        pot.ion_ion_energy(z1, z2, r.norm()),
        z1 * z2 / r.norm(),
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &((cutoff + 1.0) * rh)),
        -0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &r),
        -0.04914933837,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &((cutoff + 1.0) * rh)),
        -0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &r),
        -0.02630064930,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.ion_quadrupole_energy(z2, &quad1, &rq),
        0.002808984511,
        epsilon = eps
    );

    // Test forces
    assert_relative_eq!(
        pot.ion_ion_force(z1, z2, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );
    let ionion = pot.ion_ion_force(z1, z2, &r);
    assert_relative_eq!(ionion[0], 0.01134215501, epsilon = eps);
    assert_relative_eq!(ionion.norm(), 0.01134215501, epsilon = eps);
    assert_relative_eq!(
        pot.ion_dipole_force(z2, &mu1, &((cutoff + 1.0) * rh))
            .norm(),
        0.0,
        epsilon = eps
    );
    let iondipole = pot.ion_dipole_force(z2, &mu1, &r);
    assert_relative_eq!(iondipole[0], 0.009369606312, epsilon = eps);
    assert_relative_eq!(iondipole[1], -0.001725980110, epsilon = eps);
    assert_relative_eq!(iondipole[2], -0.002712254459, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_dipole_force(&mu1, &mu2, &((cutoff + 1.0) * rh))
            .norm(),
        0.0,
        epsilon = eps
    );
    let dipoledipole = pot.dipole_dipole_force(&mu1, &mu2, &r);
    assert_relative_eq!(dipoledipole[0], 0.003430519474, epsilon = eps);
    assert_relative_eq!(dipoledipole[1], -0.004438234569, epsilon = eps);
    assert_relative_eq!(dipoledipole[2], -0.002551448858, epsilon = eps);
}
