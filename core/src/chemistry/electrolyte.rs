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

use anyhow::Result;

use crate::{BOLTZMANN, UNIT_CHARGE, VACUUM_PERMITTIVITY};

/// Stores information about salts for calculation of Debye screening length etc.
///
/// In this context a _salt_ is an arbitrary set of cations and anions, combined to form
/// a net-neutral compound. The object state is _temperature independent_.
/// The stoichiometry is automatically worked out.
///
/// # Example valencies:
///
/// Salt      | `valencies`
/// --------- | -------------
/// NaCl      | `[1, -1]`
/// CaCl₂     | `[2, -1]`
/// KAl(SO₄)₂ | `[1, 3, -2]`
///
#[derive(Debug, PartialEq, Clone)]
pub struct Electrolyte {
    /// Molar salt concentration
    pub molarity: f64,
    /// Molar ionic strength concentration
    pub ionic_strength: f64,
    /// valencies for participating ions
    pub valencies: Vec<isize>,
}

impl Electrolyte {
    pub fn new(molarity: f64, valencies: &[isize]) -> Result<Electrolyte> {
        let sum_positive: isize = valencies.iter().filter(|i| i.is_positive()).sum();
        let sum_negative: isize = valencies.iter().filter(|i| i.is_negative()).sum();
        let gcd = num::integer::gcd(sum_positive, sum_negative);
        if sum_positive == 0 || sum_negative == 0 || gcd == 0 {
            anyhow::bail!("cannot resolve stoichiometry; did you provide both + and - ions?")
        }
        let nu_times_squared_valency = valencies.iter().map(|valency| {
            let nu = match valency.is_positive() {
                true => -sum_negative,
                false => sum_positive,
            } / gcd;
            (nu * valency * valency) as f64
        });
        let ionic_strength = 0.5 * molarity * nu_times_squared_valency.sum::<f64>();

        Ok(Electrolyte {
            molarity,
            ionic_strength,
            valencies: Vec::from(valencies),
        })
    }
}

#[test]
fn test_electrolyte() {
    let molarity = 0.1;

    // NaCl
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &[1, -1]).unwrap().ionic_strength,
        0.1
    );
    // CaSO₄
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &[2, -2]).unwrap().ionic_strength,
        0.5 * (0.1 * 4.0 + 0.1 * 4.0)
    );

    // CaCl₂
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &[2, -1]).unwrap().ionic_strength,
        0.5 * (0.1 * 4.0 + 0.2)
    );

    // KAl(SO₄)₂
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &[1, 3, -2])
            .unwrap()
            .ionic_strength,
        0.5 * (0.1 * 1.0 + 0.1 * 9.0 + 0.1 * 2.0 * 4.0)
    );

    // Invalid combinations
    assert!(Electrolyte::new(molarity, &[1, 1]).is_err());
    assert!(Electrolyte::new(molarity, &[-1, -1]).is_err());
    assert!(Electrolyte::new(molarity, &[0, 0]).is_err());
}

/// Calculates the Bjerrum length, lB = e²/4πεkT commonly used in electrostatics (ångström).
///
/// More information [here](https://en.wikipedia.org/wiki/Bjerrum_length).
///
/// # Examples
/// ~~~
/// use faunus::chemistry::electrolyte::bjerrum_length;
/// let lB = bjerrum_length(298.15, 80.0); // angstroms
/// assert_eq!(lB, 7.00574152684418);
/// ~~~
pub fn bjerrum_length(kelvin: f64, relative_dielectric_const: f64) -> f64 {
    UNIT_CHARGE.powi(2) * 1e10
        / (4.0
            * std::f64::consts::PI
            * relative_dielectric_const
            * VACUUM_PERMITTIVITY
            * BOLTZMANN
            * kelvin)
}
