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

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::chemistry::RelativePermittivity;
use crate::{AVOGADRO, BOLTZMANN, UNIT_CHARGE, VACUUM_PERMITTIVITY};

/// Empirical model for relative permittivity according to Raspo and Neau (NR).
///
/// https://doi.org/10.1016/j.fluid.2019.112371
///
/// # Example
/// ~~~
/// use faunus::chemistry::{PermittivityNR, RelativePermittivity};
/// assert_eq!(PermittivityNR::WATER.relative_permittivity(298.15).unwrap(), 78.35565171480539);
/// assert_eq!(PermittivityNR::METHANOL.relative_permittivity(298.15).unwrap(), 33.081980713895064);
/// assert_eq!(PermittivityNR::ETHANOL.relative_permittivity(298.15).unwrap(), 24.33523434183735);
/// ~~~
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct PermittivityNR {
    coeffs: [f64; 5],
    temperature_interval: (f64, f64),
}

impl PermittivityNR {
    /// Creates a new instance of the NR model
    pub const fn new(coeffs: &[f64; 5], temperature_interval: (f64, f64)) -> PermittivityNR {
        PermittivityNR {
            coeffs: *coeffs,
            temperature_interval,
        }
    }
    /// Relative permittivity of water
    pub const WATER: PermittivityNR = PermittivityNR::new(
        &[-1664.4988, -0.884533, 0.0003635, 64839.1736, 308.3394],
        (273.0, 403.0),
    );
    /// Relative permittivity of methanol
    pub const METHANOL: PermittivityNR = PermittivityNR::new(
        &[-1750.3069, -0.99026, 0.0004666, 51360.2652, 327.3124],
        (176.0, 318.0),
    );
    /// Relative permittivity of ethanol
    pub const ETHANOL: PermittivityNR = PermittivityNR::new(
        &[-1522.2782, -1.00508, 0.0005211, 38733.9481, 293.1133],
        (288.0, 328.0),
    );
}

impl RelativePermittivity for PermittivityNR {
    fn relative_permittivity(&self, temperature: f64) -> Result<f64> {
        if temperature < self.temperature_interval.0 || temperature > self.temperature_interval.1 {
            anyhow::bail!("temperature out of range")
        }
        Ok(self.coeffs[0]
            + self.coeffs[1] * temperature
            + self.coeffs[2] * temperature.powi(2)
            + self.coeffs[3] / temperature
            + self.coeffs[4] * temperature.ln())
    }
}

/// Stores information about salts for calculation of Debye screening length etc.
///
/// In this context a _salt_ is an arbitrary set of cations and anions, combined to form
/// a net-neutral compound. The object state is _temperature independent_.
/// The stoichiometry is automatically worked out.
///
/// # Example usage:
/// ~~~
/// use faunus::chemistry::Electrolyte;
/// let molarity = 0.1;
/// let salt = Electrolyte::new(molarity, &Electrolyte::SODIUM_CHLORIDE).unwrap();    // Nacl
/// assert_eq!(salt.ionic_strength, 0.1);
/// assert_eq!(salt.stoichiometry, [1, 1]);
/// let alum = Electrolyte::new(molarity, &[1, 3, -2]).unwrap(); // KAl(SO₄)₂
/// assert_eq!(alum.ionic_strength, 0.9);
/// assert_eq!(alum.stoichiometry, [1, 1, 2]);
/// ~~~
///
/// # Example valencies:
///
/// Salt      | `valencies`
/// --------- | ---------------------------------
/// NaCl      | `[1, -1]` or `SODIUM_CHLORIDE`
/// CaCl₂     | `[2, -1]` or `CALCIUM_CHLORIDE`
/// KAl(SO₄)₂ | `[1, 3, -2]` or `POTASSIUM_ALUM`
///
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct Electrolyte {
    /// Molar salt concentration
    pub molarity: f64,
    /// Molar ionic strength
    pub ionic_strength: f64,
    /// Valencies for participating ions
    pub valencies: Vec<isize>,
    /// Stoichiometric coefficients for participating ions
    pub stoichiometry: Vec<usize>,
}

impl Electrolyte {
    /// Valencies for sodium chloride, NaCl
    pub const SODIUM_CHLORIDE: [isize; 2] = [1, -1];
    /// Valencies for calcium chloride, CaCl₂
    pub const CALCIUM_CHLORIDE: [isize; 2] = [2, -1];
    /// Valencies for calcium sulfate, CaSO₄
    pub const CALCIUM_SULFATE: [isize; 2] = [2, -2];
    /// Valencies for potassium alum, KAl(SO₄)₂
    pub const POTASSIUM_ALUM: [isize; 3] = [1, 3, -2];
    /// Valencies for sodium sulfate, Na₂SO₄
    pub const SODIUM_SULFATE: [isize; 2] = [1, -2];
    /// Valencies for lanthanum chloride, LaCl₃
    pub const LANTHANUM_CHLORIDE: [isize; 2] = [3, -1];

    pub fn new(molarity: f64, valencies: &[isize]) -> Result<Electrolyte> {
        let sum_positive: isize = valencies.iter().filter(|i| i.is_positive()).sum();
        let sum_negative: isize = valencies.iter().filter(|i| i.is_negative()).sum();
        let gcd = num::integer::gcd(sum_positive, sum_negative);
        if sum_positive == 0 || sum_negative == 0 || gcd == 0 {
            anyhow::bail!("cannot resolve stoichiometry; did you provide both + and - ions?")
        }

        let stoichiometry: Vec<usize> = valencies
            .iter()
            .map(|valency| {
                ((match valency.is_positive() {
                    true => -sum_negative,
                    false => sum_positive,
                }) / gcd) as usize
            })
            .collect();

        let nu_times_squared_valency_sum: usize = std::iter::zip(valencies, stoichiometry.iter())
            .map(|(valency, nu)| (*nu * valency.pow(2) as usize))
            .sum();

        let ionic_strength = 0.5 * molarity * nu_times_squared_valency_sum as f64;

        Ok(Electrolyte {
            molarity,
            ionic_strength,
            valencies: Vec::from(valencies),
            stoichiometry,
        })
    }

    /// Calculates the Debye screening length, given the Bjerrum length (angstrom)
    pub fn debye_length(&self, bjerrum_length: f64) -> f64 {
        const LITER_PER_ANGSTROM3: f64 = 1e-27;
        (8.0 * PI * bjerrum_length * self.ionic_strength * AVOGADRO * LITER_PER_ANGSTROM3)
            .sqrt()
            .recip()
    }
}

#[test]
fn test_electrolyte() {
    let molarity = 0.15;

    // NaCl
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &Electrolyte::SODIUM_CHLORIDE)
            .unwrap()
            .ionic_strength,
        molarity
    );
    // CaSO₄
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &Electrolyte::CALCIUM_SULFATE)
            .unwrap()
            .ionic_strength,
        0.5 * (molarity * 4.0 + molarity * 4.0)
    );

    // CaCl₂
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &Electrolyte::CALCIUM_CHLORIDE)
            .unwrap()
            .ionic_strength,
        0.5 * (molarity * 4.0 + 2.0 * molarity)
    );

    // KAl(SO₄)₂
    approx::assert_abs_diff_eq!(
        Electrolyte::new(molarity, &Electrolyte::POTASSIUM_ALUM)
            .unwrap()
            .ionic_strength,
        0.5 * (molarity * 1.0 + molarity * 9.0 + 2.0 * molarity * 4.0)
    );

    // Invalid combinations
    assert!(Electrolyte::new(molarity, &[1, 1]).is_err());
    assert!(Electrolyte::new(molarity, &[-1, -1]).is_err());
    assert!(Electrolyte::new(molarity, &[0, 0]).is_err());
    assert!(Electrolyte::new(molarity, &[0, 1]).is_err());

    // Debye length
    let bjerrum_length = bjerrum_length(293.0, 80.0);
    assert_eq!(
        Electrolyte::new(0.03, &[1, -1])
            .unwrap()
            .debye_length(bjerrum_length),
        17.576538097378368
    );
}

/// Calculates the Bjerrum length, lB = e²/4πεkT commonly used in electrostatics (ångström).
///
/// More information at <https://en.wikipedia.org/wiki/Bjerrum_length>.
///
/// # Examples
/// ~~~
/// use faunus::chemistry::bjerrum_length;
/// let lB = bjerrum_length(293.0, 80.0); // angstroms
/// assert_eq!(lB, 7.1288799871283);
/// ~~~
pub fn bjerrum_length(kelvin: f64, relative_dielectric_const: f64) -> f64 {
    const ANGSTROM_PER_METER: f64 = 1e10;
    UNIT_CHARGE.powi(2) * ANGSTROM_PER_METER
        / (4.0 * PI * relative_dielectric_const * VACUUM_PERMITTIVITY * BOLTZMANN * kelvin)
}
