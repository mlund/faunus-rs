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

use crate::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, VACUUM_ELECTRIC_PERMITTIVITY,
};
use anyhow::Result;
use core::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::Temperature;

/// Trait for objects that has a relative permittivity
pub trait RelativePermittivity {
    /// Get the relative permittivity. May error if the temperature is out of range.
    fn permittivity(&self, temperature: f64) -> Result<f64>;
    /// Set the relative permittivity
    fn set_permittivity(&mut self, _permittivity: f64) -> Result<()> {
        Err(anyhow::anyhow!("Setting permittivity is not implemented"))
    }
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

/// Empirical model for relative permittivity according to Neau and Raspo (NR).
///
/// <https://doi.org/10.1016/j.fluid.2019.112371>
///
/// # Example
/// ~~~
/// use faunus::electrolyte::{PermittivityNR, RelativePermittivity};
/// assert_eq!(PermittivityNR::WATER.permittivity(298.15).unwrap(), 78.35565171480539);
/// assert_eq!(PermittivityNR::METHANOL.permittivity(298.15).unwrap(), 33.081980713895064);
/// assert_eq!(PermittivityNR::ETHANOL.permittivity(298.15).unwrap(), 24.33523434183735);
/// ~~~
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct PermittivityNR {
    /// Coefficients for the model
    coeffs: [f64; 5],
    /// Closed temperature interval in which the model is valid
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
    fn permittivity(&self, temperature: f64) -> Result<f64> {
        if temperature < self.temperature_interval.0 || temperature > self.temperature_interval.1 {
            Err(anyhow::anyhow!(
                "Temperature out of range for permittivity model"
            ))
        } else {
            Ok(self.coeffs[0]
                + self.coeffs[1] * temperature
                + self.coeffs[2] * temperature.powi(2)
                + self.coeffs[3] / temperature
                + self.coeffs[4] * temperature.ln())
        }
    }
}

/// # Common salts and valencies
///
/// Stores the valencies ions in a salt and is used to calculate
/// the ionic strength of arbitrary salt types.
/// Below are some common salts and their valencies:
///
/// Salt      | `valencies`
/// --------- | -----------
/// NaCl      | `[1, -1]`
/// CaCl₂     | `[2, -1]`
/// KAl(SO₄)₂ | `[1, 3, -2]`
///
/// # Examples:
/// ~~~
/// use faunus::electrolyte::Salt;
/// let molarity = 0.1;
///
/// let salt = Salt::SodiumChloride;
/// assert_eq!(salt.valencies(), [1, -1]);
/// assert_eq!(salt.stoichiometry(), [1, 1]);
/// assert_eq!(salt.ionic_strength(molarity), 0.1);
///
/// let alum = Salt::Custom(vec![1, 3, -2]); // e.g. KAl(SO₄)₂
/// assert_eq!(alum.stoichiometry(), [1, 1, 2]);
/// assert_eq!(alum.ionic_strength(molarity), 0.9);
/// ~~~
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub enum Salt {
    /// Sodium chloride, NaCl. This is an example of a 1:1 electrolyte and is the default salt type.
    #[serde(rename = "NaCl")]
    #[default]
    SodiumChloride,
    /// Calcium chloride, CaCl₂
    #[serde(rename = "CaCl₂")]
    CalciumChloride,
    /// Calcium sulfate, CaSO₄
    #[serde(rename = "CaSO₄")]
    CalciumSulfate,
    /// Potassium alum, KAl(SO₄)₂
    #[serde(rename = "KAl(SO₄)₂")]
    PotassiumAlum,
    /// Sodium sulfate, Na₂SO₄
    #[serde(rename = "Na₂SO₄")]
    SodiumSulfate,
    /// Lanthanum chloride, LaCl₃
    #[serde(rename = "LaCl₃")]
    LanthanumChloride,
    /// Salt with custom valencies
    Custom(Vec<isize>),
}

impl Salt {
    /// Valencies of participating ions, zᵢ
    pub fn valencies(&self) -> Vec<isize> {
        match self {
            Salt::SodiumChloride => vec![1, -1],
            Salt::CalciumChloride => vec![2, -1],
            Salt::CalciumSulfate => vec![2, -2],
            Salt::PotassiumAlum => vec![1, 3, -2],
            Salt::SodiumSulfate => vec![1, -2],
            Salt::LanthanumChloride => vec![3, -1],
            Salt::Custom(valencies) => valencies.clone(),
        }
    }

    /// Deduce stoichiometry of the salt, νᵢ
    pub fn stoichiometry(&self) -> Vec<usize> {
        let valencies = self.valencies();
        let sum_positive: isize = valencies.iter().filter(|i| i.is_positive()).sum();
        let sum_negative: isize = valencies.iter().filter(|i| i.is_negative()).sum();
        let gcd = num::integer::gcd(sum_positive, sum_negative);
        if sum_positive == 0 || sum_negative == 0 || gcd == 0 {
            panic!("cannot resolve stoichiometry; did you provide both + and - ions?")
        }
        valencies
            .iter()
            .map(|valency| {
                ((match valency.is_positive() {
                    true => -sum_negative,
                    false => sum_positive,
                }) / gcd) as usize
            })
            .collect()
    }

    /// Calculate ionic strength from the salt molarity (mol/l), I = ½m∑(νᵢzᵢ²)
    pub fn ionic_strength(&self, molarity: f64) -> f64 {
        0.5 * molarity
            * std::iter::zip(self.valencies(), self.stoichiometry().iter().copied())
                .map(|(valency, nu)| (nu * valency.pow(2) as usize))
                .sum::<usize>() as f64
    }
}

impl Display for Salt {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Salt::SodiumChloride => write!(f, "NaCl🧂"),
            Salt::CalciumChloride => write!(f, "CaCl₂"),
            Salt::CalciumSulfate => write!(f, "CaSO₄"),
            Salt::PotassiumAlum => write!(f, "KAl(SO₄)₂"),
            Salt::SodiumSulfate => write!(f, "Na₂SO₄"),
            Salt::LanthanumChloride => write!(f, "LaCl₃"),
            Salt::Custom(valencies) => {
                write!(f, "Custom(")?;
                for (i, valency) in valencies.iter().enumerate() {
                    write!(f, "{}", valency)?;
                    if i < valencies.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}

#[test]
fn test_salt() {
    let molarity = 0.15;

    // NaCl
    assert_eq!(Salt::SodiumChloride.valencies(), [1, -1]);
    assert_eq!(Salt::SodiumChloride.stoichiometry(), [1, 1]);
    approx::assert_abs_diff_eq!(Salt::SodiumChloride.ionic_strength(molarity), molarity);

    // CaSO₄
    assert_eq!(Salt::CalciumSulfate.valencies(), [2, -2]);
    assert_eq!(Salt::CalciumSulfate.stoichiometry(), [1, 1]);
    approx::assert_abs_diff_eq!(
        Salt::CalciumSulfate.ionic_strength(molarity),
        0.5 * (molarity * 4.0 + molarity * 4.0)
    );

    // CaCl₂
    assert_eq!(Salt::CalciumChloride.valencies(), [2, -1]);
    assert_eq!(Salt::CalciumChloride.stoichiometry(), [1, 2]);
    approx::assert_abs_diff_eq!(
        Salt::CalciumChloride.ionic_strength(molarity),
        0.5 * (molarity * 4.0 + 2.0 * molarity)
    );

    // KAl(SO₄)₂
    assert_eq!(Salt::PotassiumAlum.valencies(), [1, 3, -2]);
    assert_eq!(Salt::PotassiumAlum.stoichiometry(), [1, 1, 2]);
    approx::assert_abs_diff_eq!(
        Salt::PotassiumAlum.ionic_strength(molarity),
        0.5 * (molarity * 1.0 + molarity * 9.0 + 2.0 * molarity * 4.0)
    );
}

/// # Implicit solvent medium such as water or a salt solution
///
/// Stores the following properties from which ionic strength,
/// Debye and Bjerrum lengths can be obtained through traits:
///
/// - Relative permittivity
/// - Salt type
/// - Salt molarity
/// - Temperature
///
/// # Examples
/// ~~~
/// use faunus::electrolyte::{Medium, Salt, DebyeLength, RelativePermittivity, IonicStrength};
/// let medium = Medium::neat_water(298.15);
/// assert_eq!(medium.permittivity(298.15).unwrap(), 78.35565171480539);
/// assert_eq!(medium.ionic_strength(), 0.0);
/// assert!(medium.debye_length().is_none());
///
/// let medium = Medium::salt_water(298.15, Salt::CalciumChloride, 0.1);
/// assert_eq!(medium.permittivity(298.15).unwrap(), 78.35565171480539);
/// approx::assert_abs_diff_eq!(medium.ionic_strength(), 0.3);
/// approx::assert_abs_diff_eq!(medium.debye_length().unwrap(), 5.548902662386284);
/// ~~~
pub struct Medium {
    /// Relative permittivity of the medium
    permittivity: Box<dyn RelativePermittivity>,
    /// Salt type
    salt: Option<Salt>,
    /// Salt molarity in mol/l
    molarity: f64,
    /// Temperature in Kelvin
    temperature: f64,
}

impl DebyeLength for Medium {}

impl Medium {
    /// Creates a new medium
    pub fn new(
        temperature: f64,
        permittivity: Box<dyn RelativePermittivity>,
        molarity: f64,
        salt: Option<Salt>,
    ) -> Self {
        Self {
            permittivity,
            salt,
            molarity,
            temperature,
        }
    }
    /// Medium with neat water using the `PermittivityNR::WATER` model
    pub fn neat_water(temperature: f64) -> Self {
        Self {
            permittivity: Box::new(PermittivityNR::WATER),
            salt: None,
            molarity: 0.0,
            temperature,
        }
    }
    /// Medium with salt water using the `PermittivityNR::WATER` model
    pub fn salt_water(temperature: f64, salt: Salt, molarity: f64) -> Self {
        Self {
            permittivity: Box::new(PermittivityNR::WATER),
            salt: Some(salt),
            molarity,
            temperature,
        }
    }
    /// Change the molarity of the salt solution. Error if no salt type is defined.
    pub fn set_molarity(&mut self, molality: f64) -> Result<()> {
        if self.salt.is_some() {
            self.molarity = molality;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Cannot set molarity without a salt"))
        }
    }
    /// Bjerrum length in angstrom, lB = e²/4πεkT
    pub fn bjerrum_length(&self) -> f64 {
        bjerrum_length(
            self.temperature,
            self.permittivity.permittivity(self.temperature).unwrap(),
        )
    }
}

impl Display for Medium {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Medium(T={:.2} K, εᵣ={:.1}, salt={:?}, I={:.2} M, λᴮ={:.1} Å, λᴰ={:.1} Å)",
            self.temperature,
            self.permittivity.permittivity(self.temperature).unwrap(),
            self.salt,
            self.ionic_strength(),
            self.bjerrum_length(),
            self.debye_length().unwrap_or(f64::INFINITY)
        )
    }
}

impl Temperature for Medium {
    fn temperature(&self) -> f64 {
        self.temperature
    }
    fn set_temperature(&mut self, temperature: f64) -> anyhow::Result<()> {
        self.temperature = temperature;
        Ok(())
    }
}
impl RelativePermittivity for Medium {
    fn permittivity(&self, temperature: f64) -> Result<f64> {
        self.permittivity.permittivity(temperature)
    }
}
impl IonicStrength for Medium {
    fn ionic_strength(&self) -> f64 {
        self.salt
            .as_ref()
            .map(|salt| salt.ionic_strength(self.molarity))
            .unwrap_or(0.0)
    }
}

/// Calculates the Bjerrum length, lB = e²/4πεkT commonly used in electrostatics (ångström).
///
/// More information at <https://en.wikipedia.org/wiki/Bjerrum_length>.
///
/// # Examples
/// ~~~
/// use faunus::electrolyte::bjerrum_length;
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
/// use faunus::electrolyte::debye_length;
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