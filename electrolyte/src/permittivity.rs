use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Trait for objects that has a relative permittivity
pub trait RelativePermittivity {
    /// Get the relative permittivity. May error if the temperature is out of range.
    fn permittivity(&self, temperature: f64) -> Result<f64>;
    /// Set the relative permittivity
    fn set_permittivity(&mut self, _permittivity: f64) -> Result<()> {
        Err(anyhow::anyhow!("Setting permittivity is not implemented"))
    }
}

/// Empirical model for relative permittivity according to Neau and Raspo (NR).
///
/// <https://doi.org/10.1016/j.fluid.2019.112371>
///
/// # Example
/// ~~~
/// use electrolyte::{PermittivityNR, RelativePermittivity};
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
