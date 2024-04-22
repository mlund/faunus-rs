use anyhow::Result;
use core::fmt;
use core::fmt::{Display, Formatter};
use dyn_clone::DynClone;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for objects that has a relative permittivity
pub trait RelativePermittivity: DynClone {
    /// Get the relative permittivity. May error if the temperature is out of range.
    fn permittivity(&self, temperature: f64) -> Result<f64>;

    /// Test is temperature is within range
    fn temperature_is_ok(&self, temperature: f64) -> bool {
        self.permittivity(temperature).is_ok()
    }
}

dyn_clone::clone_trait_object!(RelativePermittivity);

/// Empirical model for the temperature dependent relative permittivity, εᵣ(𝑇),
///
/// For more information, see
/// [Neau and Raspo](https://doi.org/10.1016/j.fluid.2019.112371).
///
/// # Example
/// ~~~
/// use coulomb::{EmpiricalPermittivity, RelativePermittivity};
/// assert_eq!(EmpiricalPermittivity::WATER.permittivity(298.15).unwrap(), 78.35565171480539);
/// assert_eq!(EmpiricalPermittivity::METHANOL.permittivity(298.15).unwrap(), 33.081980713895064);
/// assert_eq!(EmpiricalPermittivity::ETHANOL.permittivity(298.15).unwrap(), 24.33523434183735);
/// ~~~
///
/// We can also pretty print the model:
/// ~~~
/// # use coulomb::EmpiricalPermittivity;
/// assert_eq!(EmpiricalPermittivity::WATER.to_string(),
///            "εᵣ(𝑇) = -1.66e3 + -8.85e-1𝑇 + 3.63e-4𝑇² + 6.48e4/𝑇 + 3.08e2㏑(𝑇); 𝑇 = [273.0, 403.0]");
/// ~~~
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EmpiricalPermittivity {
    /// Coefficients for the model
    coeffs: [f64; 5],
    /// Closed temperature interval in which the model is valid
    temperature_interval: (f64, f64),
}

impl EmpiricalPermittivity {
    /// Creates a new instance of the NR model
    pub const fn new(coeffs: &[f64; 5], temperature_interval: (f64, f64)) -> EmpiricalPermittivity {
        EmpiricalPermittivity {
            coeffs: *coeffs,
            temperature_interval,
        }
    }
    /// Relative permittivity of water
    pub const WATER: EmpiricalPermittivity = EmpiricalPermittivity::new(
        &[-1664.4988, -0.884533, 0.0003635, 64839.1736, 308.3394],
        (273.0, 403.0),
    );
    /// Relative permittivity of methanol
    pub const METHANOL: EmpiricalPermittivity = EmpiricalPermittivity::new(
        &[-1750.3069, -0.99026, 0.0004666, 51360.2652, 327.3124],
        (176.0, 318.0),
    );
    /// Relative permittivity of ethanol
    pub const ETHANOL: EmpiricalPermittivity = EmpiricalPermittivity::new(
        &[-1522.2782, -1.00508, 0.0005211, 38733.9481, 293.1133],
        (288.0, 328.0),
    );
}

impl RelativePermittivity for EmpiricalPermittivity {
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

impl Display for EmpiricalPermittivity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "εᵣ(𝑇) = {:.2e} + {:.2e}𝑇 + {:.2e}𝑇² + {:.2e}/𝑇 + {:.2e}㏑(𝑇); 𝑇 = [{:.1}, {:.1}]",
            self.coeffs[0],
            self.coeffs[1],
            self.coeffs[2],
            self.coeffs[3],
            self.coeffs[4],
            self.temperature_interval.0,
            self.temperature_interval.1
        )
    }
}
