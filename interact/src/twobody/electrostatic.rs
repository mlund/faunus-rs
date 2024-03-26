use super::Info;
use crate::multipole::Potential;
use serde::Serialize;

/// Monopole-monopole interaction energy
#[derive(Serialize, Clone, PartialEq, Debug)]
pub struct IonIon<T: Potential + 'static> {
    /// Charge number product of the two particles, z₁ × z₂
    charge_product: f64,
    /// Reference to the potential energy function
    #[serde(skip)]
    potential: &'static T,
}

impl<T: Potential + 'static> IonIon<T> {
    /// Create a new ion-ion interaction
    pub fn new(charge_product: f64, potential: &'static T) -> Self {
        Self {
            charge_product,
            potential,
        }
    }
}

impl<T: Potential + 'static> Info for IonIon<T> {
    fn short_name(&self) -> Option<&'static str> {
        Some("ion-ion")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Ion-ion interaction")
    }
    fn citation(&self) -> Option<&'static str> {
        None
    }
}

// #[typetag::serialize]
// impl<T: Potential + 'static> TwobodyEnergy for IonIon<T> {
//     fn twobody_energy(&self, distance_squared: f64) -> f64 {
//         0.0
//         // self.charge_product * self.potential.potential(distance_squared)
//     }
// }
