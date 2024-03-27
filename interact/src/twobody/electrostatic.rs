use super::Info;
use crate::multipole::MultipoleEnergy;
use crate::twobody::TwobodyEnergy;
use serde::Serialize;

/// Monopole-monopole interaction energy
#[derive(Serialize, Clone, PartialEq, Debug)]
pub struct IonIon<T: MultipoleEnergy + 'static> {
    /// Charge number product of the two particles, z₁ × z₂
    #[serde(rename = "z₁z₂")]
    charge_product: f64,
    /// Reference to the potential energy function
    #[serde(skip)]
    multipole: &'static T,
}

impl<T: MultipoleEnergy + 'static> IonIon<T> {
    /// Create a new ion-ion interaction
    pub fn new(charge_product: f64, potential: &'static T) -> Self {
        Self {
            charge_product,
            multipole: potential,
        }
    }
}

impl<T: MultipoleEnergy + 'static> Info for IonIon<T> {
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

impl<T: MultipoleEnergy + 'static + std::fmt::Debug> TwobodyEnergy for IonIon<T> {
    fn twobody_energy(&self, distance_squared: f64) -> f64 {
        self.multipole
            .ion_ion_energy(self.charge_product, 1.0, distance_squared.sqrt())
    }
}
