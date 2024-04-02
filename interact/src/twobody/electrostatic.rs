use super::Info;
use crate::multipole::MultipoleEnergy;
use crate::twobody::IsotropicTwobodyEnergy;
use serde::Serialize;

/// Monopole-monopole interaction energy
#[derive(Serialize, Clone, PartialEq, Debug)]
pub struct IonIon<'a, T: MultipoleEnergy> {
    /// Charge number product of the two particles, z₁ × z₂
    #[serde(rename = "z₁z₂")]
    charge_product: f64,
    /// Reference to the potential energy function
    #[serde(skip)]
    multipole: &'a T,
}

impl<'a, T: MultipoleEnergy> IonIon<'a, T> {
    /// Create a new ion-ion interaction
    pub fn new(charge_product: f64, potential: &'static T) -> Self {
        Self {
            charge_product,
            multipole: potential,
        }
    }
}

impl<'a, T: MultipoleEnergy> Info for IonIon<'a, T> {
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

impl<T: MultipoleEnergy + 'static + std::fmt::Debug> IsotropicTwobodyEnergy for IonIon<'_, T> {
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        self.multipole
            .ion_ion_energy(self.charge_product, 1.0, distance_squared.sqrt())
    }
}

/// Alias for ion-ion with Yukawa
pub type IonIonYukawa<'a> = IonIon<'a, crate::multipole::Yukawa>;
