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
    pub fn new(charge_product: f64, multipole: &'a T) -> Self {
        Self {
            charge_product,
            multipole,
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

impl<T: MultipoleEnergy + std::fmt::Debug> IsotropicTwobodyEnergy for IonIon<'_, T> {
    /// Calculate the isotropic twobody energy (kJ/mol)
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        self.multipole.prefactor()
            * self
                .multipole
                .ion_ion_energy(self.charge_product, 1.0, distance_squared.sqrt())
    }
}

/// Alias for ion-ion with Yukawa
pub type IonIonYukawa<'a> = IonIon<'a, crate::multipole::Yukawa>;

/// Alias for ion-ion with a plain Coulomb potential that can be screened
pub type IonIonPlain<'a> = IonIon<'a, crate::multipole::Coulomb>;

// Test ion-ion energy
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::multipole::Coulomb;

    #[test]
    fn test_ion_ion() {
        let r: f64 = 7.0;
        let cutoff = f64::INFINITY;
        let permittivity = 80.0;
        let scheme = Coulomb::new(80.0, cutoff, None);
        let ionion = IonIon::new(1.0, &scheme);
        let unscreened_energy = ionion.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(unscreened_energy, 2.48099031507825);
        let debye_length = 30.0;
        let scheme = Coulomb::new(permittivity, cutoff, Some(debye_length));
        let ionion = IonIon::new(1.0, &scheme);
        let screened_energy = ionion.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(
            screened_energy,
            unscreened_energy * (-r / debye_length).exp()
        );
    }
}
