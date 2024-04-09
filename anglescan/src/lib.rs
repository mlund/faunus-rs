pub use nalgebra::Vector3;
pub mod anglescan;
pub mod energy;
pub mod structure;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;

use faunus;
use std::iter::Sum;
use std::ops::{Add, Neg};

/// Structure to store energy samples
#[derive(Debug, Default, Clone)]
pub struct Sample {
    /// Number of samples
    n: u64,
    /// Thermal energy, RT in kJ/mol
    pub thermal_energy: f64,
    /// Boltzmann weighted energy, U * exp(-U/kT)
    pub mean_energy: f64,
    /// Boltzmann factored energy, exp(-U/kT)
    pub exp_energy: f64,
}

impl Sample {
    /// New from energy in kJ/mol and temperature in K
    pub fn new(energy: f64, temperature: f64) -> Self {
        let thermal_energy = faunus::MOLAR_GAS_CONSTANT * temperature * 1e-3; // kJ/mol
        let exp_energy = (-energy / thermal_energy).exp();
        Self {
            n: 1,
            thermal_energy,
            mean_energy: energy * exp_energy,
            exp_energy,
        }
    }
    /// Mean energy (kJ/mol)
    pub fn mean_energy(&self) -> f64 {
        self.mean_energy / self.exp_energy
    }
    /// Free energy (kJ / mol)
    pub fn free_energy(&self) -> f64 {
        (self.exp_energy / self.n as f64).ln().neg() * self.thermal_energy
    }
}

impl Sum for Sample {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Sample::default(), |sum, s| sum + s)
    }
}

impl Add for Sample {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            n: self.n + other.n,
            thermal_energy: f64::max(self.thermal_energy, other.thermal_energy),
            mean_energy: self.mean_energy + other.mean_energy,
            exp_energy: self.exp_energy + other.exp_energy,
        }
    }
}
