pub use nalgebra::{Matrix3, UnitQuaternion, Vector3};
mod anglescan;
pub mod energy;
pub mod structure;
pub mod table;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;

use std::iter::Sum;
use std::ops::{Add, Neg};

pub use anglescan::{
    make_fibonacci_sphere, make_icosphere, make_icosphere_vertices, TwobodyAngles,
    IcoSphereWithNeighbors,
};

/// RMSD angle between two quaternion rotations
///
/// The root-mean-square deviation (RMSD) between two quaternion rotations is
/// defined as the square of the angle between the two quaternions.
///
/// - <https://fr.mathworks.com/matlabcentral/answers/415936-angle-between-2-quaternions>
/// - <https://github.com/charnley/rmsd>
/// - <https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.20296>
/// - <https://www.ams.stonybrook.edu/~coutsias/papers/2004-rmsd.pdf>
pub fn rmsd_angle(q1: &UnitQuaternion<f64>, q2: &UnitQuaternion<f64>) -> f64 {
    // let q = q1 * q2.inverse();
    // q.angle().powi(2)
    q1.angle_to(q2).powi(2)
}

#[allow(non_snake_case)]
pub fn rmsd2(Q: &UnitQuaternion<f64>, inertia: &Matrix3<f64>, total_mass: f64) -> f64 {
    let q = Q.vector();
    4.0 / total_mass * (q.transpose() * inertia * q)[0]
}

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
