use super::IsotropicTwobodyEnergy;
#[cfg(feature = "serde")]
use crate::{sqrt_serialize, square_deserialize};
use crate::{Cutoff, Info};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Hardsphere potential
///
/// More information [here](http://www.sklogwiki.org/SklogWiki/index.php/Hard_sphere_model).
/// # Examples
/// ~~~
/// use interatomic::twobody::{HardSphere, IsotropicTwobodyEnergy};
/// let hardsphere = HardSphere::new(1.0);
/// let distance: f64 = 0.9; // smaller than the minimum distance
/// assert!(hardsphere.isotropic_twobody_energy(distance.powi(2)).is_infinite());
/// let distance: f64 = 1.1; // greater than the minimum distance
/// assert_eq!(hardsphere.isotropic_twobody_energy(distance.powi(2)), 0.0);
/// ~~~
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct HardSphere {
    /// Minimum distance
    #[cfg_attr(
        feature = "serde",
        serde(
            rename = "σ",
            serialize_with = "sqrt_serialize",
            deserialize_with = "square_deserialize"
        )
    )]
    min_distance_squared: f64,
}

impl HardSphere {
    /// Create by giving the minimum distance where if smaller, the energy is infinite or zero otherwise
    pub fn new(min_distance: f64) -> Self {
        Self {
            min_distance_squared: min_distance.powi(2),
        }
    }
}

impl IsotropicTwobodyEnergy for HardSphere {
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared < self.min_distance_squared {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

impl Cutoff for HardSphere {
    fn cutoff(&self) -> f64 {
        self.cutoff_squared().sqrt()
    }
    fn cutoff_squared(&self) -> f64 {
        self.min_distance_squared
    }
}

impl Info for HardSphere {
    fn short_name(&self) -> Option<&'static str> {
        Some("hardsphere")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("https://en.wikipedia.org/wiki/Hard_spheres")
    }
}
