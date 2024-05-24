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

//! # Spherical cell with hard walls, i.e. no periodic boundary conditions

use crate::{
    cell::{BoundaryConditions, Shape, SimulationCell, VolumeScale, VolumeScalePolicy},
    Point,
};
use anyhow::Ok;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Spherical cell with hard walls, i.e. no periodic boundary conditions
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq)]
pub struct Sphere {
    /// The center of the sphere
    radius: f64,
}

impl Sphere {
    /// Create new sphere with given radius
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
    /// Create new sphere with given volume
    pub fn from_volume(volume: f64) -> Self {
        Self {
            radius: (volume / (4.0 / 3.0 * std::f64::consts::PI)).cbrt(),
        }
    }
}

impl BoundaryConditions for Sphere {
    fn pbc(&self) -> super::PeriodicDirections {
        super::PeriodicDirections::None
    }
    fn boundary(&self, _point: &mut Point) {}
    #[inline]
    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        point1 - point2
    }
}

impl Shape for Sphere {
    fn volume(&self) -> Option<f64> {
        Some(4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3))
    }
    fn is_inside(&self, point: &Point) -> bool {
        point.norm_squared() < self.radius.powi(2)
    }
    /// Creates a box which volume fits the sphere.
    fn bounding_box(&self) -> Option<Point> {
        Some(Point::from_element(2.0 * self.radius))
    }
    /// Get random point located inside the Sphere.
    fn get_point_inside(&self, rng: &mut rand::prelude::ThreadRng) -> Point {
        let r2 = self.radius * self.radius;
        let d = 2.0 * self.radius;
        let mut point;
        loop {
            point = Point::new(
                (rng.gen::<f64>() - 0.5) * d,
                (rng.gen::<f64>() - 0.5) * d,
                (rng.gen::<f64>() - 0.5) * d,
            );
            if point.norm_squared() < r2 {
                return point;
            }
        }
    }
}

impl SimulationCell for Sphere {}

impl VolumeScale for Sphere {
    fn scale_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()> {
        if policy != VolumeScalePolicy::Isotropic {
            anyhow::bail!("Sphere only supports isotropic volume scaling")
        }
        *self = Self::from_volume(new_volume);
        Ok(())
    }
    fn scale_position(
        &self,
        new_volume: f64,
        point: &mut Point,
        policy: VolumeScalePolicy,
    ) -> Result<(), anyhow::Error> {
        match policy {
            VolumeScalePolicy::Isotropic => {
                let new_radius = (new_volume / (4.0 / 3.0 * std::f64::consts::PI)).cbrt();
                *point *= new_radius / self.radius;
            }
            _ => {
                anyhow::bail!("Sphere only supports isotropic volume scaling")
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::cell::Shape;

    use super::Sphere;

    #[test]
    fn generate_points() {
        let shape = Sphere::new(2.5);
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let point = shape.get_point_inside(&mut rng);
            assert!(shape.is_inside(&point));
        }
    }
}
