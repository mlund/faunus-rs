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

//! # Spherical unit cell with no periodic boundary conditions

use crate::{
    cell::{BoundaryConditions, Shape, SimulationCell, VolumeScale, VolumeScalePolicy},
    Point,
};
use anyhow::Ok;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq)]
pub struct Sphere {
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
    fn center(&self) -> Point {
        Point::zeros()
    }
    fn volume(&self) -> Option<f64> {
        Some(4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3))
    }
    fn is_inside(&self, point: &Point) -> bool {
        point.norm_squared() < self.radius.powi(2)
    }
    fn bounding_box(&self) -> Option<Point> {
        Some(Point::from_element(2.0 * self.radius))
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
