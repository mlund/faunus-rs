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

//! # Cuboidal, orthorhombic unit cell

use crate::{
    cell::{BoundaryConditions, Shape, SimulationCell, VolumeScale, VolumeScalePolicy},
    Point,
};
use anyhow::Ok;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Cuboidal unit cell
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Cuboid {
    /// Unit cell vectors
    cell: Point,
    /// Half of the cell vectors
    #[serde(skip)]
    half_cell: Point,
}

impl Cuboid {
    /// Create new cuboidal cell with side lengths `a`, `b`, and `c`
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        let cell = Point::new(a, b, c);
        Self {
            cell,
            half_cell: cell.scale(0.5),
        }
    }
    /// Cube with side length `a`
    pub fn cubic(a: f64) -> Self {
        Self::new(a, a, a)
    }
    /// Create new cube with given volume
    pub fn from_volume(volume: f64) -> Self {
        let a = volume.cbrt();
        Self::new(a, a, a)
    }

    /// Sets `half_cell` based on the current cell size.
    pub(super) fn set_half_cell(&mut self) {
        self.half_cell = self.cell.scale(0.5);
    }
}

impl Shape for Cuboid {
    fn volume(&self) -> Option<f64> {
        Some(self.cell.x * self.cell.y * self.cell.z)
    }
    fn is_inside(&self, point: &Point) -> bool {
        point.x.abs() <= self.half_cell.x
            && point.y.abs() <= self.half_cell.y
            && point.z.abs() <= self.half_cell.z
    }
    fn bounding_box(&self) -> Option<Point> {
        Some(self.cell)
    }
    fn get_point_inside(&self, rng: &mut rand::prelude::ThreadRng) -> Point {
        Point::new(
            rng.gen_range(-self.half_cell.x..self.half_cell.x),
            rng.gen_range(-self.half_cell.y..self.half_cell.y),
            rng.gen_range(-self.half_cell.z..self.half_cell.z),
        )
    }
}

impl BoundaryConditions for Cuboid {
    fn pbc(&self) -> super::PeriodicDirections {
        super::PeriodicDirections::PeriodicXYZ
    }
    #[inline(always)]
    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        let mut delta = *point1 - *point2;
        if delta.x > self.half_cell.x {
            delta.x -= self.cell.x;
        } else if delta.x < -self.half_cell.x {
            delta.x += self.cell.x;
        }
        if delta.y > self.half_cell.y {
            delta.y -= self.cell.y;
        } else if delta.y < -self.half_cell.y {
            delta.y += self.cell.y;
        }
        if delta.z > self.half_cell.z {
            delta.z -= self.cell.z;
        } else if delta.z < -self.half_cell.z {
            delta.z += self.cell.z;
        }
        delta
    }
    fn boundary(&self, point: &mut Point) {
        point.x -= self.cell.x * (point.x / self.cell.x).round();
        point.y -= self.cell.y * (point.y / self.cell.y).round();
        point.z -= self.cell.z * (point.z / self.cell.z).round();
    }
}

impl VolumeScale for Cuboid {
    fn scale_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()> {
        if let Some(_old_volume) = self.volume() {
            let mut cell = self.cell;
            self.scale_position(new_volume, &mut cell, policy)?;
            *self = Self::new(cell.x, cell.y, cell.z);
            Ok(())
        } else {
            anyhow::bail!("Cannot set volume of undefined cell volume");
        }
    }
    fn scale_position(
        &self,
        new_volume: f64,
        point: &mut Point,
        policy: VolumeScalePolicy,
    ) -> Result<(), anyhow::Error> {
        let old_volume = self.volume().unwrap();
        match policy {
            VolumeScalePolicy::Isotropic => {
                let scale = (new_volume / old_volume).cbrt();
                *point *= scale;
            }
            VolumeScalePolicy::IsochoricZ => {
                let scale = (new_volume / old_volume).sqrt();
                point.z *= scale;
            }
            VolumeScalePolicy::ScaleZ => {
                let scale = new_volume / old_volume;
                point.z *= scale;
            }
            VolumeScalePolicy::ScaleXY => {
                let scale = (new_volume / old_volume).sqrt();
                point.x *= scale;
                point.y *= scale;
            }
        }
        Ok(())
    }
}

impl SimulationCell for Cuboid {}

#[cfg(test)]
mod tests {
    use crate::cell::Shape;

    use super::Cuboid;

    #[test]
    fn generate_points() {
        let shape = Cuboid::new(10.0, 5.0, 2.5);
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let point = shape.get_point_inside(&mut rng);
            assert!(shape.is_inside(&point));
        }
    }
}
