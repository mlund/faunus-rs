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
use serde::{Deserialize, Serialize};

/// Cuboidal unit cell
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cuboid {
    /// Unit cell vectors
    cell: Point,
    /// Half of the cell vectors
    half_cell: Point,
}

impl Cuboid {
    /// Create new cuboidal cell with side lengths `a`, `b`, and `c`
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self {
            cell: Point::new(a, b, c),
            half_cell: Point::new(a / 2.0, b / 2.0, c / 2.0),
        }
    }
    /// Create new cube with given volume
    pub fn from_volume(volume: f64) -> Self {
        let a = volume.cbrt();
        Self::new(a, a, a)
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
}

impl BoundaryConditions for Cuboid {
    fn pbc(&self) -> super::PeriodicDirections {
        super::PeriodicDirections::PeriodicXYZ
    }
    #[inline]
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
        if point.x.abs() > self.half_cell.x {
            point.x -= self.cell.x * (point.x / self.cell.x).round();
        }
        if point.y.abs() > self.half_cell.y {
            point.y -= self.cell.y * (point.y / self.cell.y).round();
        }
        if point.z.abs() > self.half_cell.z {
            point.z -= self.cell.z * (point.z / self.cell.z).round();
        }
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
