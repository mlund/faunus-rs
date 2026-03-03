// Copyright 2023-2026 Mikael Lund
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

//! # Cuboidal slit geometry with periodic boundary conditions in XY only

use crate::{
    cell::{BoundaryConditions, Cuboid, Shape, SimulationCell, VolumeScale, VolumeScalePolicy},
    Point,
};
use serde::{Deserialize, Serialize};

/// Cuboidal slit with periodic boundaries in XY and hard walls in Z.
///
/// Shares geometry and volume scaling with [`Cuboid`], but applies
/// minimum image convention and boundary wrapping only in X and Y.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Slit(Cuboid);

impl Slit {
    /// Create a new slit with side lengths `a`, `b` (periodic) and height `c` (hard walls)
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self(Cuboid::new(a, b, c))
    }
    /// Square slit with side length `a` in XY and height `c`
    pub fn square(a: f64, c: f64) -> Self {
        Self::new(a, a, c)
    }
}

impl Shape for Slit {
    fn volume(&self) -> Option<f64> {
        self.0.volume()
    }
    fn is_inside(&self, point: &Point) -> bool {
        self.0.is_inside(point)
    }
    fn bounding_box(&self) -> Option<Point> {
        self.0.bounding_box()
    }
    fn get_point_inside(&self, rng: &mut rand::prelude::ThreadRng) -> Point {
        self.0.get_point_inside(rng)
    }
}

impl BoundaryConditions for Slit {
    fn pbc(&self) -> super::PeriodicDirections {
        super::PeriodicDirections::PeriodicXY
    }
    #[inline(always)]
    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        let mut delta = *point1 - *point2;
        if delta.x > self.0.half_cell.x {
            delta.x -= self.0.cell.x;
        } else if delta.x < -self.0.half_cell.x {
            delta.x += self.0.cell.x;
        }
        if delta.y > self.0.half_cell.y {
            delta.y -= self.0.cell.y;
        } else if delta.y < -self.0.half_cell.y {
            delta.y += self.0.cell.y;
        }
        delta
    }
    fn boundary(&self, point: &mut Point) {
        point.x -= self.0.cell.x * (point.x / self.0.cell.x).round();
        point.y -= self.0.cell.y * (point.y / self.0.cell.y).round();
    }
}

impl VolumeScale for Slit {
    fn scale_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()> {
        self.0.scale_volume(new_volume, policy)
    }
    fn scale_position(
        &self,
        new_volume: f64,
        point: &mut Point,
        policy: VolumeScalePolicy,
    ) -> anyhow::Result<()> {
        self.0.scale_position(new_volume, point, policy)
    }
}

impl SimulationCell for Slit {}

#[cfg(test)]
mod tests {
    use crate::cell::{BoundaryConditions, PeriodicDirections, Shape};

    use super::Slit;

    #[test]
    fn generate_points() {
        let shape = Slit::new(10.0, 5.0, 2.5);
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let point = shape.get_point_inside(&mut rng);
            assert!(shape.is_inside(&point));
        }
    }

    #[test]
    fn pbc_xy_only() {
        let slit = Slit::new(10.0, 10.0, 20.0);
        assert_eq!(slit.pbc(), PeriodicDirections::PeriodicXY);

        // XY should use minimum image: 4.5 - (-4.5) = 9.0, wraps to -1.0
        let p1 = crate::Point::new(4.5, 0.0, 0.0);
        let p2 = crate::Point::new(-4.5, 0.0, 0.0);
        let d = slit.distance(&p1, &p2);
        assert!((d.x - (-1.0)).abs() < 1e-10);

        // Z should NOT use minimum image
        let p1 = crate::Point::new(0.0, 0.0, 9.0);
        let p2 = crate::Point::new(0.0, 0.0, -9.0);
        let d = slit.distance(&p1, &p2);
        assert!((d.z - 18.0).abs() < 1e-10);
    }

    #[test]
    fn boundary_wraps_xy_only() {
        let slit = Slit::new(10.0, 10.0, 20.0);
        let mut point = crate::Point::new(6.0, -7.0, 15.0);
        slit.boundary(&mut point);
        assert!((point.x - (-4.0)).abs() < 1e-10);
        assert!((point.y - 3.0).abs() < 1e-10);
        // z unchanged
        assert!((point.z - 15.0).abs() < 1e-10);
    }
}
