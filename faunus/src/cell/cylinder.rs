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

//! # Cylindrical cell with periodic boundary conditions in Z only

use crate::{
    cell::{BoundaryConditions, Shape, SimulationCell, VolumeScale, VolumeScalePolicy},
    Point,
};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

/// Cylindrical cell with hard walls in XY and periodic boundaries in Z.
///
/// The cylinder axis is along Z. Particles are confined within radius `r`
/// in the XY plane, and minimum image convention is applied along Z.
#[derive(Clone, Debug, Serialize)]
pub struct Cylinder {
    radius: f64,
    height: f64,
    #[serde(skip)]
    radius_squared: f64,
    #[serde(skip)]
    half_height: f64,
}

impl<'de> Deserialize<'de> for Cylinder {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            radius: f64,
            height: f64,
        }
        let raw = Raw::deserialize(deserializer)?;
        Ok(Self::new(raw.radius, raw.height))
    }
}

impl Cylinder {
    pub fn new(radius: f64, height: f64) -> Self {
        Self {
            radius,
            height,
            radius_squared: radius * radius,
            half_height: height * 0.5,
        }
    }
}

impl Shape for Cylinder {
    fn volume(&self) -> Option<f64> {
        Some(std::f64::consts::PI * self.radius_squared * self.height)
    }
    fn is_inside(&self, point: &Point) -> bool {
        point.x * point.x + point.y * point.y <= self.radius_squared
            && point.z.abs() <= self.half_height
    }
    fn bounding_box(&self) -> Option<Point> {
        let d = 2.0 * self.radius;
        Some(Point::new(d, d, self.height))
    }
    /// Random point via rejection sampling in the bounding box
    fn get_point_inside(&self, rng: &mut rand::prelude::ThreadRng) -> Point {
        loop {
            let point = Point::new(
                rng.gen_range(-self.radius..self.radius),
                rng.gen_range(-self.radius..self.radius),
                rng.gen_range(-self.half_height..self.half_height),
            );
            if point.x * point.x + point.y * point.y <= self.radius_squared {
                return point;
            }
        }
    }
}

impl BoundaryConditions for Cylinder {
    fn pbc(&self) -> super::PeriodicDirections {
        super::PeriodicDirections::PeriodicZ
    }
    #[inline(always)]
    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        let mut delta = *point1 - *point2;
        if delta.z > self.half_height {
            delta.z -= self.height;
        } else if delta.z < -self.half_height {
            delta.z += self.height;
        }
        delta
    }
    fn boundary(&self, point: &mut Point) {
        point.z -= self.height * (point.z / self.height).round();
    }
}

impl VolumeScale for Cylinder {
    fn scale_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()> {
        let old_volume = self.volume().unwrap();
        let (new_radius, new_height) = match policy {
            VolumeScalePolicy::Isotropic => {
                let scale = (new_volume / old_volume).cbrt();
                (self.radius * scale, self.height * scale)
            }
            VolumeScalePolicy::ScaleZ => (self.radius, self.height * new_volume / old_volume),
            VolumeScalePolicy::ScaleXY => {
                let scale = (new_volume / old_volume).sqrt();
                (self.radius * scale, self.height)
            }
            VolumeScalePolicy::IsochoricZ => {
                let scale = (new_volume / old_volume).sqrt();
                (self.radius, self.height * scale)
            }
        };
        *self = Self::new(new_radius, new_height);
        Ok(())
    }

    fn scale_position(
        &self,
        new_volume: f64,
        point: &mut Point,
        policy: VolumeScalePolicy,
    ) -> anyhow::Result<()> {
        let old_volume = self.volume().unwrap();
        match policy {
            VolumeScalePolicy::Isotropic => {
                let scale = (new_volume / old_volume).cbrt();
                *point *= scale;
            }
            VolumeScalePolicy::ScaleZ => {
                point.z *= new_volume / old_volume;
            }
            VolumeScalePolicy::ScaleXY => {
                let scale = (new_volume / old_volume).sqrt();
                point.x *= scale;
                point.y *= scale;
            }
            VolumeScalePolicy::IsochoricZ => {
                let scale = (new_volume / old_volume).sqrt();
                point.z *= scale;
            }
        }
        Ok(())
    }
}

impl SimulationCell for Cylinder {}

#[cfg(test)]
mod tests {
    use crate::cell::{
        BoundaryConditions, PeriodicDirections, Shape, VolumeScale, VolumeScalePolicy,
    };

    use super::Cylinder;

    #[test]
    fn generate_points() {
        let shape = Cylinder::new(5.0, 10.0);
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let point = shape.get_point_inside(&mut rng);
            assert!(shape.is_inside(&point));
        }
    }

    #[test]
    fn pbc_z_only() {
        let cyl = Cylinder::new(5.0, 10.0);
        assert_eq!(cyl.pbc(), PeriodicDirections::PeriodicZ);

        // Z should use minimum image: 4.5 - (-4.5) = 9.0, wraps to -1.0
        let p1 = crate::Point::new(0.0, 0.0, 4.5);
        let p2 = crate::Point::new(0.0, 0.0, -4.5);
        let d = cyl.distance(&p1, &p2);
        assert!((d.z - (-1.0)).abs() < 1e-10);

        // XY should NOT use minimum image
        let p1 = crate::Point::new(4.0, 0.0, 0.0);
        let p2 = crate::Point::new(-4.0, 0.0, 0.0);
        let d = cyl.distance(&p1, &p2);
        assert!((d.x - 8.0).abs() < 1e-10);
    }

    #[test]
    fn boundary_wraps_z_only() {
        let cyl = Cylinder::new(5.0, 10.0);
        let mut point = crate::Point::new(3.0, -2.0, 7.0);
        cyl.boundary(&mut point);
        assert!((point.x - 3.0).abs() < 1e-10);
        assert!((point.y - (-2.0)).abs() < 1e-10);
        assert!((point.z - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn volume_and_scaling() {
        let mut cyl = Cylinder::new(5.0, 10.0);
        let vol = cyl.volume().unwrap();
        let expected = std::f64::consts::PI * 25.0 * 10.0;
        assert!((vol - expected).abs() < 1e-10);

        let new_vol = vol * 2.0;
        cyl.scale_volume(new_vol, VolumeScalePolicy::Isotropic)
            .unwrap();
        assert!((cyl.volume().unwrap() - new_vol).abs() < 1e-6);
    }
}
