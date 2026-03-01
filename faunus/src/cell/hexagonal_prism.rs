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

//! # Hexagonal prism cell with full 3D periodic boundary conditions

use crate::{
    cell::{
        BoundaryConditions, Cuboid, OrthorhombicExpansion, Shape, SimulationCell, VolumeScale,
        VolumeScalePolicy,
    },
    Point,
};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

const SQRT_3: f64 = 1.732_050_807_568_877_2;
const SQRT_3_OVER_2: f64 = SQRT_3 / 2.0;

/// Hexagonal prism cell with full 3D periodic boundary conditions.
///
/// Parameterized by hexagon side length `side` (= outer radius) and prism `height`.
/// The hexagon axis is along Z.
///
/// # Geometry
///
/// - Inner radius (apothem): `√3/2 · side`
/// - Volume: `3√3/2 · side² · height`
/// - Bounding box: `[√3·side, 2·side, height]`
/// - Lattice vectors: **a₁** = `(√3·a, 0, 0)`, **a₂** = `(√3·a/2, 3·a/2, 0)`, **a₃** = `(0, 0, h)`
///
/// Minimum image convention is implemented via Wigner-Seitz reduction:
/// sequential projection onto three hexagonal lattice directions,
/// following the C++ Faunus implementation.
#[derive(Clone, Debug, Serialize)]
pub struct HexagonalPrism {
    side: f64,
    height: f64,
    /// √3 × side — distance between parallel edges
    #[serde(skip)]
    box_x: f64,
    #[serde(skip)]
    half_box_x: f64,
    #[serde(skip)]
    half_height: f64,
}

impl<'de> Deserialize<'de> for HexagonalPrism {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            side: f64,
            height: f64,
        }
        let raw = Raw::deserialize(deserializer)?;
        Ok(Self::new(raw.side, raw.height))
    }
}

impl HexagonalPrism {
    pub fn new(side: f64, height: f64) -> Self {
        let box_x = SQRT_3 * side;
        Self {
            side,
            height,
            box_x,
            half_box_x: box_x * 0.5,
            half_height: height * 0.5,
        }
    }
}

impl Shape for HexagonalPrism {
    fn volume(&self) -> Option<f64> {
        Some(1.5 * SQRT_3 * self.side * self.side * self.height)
    }

    /// Point lies inside the hexagonal prism if all three conditions hold:
    /// `|x| ≤ √3a/2`, `|x| + √3|y| ≤ √3a`, and `|z| ≤ h/2`.
    fn is_inside(&self, point: &Point) -> bool {
        let ax = point.x.abs();
        point.z.abs() <= self.half_height
            && ax <= self.half_box_x
            && ax + SQRT_3 * point.y.abs() <= self.box_x
    }

    fn bounding_box(&self) -> Option<Point> {
        Some(Point::new(self.box_x, 2.0 * self.side, self.height))
    }

    /// Supercell `[√3a, 3a, h]` containing two hex unit cells, with one image translation.
    fn orthorhombic_expansion(&self) -> Option<OrthorhombicExpansion> {
        Some(OrthorhombicExpansion {
            box_lengths: Point::new(self.box_x, 3.0 * self.side, self.height),
            translations: vec![Point::new(self.half_box_x, 1.5 * self.side, 0.0)],
        })
    }

    /// Random point via rejection sampling in the bounding box (~75% acceptance)
    fn get_point_inside(&self, rng: &mut rand::prelude::ThreadRng) -> Point {
        loop {
            let point = Point::new(
                rng.gen_range(-self.half_box_x..self.half_box_x),
                rng.gen_range(-self.side..self.side),
                rng.gen_range(-self.half_height..self.half_height),
            );
            if self.is_inside(&point) {
                return point;
            }
        }
    }
}

impl BoundaryConditions for HexagonalPrism {
    fn pbc(&self) -> super::PeriodicDirections {
        super::PeriodicDirections::PeriodicXYZ
    }

    #[inline(always)]
    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        let mut delta = *point1 - *point2;
        self.boundary(&mut delta);
        delta
    }

    /// Wrap a displacement vector into the Wigner-Seitz cell of the hexagonal lattice.
    ///
    /// Projects onto three hexagonal lattice directions and wraps sequentially,
    /// following the C++ Faunus implementation.
    #[inline]
    fn boundary(&self, point: &mut Point) {
        // Wrap along x-axis (lattice direction a1)
        if point.x.abs() > self.half_box_x {
            point.x -= self.box_x * (point.x / self.box_x).round();
        }

        // Wrap along unitvY = (1/2, √3/2, 0) direction
        let dot_y = 0.5 * point.x + SQRT_3_OVER_2 * point.y;
        if dot_y > self.half_box_x {
            point.x -= self.half_box_x;
            point.y -= self.box_x * SQRT_3_OVER_2;
            if point.x < -self.half_box_x {
                point.x += self.box_x;
            }
        } else if dot_y < -self.half_box_x {
            point.x += self.half_box_x;
            point.y += self.box_x * SQRT_3_OVER_2;
            if point.x > self.half_box_x {
                point.x -= self.box_x;
            }
        }

        // Wrap along unitvZ = (-1/2, √3/2, 0) direction
        let dot_z = -0.5 * point.x + SQRT_3_OVER_2 * point.y;
        if dot_z.abs() > self.half_box_x {
            let n = (dot_z / self.box_x).round();
            point.x += self.half_box_x * n;
            point.y -= self.box_x * SQRT_3_OVER_2 * n;
        }

        // Wrap along z-axis
        if point.z.abs() > self.half_height {
            point.z -= self.height * (point.z / self.height).round();
        }
    }
}

impl VolumeScale for HexagonalPrism {
    fn scale_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()> {
        let old_volume = self.volume().unwrap();
        let (new_side, new_height) = match policy {
            VolumeScalePolicy::Isotropic => {
                let scale = (new_volume / old_volume).cbrt();
                (self.side * scale, self.height * scale)
            }
            VolumeScalePolicy::ScaleZ => (self.side, self.height * new_volume / old_volume),
            VolumeScalePolicy::ScaleXY => {
                let scale = (new_volume / old_volume).sqrt();
                (self.side * scale, self.height)
            }
            VolumeScalePolicy::IsochoricZ => {
                let alpha = (new_volume / old_volume).cbrt();
                (self.side * alpha, self.height / (alpha * alpha))
            }
        };
        *self = Self::new(new_side, new_height);
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
                let alpha = (new_volume / old_volume).cbrt();
                point.x *= alpha;
                point.y *= alpha;
                point.z /= alpha * alpha;
            }
        }
        Ok(())
    }
}

/// Convert to the smallest orthorhombic supercell `[√3·a, 3·a, h]`
/// containing two hexagonal prism unit cells.
impl From<HexagonalPrism> for Cuboid {
    fn from(hex: HexagonalPrism) -> Self {
        Cuboid::new(hex.box_x, 3.0 * hex.side, hex.height)
    }
}

impl SimulationCell for HexagonalPrism {}

#[cfg(test)]
mod tests {
    use super::HexagonalPrism;
    use crate::cell::{
        BoundaryConditions, Cuboid, PeriodicDirections, Shape, VolumeScale, VolumeScalePolicy,
    };
    use crate::Point;
    use rand::Rng;

    /// Side=1, height chosen so volume equals exactly 1.0
    fn unit_volume_hex() -> HexagonalPrism {
        let side = 1.0;
        let height = 2.0 / (3.0 * 3.0_f64.sqrt());
        HexagonalPrism::new(side, height)
    }

    #[test]
    fn volume() {
        let hex = unit_volume_hex();
        assert!((hex.volume().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn is_inside() {
        let hex = unit_volume_hex();
        let side = 1.0;
        let inner_radius = 3.0_f64.sqrt() / 2.0 * side;
        let outer_radius = side;

        // x-axis: inner_radius is the apothem
        assert!(!hex.is_inside(&Point::new(-1.01 * inner_radius, 0.0, 0.0)));
        assert!(hex.is_inside(&Point::new(0.99 * inner_radius, 0.0, 0.0)));

        // y-axis: outer_radius is the circumradius
        assert!(!hex.is_inside(&Point::new(0.0, -1.01 * outer_radius, 0.0)));
        assert!(hex.is_inside(&Point::new(0.0, 0.99 * outer_radius, 0.0)));

        // 60° direction
        let angle = std::f64::consts::PI / 3.0;
        assert!(hex.is_inside(&Point::new(
            0.99 * angle.cos() * inner_radius,
            0.99 * angle.sin() * inner_radius,
            0.0,
        )));
        assert!(!hex.is_inside(&Point::new(
            1.01 * angle.cos() * inner_radius,
            1.01 * angle.sin() * inner_radius,
            0.0,
        )));

        // z boundaries
        let height = hex.height;
        assert!(!hex.is_inside(&Point::new(0.0, 0.0, -0.51 * height)));
        assert!(hex.is_inside(&Point::new(0.0, 0.0, 0.49 * height)));
    }

    #[test]
    fn bounding_box() {
        let side = 5.0;
        let height = 10.0;
        let hex = HexagonalPrism::new(side, height);
        let bb = hex.bounding_box().unwrap();
        assert!((bb.x - 3.0_f64.sqrt() * side).abs() < 1e-10);
        assert!((bb.y - 2.0 * side).abs() < 1e-10);
        assert!((bb.z - height).abs() < 1e-10);
    }

    #[test]
    fn random_points_inside() {
        let hex = HexagonalPrism::new(5.0, 10.0);
        let mut rng = rand::thread_rng();
        for _ in 0..10_000 {
            let point = hex.get_point_inside(&mut rng);
            assert!(hex.is_inside(&point));
        }
    }

    /// Displacement vectors between points inside the cell are at most ±bounding_box.
    /// Verify that boundary() maps all such vectors back inside the WS cell.
    #[test]
    fn boundary_wraps_inside() {
        let hex = HexagonalPrism::new(5.0, 20.0);
        let mut rng = rand::thread_rng();
        let bb = hex.bounding_box().unwrap();

        for _ in 0..10_000 {
            let mut point = Point::new(
                rng.gen_range(-bb.x..bb.x),
                rng.gen_range(-bb.y..bb.y),
                rng.gen_range(-bb.z..bb.z),
            );
            hex.boundary(&mut point);
            assert!(
                hex.is_inside(&point),
                "Point {point:?} not inside after boundary()"
            );
        }
    }

    /// Verify minimum image by exhaustive search over 27 periodic images
    #[test]
    fn brute_force_minimum_image() {
        let side = 5.0;
        let height = 20.0;
        let hex = HexagonalPrism::new(side, height);
        let mut rng = rand::thread_rng();
        let sqrt3 = 3.0_f64.sqrt();

        let a1 = Point::new(sqrt3 * side, 0.0, 0.0);
        let a2 = Point::new(sqrt3 * side / 2.0, 1.5 * side, 0.0);
        let a3 = Point::new(0.0, 0.0, height);

        for _ in 0..10_000 {
            let p1 = hex.get_point_inside(&mut rng);
            let p2 = hex.get_point_inside(&mut rng);

            let dist_sq = hex.distance_squared(&p1, &p2);

            let delta = p1 - p2;
            let mut min_dist_sq = f64::MAX;
            for i in -1..=1_i32 {
                for j in -1..=1_i32 {
                    for k in -1..=1_i32 {
                        let image = delta + (i as f64) * a1 + (j as f64) * a2 + (k as f64) * a3;
                        min_dist_sq = min_dist_sq.min(image.norm_squared());
                    }
                }
            }

            assert!(
                (dist_sq - min_dist_sq).abs() < 1e-6,
                "boundary={:.6}, brute_force={:.6}",
                dist_sq.sqrt(),
                min_dist_sq.sqrt(),
            );
        }
    }

    #[test]
    fn volume_scaling() {
        let original = HexagonalPrism::new(5.0, 10.0);
        let vol = original.volume().unwrap();
        let new_vol = vol * 2.0;

        // Isotropic
        let mut hex = original.clone();
        hex.scale_volume(new_vol, VolumeScalePolicy::Isotropic)
            .unwrap();
        assert!((hex.volume().unwrap() - new_vol).abs() < 1e-6);

        // ScaleZ
        let mut hex = original.clone();
        hex.scale_volume(new_vol, VolumeScalePolicy::ScaleZ)
            .unwrap();
        assert!((hex.volume().unwrap() - new_vol).abs() < 1e-6);

        // ScaleXY
        let mut hex = original.clone();
        hex.scale_volume(new_vol, VolumeScalePolicy::ScaleXY)
            .unwrap();
        assert!((hex.volume().unwrap() - new_vol).abs() < 1e-6);

        // IsochoricZ preserves volume
        let mut hex = original.clone();
        hex.scale_volume(new_vol, VolumeScalePolicy::IsochoricZ)
            .unwrap();
        assert!((hex.volume().unwrap() - vol).abs() < 1e-6);
    }

    #[test]
    fn from_hexagonal_prism_for_cuboid() {
        let hex = HexagonalPrism::new(5.0, 10.0);
        let hex_vol = hex.volume().unwrap();
        let cuboid: Cuboid = hex.into();
        let cuboid_vol = cuboid.volume().unwrap();
        assert!((cuboid_vol - 2.0 * hex_vol).abs() < 1e-6);
    }

    #[test]
    fn pbc_xyz() {
        let hex = HexagonalPrism::new(5.0, 10.0);
        assert_eq!(hex.pbc(), PeriodicDirections::PeriodicXYZ);
    }
}
