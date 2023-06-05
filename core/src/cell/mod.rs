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

//! # Simulation cells with or without periodic boundary conditions
//!
//! This module contains the interface for the simulation cell, which describes the geometry of the simulation system.
//! The simulation cell is a geometric [`Shape`], e.g. a cube, sphere, etc., with defined [`BoundaryConditions`].
//! Some statistical thermodynamic ensembles require volume fluctuations, which is implemented by scaling the simulation cell
//! through the [`VolumeScale`] trait.

mod cuboid;
mod endless;
//pub(crate) mod lumol;
mod sphere;

use crate::Point;
pub use cuboid::Cuboid;
pub use endless::Endless;
use serde::{Deserialize, Serialize};
pub use sphere::Sphere;

/// Final interface for a unit cell used to describe the geometry of a simulation system.
///
/// It is a combination of a [`Shape`], [`BoundaryConditions`] and [`VolumeScale`].
pub trait SimulationCell: Shape + BoundaryConditions + VolumeScale {}

/// Geometric shape like a sphere, cube, etc.
pub trait Shape {
    /// Get volume
    fn volume(&self) -> Option<f64>;
    /// Position of the geometric center of the shape. For a cube, this is the center of the box; for a sphere, the center of the sphere etc.
    fn center(&self) -> Point {
        Point::zeros()
    }
    /// Determines if a point lies inside the boundaries of the shape
    fn is_inside(&self, point: &Point) -> bool;
    /// Bounding box of the shape centered at `center()`
    fn bounding_box(&self) -> Option<Point> {
        None
    }
}

/// Periodic boundary conditions in various directions
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum PeriodicDirections {
    /// Periodic boundary conditions in Z direction
    PeriodicZ,
    /// 2d periodic boundary conditions in the XY plane, e.g. a slab
    PeriodicXY,
    /// 3d periodic boundary conditions in XYZ directions
    PeriodicXYZ,
    /// No periodic boundaries in any direction
    None,
}

impl PeriodicDirections {
    /// True if periodic in some direction
    pub fn is_some(&self) -> bool {
        *self != PeriodicDirections::None
    }
}

/// Interface for periodic boundary conditions and minimum image convention
pub trait BoundaryConditions {
    /// Report on periodic boundary conditions
    fn pbc(&self) -> PeriodicDirections;
    /// Wrap a point to fit within boundaries, if appropriate
    fn boundary(&self, point: &mut Point);
    /// Minimum image distance between two points inside a cell
    fn distance(&self, point1: &Point, point2: &Point) -> Point;
    /// Get the minimum squared distance between two points
    #[inline]
    fn distance_squared(&self, point1: &Point, point2: &Point) -> f64 {
        self.distance(point1, point2).norm_squared()
    }
}

/// Policies for how to scale a volume
///
/// This is used to scale an old volume to a new volume.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum VolumeScalePolicy {
    /// Isotropic scaling (equal scaling in all directions)
    Isotropic,
    /// Isochoric scaling of z and the xy-plane (constant volume)
    IsochoricZ,
    /// Scale along z-axis only
    ScaleZ,
    /// Scale the XY plane
    ScaleXY,
}

/// Trait for scaling a position or the simulation cell according to a scaling policy.
pub trait VolumeScale {
    /// Scale a `position` inside a simulation cell according to a scaling policy.
    /// Errors if the scaling policy is unsupported.
    fn scale_position(
        &self,
        new_volume: f64,
        position: &mut Point,
        policy: VolumeScalePolicy,
    ) -> Result<(), anyhow::Error>;

    /// Scale cell volume to a new volume according to a scaling policy.
    /// This should typically be followed by a call to `scale_position` for each particle or mass center.
    /// Errors if the scaling policy is unsupported.
    fn scale_volume(
        &mut self,
        new_volume: f64,
        policy: VolumeScalePolicy,
    ) -> Result<(), anyhow::Error>;
}
