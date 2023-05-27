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

pub mod cuboid;
pub mod lumol;
pub mod sphere;

use crate::{transform::VolumeScalePolicy, Point};
//type Matrix3 = nalgebra::Matrix3<f64>;
//type Vector3D = Point;

/// Interface for a unit cell used to describe the geometry of the simulation system
pub trait SimulationCell {
    /// Get volume of system
    fn volume(&self) -> Option<f64>;
    /// Set volume of system
    fn set_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()>;
    /// Apply periodic boundary conditions to a point
    fn boundary(&self, point: &mut Point);
    /// Calculate the minimum image distance between two points
    fn distance(&self, point1: &Point, point2: &Point) -> Point;
    /// Get the minimum squared distance between two points
    fn distance_squared(&self, point1: &Point, point2: &Point) -> f64;
    /// Determines if a point is indide the unit cell
    fn is_inside(&self, point: &Point) -> bool;
}
