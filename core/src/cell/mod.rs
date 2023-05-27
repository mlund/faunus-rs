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

use crate::Point;
use serde::{Deserialize, Serialize};

/// Interface for a unit cell used to describe the geometry of the simulation system
pub trait SimulationCell {
    /// Get volume of system
    fn volume(&self) -> Option<f64>;
    /// Apply periodic boundary conditions to a point, wrapping around PBC if necessary
    fn boundary(&self, point: &mut Point);
    /// Calculate the minimum distance between two points
    fn distance(&self, point1: &Point, point2: &Point) -> Point;
    /// Get the minimum squared distance between two points
    fn distance_squared(&self, point1: &Point, point2: &Point) -> f64;
    /// Determines if a point is inside the unit cell
    fn is_inside(&self, point: &Point) -> bool;
}

/// Policies for how to scale a volume
///
/// This is used to scale an old volume to a new volume.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum VolumeScalePolicy {
    /// Isotropic scaling (equal scaling in all directions)
    Isotropic,
    /// Isochoric scaling along z (constant volume)
    IsochoricZ,
    /// Scale along z-axis only
    ScaleZ,
    /// Scale along x and y
    ScaleXY,
}

/// Trait for scaling a position in a simulation cell according to a volume scaling policy.
/// Typically implemented for a unit cell.
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
