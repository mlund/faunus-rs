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

//! # Endless simulation cell with no boundaries and infinite volume

use crate::{
    cell::{BoundaryConditions, Shape, VolumeScale, VolumeScalePolicy},
    Point,
};
use serde::{Deserialize, Serialize};

use super::SimulationCell;

/// Endless simulation cell with no boundaries and infinite volume
#[derive(Clone, Debug, Serialize, Deserialize, Copy, Default)]
pub struct Endless {}

impl Shape for Endless {
    fn volume(&self) -> Option<f64> {
        Some(f64::INFINITY)
    }
    fn is_inside(&self, _point: &Point) -> bool {
        true
    }
    fn bounding_box(&self) -> Option<Point> {
        None
    }
    fn get_point_inside(&self, _rng: &mut rand::prelude::ThreadRng) -> Point {
        todo!("Implement get_point_inside for endless box");
    }
}

impl BoundaryConditions for Endless {
    fn pbc(&self) -> super::PeriodicDirections {
        super::PeriodicDirections::None
    }
    fn boundary(&self, _point: &mut Point) {}
    #[inline]
    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        *point1 - *point2
    }
}

impl VolumeScale for Endless {
    fn scale_volume(
        &mut self,
        _new_volume: f64,
        _policyy: VolumeScalePolicy,
    ) -> anyhow::Result<()> {
        anyhow::bail!("Cannot scale position in endless cell")
    }
    fn scale_position(
        &self,
        _new_volume: f64,
        _point: &mut Point,
        _policy: VolumeScalePolicy,
    ) -> Result<(), anyhow::Error> {
        anyhow::bail!("Cannot scale position in endless cell")
    }
}

impl SimulationCell for Endless {}
