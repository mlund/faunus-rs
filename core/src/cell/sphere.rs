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

use crate::{
    cell::SimulationCell,
    transform::{VolumeScale, VolumeScalePolicy},
    Point,
};
use anyhow::Ok;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sphere {
    radius: f64,
}

impl SimulationCell for Sphere {
    fn volume(&self) -> Option<f64> {
        Some(4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3))
    }

    fn set_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()> {
        if !matches!(policy, VolumeScalePolicy::Isotropic) {
            anyhow::bail!("Sphere only supports isotropic volume scaling")
        }
        self.radius = (new_volume / (4.0 / 3.0 * std::f64::consts::PI)).cbrt();
        Ok(())
    }

    fn boundary(&self, _point: &mut Point) {}

    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        point1 - point2
    }

    fn distance_squared(&self, point1: &Point, point2: &Point) -> f64 {
        self.distance(point1, point2).norm_squared()
    }

    fn is_inside(&self, point: &Point) -> bool {
        point.norm_squared() < self.radius.powi(2)
    }
}

impl VolumeScale for Sphere {
    fn scale_position(
        &self,
        policy: VolumeScalePolicy,
        new_volume: f64,
        point: &mut Point,
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
