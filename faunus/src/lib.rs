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

use crate::group::Group;

use nalgebra::DMatrix;
use topology::Topology;

pub type Point = interatomic::Vector3;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

mod info;
pub use info::*;
pub mod cell;
mod change;
pub use self::change::{Change, GroupChange};
pub mod analysis;
pub mod aux;
pub mod chemistry;
pub mod dimension;
pub mod energy;
pub mod group;
pub mod montecarlo;
pub mod platform;
pub mod propagate;
pub mod time;
pub mod topology;
pub mod transform;

mod particle;
pub use particle::{Particle, PointParticle};

mod context;
pub use context::*;

pub trait SyncFrom {
    /// Synchronize internal state from another object of the same type
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()>;
}
