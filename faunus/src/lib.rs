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

use topology::Topology;

/// Molar gas constant in kJ/(mol·K).
pub const R_IN_KJ_PER_MOL: f64 = physical_constants::MOLAR_GAS_CONSTANT * 1e-3;

pub type Point = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

mod info;
pub use info::*;
pub mod cell;
pub mod celllist;
mod change;
pub mod collective_variable;
pub use self::change::{Change, GroupChange};
pub mod analysis;
pub mod auxiliary;
pub mod backend;
pub mod chemistry;
#[cfg(feature = "cli")]
pub mod cli;
pub mod dimension;
pub mod energy;
pub(crate) mod geometry;
pub mod group;
pub(crate) mod histogram;
pub mod montecarlo;
pub mod propagate;
pub mod selection;
pub mod simulation;
pub mod state;
pub mod time;
pub mod topology;
pub mod transform;
#[cfg(feature = "cli")]
pub mod umbrella;

mod particle;
pub use particle::{Particle, PointParticle};

/// Re-export interatomic to avoid diamond dependency conflicts in downstream crates.
pub use interatomic;

mod context;
pub use context::*;
