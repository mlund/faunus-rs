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

/// 1 M standard-state concentration in 1/ų: N_A · 10⁻²⁷.
pub const MOLAR_TO_INV_ANGSTROM3: f64 = physical_constants::AVOGADRO_CONSTANT * 1e-27;

pub type Point = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

mod info;
pub use info::*;
pub mod cell;
#[cfg(feature = "cli")]
pub mod cli;
pub mod energy;
pub mod topology;
pub mod transform;

// Simulation machinery. Nothing outside the crate drives a simulation through these types — the
// supported entry point is `cli::do_main` — and exposing them would freeze the internal interfaces
// that Tier 4 exists to keep narrow.
pub(crate) mod analysis;
pub(crate) mod auxiliary;
pub(crate) mod axes;
pub(crate) mod backend;
pub(crate) mod celllist;
pub(crate) mod change;
pub(crate) mod chemistry;
pub(crate) mod collective_variable;
pub(crate) mod flat_histogram;
pub(crate) mod geometry;
pub(crate) mod group;
pub(crate) mod histogram;
pub(crate) mod montecarlo;
pub(crate) mod propagate;
pub(crate) mod selection;
pub(crate) mod simulation;
pub(crate) mod state;
pub(crate) mod time;
#[cfg(feature = "cli")]
pub(crate) mod umbrella;
#[cfg(feature = "cli")]
pub(crate) mod wang_landau;
pub(crate) mod z_grid;

pub(crate) use self::change::{Change, GroupChange};

mod particle;
pub use particle::{Particle, PointParticle};

/// Re-export interatomic to avoid diamond dependency conflicts in downstream crates.
pub use interatomic;

pub(crate) mod context;
pub(crate) use context::*;
