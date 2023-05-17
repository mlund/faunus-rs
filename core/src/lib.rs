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

extern crate chemfiles;

use crate::group::{Group, GroupCollection};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

pub type Point = Vector3<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

pub use interact::Info;
pub mod cell;
mod change;
pub use self::change::{Change, GroupChange};
pub mod chemistry;
pub mod energy;
pub mod group;
pub mod topology;
pub mod montecarlo;
pub mod platform;
pub mod time;
pub mod transform;

/// Boltzmann constant in J/K
pub const BOLTZMANN: f64 = 1.380649e-23;
/// Avogadro's number in 1/mol
pub const AVOGADRO: f64 = 6.02214076e23;
/// Gas constant in J/(mol K)
pub const GAS_CONSTANT: f64 = BOLTZMANN * AVOGADRO;
/// Electron unit charge in C
pub const UNIT_CHARGE: f64 = 1.602176634e-19;
/// Vacuum permittivity in F/m
pub const VACUUM_PERMITTIVITY: f64 = 8.8541878128e-12;

trait PointParticle {
    /// Type of the particle identifier
    type Idtype;
    /// Type of the particle position
    type Positiontype;
    /// Identifier for the particle type
    fn id(&self) -> Self::Idtype;
    /// Get position
    fn pos(&self) -> &Self::Positiontype;
    /// Get mutable position
    fn pos_mut(&mut self) -> &mut Self::Positiontype;
    /// Index in main list of particle (immutable)
    fn index(&self) -> usize;
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct Particle {
    id: usize,
    index: usize,
    pos: Point,
}

impl PointParticle for Particle {
    type Idtype = usize;
    type Positiontype = Point;
    fn id(&self) -> Self::Idtype {
        self.id
    }
    fn pos(&self) -> &Self::Positiontype {
        &self.pos
    }
    fn pos_mut(&mut self) -> &mut Self::Positiontype {
        &mut self.pos
    }
    fn index(&self) -> usize {
        self.index
    }
}

/// Trait for synchronizing internal state from another object, given a change.
pub trait SyncFrom {
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()>;
}

pub trait SyncFromAny {
    fn sync_from(&mut self, other: &dyn as_any::AsAny, change: &Change) -> anyhow::Result<()>;
}

pub trait Context: GroupCollection + cell::SimulationCell + Clone + std::fmt::Debug {
    /// Get list of energies in the system
    fn energies(&self) {}
}
