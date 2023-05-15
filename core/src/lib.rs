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

use crate::group::{Group, GroupCollection};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

pub type Point = Vector3<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

pub mod cell;
pub mod cite;
pub mod energy;
pub mod group;
pub mod montecarlo;
pub mod platform;
pub mod time;
pub mod transform;

use crate::group::GroupChange;

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

/// Describes a change in the system. This can for example be used to
/// describe a change in the volume of the system, or a change in the
/// number of particles in a group.
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub enum Change {
    /// Everything has changed
    Everything,

    /// The volume has changed
    Volume,

    /// Some groups have changed
    Groups(Vec<GroupChange>),

    /// A single group has changed
    SingleGroup(GroupChange),

    /// Single particle has changed
    /// (group index, particle index relative to group)
    SingleParticle(usize, usize),

    /// No change
    #[default]
    None,
}

/// Trait for synchronizing internal state from another object, given a change.
pub trait SyncFrom {
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()>;
}

pub trait SyncFromAny {
    fn sync_from(&mut self, other: &dyn as_any::AsAny, change: &Change) -> anyhow::Result<()>;
}

pub trait Context:
    GroupCollection + cell::SimulationCell + Clone + Default + std::fmt::Debug + Sized
{
    /// Get list of energies in the system
    fn energies(&self) {}
}
