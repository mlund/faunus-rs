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
extern crate serde_json;

use crate::group::{Group, GroupCollection};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

pub type Point = Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

mod info;
pub use info::*;
pub mod cell;
mod change;
pub use self::change::{Change, GroupChange};
pub mod analysis;
pub mod chemistry;
pub mod energy;
pub mod group;
pub mod montecarlo;
pub mod platform;
pub mod time;
pub mod topology;
pub mod transform;

use physical_constants::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, VACUUM_ELECTRIC_PERMITTIVITY,
};

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
    /// Type of the particle
    id: usize,
    /// Index in main list of particles
    index: usize,
    /// Position of the particle
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
pub trait SyncFrom {
    /// Synchronize internal state from another object
    fn sync_from(&mut self, other: &dyn as_any::AsAny, change: &Change) -> anyhow::Result<()>;
}

/// Context stores the state of a single simulation system
///
/// There can be multiple contexts in a simulation, e.g. one for a trial move and one for the current state.
pub trait Context: GroupCollection + Clone + std::fmt::Debug + Sized + SyncFrom {
    /// Simulation cell type
    type Cell: cell::SimulationCell;
    /// Get reference to simulation cell
    fn cell(&self) -> &Self::Cell;
    /// Get mutable reference to simulation cell
    fn cell_mut(&mut self) -> &mut Self::Cell;
    /// Get reference to the topology
    fn topology(&self) -> Rc<topology::Topology>;
    /// Reference to Hamiltonian
    fn hamiltonian(&self) -> &energy::Hamiltonian;
    /// Mutable reference to Hamiltonian
    fn hamiltonian_mut(&mut self) -> &mut energy::Hamiltonian;

    /// Update the internal state to match a recently applied change
    ///
    /// By default, this function tries to update the Hamiltonian.
    /// For e.g. Ewald summation, the reciprocal space energy needs to be updated.
    #[allow(unused_variables)]
    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        use crate::energy::EnergyTerm;
        self.hamiltonian_mut().update(change)?;
        Ok(())
    }
}

/// A trait for objects that have a temperature
pub trait Temperature {
    /// Get the temperature in K
    fn temperature(&self) -> f64;
    /// Set the temperature in K
    fn set_temperature(&mut self, _temperature: f64) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "Setting the temperature is not implemented"
        ))
    }
}
