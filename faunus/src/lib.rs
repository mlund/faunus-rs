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
use cell::SimulationCell;
use energy::Hamiltonian;
use nalgebra::Vector3;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use std::{
    cell::{Ref, RefCell, RefMut},
    path::Path,
    rc::Rc,
};
use topology::Topology;

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
pub mod basic;
pub mod chemistry;
pub mod dimension;
pub mod energy;
pub mod group;
pub mod montecarlo;
pub mod platform;
pub mod time;
pub mod topology;
pub mod transform;

pub use physical_constants::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, MOLAR_GAS_CONSTANT,
    VACUUM_ELECTRIC_PERMITTIVITY,
};

trait PointParticle {
    /// Type of the particle identifier
    type Idtype;
    /// Type of the particle position
    type Positiontype;
    /// Identifier for the particle type
    fn atom_id(&self) -> Self::Idtype;
    /// Get position
    fn pos(&self) -> &Self::Positiontype;
    /// Get mutable position
    fn pos_mut(&mut self) -> &mut Self::Positiontype;
    /// Index in main list of particle (immutable)
    fn index(&self) -> usize;
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct Particle {
    /// Type of the particle (index of the atom kind)
    atom_id: usize,
    /// Index in main list of particles
    index: usize,
    /// Position of the particle
    pos: Point,
}

impl Particle {
    pub(crate) fn new(atom_id: usize, index: usize, pos: Point) -> Particle {
        Particle {
            atom_id,
            index,
            pos,
        }
    }
}

impl PointParticle for Particle {
    type Idtype = usize;
    type Positiontype = Point;
    fn atom_id(&self) -> Self::Idtype {
        self.atom_id
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
    /// Synchronize internal state from another object of the same type
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()>;
}

/// Context stores the state of a single simulation system
///
/// There can be multiple contexts in a simulation, e.g. one for a trial move and one for the current state.
#[cfg(feature = "chemfiles")]
pub trait Context:
    ParticleSystem
    + WithHamiltonian
    + Clone
    + std::fmt::Debug
    + SyncFrom
    + crate::topology::chemfiles_interface::ChemFrameConvert
{
    /// Update the internal state to match a recently applied change
    ///
    /// By default, this function tries to update the Hamiltonian.
    /// For e.g. Ewald summation, the reciprocal space energy needs to be updated.
    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian_mut().update(self, change)?;
        Ok(())
    }

    /// Construct a new simulation system.
    ///
    /// ## Parameters
    /// - `faunus_file` Path to the input file with Faunus topology, hamiltonian etc.
    /// - `structure_file` Path to optional external structure file.
    /// - `rng` Random number generator.
    fn new(
        faunus_file: impl AsRef<Path> + Clone,
        structure_file: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self>;

    /// Construct a new simulation system from raw parts.
    fn from_raw_parts(
        topology: Rc<Topology>,
        cell: Box<dyn SimulationCell>,
        hamiltonian: RefCell<Hamiltonian>,
        structure_file: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

/// Context stores the state of a single simulation system
///
/// There can be multiple contexts in a simulation, e.g. one for a trial move and one for the current state.
#[cfg(not(feature = "chemfiles"))]
pub trait Context:
    ParticleSystem + WithHamiltonian + Clone + std::fmt::Debug + Sized + SyncFrom
{
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

    /// Construct a new simulation system.
    ///
    /// ## Parameters
    /// - `faunus_file` Path to the input file with Faunus topology, hamiltonian etc.
    /// - `structure_file` Optional external structure file.
    /// - `rng` Random number generator.
    fn new(
        faunus_file: impl AsRef<Path> + Clone,
        structure_file: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self>;

    /// Construct a new simulation system from raw parts.
    fn from_raw_parts(
        topology: Rc<Topology>,
        cell: Self::Cell,
        hamiltonian: RefCell<Hamiltonian>,
        structure_file: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self>;
}

/// A trait for objects that have a simulation cell.
pub trait WithCell {
    /// Get reference to simulation cell.
    fn cell(&self) -> &dyn SimulationCell;
    /// Get mutable reference to simulation cell.
    fn cell_mut(&mut self) -> &mut dyn SimulationCell;
}

/// A trait for objects that have a topology.
pub trait WithTopology {
    /// Get reference-counted topology of the system.
    fn topology(&self) -> Rc<Topology>;

    /// Get reference to the topology of the system.
    ///
    /// This does not increase the counter of Rc<Topology>
    /// and should therefore be faster than using `WithTopology::topology`.
    fn topology_ref(&self) -> &Rc<Topology>;
}

/// A trait for objects that have a hamiltonian.
pub trait WithHamiltonian: GroupCollection + Sized {
    /// Reference to Hamiltonian.
    ///
    /// Hamiltonian must be stored as `RefCell<Hamiltonian>`.
    fn hamiltonian(&self) -> Ref<Hamiltonian>;

    /// Mutable reference to Hamiltonian.
    ///
    /// Hamiltonian must be stored as `RefCell<Hamiltonian>`.
    fn hamiltonian_mut(&self) -> RefMut<Hamiltonian>;
}

/// A trait for objects that have a temperature
pub trait WithTemperature {
    /// Get the temperature in K.
    fn temperature(&self) -> f64;
    /// Set the temperature in K.
    fn set_temperature(&mut self, _temperature: f64) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "Setting the temperature is not implemented"
        ))
    }
}

/// A trait for objects which contains groups of particles with defined topology in defined cell.
pub trait ParticleSystem: GroupCollection + WithCell + WithTopology {
    /// Get distance between two particles with the given indices.
    ///
    /// ## Warning
    /// The default implementation of this method may be slow since it involves copying the particles.
    /// It is recommended to implement the method specifically for your platform.
    fn get_distance(&self, i: usize, j: usize) -> Point {
        self.cell()
            .distance(self.particle(i).pos(), self.particle(j).pos())
    }

    /// Get squared distance between two particles with the given indices.
    fn get_distance_squared(&self, i: usize, j: usize) -> f64 {
        self.get_distance(i, j).norm_squared()
    }

    /// Get index of the atom kind of the particle with the given index.
    ///
    /// ## Warning
    /// The default implementation of this method may be slow since it involves copying the particles.
    /// It is recommended to implement the method specifically for your platform.
    fn get_atomkind(&self, i: usize) -> usize {
        self.particle(i).atom_id
    }

    /// Get angle (in degrees) between three particles with the given indices.
    /// `i`, `j`, `k` are consecutively bonded atoms (`j` is the vertex of the angle).
    ///
    /// ## Warning
    /// The default implementation of this method may be slow since it involves copying the particles.
    /// It is recommended to implement the method specifically for your platform.
    fn get_angle(&self, i: usize, j: usize, k: usize) -> f64 {
        let p1 = self.particle(i);
        let p2 = self.particle(j);
        let p3 = self.particle(k);

        crate::basic::angle_points(p1.pos(), p2.pos(), p3.pos(), self.cell())
    }

    /// Get dihedral angle (in degrees) between four particles with the given indices.
    ///
    /// ## Details
    /// - This method returns an angle between the plane formed by atoms `i`, `j`, `k` and the plane formed by
    /// atoms `j`, `k`, `l`.
    /// - In case of a **proper** dihedral, `i`, `j`, `k`, `l` are (considered to be) consecutively bonded atoms.
    /// - In case of an **improper** dihedral, `i` is the central atom and `j`, `k`, `l` are (considered to be) bonded to it.
    /// - The angle adopts values between −180° and +180°. If the angle represents proper dihedral,
    /// then 0° corresponds to the *cis* conformation and ±180° to the *trans* conformation
    /// in line with the IUPAC/IUB convention.
    ///
    /// ## Warning
    /// The default implementation of this method may be slow since it involves copying the particles.
    /// It is recommended to implement the method specifically for your platform.
    fn get_dihedral(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let p1 = self.particle(i);
        let p2 = self.particle(j);
        let p3 = self.particle(k);
        let p4 = self.particle(l);

        crate::basic::dihedral_points(p1.pos(), p2.pos(), p3.pos(), p4.pos(), self.cell())
    }

    /// Shift positions of selected particles by target vector and apply periodic boundary conditions.
    fn translate_particles(&mut self, indices: &[usize], shift: &Vector3<f64>);
}
