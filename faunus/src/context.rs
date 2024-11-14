use crate::cell::SimulationCell;
use crate::energy::Hamiltonian;
use crate::group::GroupCollection;
use crate::Point;
use crate::{change::Change, topology::Topology, SyncFrom};
use std::{
    cell::{Ref, RefMut},
    rc::Rc,
};

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
}

/// Context stores the state of a single simulation system
///
/// There can be multiple contexts in a simulation, e.g. one for a trial move and one for the current state.
#[cfg(not(feature = "chemfiles"))]
pub trait Context: ParticleSystem + WithHamiltonian + Clone + std::fmt::Debug + SyncFrom {
    /// Update the internal state to match a recently applied change
    ///
    /// By default, this function tries to update the Hamiltonian.
    /// For e.g. Ewald summation, the reciprocal space energy needs to be updated.
    #[allow(unused_variables)]
    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian_mut().update(self, change)?;
        Ok(())
    }
}

/// A trait for objects that have a simulation cell.
pub trait WithCell {
    type SimCell: SimulationCell;
    /// Get reference to simulation cell.
    fn cell(&self) -> &Self::SimCell;
    /// Get mutable reference to simulation cell.
    fn cell_mut(&mut self) -> &mut Self::SimCell;
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
pub trait WithHamiltonian: GroupCollection {
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
    /// ## Example implementation
    /// ```ignore
    /// self.cell().distance(self.particle(i).pos(), self.particle(j).pos())
    /// ```
    fn get_distance(&self, i: usize, j: usize) -> Point;

    /// Get squared distance between two particles with the given indices.
    fn get_distance_squared(&self, i: usize, j: usize) -> f64 {
        self.get_distance(i, j).norm_squared()
    }

    /// Get index of the atom kind of the particle with the given index.
    ///
    /// ## Example implementation
    /// ```ignore
    /// self.particle(i).atom_id
    /// ```
    fn get_atomkind(&self, i: usize) -> usize;

    /// Get angle (in degrees) between three particles with the given indices.
    /// Here, the provided indices are called `i`, `j`, `k`, in this order.
    /// `i`, `j`, `k` are consecutively bonded atoms (`j` is the vertex of the angle).
    ///
    /// ## Example implementation
    /// ```ignore
    /// let p1 = self.particle(indices[0]);
    /// let p2 = self.particle(indices[1]);
    /// let p3 = self.particle(indices[2]);
    ///
    /// crate::aux::angle_points(p1.pos(), p2.pos(), p3.pos(), self.cell())
    /// ```
    fn get_angle(&self, indices: &[usize; 3]) -> f64;

    /// Get dihedral angle (in degrees) between four particles with the given indices.
    ///
    /// ## Details
    /// - In this documentation, the provided indices are called `i`, `j`, `k`, `l`, in this order.
    /// - This method returns an angle between the plane formed by atoms `i`, `j`, `k` and the plane formed by
    ///   atoms `j`, `k`, `l`.
    /// - In case of a **proper** dihedral, `i`, `j`, `k`, `l` are (considered to be) consecutively bonded atoms.
    /// - In case of an **improper** dihedral, `i` is the central atom and `j`, `k`, `l` are (considered to be) bonded to it.
    /// - The angle adopts values between −180° and +180°. If the angle represents proper dihedral,
    ///   then 0° corresponds to the *cis* conformation and ±180° to the *trans* conformation
    ///   in line with the IUPAC/IUB convention.
    ///
    /// ## Example implementation
    /// ```ignore
    /// let p1 = self.particle(indices[0]);
    /// let p2 = self.particle(indices[1]);
    /// let p3 = self.particle(indices[2]);
    /// let p4 = self.particle(indices[3]);
    ///
    /// crate::aux::dihedral_points(p1.pos(), p2.pos(), p3.pos(), p4.pos(), self.cell())
    /// ```
    fn get_dihedral_angle(&self, indices: &[usize; 4]) -> f64;

    /// Shift positions of selected particles by target vector and apply periodic boundary conditions.
    fn translate_particles(&mut self, indices: &[usize], shift: &Point);
}
