use crate::cell::SimulationCell;
use crate::energy::Hamiltonian;
use crate::group::GroupCollection;
use crate::Point;
use crate::{change::Change, topology::Topology};
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

    /// Update internal state with backup for later undo on MC reject.
    fn update_with_backup(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian_mut().update_with_backup(self, change)?;
        Ok(())
    }

    /// Synchronize state from another context after an MC accept/reject step.
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()>;

    /// Save particles at given indices and the group's mass center as backup.
    fn save_particle_backup(&mut self, group_index: usize, indices: &[usize]);

    /// Save all particles, mass centers, and cell as backup (for volume moves).
    fn save_system_backup(&mut self);

    /// Restore state from backup (reject path). Consumes the backup.
    fn undo(&mut self) -> anyhow::Result<()>;

    /// Drop backup without restoring (accept path).
    fn discard_backup(&mut self);
}

/// Context stores the state of a single simulation system
///
/// There can be multiple contexts in a simulation, e.g. one for a trial move and one for the current state.
#[cfg(not(feature = "chemfiles"))]
pub trait Context: ParticleSystem + WithHamiltonian + Clone + std::fmt::Debug {
    /// Update the internal state to match a recently applied change
    ///
    /// By default, this function tries to update the Hamiltonian.
    /// For e.g. Ewald summation, the reciprocal space energy needs to be updated.
    #[allow(unused_variables)]
    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian_mut().update(self, change)?;
        Ok(())
    }

    /// Update internal state with backup for later undo on MC reject.
    fn update_with_backup(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian_mut().update_with_backup(self, change)?;
        Ok(())
    }

    /// Synchronize state from another context after an MC accept/reject step.
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()>;

    /// Save particles at given indices and the group's mass center as backup.
    fn save_particle_backup(&mut self, group_index: usize, indices: &[usize]);

    /// Save all particles, mass centers, and cell as backup (for volume moves).
    fn save_system_backup(&mut self);

    /// Restore state from backup (reject path). Consumes the backup.
    fn undo(&mut self) -> anyhow::Result<()>;

    /// Drop backup without restoring (accept path).
    fn discard_backup(&mut self);
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
    fn hamiltonian(&self) -> Ref<'_, Hamiltonian>;

    /// Mutable reference to Hamiltonian.
    ///
    /// Hamiltonian must be stored as `RefCell<Hamiltonian>`.
    fn hamiltonian_mut(&self) -> RefMut<'_, Hamiltonian>;
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
    /// self.cell().distance(self.position(i), self.position(j))
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
    /// let [p1, p2, p3] = indices.map(|i| self.position(i));
    /// crate::auxiliary::angle_points(p1, p2, p3, self.cell())
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
    /// let [p1, p2, p3, p4] = indices.map(|i| self.position(i));
    /// crate::auxiliary::dihedral_points(p1, p2, p3, p4, self.cell())
    /// ```
    fn get_dihedral_angle(&self, indices: &[usize; 4]) -> f64;

    /// Calculate mass center of set of particles given by their indices. Periodic boundry conditions are respected.
    fn mass_center(&self, indices: &[usize]) -> Point {
        let positions: Vec<Point> = indices.iter().map(|&i| self.position(i)).collect();
        let atomids = indices.iter().map(|&i| self.get_atomkind(i));
        let masses: Vec<_> = atomids
            .map(|i| self.topology().atomkinds()[i].mass())
            .collect();
        let shift = positions.first().map_or_else(Point::zeros, |p| -*p);
        crate::auxiliary::mass_center_pbc(&positions, &masses, self.cell(), Some(shift))
    }

    /// Scale all particle positions and cell volume to a new volume.
    ///
    /// The algorithm unwraps PBC for molecular groups, scales positions using the old cell,
    /// resizes the cell, re-applies PBC with the new cell, and recomputes mass centers.
    /// Returns the old volume.
    fn scale_volume_and_positions(
        &mut self,
        new_volume: f64,
        policy: crate::cell::VolumeScalePolicy,
    ) -> anyhow::Result<f64>;

    /// Shift positions of selected particles by target vector and apply periodic boundary conditions.
    fn translate_particles(&mut self, indices: &[usize], shift: &Point);

    /// Rotate selected particles around the center of mass by the given quaternion. An optional
    /// translational shift can be provided to help remove PBC. The shift is added before rotation and
    /// subtracted after.
    fn rotate_particles(
        &mut self,
        indices: &[usize],
        quaternion: &crate::UnitQuaternion,
        center: Option<Point>,
    );
}
