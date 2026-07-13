use crate::cell::BoundaryConditions;
use crate::energy::Hamiltonian;
use crate::group::GroupCollection;
use crate::Point;
use crate::{change::Change, topology::Topology};
use std::{
    cell::{Ref, RefMut},
    sync::Arc,
};

/// Full control over a simulation system: the view the framework itself holds.
///
/// Moves, analyses and energy terms are deliberately bound to the narrower [`ObserveContext`] or
/// [`PerturbContext`] instead, so the machinery below — backups, `undo`, direct mutation — is
/// unreachable from them. Every field of the implementing type is private, so this trait list *is*
/// the mutation surface.
pub trait Context:
    PerturbContext
    + crate::group::GroupCollectionMut
    + WithHamiltonianMut
    + WithSimulationCellMut
    + std::fmt::Debug
{
    /// Save energy term backups before a move is applied.
    ///
    /// Call before `apply_with_backup` so Ewald can snapshot old positions.
    fn save_energy_backups(&mut self, change: &Change) {
        self.hamiltonian_mut().save_backups(change, self);
    }

    /// Update internal state with backup for later undo on MC reject.
    fn update_with_backup(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian_mut().update_with_backup(self, change)?;
        Ok(())
    }

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
pub trait WithSimulationCell {
    /// Get reference to simulation cell.
    fn cell(&self) -> &crate::cell::Cell;
}

/// Mutable access to the simulation cell.
///
/// Kept apart from [`WithSimulationCell`] so that an observer bound to the read half cannot resize the
/// box. The single production caller restores a saved state (`state.rs`).
pub trait WithSimulationCellMut: WithSimulationCell {
    /// Get mutable reference to simulation cell.
    fn cell_mut(&mut self) -> &mut crate::cell::Cell;
}

/// A trait for objects that have a topology.
pub trait WithTopology {
    /// Get reference-counted topology of the system.
    fn topology(&self) -> Arc<Topology>;

    /// Get reference to the topology of the system.
    ///
    /// This does not increase the counter of `Arc<Topology>`
    /// and should therefore be faster than using `WithTopology::topology`.
    fn topology_ref(&self) -> &Arc<Topology>;
}

/// A trait for objects that have a hamiltonian.
pub trait WithHamiltonian: GroupCollection {
    /// Reference to Hamiltonian.
    ///
    /// Hamiltonian must be stored as `RefCell<Hamiltonian>`.
    fn hamiltonian(&self) -> Ref<'_, Hamiltonian>;
}

/// Mutable access to the Hamiltonian, through a *shared* reference.
///
/// The `&self` receiver is unavoidable — the Hamiltonian lives behind a `RefCell` because energy
/// terms update their caches while the rest of the system is borrowed immutably. It is also why
/// this must stay off the observer and perturber views: anything holding a plain `&Context` could
/// otherwise rewrite the shared physics. Only the framework may reach it.
pub trait WithHamiltonianMut: WithHamiltonian {
    /// Mutable reference to Hamiltonian.
    ///
    /// Hamiltonian must be stored as `RefCell<Hamiltonian>`.
    fn hamiltonian_mut(&self) -> RefMut<'_, Hamiltonian>;
}

/// A trait for objects which contains groups of particles with defined topology in defined cell.
pub trait ObserveContext: GroupCollection + WithSimulationCell + WithTopology {
    /// Count independently translatable entities (mass centers).
    ///
    /// Atomic groups contribute each active atom; molecular groups contribute one.
    /// Reservoir groups are excluded (handled by `count_active_molecules`).
    /// This is the correct N for the V^N partition function factor.
    fn num_active_mass_centers(&self) -> usize {
        self.topology_ref()
            .moleculekinds()
            .iter()
            .enumerate()
            .map(|(id, _)| self.count_active_molecules(crate::group::MoleculeId::new(id)))
            .sum()
    }

    /// Count active molecules of a given kind (excluding reservoir).
    ///
    /// For atomic mega-groups, N = number of active atoms.
    /// For molecular groups, N = number of non-empty groups.
    /// Reservoir groups always return 0 since they are outside the simulation box.
    fn count_active_molecules(&self, molecule_id: crate::group::MoleculeId) -> usize {
        let kind = self.topology_ref().moleculekind(molecule_id);
        if kind.is_reservoir() {
            // Reservoir particles live outside the simulation box and must not
            // contribute to physical counts like the V^N partition function factor.
            0
        } else {
            self.count_active(molecule_id, kind.group_kind())
        }
    }

    /// Mass of the i-th particle's atom type.
    fn atom_mass(&self, index: usize) -> f64 {
        self.topology_ref().atomkind(self.atom_kind(index)).mass()
    }

    /// Charge of the i-th particle's atom type.
    fn atom_charge(&self, index: usize) -> f64 {
        self.topology_ref().atomkind(self.atom_kind(index)).charge()
    }

    /// Resolve a selection to active atom indices, using each atom's current kind.
    ///
    /// Resolution is always against runtime kinds — there is no template-based variant — so a
    /// titration or speciation swap is reflected immediately.
    fn resolve_atoms(&self, selection: &crate::selection::Selection) -> Vec<usize> {
        selection.resolve_atoms(self.topology_ref(), self.groups(), &|i| self.atom_kind(i))
    }

    /// Resolve a selection to group indices, using each atom's current kind.
    ///
    /// See [`resolve_atoms`](Self::resolve_atoms).
    fn resolve_groups(&self, selection: &crate::selection::Selection) -> Vec<usize> {
        selection.resolve_groups(self.topology_ref(), self.groups(), &|i| self.atom_kind(i))
    }

    /// Optional cell list for spatial acceleration of pair interactions.
    fn cell_list(&self) -> Option<&crate::celllist::CellList> {
        None
    }

    /// Get distance between two particles with the given indices.
    ///
    /// ## Example implementation
    /// ```ignore
    /// self.cell().distance(self.position(i), self.position(j))
    /// ```
    fn get_distance(&self, i: usize, j: usize) -> Point;

    /// Get squared distance between two particles with the given indices.
    // Called per pair in nonbonded inner loops; must inline to expose
    // the full distance→square→spline chain for the compiler to optimize.
    #[inline(always)]
    fn get_distance_squared(&self, i: usize, j: usize) -> f64 {
        self.get_distance(i, j).norm_squared()
    }

    /// Position arrays (separate x, y, z) for batch evaluation.
    fn positions(&self) -> (&[f64], &[f64], &[f64]);

    /// Contiguous atom kind array (u32 for SIMD gather).
    fn atom_kinds_u32(&self) -> &[u32];

    /// Optional cached PBC parameters for branchless minimum image distance.
    fn pbc_params(&self) -> Option<crate::cell::PbcParams> {
        None
    }

    /// Get angle (in degrees) between three particles with the given indices.
    /// Here, the provided indices are called `i`, `j`, `k`, in this order.
    /// `i`, `j`, `k` are consecutively bonded atoms (`j` is the vertex of the angle).
    ///
    /// ## Example implementation
    /// ```ignore
    /// let [p1, p2, p3] = indices.map(|i| self.position(i));
    /// crate::geometry::angle_points(p1, p2, p3, self.cell())
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
    /// crate::geometry::dihedral_points(p1, p2, p3, p4, self.cell())
    /// ```
    fn get_dihedral_angle(&self, indices: &[usize; 4]) -> f64;

    /// Calculate mass center of set of particles given by their indices. Periodic boundary conditions are respected.
    fn mass_center(&self, indices: &[usize]) -> Point {
        if indices.is_empty() {
            return Point::zeros();
        }
        let ref_pos = self.position(indices[0]);
        let first_mass = self.atom_mass(indices[0]);
        let mut total_mass = first_mass;
        let mut com = ref_pos * first_mass;
        for &i in &indices[1..] {
            let mass = self.atom_mass(i);
            let unwrapped = ref_pos + self.cell().distance(&self.position(i), &ref_pos);
            com += unwrapped * mass;
            total_mass += mass;
        }
        com /= total_mass;
        self.cell().boundary(&mut com);
        com
    }
}

/// A system that can be cloned and perturbed, for analyses that need a trial move.
///
/// This is the whole of what a virtual move requires: displace particles, rotate them, rescale the
/// volume, and rebuild the energy caches afterwards. Everything else — backups, `undo`, group
/// resizing, atom-kind swaps — stays on [`Context`], so an analysis cannot desynchronise energy
/// caches it does not own.
///
/// [`hamiltonian`](WithHamiltonian::hamiltonian) is reachable here, but `hamiltonian_mut` is not.
/// That is why [`update`](Self::update) is a *required* method rather than a provided one: its
/// natural body would call `hamiltonian_mut`, which would hand the mutation back.
pub trait PerturbContext: ObserveContext + WithHamiltonian + Clone {
    /// Update internal state after a change (e.g. reciprocal-space energy for Ewald).
    ///
    /// Implementors reach their own Hamiltonian directly; there is no shared default body.
    fn update(&mut self, change: &Change) -> anyhow::Result<()>;

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

    /// Rigidly translate a whole group, carrying its mass center with it.
    ///
    /// Group-scoped rather than index-scoped so that it *can* maintain the group's derived
    /// state. A displaced molecule whose cached mass center stays behind is not a bookkeeping
    /// detail: the bounding-sphere cull in `energy/nonbonded` reads that centre to decide
    /// whether two groups interact at all, so a stale one silently drops the pair.
    fn translate_group(&mut self, group_index: usize, shift: &Point) -> anyhow::Result<()>;

    /// Rigidly rotate a whole group about its own mass center, composing its orientation.
    ///
    /// The mass center is invariant under a rotation about itself; the orientation is not, and
    /// composing it here is what keeps it describing the coordinates. Being exact and
    /// incremental, it also preserves the continuity that a best fit could not: for a symmetric
    /// molecule, re-deriving the frame each step could jump between equivalent orientations.
    fn rotate_group(
        &mut self,
        group_index: usize,
        quaternion: &crate::UnitQuaternion,
    ) -> anyhow::Result<()>;

    /// Reshape a group: move some of its atoms, leaving its rigid-body frame alone.
    ///
    /// The counterpart to [`translate_group`](Self::translate_group) and
    /// [`rotate_group`](Self::rotate_group): those move the molecule, this changes its shape.
    /// The internal-coordinate moves — pivot, crankshaft — apply here. The mass center and
    /// bounding radius are recomputed, since the atoms moved; the orientation is not, because a
    /// conformational change is not a rotation of the body. That is a fact about the operation,
    /// not something a caller has to remember.
    ///
    /// Positions are given outright rather than as a rotation because a sub-tree of a chain must
    /// be unwrapped by *following bonds*: taking the minimum image of each atom independently, as
    /// a rotation about a centre would, folds the part of the chain lying more than half a box
    /// from that centre into the wrong periodic image and tears it apart.
    fn set_group_conformation(
        &mut self,
        group_index: usize,
        indices: &[usize],
        positions: &[Point],
    ) -> anyhow::Result<()>;

    /// Shift a subset of a group's atoms, leaving its rigid-body frame alone.
    ///
    /// Like [`set_group_conformation`](Self::set_group_conformation), but expressed as a shift.
    fn translate_group_atoms(
        &mut self,
        group_index: usize,
        indices: &[usize],
        shift: &Point,
    ) -> anyhow::Result<()>;

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

    /// Move selected particles to the given positions and apply periodic boundary conditions.
    ///
    /// For geometry that must be built by following bonds — a rotated sub-tree of a chain, say —
    /// the caller has to supply positions outright: [`rotate_particles`](Self::rotate_particles)
    /// takes the minimum image of each particle independently, which folds any part of the
    /// molecule lying more than half a box length from the rotation centre into the wrong image.
    fn set_particle_positions(&mut self, indices: &[usize], positions: &[Point]);
}
