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
    PerturbContext + crate::group::GroupCollectionMut + WithHamiltonianMut + std::fmt::Debug
{
    /// Update internal state after a change (e.g. reciprocal-space energy for Ewald).
    ///
    /// The framework calls this itself because it drives a trial in stages — propose, apply,
    /// evaluate, accept or undo — and must place the refresh between them. An analysis has no such
    /// stages: it perturbs and reads, so [`PerturbContext::measure`] refreshes on its behalf.
    fn update(&mut self, change: &Change) -> anyhow::Result<()>;

    /// Restore a whole configuration — a checkpoint, or a replayed trajectory frame.
    ///
    /// One verb rather than a box-setter and a coordinate-setter, because the order is load-bearing
    /// and getting it wrong is silent: orientations are fitted against the cell in force at the
    /// time, so coordinates applied while the outgoing box is still installed shatter any molecule
    /// straddling the new boundary, and the stored quaternion then describes that wreck. `cell` is
    /// `None` when the box is unchanged.
    fn restore_configuration(
        &mut self,
        cell: Option<crate::cell::Cell>,
        particles: &[crate::Particle],
        sizes: &[crate::group::GroupSize],
        quaternions: &[crate::UnitQuaternion],
    ) -> anyhow::Result<()>;

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
}

/// A trait for objects that have a simulation cell.
pub trait WithSimulationCell {
    /// Get reference to simulation cell.
    fn cell(&self) -> &crate::cell::Cell;
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

    /// Counter advanced whenever any particle position changes, including a restore.
    ///
    /// Lets a consumer cache a quantity derived from the coordinates and learn, on reading it,
    /// whether they moved underneath — the contract of
    /// [`group_lists_generation`](crate::group::GroupCollection::group_lists_generation), applied to
    /// the one thing almost every trial changes. The context maintains it, so no consumer can supply
    /// a key that misses a change.
    ///
    /// It says *that* something moved, never what. Only a consumer that knows it caused the change
    /// itself may patch; anyone else rebuilds.
    fn positions_generation(&self) -> u64;

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

/// A virtual move: what an analysis asks a cloned system to do to itself.
///
/// Each variant states an intent, not a sequence of writes. The caller never names the [`Change`]
/// that follows from it, so the two cannot disagree.
#[derive(Clone, Debug)]
pub enum Perturbation {
    /// Rigidly translate a group, carrying its mass center with it.
    Translate { group: usize, shift: Point },
    /// Rigidly rotate a group about its mass center, composing its orientation.
    ///
    /// Through the group, not the bare coordinates: a 6D tabulated potential is a function of the
    /// mass center and the quaternion alone, so rotating the coordinates by themselves would leave
    /// it reading an unchanged orientation.
    Rotate {
        group: usize,
        rotation: crate::UnitQuaternion,
    },
    /// Scale the cell and every particle position to `volume`.
    ScaleVolume {
        volume: f64,
        policy: crate::cell::VolumeScalePolicy,
    },
}

impl Perturbation {
    /// The change this perturbation makes, derived from it rather than stated alongside it.
    pub(crate) fn change(&self) -> Change {
        match self {
            Self::Translate { group, .. } | Self::Rotate { group, .. } => {
                Change::SingleGroup(*group, crate::change::GroupChange::RigidBody)
            }
            // `Everything` and `Volume` are the same change to every stateful term — both drop the
            // caches and rebuild them (`energy::stateful`). They differ only in what a *read* then
            // returns: `Volume` is served from the rebuilt cache, `Everything` is recomputed from
            // scratch. A virtual volume move compares absolute totals across a box that has
            // changed size, so it wants the recompute.
            Self::ScaleVolume { .. } => Change::Everything,
        }
    }
}

/// A system that can be cloned and perturbed, for analyses that need a trial move.
///
/// [`measure`](Self::measure) is the whole mutation surface, and it hands the system back as it
/// found it. The verbs it is built from, the `update` that must follow them, and the backups that
/// roll a trial back all live on [`Context`], the framework's own view.
///
/// That asymmetry is the point: the framework drives a trial in stages because it must decide
/// between them whether to keep it, while an analysis only ever looks. A perturbation that outlives
/// its refresh — or its rollback — is therefore something only the framework can express.
pub trait PerturbContext: ObserveContext + WithHamiltonian + Clone {
    /// Apply `perturbation`, evaluate `read` on the perturbed system, and restore it exactly.
    ///
    /// `read` receives the perturbed system and the [`Change`] describing it, which is what
    /// [`hamiltonian`](WithHamiltonian::hamiltonian)`().energy()` wants. Mass centers, bounding
    /// radii, the cell list and every energy cache describe the perturbed coordinates for the
    /// duration of the call.
    ///
    /// The system is then *restored*, not un-perturbed. An inverse perturbation would re-derive
    /// each cache by running its incremental update backwards, and a neighbour that a trial pose
    /// overlapped carries `+∞` in its cached energy: subtracting that again yields `NaN`. A
    /// snapshot has no such arithmetic to undo, and costs one refresh per trial instead of two.
    /// It is restored whether or not the trial succeeded, so a failure cannot leave the system
    /// half-moved.
    fn measure<R>(
        &mut self,
        perturbation: &Perturbation,
        read: impl FnOnce(&Self, &Change) -> R,
    ) -> anyhow::Result<R>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change::GroupChange;

    /// A rigid-body perturbation must announce itself as one. Labelling it `PartialUpdate` would
    /// make the bonded term recompute an intramolecular energy that a rigid move cannot change;
    /// labelling an internal change `RigidBody` would make it skip one that did. The same pairing
    /// is asserted for the framework's own moves in `propagate::moveproposal`.
    #[test]
    fn a_rigid_perturbation_announces_a_rigid_body_change() {
        let translate = Perturbation::Translate {
            group: 3,
            shift: Point::new(0.1, 0.0, 0.0),
        };
        let rotate = Perturbation::Rotate {
            group: 3,
            rotation: crate::UnitQuaternion::identity(),
        };
        for perturbation in [translate, rotate] {
            let Change::SingleGroup(group, group_change) = perturbation.change() else {
                panic!("expected SingleGroup, got {:?}", perturbation.change());
            };
            assert_eq!(group, 3);
            assert!(matches!(group_change, GroupChange::RigidBody));
            assert!(!group_change.internal_change());
        }
    }

    /// A volume scale moves every particle and the cell, and the analyses that use it want the
    /// absolute total energy either side — which only `Everything` recomputes from scratch.
    #[test]
    fn scaling_a_volume_announces_a_whole_system_change() {
        let scale = Perturbation::ScaleVolume {
            volume: 1000.0,
            policy: crate::cell::VolumeScalePolicy::Isotropic,
        };
        assert!(matches!(scale.change(), Change::Everything));
    }
}
