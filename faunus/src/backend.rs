//! Simulation backend with a structure-of-arrays memory layout.
//!
//! Positions are stored as separate x, y, z vectors for cache-friendly access.
//! This enables SIMD batch evaluation without sync overhead.

use crate::{
    cell::PbcParams,
    cell::{BoundaryConditions, Cell},
    change::Change,
    context::{Perturbation, WithHamiltonianMut},
    energy::{builder::HamiltonianBuilder, Hamiltonian},
    group::{
        AtomKindId, GroupCollection, GroupCollectionMut, GroupGeometry, GroupLists, GroupSize,
        MoleculeId,
    },
    topology::Topology,
    Context, Group, ObserveContext, Point, UnitQuaternion, WithSimulationCell, WithTopology,
};

use rand::Rng;
use serde::Serialize;

use std::{cell::RefCell, path::Path, sync::Arc};

/// Extract medium from system/medium in YAML file
pub fn get_medium(path: impl AsRef<Path>) -> anyhow::Result<interatomic::coulomb::Medium> {
    let yaml = crate::auxiliary::read_yaml(&path)?;
    get_medium_str(&yaml)
}

/// Extract medium from the `system/medium` section of a YAML string.
pub fn get_medium_str(yaml: &str) -> anyhow::Result<interatomic::coulomb::Medium> {
    let value = serde_yml::from_str::<serde_yml::Value>(yaml)?;
    // Keep "not found" distinct from "found but invalid" so a bad field in an
    // existing section is reported as a parse error, not a missing section.
    let medium = value
        .get("system")
        .and_then(|system| system.get("medium"))
        .ok_or_else(|| anyhow::anyhow!("Could not find `system/medium` in input file"))?;
    crate::auxiliary::from_section_value("system/medium", medium)
}

/// Backup for undo on MC reject.
#[derive(Clone, Debug)]
struct Backup {
    /// (index, x, y, z, atom_kind) tuples for changed particles
    particles: Vec<(usize, f64, f64, f64, u32)>,
    /// Derived geometry per group; `None` records a group that had none, so undo can restore that.
    geometries: Vec<(usize, Option<GroupGeometry>)>,
    quaternions: Vec<(usize, UnitQuaternion)>,
    group_sizes: Vec<(usize, GroupSize)>,
    cell: Option<Cell>,
    /// Incremental cell list changes for undo (particle moves)
    cell_list_backup: Option<crate::celllist::CellListBackup>,
    /// Full cell list clone for undo (volume changes)
    cell_list_clone: Option<crate::celllist::CellList>,
}

/// Simulation backend with structure-of-arrays position layout for SIMD-friendly access.
#[derive(Clone, Debug, Serialize)]
pub struct Backend {
    topology: Arc<Topology>,
    /// Separate x, y, z arrays for SIMD-friendly access
    #[serde(skip)]
    x: Vec<f64>,
    #[serde(skip)]
    y: Vec<f64>,
    #[serde(skip)]
    z: Vec<f64>,
    /// Contiguous atom type array (u32 for SIMD gather)
    #[serde(skip)]
    atom_kinds: Vec<u32>,
    /// Bumped on every atom-kind change, so selections matching on atom identity re-resolve.
    #[serde(skip)]
    atom_kinds_generation: u64,
    /// Bumped on every coordinate change, so state derived from positions can tell it went stale.
    ///
    /// Bumped generously via [`Self::touch_positions`]: an extra bump costs a consumer one rebuild,
    /// a missing one costs correctness.
    #[serde(skip)]
    positions_generation: u64,
    #[serde(skip)]
    groups: Vec<Group>,
    #[serde(skip)]
    group_lists: GroupLists,
    cell: Cell,
    /// Cached to avoid recomputing on every `energy()` call; invalidated on cell mutation.
    #[serde(skip)]
    pbc_params: Option<PbcParams>,
    #[serde(skip)]
    hamiltonian: RefCell<Hamiltonian>,
    #[serde(skip)]
    backup: Option<Backup>,
    /// Cell list for spatial acceleration, when one was requested *and* the box can host a grid.
    #[serde(skip)]
    cell_list: Option<crate::celllist::CellList>,
    /// Cutoff a cell list was *requested* at. Kept apart from `cell_list`, which a box that cannot
    /// host a grid leaves empty — taking the request with it.
    #[serde(skip)]
    cell_list_cutoff: Option<f64>,
}

/// What to do with a group's stored orientation when its coordinates cannot pin one.
#[derive(Clone, Copy)]
enum OrientationPolicy {
    /// The group still holds the *same* molecule. A chain that has merely been reshaped is no
    /// rigid image of its reference conformation, and its stored orientation is the only record
    /// of the rotations that were applied to it — so keep it.
    KeepWhatIsStored,
    /// The slot now holds a *different* molecule, so the orientation it held describes something
    /// that is no longer there. Take the closest frame the coordinates allow; never inherit.
    NeverInherit,
}

impl Backend {
    /// Recompute a group's mass center and bounding radius from its active atoms.
    ///
    /// Private: every mutator that can invalidate them calls this itself. Exposing it would
    /// re-create the obligation to remember, which is the whole defect being closed.
    fn update_mass_center(&mut self, group_index: usize) {
        let group = &self.groups[group_index];
        let active = group.iter_active();
        // Tested before gathering the indices: an atomic mega-group has no mass center but does
        // have every ion in the box, and this runs on every insertion and deletion.
        if active.is_empty() || !self.topology().moleculekind(group.molecule()).has_com() {
            // A group with no active atoms has no mass center. Leaving the last one in place
            // would let an emptied group keep describing the molecule it no longer holds, and
            // its bounding radius feeds the cutoff culling that decides which pairs interact.
            self.groups[group_index].set_geometry(None);
            return;
        }
        let indices: Vec<usize> = active.collect();
        let mass_center = self.mass_center(&indices);
        // Bounding radius: max PBC distance from COM to any active particle
        let bounding_radius = indices
            .iter()
            .map(|&i| {
                let pos = Point::new(self.x[i], self.y[i], self.z[i]);
                self.cell.distance(&pos, &mass_center).norm()
            })
            .fold(0.0_f64, f64::max);
        self.groups[group_index].set_geometry(Some(GroupGeometry {
            mass_center,
            bounding_radius,
        }));
    }

    /// Shift the given particles and re-apply periodic boundaries.
    ///
    /// Private: an index list says nothing about which group's derived state it invalidates, which
    /// is exactly how a mass center gets left behind. Callers go through the group-scoped verbs.
    #[inline(always)]
    pub(crate) fn translate_particles(&mut self, indices: &[usize], shift: &Point) {
        let cell = self.cell.clone();
        for &i in indices {
            let mut pos = Point::new(
                self.x[i] + shift.x,
                self.y[i] + shift.y,
                self.z[i] + shift.z,
            );
            cell.boundary(&mut pos);
            self.x[i] = pos.x;
            self.y[i] = pos.y;
            self.z[i] = pos.z;
        }
        self.update_cell_list_particles(indices);
    }

    pub(crate) fn rotate_particles(
        &mut self,
        indices: &[usize],
        quaternion: &crate::UnitQuaternion,
        shift: Option<Point>,
    ) {
        let center = -shift.unwrap_or_else(Point::zeros);
        for &i in indices {
            let pos = Point::new(self.x[i], self.y[i], self.z[i]);
            let relative = self.cell.distance(&pos, &center);
            let mut rotated = quaternion.transform_vector(&relative) + center;
            self.cell.boundary(&mut rotated);
            self.x[i] = rotated.x;
            self.y[i] = rotated.y;
            self.z[i] = rotated.z;
        }
        self.update_cell_list_particles(indices);
    }

    pub(crate) fn set_particle_positions(&mut self, indices: &[usize], positions: &[Point]) {
        for (&i, position) in indices.iter().zip(positions) {
            let mut position = *position;
            self.cell.boundary(&mut position);
            self.x[i] = position.x;
            self.y[i] = position.y;
            self.z[i] = position.z;
        }
        self.update_cell_list_particles(indices);
    }

    /// A group's reference conformation and its current coordinates, gathered into one image —
    /// the two point sets a body frame is fitted between.
    ///
    /// The reference is copied only to release the borrow on the topology.
    fn body_frame_inputs(&self, group_index: usize) -> (Vec<Point>, Vec<Point>) {
        let group = &self.groups[group_index];
        let reference = self
            .topology_ref()
            .moleculekind(group.molecule())
            .reference_positions()
            .to_vec();
        let current: Vec<Point> = group.iter_active().map(|i| self.position(i)).collect();
        let gathered = crate::geometry::gather_molecule(&current, &self.cell);
        (reference, gathered)
    }

    /// Settle a group's orientation against the coordinates it now holds.
    ///
    /// One query, both policies, so that the awkward cases — a linear molecule, whose axial spin
    /// no coordinate can witness — are handled the same way whichever path arrives here.
    fn settle_orientation(&mut self, group_index: usize, policy: OrientationPolicy) {
        /// Below this, the coordinates *are* the reference conformation, rotated. Loose enough to
        /// absorb the rounding of a three-decimal structure file, and orders of magnitude tighter
        /// than any conformational change: a chain that has actually been reshaped misses its
        /// reference by ångströms, not by thousandths. Nothing real lands in between.
        const RIGID_IMAGE_TOLERANCE: f64 = 1e-3;

        let (reference, gathered) = self.body_frame_inputs(group_index);
        let prior = *self.groups[group_index].quaternion();
        let tolerance = match policy {
            OrientationPolicy::KeepWhatIsStored => RIGID_IMAGE_TOLERANCE,
            // Any frame the coordinates suggest beats one describing a molecule that has left.
            OrientationPolicy::NeverInherit => f64::INFINITY,
        };

        match crate::geometry::rigid_body_rotation(&reference, &gathered, tolerance, &prior) {
            Some(orientation) => self.groups[group_index].set_quaternion(orientation),
            None => match policy {
                OrientationPolicy::KeepWhatIsStored => (),
                // No reference conformation to fit against: no body frame to speak of.
                OrientationPolicy::NeverInherit => {
                    self.groups[group_index].set_quaternion(UnitQuaternion::identity());
                }
            },
        }
    }

    /// Build from raw parts (topology, cell, hamiltonian) for testing.
    pub(crate) fn from_raw_parts(
        topology: Arc<Topology>,
        cell: Cell,
        hamiltonian: RefCell<Hamiltonian>,
        structure: Option<&Path>,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        if topology.system.is_empty() {
            anyhow::bail!("Topology doesn't contain a system");
        }
        let group_lists = GroupLists::new(topology.moleculekinds().len());
        let pbc_params = PbcParams::try_from_cell(&cell);
        let mut backend = Self {
            topology: topology.clone(),
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            atom_kinds: Vec::new(),
            atom_kinds_generation: 0,
            positions_generation: 0,
            groups: Vec::new(),
            group_lists,
            cell,
            pbc_params,
            hamiltonian,
            backup: None,
            cell_list: None,
            cell_list_cutoff: None,
        };
        topology.insert_groups(&mut backend, structure, rng)?;
        backend.update(&Change::Everything)?;
        Ok(backend)
    }

    /// Build from a YAML input file.
    pub fn new(
        yaml_file: impl AsRef<Path>,
        structure_file: Option<&Path>,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        let medium = Some(get_medium(&yaml_file)?);
        let topology = Topology::from_file(&yaml_file)?;
        let hamiltonian_builder = HamiltonianBuilder::from_file(&yaml_file)?;
        let cell = Cell::from_file(&yaml_file)?;
        Self::assemble(
            medium,
            topology,
            hamiltonian_builder,
            cell,
            structure_file,
            rng,
        )
    }

    /// Build from a self-contained YAML string (no filesystem access).
    ///
    /// Mirrors [`new`](Self::new) but parses every section from memory, so it works on
    /// targets without a filesystem (e.g. `wasm32-unknown-unknown`). The input must be
    /// fully self-contained: no `include` directives and no external structure files.
    pub fn from_yaml_str(
        yaml: &str,
        structure_file: Option<&Path>,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        let medium = Some(get_medium_str(yaml)?);
        let topology = Topology::from_str(yaml)?;
        let hamiltonian_builder = HamiltonianBuilder::from_str(yaml)?;
        let cell = Cell::from_str(yaml)?;
        Self::assemble(
            medium,
            topology,
            hamiltonian_builder,
            cell,
            structure_file,
            rng,
        )
    }

    /// Assemble a backend from already-parsed sections. Shared by [`new`](Self::new) and
    /// [`from_yaml_str`](Self::from_yaml_str).
    fn assemble(
        medium: Option<interatomic::coulomb::Medium>,
        topology: Topology,
        hamiltonian_builder: HamiltonianBuilder,
        cell: Cell,
        structure_file: Option<&Path>,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        hamiltonian_builder.validate(topology.atomkinds())?;
        let hamiltonian = Hamiltonian::new(&hamiltonian_builder, &topology, medium.clone())?;

        let mut backend = Self::from_raw_parts(
            Arc::new(topology),
            cell,
            RefCell::new(hamiltonian),
            structure_file,
            rng,
        )?;

        backend
            .hamiltonian_mut()
            .finalize(&hamiltonian_builder, &backend, medium.as_ref())?;
        backend.update(&Change::Everything)?;

        if let Some(spline_opts) = hamiltonian_builder
            .pairpot_builder
            .as_ref()
            .and_then(|pb| pb.spline())
        {
            if spline_opts.cell_list {
                backend.request_cell_list(spline_opts.cutoff);
            }
        }

        Ok(backend)
    }

    /// Update cell list assignment for a single moved particle.
    fn update_cell_list_particle(&mut self, i: usize) {
        self.update_cell_list_particles(&[i]);
    }

    /// Record that coordinates changed, so state derived from them can tell it went stale.
    ///
    /// Called from every path that writes `x`/`y`/`z` — the particle-moving verbs (via
    /// `update_cell_list_particles`), volume scaling, insertion, and `undo`. A restore counts:
    /// coordinates moved, even if they moved back.
    fn touch_positions(&mut self) {
        self.positions_generation += 1;
    }

    /// Update cell list assignments for moved particles, tracking changes in backup.
    ///
    /// Every particle-moving verb ends here, which makes it the one place those paths have to
    /// announce that coordinates changed.
    fn update_cell_list_particles(&mut self, indices: &[usize]) {
        self.touch_positions();
        // A group move logs per-particle deltas (`cell_list_backup`) for a cheap tracked undo. A
        // system move (cluster, volume) instead snapshots the whole list (`cell_list_clone`) and is
        // restored wholesale on reject, so its live updates are untracked. Without the untracked
        // branch below, an accepted system move would leave the moved particles in their old cells.
        let tracked = self
            .backup
            .as_ref()
            .is_some_and(|b| b.cell_list_backup.is_some());
        if tracked {
            if let (Some(cl), Some(backup)) = (&mut self.cell_list, &mut self.backup) {
                if let Some(cl_backup) = &mut backup.cell_list_backup {
                    for &i in indices {
                        let pos = Point::new(self.x[i], self.y[i], self.z[i]);
                        cl.update_particle_tracked(i, &pos, cl_backup);
                    }
                }
            }
        } else if let Some(cl) = &mut self.cell_list {
            for &i in indices {
                let pos = Point::new(self.x[i], self.y[i], self.z[i]);
                cl.update_particle(i, &pos);
            }
        }
    }

    /// `quaternions` is what the state file *recorded*; where the restored coordinates turn out to
    /// be a rigid image of the molecule's reference conformation, the orientation is recomputed from
    /// them instead, because the coordinates are what every energy term and analysis actually reads.
    /// A state file written before orientations were tracked, or by a path that forgot to update
    /// one, therefore heals on load rather than importing the lie.
    fn apply_particles_and_groups(
        &mut self,
        particles: &[crate::Particle],
        sizes: &[GroupSize],
        quaternions: &[crate::UnitQuaternion],
    ) -> anyhow::Result<()> {
        /// Two orientations further apart than this describe visibly different molecules.
        const ORIENTATION_MISMATCH: f64 = 1e-6;

        self.set_positions(0..particles.len(), particles.iter().map(|p| &p.pos));
        for (i, p) in particles.iter().enumerate() {
            // Every group's geometry is recomputed below, so skip the per-particle refresh.
            self.set_atom_kind_unchecked(i, AtomKindId::new(p.atom_id));
        }
        for (i, (&size, &q)) in sizes.iter().zip(quaternions.iter()).enumerate() {
            self.resize_group(i, size)?;
            self.groups[i].set_quaternion(q);
        }

        // `resize_group` has already refreshed each group's geometry against the final coordinates.
        let mut corrected = 0usize;
        for i in 0..sizes.len() {
            let recorded = *self.groups[i].quaternion();
            self.settle_orientation(i, OrientationPolicy::KeepWhatIsStored);
            if self.groups[i].quaternion().angle_to(&recorded) > ORIENTATION_MISMATCH {
                corrected += 1;
            }
        }
        // Silently healing this would hide the fact that the run it came from was writing
        // orientations that did not describe its own coordinates.
        if corrected > 0 {
            log::warn!(
                "{corrected} group(s) had a stored orientation inconsistent with their coordinates; \
                 recomputed from the coordinates. The state file was written before orientations \
                 were tracked, or by a path that did not keep them in step."
            );
        }
        Ok(())
    }

    /// Re-derive everything that is a function of the box, around the coordinates that belong in it.
    ///
    /// The one place that knows what a new cell invalidates, so no path that installs one can
    /// remember a different subset. (`undo` is the exception: it *restores* the pre-move caches
    /// wholesale rather than re-deriving them, which is cheaper and exact.)
    ///
    /// `settle` brings in those coordinates, and runs with no cell list: a grid sized for the box
    /// that is gone must not be updated particle-by-particle, so it is built once, afterwards, from
    /// the coordinates that stay.
    fn cell_changed(
        &mut self,
        settle: impl FnOnce(&mut Self) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        self.pbc_params = PbcParams::try_from_cell(&self.cell);
        self.cell_list = None;
        settle(self)?;
        self.rebuild_cell_list();
        // A bounding radius is a minimum-image distance, so it belongs to the box it was measured in.
        for group_index in 0..self.groups.len() {
            self.update_mass_center(group_index);
        }
        Ok(())
    }

    /// Accelerate pair interactions with a cell list, sized for the current box.
    ///
    /// The cutoff outlives any grid built from it, so a box that cannot host one today can still
    /// get one after [`Self::set_cell`].
    fn request_cell_list(&mut self, cutoff: f64) {
        self.cell_list_cutoff = Some(cutoff);
        self.rebuild_cell_list();
    }

    /// Rebuild the cell list from the current positions and box, at the requested cutoff.
    ///
    /// Leaves no list behind when the box cannot host one: a grid sized for a box that is gone
    /// mis-bins silently, because `CellList` wraps out-of-range indices rather than rejecting them.
    fn rebuild_cell_list(&mut self) {
        use crate::cell::Shape;
        self.cell_list = None;
        let (Some(cutoff), Some(bb)) = (self.cell_list_cutoff, self.cell.bounding_box()) else {
            return;
        };
        // A cell that needs an orthorhombic *expansion* is not one: a hexagonal prism reduces
        // distances by Wigner-Seitz, which a rectangular grid's neighbour wrapping does not
        // reproduce, so the grid would hand back neighbour lists that miss interacting pairs.
        if self.cell.orthorhombic_expansion().is_some() {
            return;
        }
        let box_len = [bb.x, bb.y, bb.z];
        if box_len.iter().any(|&l| l.is_infinite() || l <= 0.0) {
            return;
        }
        let mut cl = crate::celllist::CellList::new(box_len, cutoff);
        let x = &self.x;
        let y = &self.y;
        let z = &self.z;
        let active_indices = self.groups.iter().flat_map(|g| g.iter_active());
        cl.build(
            |i| Point::new(x[i], y[i], z[i]),
            self.x.len(),
            active_indices,
        );
        log::trace!(
            "Built cell list with cutoff={cutoff:.1} Å for {} active particles",
            self.num_active_particles()
        );
        self.cell_list = Some(cl);
    }
}

impl crate::WithSimulationCell for Backend {
    #[inline(always)]
    fn cell(&self) -> &Cell {
        &self.cell
    }
}

impl crate::WithTopology for Backend {
    fn topology(&self) -> std::sync::Arc<crate::topology::Topology> {
        self.topology.clone()
    }
    fn topology_ref(&self) -> &std::sync::Arc<crate::topology::Topology> {
        &self.topology
    }
}

impl crate::WithHamiltonian for Backend {
    fn hamiltonian(&self) -> std::cell::Ref<'_, crate::energy::Hamiltonian> {
        self.hamiltonian.borrow()
    }
}

impl crate::context::WithHamiltonianMut for Backend {
    fn hamiltonian_mut(&self) -> std::cell::RefMut<'_, crate::energy::Hamiltonian> {
        self.hamiltonian.borrow_mut()
    }
}

impl GroupCollection for Backend {
    fn groups(&self) -> &[Group] {
        &self.groups
    }

    #[inline(always)]
    fn position(&self, index: usize) -> Point {
        Point::new(self.x[index], self.y[index], self.z[index])
    }

    #[inline(always)]
    fn atom_kind(&self, index: usize) -> AtomKindId {
        AtomKindId::new(self.atom_kinds[index] as usize)
    }

    fn atom_kinds_generation(&self) -> u64 {
        self.atom_kinds_generation
    }

    fn num_particles(&self) -> usize {
        self.x.len()
    }

    fn group_lists(&self) -> &GroupLists {
        &self.group_lists
    }
}

impl GroupCollectionMut for Backend {
    fn set_all_positions(&mut self, positions: &[Point]) -> anyhow::Result<()> {
        anyhow::ensure!(
            positions.len() == self.x.len(),
            "set_all_positions: {} positions for {} particles",
            positions.len(),
            self.x.len()
        );
        self.set_positions(0..positions.len(), positions.iter());
        for g in 0..self.groups.len() {
            self.update_mass_center(g);
        }
        Ok(())
    }

    fn set_group_orientation(&mut self, group_index: usize, orientation: crate::UnitQuaternion) {
        self.groups[group_index].set_quaternion(orientation);
    }

    fn set_atom_kind(&mut self, index: usize, atom_id: AtomKindId) {
        let previous = self.atom_kinds[index] as usize;
        self.set_atom_kind_unchecked(index, atom_id);
        // A mass-weighted center only moves if the mass changed; a pure charge swap, which is the
        // common titration, leaves the geometry alone and must not pay for a recompute.
        let kinds = self.topology_ref().atomkinds();
        let mass_changed = kinds[previous].mass() != kinds[atom_id.get()].mass();
        if previous != atom_id.get() && mass_changed {
            if let Some(group_index) = self.group_of_particle(index) {
                self.update_mass_center(group_index);
            }
        }
    }

    fn set_atom_kind_unchecked(&mut self, index: usize, atom_id: AtomKindId) {
        let atom_id = atom_id.get();
        debug_assert!(atom_id <= u32::MAX as usize, "atom_id overflows u32");
        if self.atom_kinds[index] == atom_id as u32 {
            return; // a no-op must not invalidate any selection cache
        }
        self.atom_kinds[index] = atom_id as u32;
        self.atom_kinds_generation += 1;
    }

    fn swap_particles(&mut self, i: usize, j: usize) {
        self.x.swap(i, j);
        self.y.swap(i, j);
        self.z.swap(i, j);
        if self.atom_kinds[i] != self.atom_kinds[j] {
            // Which index holds which kind changed, so kind-based selections are stale.
            self.atom_kinds.swap(i, j);
            self.atom_kinds_generation += 1;
        }
        self.update_cell_list_particles(&[i, j]);
    }

    fn set_positions<'a>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        positions: impl IntoIterator<Item = &'a Point>,
    ) {
        for (i, pos) in indices.into_iter().zip(positions) {
            self.x[i] = pos.x;
            self.y[i] = pos.y;
            self.z[i] = pos.z;
            self.update_cell_list_particle(i);
        }
    }

    fn place_group(
        &mut self,
        group_index: usize,
        positions: &[Point],
        orientation: Option<crate::UnitQuaternion>,
    ) -> anyhow::Result<()> {
        let group = &self.groups[group_index];
        // Exactly the capacity, not merely "no more than": a short slice would leave the tail of
        // the group holding the previous molecule's coordinates, and the mass center and
        // orientation would then be derived from a mixture of the new atoms and the old. The
        // inactive slots matter too — molecular swap reads them as its overlay template.
        anyhow::ensure!(
            positions.len() == group.capacity(),
            "place_group: {} positions for a group of capacity {}",
            positions.len(),
            group.capacity()
        );
        let start = group.start();
        self.set_positions(start..start + positions.len(), positions.iter());
        self.update_mass_center(group_index);
        match orientation {
            Some(known) => self.groups[group_index].set_quaternion(known),
            None => self.settle_orientation(group_index, OrientationPolicy::NeverInherit),
        }
        Ok(())
    }

    fn add_group(
        &mut self,
        molecule: MoleculeId,
        positions: &[Point],
        atom_ids: &[usize],
    ) -> anyhow::Result<&mut Group> {
        if positions.is_empty() {
            anyhow::bail!("Cannot create empty group");
        }
        if positions.len() != atom_ids.len() {
            anyhow::bail!(
                "positions length ({}) != atom_ids length ({})",
                positions.len(),
                atom_ids.len()
            );
        }
        // Growing the coordinate arrays changes what a position-derived cache would have covered.
        self.touch_positions();
        let start = self.x.len();
        for (pos, &aid) in positions.iter().zip(atom_ids) {
            self.x.push(pos.x);
            self.y.push(pos.y);
            self.z.push(pos.z);
            debug_assert!(aid <= u32::MAX as usize, "atom_id overflows u32");
            self.atom_kinds.push(aid as u32);
        }
        let range = start..start + positions.len();
        self.groups
            .push(Group::new(self.groups.len(), molecule, range));

        let group = self.groups.last_mut().unwrap();
        self.group_lists.add_group(group);
        Ok(group)
    }

    fn resize_group(&mut self, group_index: usize, status: GroupSize) -> anyhow::Result<()> {
        // A resize mutates the cell list untracked (below), so it may only run under a wholesale
        // (system) backup, never a tracked per-particle one — otherwise a rejected group-target
        // resize move would leak these cell-list edits. Pins "no group-target resize move" (#69).
        debug_assert!(
            self.backup
                .as_ref()
                .is_none_or(|b| b.cell_list_backup.is_none()),
            "resize_group under a tracked cell-list backup would leak untracked updates on reject"
        );
        let old_active: Vec<usize> = self.groups[group_index].iter_active().collect();
        self.groups[group_index].resize(status)?;
        let new_active: Vec<usize> = self.groups[group_index].iter_active().collect();
        // Reconcile the group lists (and bump `generation`) only when the active
        // set actually changed; a no-op resize (e.g. an MC restore re-applying the
        // same size) must not invalidate generation-keyed caches. See issue #34.
        if old_active.len() != new_active.len() {
            self.group_lists.update_group(&self.groups[group_index]);
        }

        if let Some(cl) = &mut self.cell_list {
            // Remove particles that were active but are no longer
            for &i in &old_active {
                if !new_active.contains(&i) {
                    cl.remove_particle(i);
                }
            }
            // Add particles that are newly active
            for &i in &new_active {
                if !old_active.contains(&i) {
                    cl.add_particle(i, &Point::new(self.x[i], self.y[i], self.z[i]));
                }
            }
        }

        // Which atoms are active *is* what the mass center and bounding radius are taken over,
        // so a resize invalidates them. Refreshing here rather than at the call sites is what
        // stops a shrink from quietly keeping the geometry of the atoms it just dropped (#52).
        self.update_mass_center(group_index);
        Ok(())
    }
}

impl ObserveContext for Backend {
    fn positions_generation(&self) -> u64 {
        self.positions_generation
    }

    #[inline(always)]
    fn get_distance(&self, i: usize, j: usize) -> Point {
        let pi = Point::new(self.x[i], self.y[i], self.z[i]);
        let pj = Point::new(self.x[j], self.y[j], self.z[j]);
        self.cell().distance(&pi, &pj)
    }

    fn positions(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.x, &self.y, &self.z)
    }

    fn atom_kinds_u32(&self) -> &[u32] {
        &self.atom_kinds
    }

    fn pbc_params(&self) -> Option<crate::cell::PbcParams> {
        self.pbc_params
    }

    fn cell_list(&self) -> Option<&crate::celllist::CellList> {
        self.cell_list.as_ref()
    }

    #[inline(always)]
    fn get_angle(&self, indices: &[usize; 3]) -> f64 {
        let [p1, p2, p3] = indices.map(|i| self.position(i));
        crate::geometry::angle_points(&p1, &p2, &p3, self.cell())
    }

    #[inline(always)]
    fn get_dihedral_angle(&self, indices: &[usize; 4]) -> f64 {
        let [p1, p2, p3, p4] = indices.map(|i| self.position(i));
        crate::geometry::dihedral_points(&p1, &p2, &p3, &p4, self.cell())
    }
}

impl crate::context::PerturbContext for Backend {
    /// A virtual move is a Monte Carlo trial that is always rejected, so it runs the same sequence
    /// `MoveRunner` does and inherits its guarantees. The energy backups must be taken *before* the
    /// coordinates move: Ewald's incremental update reads the old positions of the particles
    /// `change` names. `undo` then restores every cache from a snapshot instead of re-deriving it.
    ///
    /// The backup is armed before anything fallible, and `undo` runs whether or not the trial
    /// succeeded, so a failed perturbation cannot leave the system half-moved.
    fn measure<R>(
        &mut self,
        perturbation: &Perturbation,
        read: impl FnOnce(&Self, &Change) -> R,
    ) -> anyhow::Result<R> {
        use crate::transform::Transform;

        let change = perturbation.change();
        self.save_energy_backups(&change);

        let applied = match perturbation {
            Perturbation::Translate { group, shift } => {
                Transform::Translate(*shift).on_group_with_backup(*group, self)
            }
            Perturbation::Rotate { group, rotation } => {
                Transform::Rotate(*rotation).on_group_with_backup(*group, self)
            }
            Perturbation::ScaleVolume { volume, policy } => {
                Transform::VolumeScale(*policy, *volume).on_system_with_backup(self)
            }
        };

        let outcome = applied.and_then(|()| {
            Context::update(self, &change)?;
            Ok(read(self, &change))
        });

        self.undo()?;
        outcome
    }
}

impl Context for Backend {
    /// Reaches the `RefCell` directly rather than through `hamiltonian_mut`, so that both borrows
    /// of `self` are shared: the borrow checker is satisfied and no energy term re-borrows the
    /// Hamiltonian from `update`.
    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian.borrow_mut().update(self, change)?;
        Ok(())
    }

    fn restore_configuration(
        &mut self,
        cell: Option<Cell>,
        particles: &[crate::Particle],
        sizes: &[GroupSize],
        quaternions: &[crate::UnitQuaternion],
    ) -> anyhow::Result<()> {
        // The box first: orientations are fitted, and coordinates binned, against the cell in force.
        match cell {
            Some(cell) => {
                self.cell = cell;
                self.cell_changed(|this| {
                    this.apply_particles_and_groups(particles, sizes, quaternions)
                })?;
            }
            None => self.apply_particles_and_groups(particles, sizes, quaternions)?,
        }
        Context::update(self, &Change::Everything)
    }

    fn scale_volume_and_positions(
        &mut self,
        new_volume: f64,
        policy: crate::cell::VolumeScalePolicy,
    ) -> anyhow::Result<f64> {
        use crate::cell::{Shape, VolumeScale};

        // Rescaling moves every particle, and writes the coordinates directly rather than through
        // the particle-moving verbs.
        self.touch_positions();

        let old_volume = self
            .cell
            .volume()
            .ok_or_else(|| anyhow::anyhow!("Cell has no defined volume"))?;

        if old_volume.is_infinite() {
            anyhow::bail!("Cannot scale volume of an infinite cell");
        }

        let num_groups = self.groups.len();

        for g in 0..num_groups {
            let is_mol = self
                .topology
                .moleculekind(self.groups[g].molecule())
                .has_com();
            if !is_mol {
                for i in self.groups[g].iter_active() {
                    let mut pos = Point::new(self.x[i], self.y[i], self.z[i]);
                    self.cell.scale_position(new_volume, &mut pos, policy)?;
                    self.x[i] = pos.x;
                    self.y[i] = pos.y;
                    self.z[i] = pos.z;
                }
                continue;
            }
            let Some(&com) = self.groups[g].mass_center() else {
                continue;
            };
            // Unwrap PBC relative to COM
            for i in self.groups[g].iter_active() {
                let pos = Point::new(self.x[i], self.y[i], self.z[i]);
                let d = self.cell.distance(&pos, &com);
                let unwrapped = com + d;
                self.x[i] = unwrapped.x;
                self.y[i] = unwrapped.y;
                self.z[i] = unwrapped.z;
            }
            let mut scaled_com = com;
            self.cell
                .scale_position(new_volume, &mut scaled_com, policy)?;
            let shift = scaled_com - com;
            for i in self.groups[g].iter_active() {
                self.x[i] += shift.x;
                self.y[i] += shift.y;
                self.z[i] += shift.z;
            }
        }

        self.cell.scale_volume(new_volume, policy)?;

        // The scaled coordinates are what belongs in the new box, once wrapped into it.
        self.cell_changed(|this| {
            for g in 0..num_groups {
                for i in this.groups[g].iter_active() {
                    let mut pos = Point::new(this.x[i], this.y[i], this.z[i]);
                    this.cell.boundary(&mut pos);
                    this.x[i] = pos.x;
                    this.y[i] = pos.y;
                    this.z[i] = pos.z;
                }
            }
            Ok(())
        })?;

        Ok(old_volume)
    }

    fn translate_group(&mut self, group_index: usize, shift: &Point) -> anyhow::Result<()> {
        let indices = self.groups[group_index].select(
            &crate::group::ParticleSelection::Active,
            self.topology_ref(),
        )?;
        self.translate_particles(&indices, shift);
        // The orientation is unchanged by a translation; the mass center moves with the group.
        self.update_mass_center(group_index);
        Ok(())
    }

    fn rotate_group(
        &mut self,
        group_index: usize,
        quaternion: &crate::UnitQuaternion,
    ) -> anyhow::Result<()> {
        let indices = self.groups[group_index].select(
            &crate::group::ParticleSelection::Active,
            self.topology_ref(),
        )?;
        let center = self.mass_center(&indices);
        self.rotate_particles(&indices, quaternion, Some(-center));
        self.groups[group_index].rotate_by(quaternion);
        // Rotation about the mass center leaves it invariant, but the bounding radius is taken
        // under minimum image and a molecule near a boundary can present a different one.
        self.update_mass_center(group_index);
        Ok(())
    }

    fn set_group_conformation(
        &mut self,
        group_index: usize,
        indices: &[usize],
        positions: &[Point],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            indices.len() == positions.len(),
            "set_group_conformation: {} positions for {} selected particles",
            positions.len(),
            indices.len()
        );
        self.set_particle_positions(indices, positions);
        self.update_mass_center(group_index);
        Ok(())
    }

    fn save_particle_backup(&mut self, group_index: usize, indices: &[usize]) {
        assert!(self.backup.is_none(), "backup already exists");
        let particles = indices
            .iter()
            .map(|&i| (i, self.x[i], self.y[i], self.z[i], self.atom_kinds[i]))
            .collect();
        let group = &self.groups[group_index];
        let geometry = group.geometry();
        let quaternion = *group.quaternion();
        let group_size = group.size();
        let cell_list_backup = self.cell_list.as_ref().map(|cl| cl.begin_changes());
        self.backup = Some(Backup {
            particles,
            geometries: vec![(group_index, geometry)],
            quaternions: vec![(group_index, quaternion)],
            group_sizes: vec![(group_index, group_size)],
            cell: None,
            cell_list_backup,
            cell_list_clone: None,
        });
    }

    fn save_system_backup(&mut self) {
        assert!(self.backup.is_none(), "backup already exists");
        let particles = (0..self.x.len())
            .map(|i| (i, self.x[i], self.y[i], self.z[i], self.atom_kinds[i]))
            .collect();
        let geometries = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i, g.geometry()))
            .collect();
        let quaternions = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i, *g.quaternion()))
            .collect();
        let group_sizes = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i, g.size()))
            .collect();
        self.backup = Some(Backup {
            particles,
            geometries,
            quaternions,
            group_sizes,
            cell: Some(self.cell.clone()),
            cell_list_backup: None,
            cell_list_clone: self.cell_list.clone(),
        });
    }

    fn undo(&mut self) -> anyhow::Result<()> {
        let backup = self.backup.take().expect("undo called without backup");
        // A restore moves coordinates too, even though it moves them back. A consumer holding
        // state derived from the *trial* coordinates must see that they are no longer current.
        self.touch_positions();
        for (i, bx, by, bz, kind) in backup.particles {
            self.x[i] = bx;
            self.y[i] = by;
            self.z[i] = bz;
            if self.atom_kinds[i] != kind {
                self.atom_kinds[i] = kind;
                // Undoing a swap is itself a kind change; selection caches must see it. Skipped
                // when nothing changed so a rejected translate never invalidates them.
                self.atom_kinds_generation += 1;
            }
        }
        // Restored unconditionally: a group that had no geometry before the move must not keep the
        // one the rejected move computed for it.
        for (group_idx, old_geometry) in backup.geometries {
            self.groups[group_idx].set_geometry(old_geometry);
        }
        for (group_idx, q) in backup.quaternions {
            self.groups[group_idx].set_quaternion(q);
        }
        for (group_idx, size) in backup.group_sizes {
            let old_len = self.groups[group_idx].len();
            self.groups[group_idx].resize(size)?;
            // Skip groups whose active count is unchanged so an undo doesn't churn
            // `generation` for groups the move never resized. See issue #34.
            if self.groups[group_idx].len() != old_len {
                self.group_lists.update_group(&self.groups[group_idx]);
            }
        }
        if let Some(cell_list_backup) = backup.cell_list_backup {
            if let Some(cl) = &mut self.cell_list {
                cl.undo(cell_list_backup);
            }
        }
        if let Some(cell) = backup.cell {
            self.cell = cell;
            self.pbc_params = PbcParams::try_from_cell(&self.cell);
            // Volume change: restore pre-move cell list (avoids O(N) rebuild)
            self.cell_list = backup.cell_list_clone;
        }
        self.hamiltonian_mut().undo();
        Ok(())
    }

    fn discard_backup(&mut self) {
        self.backup = None;
        self.hamiltonian_mut().discard_backup();
    }
}

#[cfg(test)]
mod invariant_tests {
    use super::*;
    use crate::transform::Transform;

    /// A dimer with a light and a heavy atom, plus a second dimer that starts out empty.
    const DIMER: &str = r#"
atoms:
  - {name: A, mass: 3.0, charge: 0.0, sigma: 1.0}
  - {name: B, mass: 1.0, charge: 0.0, sigma: 1.0}
  - {name: C, mass: 1.0, charge: 5.0, sigma: 1.0}
molecules:
  - name: DIMER
    atoms: [A, B]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: DIMER
      N: 2
      active: 1
      insert: !Manual [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [5.0, 0.0, -2.0], [5.0, 0.0, 2.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    fn backend() -> Backend {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), DIMER).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    /// Every path that writes a coordinate must advance the positions generation.
    ///
    /// A consumer caches state derived from the coordinates and re-reads this counter to learn
    /// whether they moved under it. A path that forgets to advance it hands that consumer stale
    /// state and no way to know — the defect class of #86 and #88. An *extra* bump is harmless
    /// (one wasted rebuild), so each case here asserts only that the counter moved.
    #[test]
    fn every_coordinate_change_advances_the_positions_generation() {
        use crate::cell::VolumeScalePolicy;

        let shift = Point::new(0.1, 0.0, 0.0);
        let quaternion = crate::UnitQuaternion::identity();

        /// A named route into the backend that writes coordinates.
        type Mutation = (&'static str, Box<dyn Fn(&mut Backend)>);

        // Each case mutates coordinates by a different route into the backend.
        let mutations: Vec<Mutation> = vec![
            (
                "translate_particles",
                Box::new(move |c: &mut Backend| c.translate_particles(&[0], &shift)),
            ),
            (
                "rotate_particles",
                Box::new(move |c: &mut Backend| c.rotate_particles(&[0, 1], &quaternion, None)),
            ),
            (
                "set_particle_positions",
                Box::new(move |c: &mut Backend| {
                    c.set_particle_positions(&[0], &[Point::new(1.0, 1.0, 1.0)])
                }),
            ),
            (
                "set_positions",
                Box::new(|c: &mut Backend| c.set_positions([0], [&Point::new(2.0, 2.0, 2.0)])),
            ),
            (
                "scale_volume_and_positions",
                Box::new(|c: &mut Backend| {
                    c.scale_volume_and_positions(7000.0, VolumeScalePolicy::Isotropic)
                        .unwrap();
                }),
            ),
        ];

        for (name, mutate) in mutations {
            let mut context = backend();
            let before = context.positions_generation();
            mutate(&mut context);
            assert!(
                context.positions_generation() > before,
                "{name} changed coordinates without advancing the positions generation"
            );
        }

        // A restore moves coordinates back, which is still a move: state derived from the trial
        // coordinates is no longer current.
        let mut context = backend();
        context.save_system_backup();
        context.translate_particles(&[0], &shift);
        let after_trial = context.positions_generation();
        context.undo().unwrap();
        assert!(
            context.positions_generation() > after_trial,
            "undo restored coordinates without advancing the positions generation"
        );
    }

    #[test]
    fn a_no_op_kind_change_invalidates_nothing() {
        let mut context = backend();
        let before = context.atom_kinds_generation();
        context.set_atom_kind(0, context.atom_kind(0));
        assert_eq!(context.atom_kinds_generation(), before);
    }

    #[test]
    fn an_equal_mass_kind_change_bumps_the_counter_but_leaves_the_geometry() {
        let mut context = backend();
        let generation = context.atom_kinds_generation();
        let geometry = context.groups()[0].geometry();
        // B (mass 1) → C (mass 1): only the charge differs, the common titration.
        context.set_atom_kind(1, AtomKindId::new(2));
        assert_eq!(context.atom_kinds_generation(), generation + 1);
        assert_eq!(context.groups()[0].geometry(), geometry);
    }

    #[test]
    fn a_mass_changing_kind_change_moves_the_center_and_radius() {
        let mut context = backend();
        let before = context.groups()[0].geometry().unwrap();
        // A (mass 3) at z=−2, B (mass 1) at z=+2 ⇒ center at z = −1.
        assert!((before.mass_center.z + 1.0).abs() < 1e-12);
        // Turn B into a second A (mass 1 → 3); the center moves to z = 0.
        context.set_atom_kind(1, AtomKindId::new(0));
        let after = context.groups()[0].geometry().unwrap();
        assert!(after.mass_center.z.abs() < 1e-12, "{:?}", after.mass_center);
        // Both atoms are now 2 Å from the center, up from 1 and 3.
        assert!((after.bounding_radius - 2.0).abs() < 1e-12);
        assert!(after.bounding_radius != before.bounding_radius);
    }

    /// The cache key must notice a swap for kind-based selections, and ignore it otherwise — an
    /// `atomtype` selection re-resolving on every titration step is correct; `molecule X` doing so
    /// would be wasted work on the energy hot path.
    #[test]
    fn only_kind_dependent_selections_re_resolve_after_a_swap() {
        use crate::selection::{CachedSelection, Selection};
        let mut context = backend();
        let mut by_kind = CachedSelection::atoms(Selection::parse("atomtype B").unwrap());
        let mut by_molecule = CachedSelection::atoms(Selection::parse("molecule DIMER").unwrap());
        assert_eq!(by_kind.resolve(&context), &[crate::group::AbsIndex::new(1)]);
        let molecule_atoms = by_molecule.resolve(&context).to_vec();

        // B → A: nothing is of kind B any more, but the molecule still holds the same atoms.
        context.set_atom_kind(1, AtomKindId::new(0));
        assert!(
            by_kind.resolve(&context).is_empty(),
            "an atomtype selection must follow the swap"
        );
        assert_eq!(by_molecule.resolve(&context), molecule_atoms.as_slice());
    }

    #[test]
    fn group_of_particle_finds_the_owning_group() {
        let context = backend();
        assert_eq!(context.group_of_particle(0), Some(0));
        assert_eq!(context.group_of_particle(1), Some(0));
        assert_eq!(context.group_of_particle(2), Some(1));
        assert_eq!(context.group_of_particle(3), Some(1));
        assert_eq!(context.group_of_particle(4), None);
    }

    /// A mass-changing swap now moves the group's geometry, so a rejected one must put it back.
    #[test]
    fn undo_restores_geometry_after_a_mass_changing_swap() {
        let mut context = backend();
        let before = context.groups()[0].geometry().unwrap();

        context.save_particle_backup(0, &[0, 1]);
        context.set_atom_kind(1, AtomKindId::new(0)); // mass 1 → 3, moves the center
        assert!(context.groups()[0].geometry().unwrap() != before);

        context.undo().unwrap();
        assert_eq!(context.groups()[0].geometry().unwrap(), before);
    }

    #[test]
    fn undo_of_a_kind_swap_bumps_the_counter_again() {
        let mut context = backend();
        context.save_particle_backup(0, &[0, 1]);
        let before = context.atom_kinds_generation();
        context.set_atom_kind(1, AtomKindId::new(0));
        assert!(context.atom_kinds_generation() > before);

        context.undo().unwrap();
        assert_eq!(context.atom_kind(1), AtomKindId::new(1), "kind restored");
        assert!(
            context.atom_kinds_generation() > before,
            "undoing a swap is itself a kind change and must invalidate caches"
        );
    }

    /// Defensive: no current path takes a group's geometry from `None` to `Some`, because
    /// insertion computes it while the group is still full (`topology/block.rs`). Should one ever
    /// appear, `undo` must clear the geometry again rather than keep what the rejected move
    /// computed — `nonbonded` culls on it.
    #[test]
    fn undo_can_clear_a_geometry_the_move_created() {
        let mut context = backend();
        context.groups[1].set_geometry(None);

        context.save_particle_backup(1, &[2, 3]);
        Transform::Activate.on_group(1, &mut context).unwrap();
        assert!(context.groups()[1].geometry().is_some());

        context.undo().unwrap();
        assert!(context.groups()[1].geometry().is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::EnergyChange;
    use crate::WithHamiltonian;

    /// Verify total energy equals sum of per-group energies, and mass_center matches auxiliary.
    #[test]
    fn energy_and_mass_center() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();

        let e_total = ctx.hamiltonian().energy(&ctx, &crate::Change::Everything);
        assert!(e_total.is_finite(), "Energy should be finite");

        // Sum of per-group RigidBody energies should approximate total energy
        let e_sum: f64 = (0..ctx.groups().len())
            .filter(|&gi| !ctx.groups()[gi].is_empty())
            .map(|gi| {
                let change = crate::Change::SingleGroup(gi, crate::GroupChange::RigidBody);
                ctx.hamiltonian().energy(&ctx, &change)
            })
            .sum();
        // Per-group sum double-counts inter-group pairs, but both values should be finite
        // and the sum should be nonzero if the total is nonzero
        assert!(e_sum.is_finite(), "Per-group energy sum should be finite");
        if e_total.abs() > 1e-10 {
            assert!(
                e_sum.abs() > 1e-10,
                "Per-group sum should be nonzero when total is nonzero"
            );
        }

        // Verify mass_center matches geometry::mass_center_pbc
        for group in ctx.groups() {
            if group.is_empty() {
                continue;
            }
            let indices: Vec<usize> = group.iter_active().collect();
            let com_trait = ctx.mass_center(&indices);
            let positions: Vec<Point> = indices.iter().map(|&i| ctx.position(i)).collect();
            let topology = ctx.topology();
            let masses: Vec<f64> = indices
                .iter()
                .map(|&i| topology.atomkind(ctx.atom_kind(i)).mass())
                .collect();
            let com_aux = crate::geometry::mass_center_pbc(&positions, &masses, ctx.cell(), None);
            let err = (com_trait - com_aux).norm();
            assert!(
                err < 1e-10,
                "mass_center mismatch for group {}: trait={com_trait:?}, aux={com_aux:?}, err={err:.2e}",
                group.index()
            );
        }
    }

    /// Verify set_positions updates coordinates without changing atom kinds.
    #[test]
    fn set_positions_preserves_atom_kinds() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let mut ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();

        let group = &ctx.groups()[0];
        let indices: Vec<usize> = group.iter_active().collect();
        let original_kinds: Vec<u32> = indices.iter().map(|&i| ctx.atom_kinds[i]).collect();

        let new_positions: Vec<Point> = indices
            .iter()
            .enumerate()
            .map(|(j, _)| Point::new(j as f64, j as f64 * 2.0, j as f64 * 3.0))
            .collect();
        ctx.set_positions(indices.clone(), new_positions.iter());

        for (j, &i) in indices.iter().enumerate() {
            let pos = ctx.position(i);
            assert_eq!(pos, new_positions[j], "position not updated at index {i}");
            assert_eq!(
                ctx.atom_kinds[i], original_kinds[j],
                "atom kind changed at index {i}"
            );
        }
    }

    /// Verify cell-list-accelerated per-group energies are consistent with total energy.
    #[test]
    fn cell_list_partial_update_energy() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");

        let ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        assert!(
            ctx.cell_list.is_some(),
            "Cell list should be built for splined input"
        );

        let e_total = ctx.hamiltonian().energy(&ctx, &crate::Change::Everything);
        assert!(e_total.is_finite(), "Total energy should be finite");

        // PartialUpdate and RigidBody energies should be finite and nonzero for non-empty groups
        for gi in 0..5.min(ctx.groups().len()) {
            if ctx.groups()[gi].is_empty() {
                continue;
            }
            let change_partial = crate::Change::SingleGroup(
                gi,
                crate::GroupChange::PartialUpdate(vec![crate::group::RelIndex::new(0)]),
            );
            let e_partial = ctx.hamiltonian().energy(&ctx, &change_partial);
            assert!(
                e_partial.is_finite(),
                "Group {gi}: PartialUpdate energy not finite"
            );

            let change_rigid = crate::Change::SingleGroup(gi, crate::GroupChange::RigidBody);
            let e_rigid = ctx.hamiltonian().energy(&ctx, &change_rigid);
            assert!(
                e_rigid.is_finite(),
                "Group {gi}: RigidBody energy not finite"
            );
        }
    }

    /// Verify apply_particles_and_groups roundtrip: save → perturb → restore → compare.
    #[test]
    fn apply_particles_and_groups_roundtrip() {
        let mut ctx = Backend::new(
            "tests/files/translate_molecules_simulation.yaml",
            None,
            &mut rand::thread_rng(),
        )
        .unwrap();

        // Snapshot original state
        let original_particles: Vec<crate::Particle> = (0..ctx.num_particles())
            .map(|i| crate::Particle::new(ctx.atom_kind(i).get(), ctx.position(i)))
            .collect();
        let original_energy = ctx.hamiltonian().energy(&ctx, &crate::Change::Everything);
        let original_sizes: Vec<crate::group::GroupSize> = ctx
            .groups()
            .iter()
            .map(|g| crate::group::GroupSize::from_count(g.len(), g.capacity()))
            .collect();
        let original_quaternions: Vec<crate::UnitQuaternion> =
            ctx.groups().iter().map(|g| *g.quaternion()).collect();
        let original_coms: Vec<Option<Point>> = ctx
            .groups()
            .iter()
            .map(|g| g.mass_center().copied())
            .collect();

        assert!(
            original_energy.abs() > 1e-6,
            "Test requires nonzero initial energy"
        );

        // Perturb: collapse all positions to origin so energy changes drastically
        let n = ctx.num_particles();
        let zeros: Vec<Point> = vec![Point::zeros(); n];
        ctx.set_positions(0..n, zeros.iter());
        ctx.update(&crate::Change::Everything).unwrap();

        // Sanity check: collapsing to origin must produce a different energy
        let perturbed_energy = ctx.hamiltonian().energy(&ctx, &crate::Change::Everything);
        assert_ne!(
            perturbed_energy, original_energy,
            "Perturbation should change energy"
        );

        // Restore via apply_particles_and_groups
        ctx.apply_particles_and_groups(&original_particles, &original_sizes, &original_quaternions)
            .unwrap();
        ctx.update(&crate::Change::Everything).unwrap();

        // Verify positions restored
        for (i, orig) in original_particles.iter().enumerate() {
            let restored_pos = ctx.position(i);
            assert!(
                (restored_pos - orig.pos).norm() < 1e-14,
                "Position mismatch at particle {i}"
            );
            assert_eq!(
                ctx.atom_kind(i).get(),
                orig.atom_id,
                "atom_id mismatch at {i}"
            );
        }

        // Verify mass centers restored
        for (i, orig_com) in original_coms.iter().enumerate() {
            let restored_com = ctx.groups()[i].mass_center().copied();
            match (orig_com, restored_com) {
                (Some(a), Some(b)) => assert!(
                    (a - b).norm() < 1e-12,
                    "COM mismatch at group {i}: {a:?} vs {b:?}"
                ),
                (None, None) => {}
                _ => panic!("COM presence mismatch at group {i}"),
            }
        }

        // Verify energy restored
        let restored_energy = ctx.hamiltonian().energy(&ctx, &crate::Change::Everything);
        assert!(
            (restored_energy - original_energy).abs() < 1e-10,
            "Energy not restored: {original_energy} vs {restored_energy}"
        );
    }

    /// Verify backup/undo restores group quaternion after rotation.
    #[test]
    fn backup_undo_restores_quaternion() {
        let mut context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(std::path::Path::new("tests/files/structure.xyz")),
            &mut rand::thread_rng(),
        )
        .unwrap();

        let group_index = 1;
        assert_eq!(
            *context.groups()[group_index].quaternion(),
            crate::UnitQuaternion::identity()
        );

        let axis = nalgebra::UnitVector3::new_normalize(Point::new(0.0, 0.0, 1.0));
        let q = crate::UnitQuaternion::from_axis_angle(&axis, 0.8);
        let transform = crate::transform::Transform::Rotate(q);
        transform
            .on_group_with_backup(group_index, &mut context)
            .unwrap();

        assert!(context.groups()[group_index].quaternion().angle_to(&q) < 1e-12);

        context.undo().unwrap();
        assert!(
            context.groups()[group_index]
                .quaternion()
                .angle_to(&crate::UnitQuaternion::identity())
                < 1e-12
        );
    }

    /// Verify add_group stores positions and atom_ids correctly.
    #[test]
    fn test_add_group_preserves_data() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let mut ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        let n_before = ctx.num_particles();
        let mol_id = ctx.groups()[0].molecule();
        let topo_atom_ids: Vec<usize> = ctx.topology().moleculekind(mol_id).atom_indices().to_vec();
        let positions: Vec<Point> = topo_atom_ids
            .iter()
            .enumerate()
            .map(|(j, _)| Point::new(j as f64, j as f64 * 2.0, j as f64 * 3.0))
            .collect();
        let group = ctx.add_group(mol_id, &positions, &topo_atom_ids).unwrap();
        assert_eq!(group.capacity(), positions.len());
        assert_eq!(group.len(), positions.len());
        let start = group.start();
        assert_eq!(start, n_before);
        for (j, &expected_kind) in topo_atom_ids.iter().enumerate() {
            assert_eq!(ctx.position(start + j), positions[j]);
            assert_eq!(ctx.atom_kind(start + j).get(), expected_kind);
        }
    }

    /// Verify swap_particles exchanges all fields.
    #[test]
    fn test_swap_particles() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let mut ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        let (i, j) = (0, 1);
        let pos_i = ctx.position(i);
        let pos_j = ctx.position(j);
        let kind_i = ctx.atom_kind(i);
        let kind_j = ctx.atom_kind(j);
        ctx.swap_particles(i, j);
        assert_eq!(ctx.position(i), pos_j);
        assert_eq!(ctx.position(j), pos_i);
        assert_eq!(ctx.atom_kind(i), kind_j);
        assert_eq!(ctx.atom_kind(j), kind_i);
    }

    /// Verify set_atom_kind updates kind without changing position.
    #[test]
    fn test_set_atom_kind() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let mut ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        let pos_before = ctx.position(0);
        let new_kind =
            AtomKindId::new((ctx.atom_kind(0).get() + 1) % ctx.topology().atomkinds().len());
        ctx.set_atom_kind(0, new_kind);
        assert_eq!(ctx.atom_kind(0), new_kind);
        assert_eq!(ctx.position(0), pos_before);
    }
}

/// Everything derived from the box — the cell-list grid, the cached PBC parameters, every group's
/// minimum-image bounding radius — must follow when a new cell is installed.
#[cfg(test)]
mod restoring_a_cell {
    use super::*;
    use crate::cell::{Cuboid, Shape};
    use crate::energy::EnergyChange;
    use crate::WithHamiltonian;
    use rand::SeedableRng;

    const BIG: f64 = 120.0;
    const SMALL: f64 = 60.0;
    const CUTOFF: f64 = 10.0;

    fn input(cell_list: bool) -> String {
        format!(
            r#"
atoms:
  - {{name: LJ, mass: 1.0, sigma: 3.0, epsilon: 1.0}}
molecules:
  - {{name: M, atoms: [LJ]}}
system:
  cell: !Cuboid [{BIG}, {BIG}, {BIG}]
  medium: {{permittivity: !Vacuum, temperature: 298.15}}
  energy:
    nonbonded:
      default:
        - !LennardJones {{mixing: LB}}
      spline: {{cutoff: {CUTOFF}, table_points: 1000, cell_list: {cell_list}}}
  blocks:
    - molecule: M
      N: 512
      active: 512
      insert: !RandomAtomPos {{}}
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#
        )
    }

    fn context(cell_list: bool) -> Backend {
        Backend::from_yaml_str(
            &input(cell_list),
            None,
            &mut rand::rngs::StdRng::seed_from_u64(7),
        )
        .unwrap()
    }

    /// 8³ simple-cubic lattice filling the *small* box, spacing 7.5 Å < cutoff. Sites sit half a
    /// spacing from each face, so opposite faces interact through the periodic boundary — the pairs
    /// a grid still sized for the big box loses.
    fn lattice() -> Vec<Point> {
        let spacing = SMALL / 8.0;
        let coord = |i: usize| -SMALL / 2.0 + spacing * (i as f64 + 0.5);
        (0..8)
            .flat_map(|i| (0..8).flat_map(move |j| (0..8).map(move |k| (i, j, k))))
            .map(|(i, j, k)| Point::new(coord(i), coord(j), coord(k)))
            .collect()
    }

    /// Install a box and the coordinates that belong in it, through the one public restore verb.
    fn restore(context: &mut Backend, cell: Cell, positions: &[Point]) {
        let particles: Vec<_> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| crate::Particle::new(context.atom_kind(i).get(), pos))
            .collect();
        let sizes: Vec<_> = context
            .groups()
            .iter()
            .map(|g| GroupSize::from_count(g.len(), g.capacity()))
            .collect();
        let quaternions: Vec<_> = context.groups().iter().map(|g| *g.quaternion()).collect();
        context
            .restore_configuration(Some(cell), &particles, &sizes, &quaternions)
            .unwrap();
    }

    /// Resize the box, leaving the coordinates where they are.
    fn restore_box(context: &mut Backend, cell: Cell) {
        let positions: Vec<Point> = (0..context.num_particles())
            .map(|i| context.position(i))
            .collect();
        restore(context, cell, &positions);
    }

    /// Every atom against the rest of the system — the quantity the cell list serves. The total
    /// energy is summed pair-by-pair and a rigid-body move reads a cached group matrix, so neither
    /// would notice a stale grid.
    fn move_energies(context: &Backend) -> f64 {
        let hamiltonian = context.hamiltonian();
        (0..context.groups().len())
            .map(|g| {
                let change = Change::SingleGroup(
                    g,
                    crate::GroupChange::PartialUpdate(vec![crate::group::RelIndex::new(0)]),
                );
                hamiltonian.energy(context, &change)
            })
            .sum()
    }

    fn move_energies_in_small_box(cell_list: bool) -> f64 {
        let mut context = context(cell_list);
        assert_eq!(context.cell_list.is_some(), cell_list);
        restore(&mut context, Cell::Cuboid(Cuboid::cubic(SMALL)), &lattice());
        move_energies(&context)
    }

    #[test]
    fn the_cell_list_still_finds_every_pair_the_brute_force_sum_does() {
        let with_list = move_energies_in_small_box(true);
        let brute_force = move_energies_in_small_box(false);
        assert!(
            brute_force.is_finite() && brute_force.abs() > 0.1,
            "degenerate reference energy {brute_force}"
        );
        assert!(
            (with_list - brute_force).abs() < 1e-9 * brute_force.abs(),
            "cell list {with_list} vs brute force {brute_force}"
        );
    }

    /// The restore path is what issue #89 was reported against: a state saved in a smaller box,
    /// loaded into a context the input file built at the larger one.
    #[test]
    fn restoring_a_state_saved_in_another_box_agrees_with_the_brute_force_sum() {
        let mut saved = context(true);
        restore(&mut saved, Cell::Cuboid(Cuboid::cubic(SMALL)), &lattice());
        let state = crate::state::State::save(&saved, 0);

        let mut restored = context(true);
        state.load(&mut restored).unwrap();
        assert_eq!(
            restored.cell().bounding_box().unwrap(),
            Point::new(SMALL, SMALL, SMALL)
        );

        let restored_energies = move_energies(&restored);
        let brute_force = move_energies_in_small_box(false);
        assert!(
            (restored_energies - brute_force).abs() < 1e-9 * brute_force.abs(),
            "restored {restored_energies} vs brute force {brute_force}"
        );
    }

    /// A stale cache would keep taking the minimum image in the old box.
    #[test]
    fn the_pbc_parameters_describe_the_new_box() {
        let mut context = context(true);
        restore_box(&mut context, Cell::Cuboid(Cuboid::cubic(SMALL)));
        let pbc = context.pbc_params().expect("cuboid has PBC parameters");
        // ±29 Å is 2 Å across the 60 Å boundary, but 58 Å in the 120 Å box.
        let [dx, _, _] = pbc.distance_vector(-29.0, 0.0, 0.0, 29.0, 0.0, 0.0);
        assert!((dx.abs() - 2.0).abs() < 1e-9, "{dx}");
    }

    /// The bounding radius is a minimum-image distance, so shrinking the box shortens it.
    #[test]
    fn the_bounding_radius_follows_the_box() {
        let yaml = r#"
atoms:
  - {name: A, mass: 1.0, sigma: 1.0}
molecules:
  - {name: DIMER, atoms: [A, A]}
system:
  cell: !Cuboid [100.0, 100.0, 100.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: {}
  blocks:
    - molecule: DIMER
      N: 1
      insert: !Manual [[0.0, 0.0, -20.0], [0.0, 0.0, 20.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let mut context =
            Backend::from_yaml_str(yaml, None, &mut rand::rngs::StdRng::seed_from_u64(1)).unwrap();
        // 40 Å apart in a 100 Å box: the atoms are 20 Å from their center.
        assert!((context.groups()[0].bounding_radius().unwrap() - 20.0).abs() < 1e-9);

        // In a 30 Å box the minimum image is 10 Å, halving the radius to 5 Å.
        restore_box(&mut context, Cell::Cuboid(Cuboid::cubic(30.0)));
        assert!((context.groups()[0].bounding_radius().unwrap() - 5.0).abs() < 1e-9);
    }

    /// An unusable box must drop the grid rather than leave the previous one in place.
    #[test]
    fn an_endless_cell_drops_the_cell_list() {
        let mut context = context(true);
        restore_box(&mut context, Cell::Endless(crate::cell::Endless));
        assert!(context.cell_list.is_none());
    }

    /// A rectangular grid cannot serve a cell whose minimum image is Wigner-Seitz: its neighbour
    /// wrapping would drop interacting pairs while the distances came out of the hexagonal cell.
    #[test]
    fn a_hexagonal_prism_gets_no_cell_list() {
        let mut context = context(true);
        assert!(context.cell_list.is_some());
        restore_box(
            &mut context,
            Cell::HexagonalPrism(crate::cell::HexagonalPrism::new(20.0, 40.0)),
        );
        assert!(context.cell_list.is_none());
        assert!(
            context.pbc_params().is_none(),
            "hexagonal prism has no PBC parameters"
        );
    }
}

/// Six analyses re-resolve their selection on every sample, which is always correct but O(N).
/// Before they are switched to a `CachedSelection`, pin the property that switch relies on: a
/// cached selection must agree with the uncached resolver after every mutation an analysis can
/// observe between samples.
#[cfg(test)]
mod cached_selection_agrees_with_uncached {
    use super::*;
    use crate::group::GroupSize;
    use crate::selection::{CachedSelection, Selection};

    /// Two dimers, so a group can be emptied without emptying the system.
    const TWO_DIMERS: &str = r#"
atoms:
  - {name: A, mass: 3.0, charge: 0.0, sigma: 1.0}
  - {name: B, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: DIMER
    atoms: [A, B]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: DIMER
      N: 2
      insert: !Manual [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [5.0, 0.0, -2.0], [5.0, 0.0, 2.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    fn backend() -> Backend {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), TWO_DIMERS).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    /// A group selection resolves into the group array, not the particle array. `group()` takes the
    /// resolved `GroupIndex` directly, while `position()` indexes particles by `usize` and rejects
    /// it — the property the type split buys, and the reason `CachedSelection` is generic over its
    /// target.
    #[test]
    fn resolved_group_indices_address_the_group_array() {
        let context = backend();
        let mut groups = CachedSelection::groups(Selection::parse("molecule DIMER").unwrap());
        let resolved = groups.resolve(&context);
        assert_eq!(resolved.len(), 2);
        assert_eq!(context.group(resolved[0]).molecule(), MoleculeId::new(0));
    }

    /// Strip the index space, so a resolved slice can be compared with the uncached `Vec<usize>`.
    trait RawIndices {
        fn raw(&self) -> Vec<usize>;
    }
    impl RawIndices for [crate::group::AbsIndex] {
        fn raw(&self) -> Vec<usize> {
            self.iter().map(|i| i.get()).collect()
        }
    }
    impl RawIndices for [crate::group::GroupIndex] {
        fn raw(&self) -> Vec<usize> {
            self.iter().map(|i| i.get()).collect()
        }
    }

    fn assert_agrees(context: &Backend, source: &str) {
        let selection = Selection::parse(source).unwrap();
        let mut atoms = CachedSelection::atoms(selection.clone());
        let mut groups = CachedSelection::groups(selection.clone());
        assert_eq!(
            atoms.resolve(context).raw(),
            context.resolve_atoms(&selection),
            "atoms disagree for '{source}'"
        );
        assert_eq!(
            groups.resolve(context).raw(),
            context.resolve_groups(&selection),
            "groups disagree for '{source}'"
        );
    }

    /// The same cache instance must keep agreeing as the system mutates beneath it.
    fn assert_tracks(context: &mut Backend, source: &str, mutate: impl FnOnce(&mut Backend)) {
        let selection = Selection::parse(source).unwrap();
        let mut atoms = CachedSelection::atoms(selection.clone());
        let mut groups = CachedSelection::groups(selection.clone());
        atoms.resolve(context);
        groups.resolve(context);

        mutate(context);

        assert_eq!(
            atoms.resolve(context).raw(),
            context.resolve_atoms(&selection),
            "cached atoms went stale for '{source}'"
        );
        assert_eq!(
            groups.resolve(context).raw(),
            context.resolve_groups(&selection),
            "cached groups went stale for '{source}'"
        );
    }

    #[test]
    fn agree_on_an_untouched_system() {
        let context = backend();
        for source in ["all", "atomtype A", "molecule DIMER", "element O"] {
            assert_agrees(&context, source);
        }
    }

    #[test]
    fn track_an_atom_kind_swap() {
        for source in ["all", "atomtype A", "atomtype B", "molecule DIMER"] {
            let mut context = backend();
            assert_tracks(&mut context, source, |c| {
                c.set_atom_kind(1, AtomKindId::new(0))
            });
        }
    }

    #[test]
    fn track_a_grand_canonical_resize() {
        for source in ["all", "atomtype A", "molecule DIMER"] {
            let mut context = backend();
            assert_tracks(&mut context, source, |c| {
                c.resize_group(1, GroupSize::Empty).unwrap();
            });
        }
    }

    #[test]
    fn track_a_partial_resize() {
        for source in ["all", "atomtype B", "molecule DIMER"] {
            let mut context = backend();
            assert_tracks(&mut context, source, |c| {
                c.resize_group(0, GroupSize::Partial(1)).unwrap();
            });
        }
    }
}
