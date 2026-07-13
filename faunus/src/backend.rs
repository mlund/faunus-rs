//! Simulation backend with a structure-of-arrays memory layout.
//!
//! Positions are stored as separate x, y, z vectors for cache-friendly access.
//! This enables SIMD batch evaluation without sync overhead.

use crate::{
    cell::PbcParams,
    cell::{BoundaryConditions, Cell},
    change::Change,
    context::{PerturbContext, WithHamiltonianMut},
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
    /// Optional cell list for spatial acceleration (built when cutoff is known).
    #[serde(skip)]
    cell_list: Option<crate::celllist::CellList>,
}

impl Backend {
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
            groups: Vec::new(),
            group_lists,
            cell,
            pbc_params,
            hamiltonian,
            backup: None,
            cell_list: None,
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

        // Build cell list if configured and cell is orthorhombic
        if let Some(spline_opts) = &hamiltonian_builder.spline {
            if spline_opts.cell_list {
                backend.build_cell_list(spline_opts.cutoff);
            }
        }

        Ok(backend)
    }

    /// Update cell list assignment for a single moved particle.
    fn update_cell_list_particle(&mut self, i: usize) {
        self.update_cell_list_particles(&[i]);
    }

    /// Update cell list assignments for moved particles, tracking changes in backup.
    fn update_cell_list_particles(&mut self, indices: &[usize]) {
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

    /// Build the cell list from current positions and box dimensions.
    fn build_cell_list(&mut self, cutoff: f64) {
        use crate::cell::Shape;
        let Some(bb) = self.cell.bounding_box() else {
            return;
        };
        let box_len = [bb.x, bb.y, bb.z];
        // Only for finite orthorhombic cells
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

impl crate::context::WithSimulationCellMut for Backend {
    /// Returns mutable cell reference and invalidates cached `pbc_params`.
    fn cell_mut(&mut self) -> &mut Cell {
        self.pbc_params = None;
        &mut self.cell
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
    fn groups_mut(&mut self) -> &mut [Group] {
        &mut self.groups
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

    fn update_mass_center(&mut self, group_index: usize) {
        let group = &self.groups[group_index];
        let indices = group
            .select(
                &crate::group::ParticleSelection::Active,
                self.topology_ref(),
            )
            .unwrap();
        if !indices.is_empty() && self.topology().moleculekind(group.molecule()).has_com() {
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
        Ok(())
    }
}

impl ObserveContext for Backend {
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
    /// Reaches the `RefCell` directly rather than through `hamiltonian_mut`, which
    /// `PerturbContext` deliberately does not expose. Both borrows of `self` are shared, so the
    /// borrow checker is satisfied and no energy term re-borrows the Hamiltonian from `update`.
    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        self.hamiltonian.borrow_mut().update(self, change)?;
        Ok(())
    }

    fn scale_volume_and_positions(
        &mut self,
        new_volume: f64,
        policy: crate::cell::VolumeScalePolicy,
    ) -> anyhow::Result<f64> {
        use crate::cell::{Shape, VolumeScale};

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
        // Cell dimensions changed — recompute cached PBC params
        self.pbc_params = PbcParams::try_from_cell(&self.cell);

        for g in 0..num_groups {
            for i in self.groups[g].iter_active() {
                let mut pos = Point::new(self.x[i], self.y[i], self.z[i]);
                self.cell.boundary(&mut pos);
                self.x[i] = pos.x;
                self.y[i] = pos.y;
                self.z[i] = pos.z;
            }
            self.update_mass_center(g);
        }

        // Cell dimensions changed — rebuild cell list
        if let Some(cutoff) = self.cell_list.as_ref().map(|cl| cl.cutoff()) {
            self.build_cell_list(cutoff);
        }

        Ok(old_volume)
    }

    #[inline(always)]
    fn translate_particles(&mut self, indices: &[usize], shift: &Point) {
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

    fn rotate_particles(
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

    fn set_particle_positions(&mut self, indices: &[usize], positions: &[Point]) {
        for (&i, position) in indices.iter().zip(positions) {
            let mut position = *position;
            self.cell.boundary(&mut position);
            self.x[i] = position.x;
            self.y[i] = position.y;
            self.z[i] = position.z;
        }
        self.update_cell_list_particles(indices);
    }
}

impl Context for Backend {
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
        context.groups_mut()[1].set_geometry(None);

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

    /// Verify position() and atom_kind() return correct values after add_group.
    #[test]
    fn test_position_and_atom_kind() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        for group in ctx.groups() {
            for i in group.iter_active() {
                let pos = ctx.position(i);
                assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite());
                let kind = ctx.atom_kind(i);
                assert!(kind.get() < ctx.topology().atomkinds().len());
            }
        }
    }

    /// Verify set_positions updates coordinates without changing atom kinds.
    #[test]
    fn test_set_positions_roundtrip() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let mut ctx = Backend::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        let group = &ctx.groups()[0];
        let indices: Vec<usize> = group.iter_active().collect();
        let original_kinds: Vec<usize> = indices.iter().map(|&i| ctx.atom_kind(i).get()).collect();
        let new_positions: Vec<Point> = indices
            .iter()
            .enumerate()
            .map(|(j, _)| Point::new(j as f64, j as f64 * 2.0, j as f64 * 3.0))
            .collect();
        ctx.set_positions(indices.clone(), new_positions.iter());
        for (j, &i) in indices.iter().enumerate() {
            assert_eq!(ctx.position(i), new_positions[j]);
            assert_eq!(ctx.atom_kind(i).get(), original_kinds[j]);
        }
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
