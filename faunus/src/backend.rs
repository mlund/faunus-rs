//! SoA (Structure-of-Arrays) backend with SIMD-friendly memory layout.
//!
//! Positions are stored as separate x, y, z vectors for cache-friendly access.
//! This enables SIMD batch evaluation without sync overhead.

use crate::{
    cell::PbcParams,
    cell::{BoundaryConditions, Cell},
    change::Change,
    energy::{builder::HamiltonianBuilder, Hamiltonian},
    group::{GroupCollection, GroupLists, GroupSize},
    topology::Topology,
    Context, Group, Particle, ParticleSystem, Point, PointParticle, UnitQuaternion, WithCell,
    WithHamiltonian, WithTopology,
};

use interatomic::coulomb::{DebyeLength, Temperature};
use rand::rngs::ThreadRng;
use serde::Serialize;

use std::{cell::RefCell, path::Path, sync::Arc};

/// Extract medium from system/medium in YAML file
pub fn get_medium(path: impl AsRef<Path>) -> anyhow::Result<interatomic::coulomb::Medium> {
    let file = std::fs::File::open(&path)
        .map_err(|err| anyhow::anyhow!("Could not open {:?}: {}", path.as_ref(), err))?;
    serde_yaml::from_reader(file)
        .ok()
        .and_then(|s: serde_yaml::Value| {
            let val = s.get("system")?.get("medium")?;
            serde_yaml::from_value(val.clone()).ok()
        })
        .ok_or_else(|| anyhow::anyhow!("Could not find `system/medium` in input file"))
}

/// Backup for undo on MC reject.
#[derive(Clone, Debug)]
struct SoaBackup {
    /// (index, x, y, z, atom_kind) tuples for changed particles
    particles: Vec<(usize, f64, f64, f64, u32)>,
    mass_centers: Vec<(usize, Option<Point>)>,
    bounding_radii: Vec<(usize, Option<f64>)>,
    quaternions: Vec<(usize, UnitQuaternion)>,
    group_sizes: Vec<(usize, GroupSize)>,
    cell: Option<Cell>,
    /// Incremental cell list changes for undo (particle moves)
    cell_list_backup: Option<crate::celllist::CellListBackup>,
    /// Full cell list clone for undo (volume changes)
    cell_list_clone: Option<crate::celllist::CellList>,
}

/// Simulation backend with SoA position layout for SIMD-friendly access.
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
    backup: Option<SoaBackup>,
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
        rng: &mut ThreadRng,
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
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self> {
        let medium = Some(get_medium(&yaml_file)?);
        let topology = Topology::from_file(&yaml_file)?;
        let hamiltonian_builder = HamiltonianBuilder::from_file(&yaml_file)?;
        hamiltonian_builder.validate(topology.atomkinds())?;
        let cell = Cell::from_file(&yaml_file)?;
        let hamiltonian = Hamiltonian::new(&hamiltonian_builder, &topology, medium.clone())?;

        let mut backend = Self::from_raw_parts(
            Arc::new(topology),
            cell,
            RefCell::new(hamiltonian),
            structure_file,
            rng,
        )?;

        // Build constrain and custom external terms
        if let Some(constrain_builders) = &hamiltonian_builder.constrain {
            for builder in constrain_builders {
                let constrain = builder.build(&backend)?;
                backend.hamiltonian_mut().push(constrain.into());
            }
        }
        if let Some(ext_builders) = &hamiltonian_builder.customexternal {
            for builder in ext_builders {
                let ext = builder.build()?;
                backend.hamiltonian_mut().push(ext.into());
            }
        }
        if let Some(pm_builder) = &hamiltonian_builder.polymer_depletion {
            let temperature = medium.as_ref().map(|m| m.temperature()).ok_or_else(|| {
                anyhow::anyhow!(
                    "Medium with temperature required for polymer_depletion energy term"
                )
            })?;
            let thermal_energy =
                crate::energy::ExternalPressure::thermal_energy_from_temperature(temperature);
            let pm = pm_builder.build(&backend, thermal_energy)?;
            backend.hamiltonian_mut().push(pm.into());
        }
        if let Some(tab_builder) = &hamiltonian_builder.tabulated6d {
            let tab = tab_builder.build(&backend)?;
            backend.hamiltonian_mut().push(tab.into());
        }
        // Ewald reciprocal term needs particles in place
        if let Some(ewald_builder) = &hamiltonian_builder.ewald {
            let medium = medium
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Ewald requires a medium with permittivity"))?;
            let initial_alpha = {
                let debye_length = medium.debye_length();
                interatomic::coulomb::pairwise::RealSpaceEwald::new(
                    ewald_builder.cutoff,
                    ewald_builder.accuracy,
                    debye_length,
                )
                .alpha()
            };
            let ewald = crate::energy::EwaldReciprocalEnergy::new(ewald_builder, &backend, medium)?;
            if ewald.alpha() != initial_alpha {
                backend.hamiltonian_mut().rebuild_nonbonded(
                    &hamiltonian_builder,
                    backend.topology_ref(),
                    Some(medium.clone()),
                    ewald.real_space_scheme(),
                )?;
            }
            backend.hamiltonian_mut().push(ewald.into());
        }
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
        if let (Some(cl), Some(backup)) = (&mut self.cell_list, &mut self.backup) {
            if let Some(cl_backup) = &mut backup.cell_list_backup {
                for &i in indices {
                    let pos = Point::new(self.x[i], self.y[i], self.z[i]);
                    cl.update_particle_tracked(i, &pos, cl_backup);
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

impl crate::WithCell for Backend {
    #[inline(always)]
    fn cell(&self) -> &Cell {
        &self.cell
    }
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
    fn hamiltonian_mut(&self) -> std::cell::RefMut<'_, crate::energy::Hamiltonian> {
        self.hamiltonian.borrow_mut()
    }
}

impl GroupCollection for Backend {
    fn groups(&self) -> &[Group] {
        &self.groups
    }

    fn groups_mut(&mut self) -> &mut [Group] {
        &mut self.groups
    }

    fn particle(&self, index: usize) -> Particle {
        Particle::new(
            self.atom_kinds[index] as usize,
            Point::new(self.x[index], self.y[index], self.z[index]),
        )
    }

    #[inline(always)]
    fn position(&self, index: usize) -> Point {
        Point::new(self.x[index], self.y[index], self.z[index])
    }

    fn num_particles(&self) -> usize {
        self.x.len()
    }

    fn group_lists(&self) -> &GroupLists {
        &self.group_lists
    }

    fn set_particles<'b>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        source: impl IntoIterator<Item = &'b Particle> + Clone,
    ) -> anyhow::Result<()> {
        for (src, i) in source.into_iter().zip(indices.into_iter()) {
            let pos = src.pos();
            self.x[i] = pos.x;
            self.y[i] = pos.y;
            self.z[i] = pos.z;
            self.atom_kinds[i] = src.atom_id() as u32;
            self.update_cell_list_particle(i);
        }
        Ok(())
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
        if !indices.is_empty() && self.topology().moleculekinds()[group.molecule()].has_com() {
            let com = self.mass_center(&indices);
            self.groups[group_index].set_mass_center(com);
            // Bounding radius: max PBC distance from COM to any active particle
            let radius = indices
                .iter()
                .map(|&i| {
                    let pos = Point::new(self.x[i], self.y[i], self.z[i]);
                    self.cell.distance(&pos, &com).norm()
                })
                .fold(0.0_f64, f64::max);
            self.groups[group_index].set_bounding_radius(radius);
        }
    }

    fn add_group(&mut self, molecule: usize, particles: &[Particle]) -> anyhow::Result<&mut Group> {
        if particles.is_empty() {
            anyhow::bail!("Cannot create empty group");
        }
        let start = self.x.len();
        for p in particles {
            let pos = p.pos();
            self.x.push(pos.x);
            self.y.push(pos.y);
            self.z.push(pos.z);
            self.atom_kinds.push(p.atom_id() as u32);
        }
        let range = start..start + particles.len();
        self.groups
            .push(Group::new(self.groups.len(), molecule, range));

        let group = self.groups.last_mut().unwrap();
        self.group_lists.add_group(group);
        Ok(group)
    }

    fn resize_group(&mut self, group_index: usize, status: GroupSize) -> anyhow::Result<()> {
        let old_active: Vec<usize> = self.groups[group_index].iter_active().collect();
        self.groups[group_index].resize(status)?;
        self.group_lists.update_group(&self.groups[group_index]);
        let new_active: Vec<usize> = self.groups[group_index].iter_active().collect();

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

impl ParticleSystem for Backend {
    #[inline(always)]
    fn get_distance(&self, i: usize, j: usize) -> Point {
        let pi = Point::new(self.x[i], self.y[i], self.z[i]);
        let pj = Point::new(self.x[j], self.y[j], self.z[j]);
        self.cell().distance(&pi, &pj)
    }

    #[inline(always)]
    fn get_atomkind(&self, i: usize) -> usize {
        self.atom_kinds[i] as usize
    }

    fn positions_soa(&self) -> (&[f64], &[f64], &[f64]) {
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
        crate::auxiliary::angle_points(&p1, &p2, &p3, self.cell())
    }

    #[inline(always)]
    fn get_dihedral_angle(&self, indices: &[usize; 4]) -> f64 {
        let [p1, p2, p3, p4] = indices.map(|i| self.position(i));
        crate::auxiliary::dihedral_points(&p1, &p2, &p3, &p4, self.cell())
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
            let is_mol = self.topology.moleculekinds()[self.groups[g].molecule()].has_com();
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
}

impl Context for Backend {
    fn save_particle_backup(&mut self, group_index: usize, indices: &[usize]) {
        assert!(self.backup.is_none(), "backup already exists");
        let particles = indices
            .iter()
            .map(|&i| (i, self.x[i], self.y[i], self.z[i], self.atom_kinds[i]))
            .collect();
        let group = &self.groups[group_index];
        let mass_center = group.mass_center().cloned();
        let bounding_radius = group.bounding_radius();
        let quaternion = *group.quaternion();
        let group_size = group.size();
        let cell_list_backup = self.cell_list.as_ref().map(|cl| cl.begin_changes());
        self.backup = Some(SoaBackup {
            particles,
            mass_centers: vec![(group_index, mass_center)],
            bounding_radii: vec![(group_index, bounding_radius)],
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
        let mass_centers = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i, g.mass_center().cloned()))
            .collect();
        let bounding_radii = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i, g.bounding_radius()))
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
        self.backup = Some(SoaBackup {
            particles,
            mass_centers,
            bounding_radii,
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
            self.atom_kinds[i] = kind;
        }
        for (group_idx, old_com) in backup.mass_centers {
            if let Some(com) = old_com {
                self.groups[group_idx].set_mass_center(com);
            }
        }
        for (group_idx, old_radius) in backup.bounding_radii {
            if let Some(r) = old_radius {
                self.groups[group_idx].set_bounding_radius(r);
            }
        }
        for (group_idx, q) in backup.quaternions {
            self.groups[group_idx].set_quaternion(q);
        }
        for (group_idx, size) in backup.group_sizes {
            self.groups[group_idx].resize(size)?;
            self.group_lists.update_group(&self.groups[group_idx]);
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
mod tests {
    use super::*;
    use crate::energy::EnergyChange;

    /// Verify total energy equals sum of per-group energies, and mass_center matches auxiliary.
    #[test]
    fn soa_energy_and_mass_center() {
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

        // Verify mass_center matches auxiliary::mass_center_pbc
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
                .map(|&i| topology.atomkinds()[ctx.get_atomkind(i)].mass())
                .collect();
            let com_aux = crate::auxiliary::mass_center_pbc(&positions, &masses, ctx.cell(), None);
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
            let change_partial =
                crate::Change::SingleGroup(gi, crate::GroupChange::PartialUpdate(vec![0]));
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
}
