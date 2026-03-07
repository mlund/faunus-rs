//! SoA (Structure-of-Arrays) platform for SIMD-friendly memory layout.
//!
//! Positions are stored as separate x, y, z vectors instead of AoS `Vec<Particle>`.
//! This enables future SIMD batch evaluation without sync overhead, since the
//! SoA layout is the source of truth — no shadow copy needed.

use crate::{
    cell::{BoundaryConditions, Cell},
    change::Change,
    energy::{builder::HamiltonianBuilder, Hamiltonian, PbcParams},
    group::{GroupCollection, GroupLists, GroupSize},
    topology::Topology,
    Context, Group, Particle, ParticleSystem, Point, PointParticle, UnitQuaternion, WithCell,
    WithHamiltonian, WithTopology,
};

use interatomic::coulomb::{DebyeLength, Temperature};
use rand::rngs::ThreadRng;
use serde::Serialize;

use std::{cell::RefCell, path::Path, sync::Arc};

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

/// Platform with SoA position layout for SIMD-friendly access.
#[derive(Clone, Debug, Serialize)]
pub struct SoaPlatform {
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

impl SoaPlatform {
    /// Build from a YAML input file, same interface as AosPlatform.
    pub fn new(
        yaml_file: impl AsRef<Path>,
        structure_file: Option<&Path>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self> {
        let medium = Some(super::get_medium(&yaml_file)?);
        let topology = Topology::from_file(&yaml_file)?;
        let hamiltonian_builder = HamiltonianBuilder::from_file(&yaml_file)?;
        hamiltonian_builder.validate(topology.atomkinds())?;
        let cell = Cell::from_file(&yaml_file)?;
        let hamiltonian = Hamiltonian::new(&hamiltonian_builder, &topology, medium.clone())?;

        let group_lists = GroupLists::new(topology.moleculekinds().len());
        let pbc_params = PbcParams::try_from_cell(&cell);
        let mut platform = Self {
            topology: Arc::new(topology.clone()),
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            atom_kinds: Vec::new(),
            groups: Vec::new(),
            group_lists,
            cell,
            pbc_params,
            hamiltonian: RefCell::new(hamiltonian),
            backup: None,
            cell_list: None,
        };

        platform.update(&Change::Everything)?;
        Arc::new(topology).insert_groups(&mut platform, structure_file, rng)?;

        // Build constrain and custom external terms
        if let Some(constrain_builders) = &hamiltonian_builder.constrain {
            for builder in constrain_builders {
                let constrain = builder.build(&platform)?;
                platform.hamiltonian_mut().push(constrain.into());
            }
        }
        if let Some(ext_builders) = &hamiltonian_builder.customexternal {
            for builder in ext_builders {
                let ext = builder.build()?;
                platform.hamiltonian_mut().push(ext.into());
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
            let pm = pm_builder.build(&platform, thermal_energy)?;
            platform.hamiltonian_mut().push(pm.into());
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
            let ewald =
                crate::energy::EwaldReciprocalEnergy::new(ewald_builder, &platform, medium)?;
            if ewald.alpha() != initial_alpha {
                platform.hamiltonian_mut().rebuild_nonbonded(
                    &hamiltonian_builder,
                    platform.topology_ref(),
                    Some(medium.clone()),
                    ewald.real_space_scheme(),
                )?;
            }
            platform.hamiltonian_mut().push(ewald.into());
        }
        platform.update(&Change::Everything)?;

        // Build cell list if configured and cell is orthorhombic
        if let Some(spline_opts) = &hamiltonian_builder.spline {
            if spline_opts.cell_list {
                platform.build_cell_list(spline_opts.cutoff);
            }
        }

        Ok(platform)
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
        log::info!(
            "Built cell list with cutoff={cutoff:.1} Å for {} active particles",
            self.num_active_particles()
        );
        self.cell_list = Some(cl);
    }
}

impl crate::WithCell for SoaPlatform {
    type SimCell = crate::cell::Cell;
    #[inline(always)]
    fn cell(&self) -> &Self::SimCell {
        &self.cell
    }
    /// Returns mutable cell reference and invalidates cached `pbc_params`.
    fn cell_mut(&mut self) -> &mut Self::SimCell {
        self.pbc_params = None;
        &mut self.cell
    }
}

super::impl_platform_topology_hamiltonian!(SoaPlatform);

impl GroupCollection for SoaPlatform {
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
        }
    }

    fn update_mass_center(&mut self, group_index: usize) {
        let group = &self.groups[group_index];
        let indices = group
            .select(&crate::group::ParticleSelection::Active, self)
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
            anyhow::bail!("Cannot create empty group on SoaPlatform");
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
        self.groups[group_index].resize(status)?;
        self.group_lists.update_group(&self.groups[group_index]);
        Ok(())
    }
}

impl ParticleSystem for SoaPlatform {
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

    fn positions_soa(&self) -> Option<(&[f64], &[f64], &[f64])> {
        Some((&self.x, &self.y, &self.z))
    }

    fn atom_kinds_u32(&self) -> Option<&[u32]> {
        Some(&self.atom_kinds)
    }

    fn pbc_params(&self) -> Option<crate::energy::PbcParams> {
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

impl Context for SoaPlatform {
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
    use crate::platform::aos::AosPlatform;

    /// Verify SoA energy path matches scalar path for identical positions.
    ///
    /// Builds both platforms from the same input, copies positions from
    /// AosPlatform into SoaPlatform, then compares energies.
    #[test]
    fn soa_energy_matches_scalar() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let ref_ctx = AosPlatform::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        let mut simd_ctx = SoaPlatform::new(&yaml, None, &mut rand::thread_rng()).unwrap();

        assert_eq!(ref_ctx.num_particles(), simd_ctx.num_particles());
        assert_eq!(ref_ctx.groups().len(), simd_ctx.groups().len());

        // Copy positions from reference into SoA arrays for exact comparison
        for i in 0..ref_ctx.num_particles() {
            let p = &ref_ctx.particles()[i];
            simd_ctx.x[i] = p.pos().x;
            simd_ctx.y[i] = p.pos().y;
            simd_ctx.z[i] = p.pos().z;
            simd_ctx.atom_kinds[i] = p.atom_id() as u32;
        }

        let e_ref = ref_ctx
            .hamiltonian()
            .energy(&ref_ctx, &crate::Change::Everything);
        let e_simd = simd_ctx
            .hamiltonian()
            .energy(&simd_ctx, &crate::Change::Everything);

        assert!(e_ref.is_finite(), "Reference energy should be finite");
        let rel_err = ((e_ref - e_simd) / e_ref).abs();
        assert!(
            rel_err < 1e-12,
            "SoA path should match scalar: ref={}, soa={}, rel_err={:.2e}",
            e_ref,
            e_simd,
            rel_err
        );

        // Verify allocation-free mass_center matches auxiliary::mass_center_pbc
        for group in simd_ctx.groups() {
            if group.is_empty() {
                continue;
            }
            let indices: Vec<usize> = group.iter_active().collect();
            let com_trait = simd_ctx.mass_center(&indices);
            let positions: Vec<Point> = indices.iter().map(|&i| simd_ctx.position(i)).collect();
            let topology = simd_ctx.topology();
            let masses: Vec<f64> = indices
                .iter()
                .map(|&i| topology.atomkinds()[simd_ctx.get_atomkind(i)].mass())
                .collect();
            let com_aux =
                crate::auxiliary::mass_center_pbc(&positions, &masses, simd_ctx.cell(), None);
            let err = (com_trait - com_aux).norm();
            assert!(
                err < 1e-10,
                "mass_center mismatch for group {}: trait={com_trait:?}, aux={com_aux:?}, err={err:.2e}",
                group.index()
            );
        }

        // Also test SingleGroup RigidBody change
        let change = crate::Change::SingleGroup(0, crate::GroupChange::RigidBody);
        let e_ref = ref_ctx.hamiltonian().energy(&ref_ctx, &change);
        let e_simd = simd_ctx.hamiltonian().energy(&simd_ctx, &change);
        let rel_err = ((e_ref - e_simd) / e_ref).abs();
        assert!(
            rel_err < 1e-12,
            "SoA RigidBody should match scalar: ref={}, soa={}, rel_err={:.2e}",
            e_ref,
            e_simd,
            rel_err
        );
    }

    /// Verify set_positions updates coordinates without changing atom kinds.
    #[test]
    fn set_positions_preserves_atom_kinds() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let mut ctx = SoaPlatform::new(&yaml, None, &mut rand::thread_rng()).unwrap();

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

    /// Verify cell-list-accelerated PartialUpdate energy matches brute-force group iteration.
    #[test]
    fn cell_list_partial_update_matches_brute_force() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");

        let ref_ctx = AosPlatform::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        let mut soa_ctx = SoaPlatform::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        assert!(
            soa_ctx.cell_list.is_some(),
            "Cell list should be built for splined input"
        );

        // Copy positions for exact comparison
        for i in 0..ref_ctx.num_particles() {
            let p = &ref_ctx.particles()[i];
            soa_ctx.x[i] = p.pos().x;
            soa_ctx.y[i] = p.pos().y;
            soa_ctx.z[i] = p.pos().z;
            soa_ctx.atom_kinds[i] = p.atom_id() as u32;
        }
        // Rebuild cell list with copied positions
        let cutoff = soa_ctx.cell_list.as_ref().unwrap().cutoff();
        soa_ctx.build_cell_list(cutoff);

        // Test PartialUpdate for several single-atom groups (ion-like moves)
        for gi in 0..5.min(ref_ctx.groups().len()) {
            let group = &ref_ctx.groups()[gi];
            if group.is_empty() {
                continue;
            }
            let change = crate::Change::SingleGroup(gi, crate::GroupChange::PartialUpdate(vec![0]));

            let e_ref = ref_ctx.hamiltonian().energy(&ref_ctx, &change);
            let e_soa = soa_ctx.hamiltonian().energy(&soa_ctx, &change);

            if e_ref.abs() < 1e-10 {
                assert!(
                    e_soa.abs() < 1e-6,
                    "Group {gi}: expected ~0, got soa={e_soa}"
                );
            } else {
                let rel_err = ((e_ref - e_soa) / e_ref).abs();
                assert!(
                    rel_err < 1e-3,
                    "Group {gi}: ref={e_ref}, soa={e_soa}, rel_err={rel_err:.2e}"
                );
            }
        }

        // Test RigidBody via cell list (protein-like whole-group move)
        for gi in 0..5.min(ref_ctx.groups().len()) {
            if ref_ctx.groups()[gi].is_empty() {
                continue;
            }
            let change = crate::Change::SingleGroup(gi, crate::GroupChange::RigidBody);

            let e_ref = ref_ctx.hamiltonian().energy(&ref_ctx, &change);
            let e_soa = soa_ctx.hamiltonian().energy(&soa_ctx, &change);

            if e_ref.abs() < 1e-10 {
                assert!(
                    e_soa.abs() < 1e-6,
                    "RigidBody group {gi}: expected ~0, got soa={e_soa}"
                );
            } else {
                let rel_err = ((e_ref - e_soa) / e_ref).abs();
                assert!(
                    rel_err < 1e-3,
                    "RigidBody group {gi}: ref={e_ref}, soa={e_soa}, rel_err={rel_err:.2e}"
                );
            }
        }
    }
}
