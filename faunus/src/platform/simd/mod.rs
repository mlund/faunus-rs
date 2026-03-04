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

use rand::rngs::ThreadRng;
use serde::Serialize;

use std::{cell::RefCell, path::Path, sync::Arc};

/// Backup for undo on MC reject.
#[derive(Clone, Debug)]
struct SoaBackup {
    /// (index, x, y, z, atom_kind) tuples for changed particles
    particles: Vec<(usize, f64, f64, f64, u32)>,
    mass_centers: Vec<(usize, Option<Point>)>,
    quaternions: Vec<(usize, UnitQuaternion)>,
    cell: Option<Cell>,
}

/// Platform with SoA position layout for SIMD-friendly access.
#[derive(Clone, Debug, Serialize)]
pub struct SimdPlatform {
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
}

impl SimdPlatform {
    /// Build from a YAML input file, same interface as ReferencePlatform.
    pub fn new(
        yaml_file: impl AsRef<Path>,
        structure_file: Option<&Path>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self> {
        let medium = Some(super::reference::get_medium(&yaml_file)?);
        let topology = Topology::from_file(&yaml_file)?;
        let hamiltonian_builder = HamiltonianBuilder::from_file(&yaml_file)?;
        hamiltonian_builder.validate(topology.atomkinds())?;
        let cell = Cell::from_file(&yaml_file)?;
        let hamiltonian = Hamiltonian::new(&hamiltonian_builder, &topology, medium)?;

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
        platform.update(&Change::Everything)?;

        Ok(platform)
    }
}

impl crate::WithCell for SimdPlatform {
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

super::impl_platform_topology_hamiltonian!(SimdPlatform);

impl GroupCollection for SimdPlatform {
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

    fn update_mass_center(&mut self, group_index: usize) {
        let group = &self.groups[group_index];
        let indices = group
            .select(&crate::group::ParticleSelection::Active, self)
            .unwrap();
        if !indices.is_empty() && self.topology().moleculekinds()[group.molecule()].has_com() {
            let com = self.mass_center(&indices);
            self.groups[group_index].set_mass_center(com);
        }
    }

    fn add_group(&mut self, molecule: usize, particles: &[Particle]) -> anyhow::Result<&mut Group> {
        if particles.is_empty() {
            anyhow::bail!("Cannot create empty group on SimdPlatform");
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

impl ParticleSystem for SimdPlatform {
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
    }
}

impl Context for SimdPlatform {
    fn save_particle_backup(&mut self, group_index: usize, indices: &[usize]) {
        assert!(self.backup.is_none(), "backup already exists");
        let particles = indices
            .iter()
            .map(|&i| (i, self.x[i], self.y[i], self.z[i], self.atom_kinds[i]))
            .collect();
        let mass_center = self.groups[group_index].mass_center().cloned();
        let quaternion = *self.groups[group_index].quaternion();
        self.backup = Some(SoaBackup {
            particles,
            mass_centers: vec![(group_index, mass_center)],
            quaternions: vec![(group_index, quaternion)],
            cell: None,
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
        let quaternions = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i, *g.quaternion()))
            .collect();
        self.backup = Some(SoaBackup {
            particles,
            mass_centers,
            quaternions,
            cell: Some(self.cell.clone()),
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
        for (group_idx, q) in backup.quaternions {
            self.groups[group_idx].set_quaternion(q);
        }
        if let Some(cell) = backup.cell {
            self.cell = cell;
            // Restored pre-volume-move cell — must match cached PBC params
            self.pbc_params = PbcParams::try_from_cell(&self.cell);
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
    use crate::platform::reference::ReferencePlatform;

    /// Verify SoA energy path matches scalar path for identical positions.
    ///
    /// Builds both platforms from the same input, copies positions from
    /// ReferencePlatform into SimdPlatform, then compares energies.
    #[test]
    fn soa_energy_matches_scalar() {
        let yaml = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/files/gibbs_ensemble/input.yaml");
        let ref_ctx = ReferencePlatform::new(&yaml, None, &mut rand::thread_rng()).unwrap();
        let mut simd_ctx = SimdPlatform::new(&yaml, None, &mut rand::thread_rng()).unwrap();

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
}
