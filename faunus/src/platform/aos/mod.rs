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

//! # AoS (Array-of-Structures) platform for CPU-based simulations

use interatomic::coulomb::DebyeLength;
use rand::rngs::ThreadRng;

use crate::{
    cell::{BoundaryConditions, Cell},
    change::Change,
    energy::{builder::HamiltonianBuilder, Hamiltonian},
    group::{GroupCollection, GroupLists, GroupSize},
    topology::Topology,
    Context, Group, Particle, ParticleSystem, Point, PointParticle, UnitQuaternion, WithCell,
    WithHamiltonian, WithTopology,
};

use serde::Serialize;

use std::{cell::RefCell, path::Path, sync::Arc};

/// Lightweight backup of context state for undo on MC reject.
#[derive(Clone, Debug)]
struct ContextBackup {
    particles: Vec<(usize, Particle)>,
    mass_centers: Vec<(usize, Option<Point>)>,
    quaternions: Vec<(usize, UnitQuaternion)>,
    group_sizes: Vec<(usize, GroupSize)>,
    cell: Option<Cell>,
}

/// AoS (Array-of-Structures) platform running on the CPU.
///
/// Particles are stored in a single `Vec<Particle>`, and groups in a separate vector.
/// This follows the same layout as the original C++ Faunus code (version 2 and lower).
#[derive(Clone, Debug, Serialize)]
pub struct AosPlatform {
    /// Arc (not Rc) so that MarkovChain is Send for Gibbs ensemble scoped threads
    topology: Arc<Topology>,
    particles: Vec<Particle>,
    #[serde(skip)]
    groups: Vec<Group>,
    #[serde(skip)]
    group_lists: GroupLists,
    cell: Cell,
    #[serde(skip)]
    hamiltonian: RefCell<Hamiltonian>,
    #[serde(skip)]
    backup: Option<ContextBackup>,
}

impl AosPlatform {
    /// Create a new simulation system on a reference platform from
    /// faunus configuration file and optional structure file.
    #[must_use = "this returns a Result that should be handled"]
    pub fn new(
        yaml_file: impl AsRef<Path>,
        structure_file: Option<&Path>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self> {
        let medium = Some(super::get_medium(&yaml_file)?);
        let topology = Topology::from_file(&yaml_file)?;
        let hamiltonian_builder = HamiltonianBuilder::from_file(&yaml_file)?;
        // validate hamiltonian builder
        hamiltonian_builder.validate(topology.atomkinds())?;

        let cell = Cell::from_file(&yaml_file)?;

        let hamiltonian = Hamiltonian::new(&hamiltonian_builder, &topology, medium.clone())?;
        Self::from_raw_parts(
            Arc::new(topology),
            cell,
            RefCell::new(hamiltonian),
            structure_file,
            rng,
        )
        .and_then(|mut context| {
            // Build constrain terms after groups exist (selections need groups).
            if let Some(constrain_builders) = &hamiltonian_builder.constrain {
                for builder in constrain_builders {
                    let constrain = builder.build(&context)?;
                    context.hamiltonian_mut().push(constrain.into());
                }
            }
            if let Some(ext_builders) = &hamiltonian_builder.customexternal {
                for builder in ext_builders {
                    let ext = builder.build()?;
                    context.hamiltonian_mut().push(ext.into());
                }
            }
            // Ewald reciprocal term needs particles in place
            if let Some(ewald_builder) = &hamiltonian_builder.ewald {
                let medium = medium
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Ewald requires a medium with permittivity"))?;
                // Record initial α so we can detect if optimization changed it
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
                    crate::energy::EwaldReciprocalEnergy::new(ewald_builder, &context, medium)?;
                if ewald.alpha() != initial_alpha {
                    context.hamiltonian_mut().rebuild_nonbonded(
                        &hamiltonian_builder,
                        context.topology_ref(),
                        Some(medium.clone()),
                        ewald.real_space_scheme(),
                    )?;
                }
                context.hamiltonian_mut().push(ewald.into());
            }
            context.update(&Change::Everything)?;
            Ok(context)
        })
    }

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
        let mut context = Self {
            topology: topology.clone(),
            particles: vec![],
            groups: vec![],
            cell,
            hamiltonian,
            group_lists: GroupLists::new(topology.moleculekinds().len()),
            backup: None,
        };

        context.update(&Change::Everything)?;

        topology.insert_groups(&mut context, structure, rng)?;

        Ok(context)
    }
}

super::impl_platform_shared!(AosPlatform);

impl Context for AosPlatform {
    fn save_particle_backup(&mut self, group_index: usize, indices: &[usize]) {
        assert!(self.backup.is_none(), "backup already exists");
        let particles = indices
            .iter()
            .map(|&i| (i, self.particles[i].clone()))
            .collect();
        let mass_center = self.groups[group_index].mass_center().cloned();
        let quaternion = *self.groups[group_index].quaternion();
        let group_size = self.groups[group_index].size();
        self.backup = Some(ContextBackup {
            particles,
            mass_centers: vec![(group_index, mass_center)],
            quaternions: vec![(group_index, quaternion)],
            group_sizes: vec![(group_index, group_size)],
            cell: None,
        });
    }

    fn save_system_backup(&mut self) {
        assert!(self.backup.is_none(), "backup already exists");
        let particles = self
            .particles
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.clone()))
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
        let group_sizes = self
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i, g.size()))
            .collect();
        self.backup = Some(ContextBackup {
            particles,
            mass_centers,
            quaternions,
            group_sizes,
            cell: Some(self.cell.clone()),
        });
    }

    fn undo(&mut self) -> anyhow::Result<()> {
        let backup = self.backup.take().expect("undo called without backup");
        for (idx, particle) in backup.particles {
            self.particles[idx] = particle;
        }
        for (group_idx, old_com) in backup.mass_centers {
            if let Some(com) = old_com {
                self.groups[group_idx].set_mass_center(com);
            }
        }
        for (group_idx, q) in backup.quaternions {
            self.groups[group_idx].set_quaternion(q);
        }
        for (group_idx, size) in backup.group_sizes {
            self.groups[group_idx].resize(size)?;
            self.group_lists.update_group(&self.groups[group_idx]);
        }
        if let Some(cell) = backup.cell {
            self.cell = cell;
        }
        self.hamiltonian_mut().undo();
        Ok(())
    }

    fn discard_backup(&mut self) {
        self.backup = None;
        self.hamiltonian_mut().discard_backup();
    }
}

impl GroupCollection for AosPlatform {
    fn groups(&self) -> &[Group] {
        self.groups.as_ref()
    }

    fn groups_mut(&mut self) -> &mut [Group] {
        &mut self.groups
    }

    fn particle(&self, index: usize) -> Particle {
        self.particles[index].clone()
    }

    #[inline(always)]
    fn position(&self, index: usize) -> Point {
        self.particles[index].pos
    }

    fn num_particles(&self) -> usize {
        self.particles.len()
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
            self.particles[i] = src.clone();
        }
        Ok(())
    }

    fn set_positions<'a>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        positions: impl IntoIterator<Item = &'a Point>,
    ) {
        for (i, pos) in indices.into_iter().zip(positions) {
            self.particles[i].pos = *pos;
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
            let radius = indices
                .iter()
                .map(|&i| self.cell().distance(&self.position(i), &com).norm())
                .fold(0.0_f64, f64::max);
            self.groups[group_index].set_bounding_radius(radius);
        }
    }

    fn add_group(&mut self, molecule: usize, particles: &[Particle]) -> anyhow::Result<&mut Group> {
        if particles.is_empty() {
            let msg = "No particles defined for AoS platform; cannot create empty group";
            log::error!("{msg}");
            anyhow::bail!(msg);
        }
        let range = self.particles.len()..self.particles.len() + particles.len();
        self.particles.extend_from_slice(particles);
        self.groups
            .push(Group::new(self.groups.len(), molecule, range));

        let group = self.groups.last_mut().unwrap();
        // add group to group lists
        self.group_lists.add_group(group);
        Ok(group)
    }

    fn resize_group(&mut self, group_index: usize, status: GroupSize) -> anyhow::Result<()> {
        self.groups[group_index].resize(status)?;
        // update group in group lists
        self.group_lists.update_group(&self.groups[group_index]);
        Ok(())
    }
}

impl ParticleSystem for AosPlatform {
    /// Get distance between two particles.
    ///
    /// Faster implementation for AoS platform which does not involve particle copying.
    #[inline(always)]
    fn get_distance(&self, i: usize, j: usize) -> Point {
        self.cell()
            .distance(self.particles()[i].pos(), self.particles()[j].pos())
    }

    /// Get index of the atom kind of the particle.
    ///
    /// Faster implementation for AoS platform which does not involve particle copying.
    #[inline(always)]
    fn get_atomkind(&self, i: usize) -> usize {
        self.particles()[i].atom_id
    }

    /// Get angle between particles `i-j-k`.
    ///
    /// Faster implementation for AoS platform which does not involve particle copying.
    #[inline(always)]
    fn get_angle(&self, indices: &[usize; 3]) -> f64 {
        let p1 = self.particles()[indices[0]].pos();
        let p2 = self.particles()[indices[1]].pos();
        let p3 = self.particles()[indices[2]].pos();

        crate::auxiliary::angle_points(p1, p2, p3, self.cell())
    }

    /// Get dihedral between particles `i-j-k-l`.
    /// Dihedral is defined as an angle between planes `ijk` and `jkl`.
    ///
    /// Faster implementation for AoS platform which does not involve particle copying.
    #[inline(always)]
    fn get_dihedral_angle(&self, indices: &[usize; 4]) -> f64 {
        let [p1, p2, p3, p4] = indices.map(|x| self.particles()[x].pos());
        crate::auxiliary::dihedral_points(p1, p2, p3, p4, self.cell())
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

        // Pre-compute per-group molecular flag to avoid holding topology borrow
        let is_molecular: Vec<bool> = (0..num_groups)
            .map(|g| self.topology.moleculekinds()[self.groups[g].molecule()].has_com())
            .collect();

        // Molecular groups (has_com): scale only mass center, translate atoms
        // to preserve intramolecular geometry. Atomic groups: scale each atom.
        for (g, &is_mol) in is_molecular.iter().enumerate() {
            if !is_mol {
                for i in self.groups[g].iter_active() {
                    self.cell
                        .scale_position(new_volume, &mut self.particles[i].pos, policy)?;
                }
                continue;
            }
            let Some(&com) = self.groups[g].mass_center() else {
                continue;
            };
            for i in self.groups[g].iter_active() {
                let d = self.cell.distance(&self.particles[i].pos, &com);
                self.particles[i].pos = com + d;
            }
            let mut scaled_com = com;
            self.cell
                .scale_position(new_volume, &mut scaled_com, policy)?;
            let shift = scaled_com - com;
            for i in self.groups[g].iter_active() {
                self.particles[i].pos += shift;
            }
        }

        // Resize the cell
        self.cell.scale_volume(new_volume, policy)?;

        // Apply PBC with new cell geometry and recompute mass centers
        for g in 0..num_groups {
            for i in self.groups[g].iter_active() {
                self.cell.boundary(&mut self.particles[i].pos);
            }
            self.update_mass_center(g);
        }

        Ok(old_volume)
    }

    /// Shift positions of target particles.
    #[inline(always)]
    fn translate_particles(&mut self, indices: &[usize], shift: &Point) {
        let cell = self.cell.clone();
        indices.iter().for_each(|&i| {
            let position = self.particles_mut()[i].pos_mut();
            *position += shift;
            cell.boundary(position)
        });
    }

    fn rotate_particles(
        &mut self,
        indices: &[usize],
        quaternion: &crate::UnitQuaternion,
        shift: Option<Point>,
    ) {
        let center = -shift.unwrap_or_else(Point::zeros);
        indices.iter().for_each(|&i| {
            let position = self.particles[i].pos_mut();
            // Unwrap via MIC, rotate around center, apply PBC
            let relative = self.cell.distance(position, &center);
            *position = quaternion.transform_vector(&relative) + center;
            self.cell.boundary(position);
        });
    }
}

/// Group-wise collection of particles
///
/// Particles are grouped into groups, which are defined by a slice of particles.
/// Each group could be a rigid body, a molecule, etc.
/// The idea is to access the particle in a group-wise fashion, e.g. to update
/// the center of mass of a group, or to rotate a group of particles.
impl AosPlatform {
    /// Get vector of indices to all other *active* particles in the system, excluding `range`
    fn _other_indices(&self, range: std::ops::Range<usize>) -> Vec<usize> {
        let no_overlap = |r: &std::ops::Range<usize>| {
            usize::max(r.start, range.start) > usize::min(r.end, range.end)
        };
        self.groups
            .iter()
            .map(|g| g.iter_active())
            .filter(no_overlap)
            .flatten()
            .collect()
    }

    /// Get reference to the particles of the system.
    #[inline(always)]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Get mutable reference to the particles of the system.
    #[inline(always)]
    pub fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.particles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::Transform;

    #[test]
    fn backup_undo_restores_quaternion() {
        let mut rng = rand::thread_rng();
        let mut context = AosPlatform::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let group_index = 1;
        assert_eq!(
            *context.groups()[group_index].quaternion(),
            crate::UnitQuaternion::identity()
        );

        // Rotate with backup
        let axis = nalgebra::UnitVector3::new_normalize(crate::Point::new(0.0, 0.0, 1.0));
        let q = crate::UnitQuaternion::from_axis_angle(&axis, 0.8);
        let transform = Transform::Rotate(q);
        transform
            .on_group_with_backup(group_index, &mut context)
            .unwrap();

        // Quaternion should be updated
        assert!(context.groups()[group_index].quaternion().angle_to(&q) < 1e-12);

        // Undo should restore identity
        context.undo().unwrap();
        assert!(
            context.groups()[group_index]
                .quaternion()
                .angle_to(&crate::UnitQuaternion::identity())
                < 1e-12
        );
    }
}
