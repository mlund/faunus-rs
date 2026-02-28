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

//! # Reference platform for CPU-based simulations

use rand::rngs::ThreadRng;

use crate::{
    cell::{BoundaryConditions, Cell},
    change::Change,
    energy::{builder::HamiltonianBuilder, Hamiltonian},
    group::{GroupCollection, GroupLists, GroupSize},
    topology::Topology,
    Context, Group, Particle, ParticleSystem, Point, PointParticle, WithCell, WithHamiltonian,
    WithTopology,
};

use serde::Serialize;

use std::{
    cell::{Ref, RefCell, RefMut},
    path::Path,
    rc::Rc,
};

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

/// Default platform running on the CPU.
///
/// Particles are stored in
/// a single vector, and groups are stored in a separate vector. This mostly
/// follows the same layout as the original C++ Faunus code (version 2 and lower).
#[derive(Clone, Debug, Serialize)]
pub struct ReferencePlatform {
    topology: Rc<Topology>,
    particles: Vec<Particle>,
    #[serde(skip)]
    groups: Vec<Group>,
    #[serde(skip)]
    group_lists: GroupLists,
    cell: Cell,
    #[serde(skip)]
    hamiltonian: RefCell<Hamiltonian>,
}

impl ReferencePlatform {
    /// Create a new simulation system on a reference platform from
    /// faunus configuration file and optional structure file.
    #[must_use = "this returns a Result that should be handled"]
    pub fn new(
        yaml_file: impl AsRef<Path>,
        structure_file: Option<&Path>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self> {
        let medium = Some(get_medium(&yaml_file)?);
        let topology = Topology::from_file(&yaml_file)?;
        let hamiltonian_builder = HamiltonianBuilder::from_file(&yaml_file)?;
        // validate hamiltonian builder
        hamiltonian_builder.validate(topology.atomkinds())?;

        let cell = Cell::from_file(&yaml_file)?;

        let hamiltonian = Hamiltonian::new(&hamiltonian_builder, &topology, medium)?;
        Self::from_raw_parts(
            Rc::new(topology),
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
            context.update(&Change::Everything)?;
            Ok(context)
        })
    }

    pub(crate) fn from_raw_parts(
        topology: Rc<Topology>,
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
        };

        context.update(&Change::Everything)?;

        topology.insert_groups(&mut context, structure, rng)?;

        Ok(context)
    }
}

impl WithCell for ReferencePlatform {
    type SimCell = Cell;
    fn cell(&self) -> &Self::SimCell {
        &self.cell
    }
    fn cell_mut(&mut self) -> &mut Self::SimCell {
        &mut self.cell
    }
}

impl WithTopology for ReferencePlatform {
    fn topology(&self) -> Rc<Topology> {
        self.topology.clone()
    }

    fn topology_ref(&self) -> &Rc<Topology> {
        &self.topology
    }
}

impl WithHamiltonian for ReferencePlatform {
    fn hamiltonian(&self) -> Ref<'_, Hamiltonian> {
        self.hamiltonian.borrow()
    }

    fn hamiltonian_mut(&self) -> RefMut<'_, Hamiltonian> {
        self.hamiltonian.borrow_mut()
    }
}

impl Context for ReferencePlatform {
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()> {
        self.cell = other.cell.clone();
        self.hamiltonian_mut()
            .sync_from(&other.hamiltonian(), change)?;
        self.sync_from_groupcollection(change, other)?;
        Ok(())
    }
}

impl GroupCollection for ReferencePlatform {
    fn groups(&self) -> &[Group] {
        self.groups.as_ref()
    }

    fn particle(&self, index: usize) -> Particle {
        self.particles[index].clone()
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

    fn update_mass_center(&mut self, group_index: usize) {
        let group = &self.groups[group_index];
        let indices = group
            .select(&crate::group::ParticleSelection::Active, self)
            .unwrap();
        if self.topology().moleculekinds()[group.molecule()].has_com() {
            let com = self.mass_center(&indices);
            self.groups[group_index].set_mass_center(com);
        }
    }

    fn add_group(&mut self, molecule: usize, particles: &[Particle]) -> anyhow::Result<&mut Group> {
        if particles.is_empty() {
            let msg = "No particles defined for reference platform; cannot create empty group";
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

impl ParticleSystem for ReferencePlatform {
    /// Get distance between two particles.
    ///
    /// Faster implementation for Reference Platform which does not involve particle copying.
    #[inline(always)]
    fn get_distance(&self, i: usize, j: usize) -> Point {
        self.cell()
            .distance(self.particles()[i].pos(), self.particles()[j].pos())
    }

    /// Get index of the atom kind of the particle.
    ///
    /// Faster implementation for Reference Platform which does not involve particle copying.
    #[inline(always)]
    fn get_atomkind(&self, i: usize) -> usize {
        self.particles()[i].atom_id
    }

    /// Get angle between particles `i-j-k`.
    ///
    /// Faster implementation for Reference Platform which does not involve particle copying.
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
    /// Faster implementation for Reference Platform which does not involve particle copying.
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

        // 1. Unwrap PBC for molecular groups (>1 atom) using mass center as reference.
        for g in 0..num_groups {
            if let Some(&com) = self.groups[g].mass_center() {
                if self.groups[g].len() > 1 {
                    for i in self.groups[g].iter_active() {
                        let d = self.cell.distance(&self.particles[i].pos, &com);
                        self.particles[i].pos = com + d;
                    }
                }
            }
        }

        // 2. Scale all active particle positions (using old cell geometry)
        for g in 0..num_groups {
            for i in self.groups[g].iter_active() {
                self.cell
                    .scale_position(new_volume, &mut self.particles[i].pos, policy)?;
            }
        }

        // 3. Resize the cell
        self.cell.scale_volume(new_volume, policy)?;

        // 4. Apply PBC with new cell geometry and recompute mass centers
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
        let shift = shift.unwrap_or_else(Point::zeros);
        indices.iter().for_each(|&i| {
            let position = self.particles[i].pos_mut();
            *position += shift;
            self.cell.boundary(position);
            *position = quaternion.transform_vector(position) - shift;
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
impl ReferencePlatform {
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
