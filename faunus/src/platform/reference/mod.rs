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
    energy::{builder::HamiltonianBuilder, Hamiltonian},
    group::{GroupCollection, GroupLists, GroupSize},
    topology::Topology,
    Change, Context, Group, Particle, ParticleSystem, Point, PointParticle, SyncFrom, WithCell,
    WithHamiltonian, WithTopology,
};

use std::{
    cell::{Ref, RefCell, RefMut},
    path::Path,
    rc::Rc,
};

/// Default platform running on the CPU.
///
/// Particles are stored in
/// a single vector, and groups are stored in a separate vector. This mostly
/// follows the same layout as the original C++ Faunus code (version 2 and lower).
#[derive(Clone, Debug)]
pub struct ReferencePlatform {
    topology: Rc<Topology>,
    particles: Vec<Particle>,
    groups: Vec<Group>,
    group_lists: GroupLists,
    cell: Cell,
    hamiltonian: RefCell<Hamiltonian>,
}

impl ReferencePlatform {
    /// Create a new simulation system on a reference platform from
    /// faunus configuration file and optional structure file.
    pub fn new(
        yaml_file: impl AsRef<Path>,
        structure_file: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self> {
        let topology = Topology::from_file(&yaml_file)?;
        let hamiltonian_builder = HamiltonianBuilder::from_file(&yaml_file)?;
        // validate hamiltonian builder
        hamiltonian_builder.validate(topology.atomkinds())?;

        let cell = Cell::from_file(&yaml_file)?;

        let hamiltonian = Hamiltonian::new(&hamiltonian_builder, &topology)?;
        Self::from_raw_parts(
            Rc::new(topology),
            cell,
            RefCell::new(hamiltonian),
            structure_file,
            rng,
        )
    }

    pub(crate) fn from_raw_parts(
        topology: Rc<Topology>,
        cell: Cell,
        hamiltonian: RefCell<Hamiltonian>,
        structure: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Self> {
        if topology.system.is_empty() {
            anyhow::bail!("Topology doesn't contain a system");
        }
        let mut context = ReferencePlatform {
            topology: topology.clone(),
            particles: vec![],
            groups: vec![],
            cell,
            hamiltonian,
            group_lists: GroupLists::new(topology.moleculekinds().len()),
        };

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
    fn hamiltonian(&self) -> Ref<Hamiltonian> {
        self.hamiltonian.borrow()
    }

    fn hamiltonian_mut(&self) -> RefMut<Hamiltonian> {
        self.hamiltonian.borrow_mut()
    }
}

impl Context for ReferencePlatform {}

impl SyncFrom for ReferencePlatform {
    /// Synchronize ReferencePlatform from another ReferencePlatform.
    fn sync_from(&mut self, other: &ReferencePlatform, change: &Change) -> anyhow::Result<()> {
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

    fn add_group(&mut self, molecule: usize, particles: &[Particle]) -> anyhow::Result<&mut Group> {
        if particles.is_empty() {
            anyhow::bail!("Cannot create empty group");
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

        crate::aux::angle_points(p1, p2, p3, self.cell())
    }

    /// Get dihedral between particles `i-j-k-l`.
    /// Dihedral is defined as an angle between planes `ijk` and `jkl`.
    ///
    /// Faster implementation for Reference Platform which does not involve particle copying.
    #[inline(always)]
    fn get_dihedral_angle(&self, indices: &[usize; 4]) -> f64 {
        let [p1, p2, p3, p4] = indices.map(|x| self.particles()[x].pos());
        crate::aux::dihedral_points(p1, p2, p3, p4, self.cell())
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
