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

use crate::{
    energy::EnergyTerm,
    group::{GroupCollection, GroupSize},
    Change, Group, Particle, SyncFrom,
};

pub mod nonbonded;

/// Default platform running on the CPU.
///
/// Particles are stored in
/// a single vector, and groups are stored in a separate vector. This mostly
/// follows the same layout as the original C++ Faunus code (version 2 and lower).
#[derive(Debug)]
pub struct ReferencePlatform {
    particles: Vec<Particle>,
    groups: Vec<Group>,
    cell: crate::cell::Cuboid,
    _energies: Vec<Box<dyn EnergyTerm>>,
}

impl crate::Context for ReferencePlatform {
    type Cell = crate::cell::Cuboid;
    fn cell(&self) -> &Self::Cell {
        &self.cell
    }
    fn cell_mut(&mut self) -> &mut Self::Cell {
        &mut self.cell
    }
}

impl Clone for ReferencePlatform {
    fn clone(&self) -> Self {
        todo!("clone boxed energies")
    }
}

impl SyncFrom for ReferencePlatform {
    fn sync_from(&mut self, other: &Self, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything => {
                *self = other.clone();
            }
            _ => todo!(),
        }
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

    fn set_particles<'a>(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        source: impl Iterator<Item = &'a Particle> + Clone,
    ) -> anyhow::Result<()> {
        for (src, i) in source.zip(indices.into_iter()) {
            self.particles[i] = src.clone();
        }
        Ok(())
    }

    fn add_group(&mut self, id: Option<usize>, particles: &[Particle]) -> anyhow::Result<&Group> {
        if particles.is_empty() {
            anyhow::bail!("Cannot create empty group");
        }
        let range = self.particles.len()..self.particles.len() + particles.len();
        self.particles.extend_from_slice(particles);
        self.groups.push(Group::new(self.groups.len(), id, range));
        Ok(self.groups.last().unwrap())
    }

    fn resize_group(&mut self, group_index: usize, status: GroupSize) -> anyhow::Result<()> {
        self.groups[group_index].resize(status)
    }
}

/// Group-wise collection of particles
///
/// Particles are grouped into groups, which are defined by a slice of particles.
/// Each group could be a rigid body, a molecule, etc.
/// The idea is to access the particle in a group-wise fashion, e.g. to update
/// the center of mass of a group, or to rotate a group of particles.
impl ReferencePlatform {
    pub fn new(cell: crate::cell::Cuboid) -> Self {
        Self {
            particles: Vec::new(),
            groups: Vec::new(),
            cell,
            _energies: Vec::new(),
        }
    }

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
}
