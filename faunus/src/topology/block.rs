// Copyright 2023-2024 Mikael Lund
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

use std::{cmp::Ordering, path::Path};

use derive_getters::Getters;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::{cell::SimulationCell, group::GroupSize, Context, Particle, Point};

use super::{molecule::MoleculeKind, InputPath};

/// Describes the activation status of a MoleculeBlock.
/// Partial(n) means that only the first 'n' molecules of the block are active.
/// All means that all molecules of the block are active.
#[derive(Debug, Clone, PartialEq, Copy, Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum BlockActivationStatus {
    Partial(usize),
    #[default]
    All,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InsertionPolicy {
    /// Read molecule block from a file.
    FromFile(InputPath),
    /// Place the atoms of each molecule of the block to random positions in the simulation cell.
    RandomAtomPos {
        #[serde(default = "default_directions")]
        directions: [bool; 3],
    },
    /// Read the structure of the molecule. Then place all molecules of the block
    /// to random positions in the simulation cell.
    RandomCOM {
        filename: InputPath,
        #[serde(default)]
        rotate: bool,
        #[serde(default = "default_directions")]
        directions: [bool; 3],
    },
    /// Define the positions of the atoms of all molecules manually, directly in the topology file.
    Manual(Vec<Point>),
}

impl InsertionPolicy {
    /// Obtain or generate positions of particles of a molecule block using the target InsertionPolicy.
    fn get_positions(
        &self,
        molecule_kind: &MoleculeKind,
        number: usize,
        cell: &impl SimulationCell,
    ) -> anyhow::Result<Vec<Point>> {
        match self {
            Self::FromFile(filename) => {
                let mut trajectory = chemfiles::Trajectory::open(filename.path().unwrap(), 'r')?;
                let mut frame = chemfiles::Frame::new();
                trajectory.read(&mut frame)?;
                Ok(frame
                    .positions()
                    .iter()
                    .map(|pos| (*pos).into())
                    .collect::<Vec<Point>>())
            }
            Self::RandomAtomPos { directions } => todo!("Implement RandomAtomPos insertion policy"),
            Self::RandomCOM {
                filename,
                rotate,
                directions,
            } => todo!("Implement RandomCOM insertion policy"),
            // these should already be validated to be compatible with the topology
            Self::Manual(positions) => Ok(positions.to_owned()),
        }
    }

    /// Finalize path to the provided structure file (if it is provided) treating it either as an absolute path
    /// (if it is absolute) or as a path relative to `filename`.
    pub(super) fn finalize_path(&mut self, filename: impl AsRef<Path>) {
        match self {
            Self::FromFile(x) => x.finalize(filename),
            Self::RandomCOM { filename: x, .. } => x.finalize(filename),
            Self::RandomAtomPos { .. } | Self::Manual(_) => (),
        }
    }
}

fn default_directions() -> [bool; 3] {
    [true, true, true]
}

/// A block of molecules of the same molecule kind.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
pub struct MoleculeBlock {
    /// Name of the molecule kind of molecules in this block.
    molecule: String,
    /// Index of the molecule kind.
    /// Only defined for MoleculeBlock in a specific Topology.
    #[serde(skip)]
    molecule_index: usize,
    /// Number of molecules in this block.
    #[serde(rename = "N")]
    number: usize,
    /// Number of active molecules in this block.
    #[serde(default)]
    active: BlockActivationStatus,
    /// Specifies how the structure of the molecule block should be obtained.
    /// None => structure should be read from a separately provided structure file
    insert: Option<InsertionPolicy>,
}

impl MoleculeBlock {
    pub fn molecule(&self) -> &str {
        &self.molecule
    }

    pub fn molecule_index(&self) -> usize {
        self.molecule_index
    }

    pub fn number(&self) -> usize {
        self.number
    }

    pub fn active(&self) -> BlockActivationStatus {
        self.active
    }

    pub fn insert(&self) -> Option<&InsertionPolicy> {
        self.insert.as_ref()
    }

    /// Create groups from a MoleculeBlock.
    ///
    /// ## Parameters
    /// - `context` - structure into which the groups should be added
    /// - `molecules` - list of all molecule kinds in the system
    /// - `external_positions` - list of particle coordinates to use;
    ///    must match exactly the number of coordinates that are required
    pub(crate) fn to_groups(
        &self,
        context: &mut impl Context,
        molecules: &[MoleculeKind],
        external_positions: &[Point],
    ) -> anyhow::Result<()> {
        let molecule = &molecules[self.molecule_index];
        let mut particle_counter = context.n_particles();

        // create groups and populate them with particles
        for i in 0..self.number {
            // get positions of the particles
            let positions = match &self.insert {
                None => external_positions[(i * molecule.atom_indices().len())
                    ..((i + 1) * molecule.atom_indices().len())]
                    .to_owned(),
                Some(policy) => policy.get_positions(molecule, self.number, context.cell())?,
            };

            // create the particles
            let particles: Vec<Particle> = molecule
                .atom_indices()
                .iter()
                .zip(positions.into_iter())
                .map(|(index, position)| {
                    let particle = Particle::new(*index, particle_counter, position);
                    particle_counter += 1;
                    particle
                })
                .collect();

            let group = context.add_group(molecule.id(), &particles)?;

            // deactivate the groups that should not be active
            match self.active {
                BlockActivationStatus::Partial(x) if i >= x => {
                    group.resize(GroupSize::Empty).unwrap()
                }
                _ => (),
            }
        }

        Ok(())
    }

    /// Get the number of atoms in a block.
    /// Panics if the molecule kind defined in the block does not exist.
    pub(crate) fn n_atoms(&self, molecules: &[MoleculeKind]) -> usize {
        self.number * molecules[self.molecule_index].atom_indices().len()
    }

    /// Set index of the molecule of the block.
    pub(super) fn set_molecule_index(&mut self, index: usize) {
        self.molecule_index = index;
    }

    /// Finalize MoleculeBlock parsing.
    pub(super) fn finalize(&mut self, filename: impl AsRef<Path>) -> Result<(), ValidationError> {
        // finalize the paths to input structure files
        match self.insert.as_mut() {
            None => (),
            Some(x) => x.finalize_path(filename),
        }

        // check that the number of active particles is not higher than the total number of particles
        if let BlockActivationStatus::Partial(active_mol) = self.active {
            match active_mol.cmp(&self.number) {
                Ordering::Greater => return Err(ValidationError::new(
                    "the specified number of active molecules in a block is higher than the total number of molecules"
                )),
                Ordering::Equal => self.active = BlockActivationStatus::All,
                Ordering::Less => (),
            }
        }

        Ok(())
    }
}