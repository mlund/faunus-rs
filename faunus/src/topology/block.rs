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

use derive_getters::Getters;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::{group::GroupSize, Context, Particle, Point};

use super::molecule::MoleculeKind;

/// Describes the activation status of a MoleculeBlock.
/// Partial(n) means that only the first 'n' molecules of the block are active.
/// All means that all molecules of the block are active.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum BlockActivationStatus {
    Partial(usize),
    #[default]
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsertionPolicy {
    /// Read molecule block from a file.
    FromFile(String),
    /// Place the atoms of each molecule of the block to random positions in the simulation cell.
    RandomAtomPos {
        #[serde(default = "default_directions")]
        directions: [bool; 3],
    },
    /// Read the structure of the molecule. Then place all molecules of the block
    /// to random positions in the simulation cell.
    RandomCOM {
        filename: String,
        #[serde(default)]
        rotate: bool,
        #[serde(default = "default_directions")]
        directions: [bool; 3],
    },
    /// Define the positions of the atoms of all molecules manually, directly in the topology file.
    Manual(Vec<Point>),
}

fn default_directions() -> [bool; 3] {
    [true, true, true]
}

/// A block of molecules of the same molecule kind.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Getters, Validate)]
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
    /// Create groups from a MoleculeBlock.
    pub(crate) fn to_groups(
        &self,
        context: &mut impl Context,
        molecules: &[MoleculeKind],
    ) -> anyhow::Result<()> {
        let molecule = &molecules[self.molecule_index];
        let mut particle_counter = context.n_particles();

        for i in 0..self.number {
            let particles: Vec<Particle> = molecule
                .atom_indices()
                .iter()
                .map(|a| {
                    let particle = Particle::new(*a, particle_counter, Point::default());
                    particle_counter += 1;
                    particle
                })
                .collect();

            let group = context.add_group(*molecule.id(), &particles)?;

            match self.active {
                BlockActivationStatus::Partial(x) if i >= x => {
                    group.resize(GroupSize::Empty).unwrap()
                }
                _ => (),
            }
        }

        // TODO: set coordinates of the particles

        Ok(())
    }

    /// Set index of the molecule of the block.
    pub(super) fn set_molecule_index(&mut self, index: usize) {
        self.molecule_index = index;
    }

    /// Finalize MoleculeBlock parsing.
    pub(super) fn finalize(&mut self) -> Result<(), ValidationError> {
        // check that the number of active particles is not higher than the total number of particles
        if let BlockActivationStatus::Partial(active_mol) = self.active {
            if active_mol > self.number {
                // TODO! this might be a warning instead
                return Err(ValidationError::new(
                    "the specified number of active molecules in a block is higher than the total number of molecules"
                ));
            } else if active_mol == self.number {
                self.active = BlockActivationStatus::All;
            }
        }

        Ok(())
    }
}
