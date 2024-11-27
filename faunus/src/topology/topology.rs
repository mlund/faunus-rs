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

//! Topology struct

use rand::rngs::ThreadRng;
use std::fmt::Debug;
use std::path::Path;

use anyhow::Ok;
use validator::Validate;

use super::*;
use crate::Context;
use serde::{Deserialize, Serialize};

pub use super::block::{InsertionPolicy, MoleculeBlock};
pub use super::molecule::MoleculeKind;
pub use super::structure::positions_from_structure_file;

/// Trait implemented by any structure resembling a Topology.
pub trait TopologyLike {}

/// Topology of the molecular system.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
pub struct Topology {
    /// Other yaml files that should be included in the topology.
    #[serde(skip_serializing, default)]
    include: Vec<InputPath>,
    /// All possible atom types.
    #[serde(default, rename = "atoms")] // can be defined in an include
    atomkinds: Vec<AtomKind>,
    /// All possible molecule types.
    #[serde(default, rename = "molecules")] // can be defined in an include
    moleculekinds: Vec<MoleculeKind>,
    /// Properties of the system.
    /// Must always be provided.
    #[validate(nested)]
    #[serde(default)]
    pub system: System,
}

impl Topology {
    /// Create a new Topology structure. This function performs no sanity checks.
    #[allow(dead_code)]
    pub(crate) fn new(
        atomkinds: &[AtomKind],
        moleculekinds: &[MoleculeKind],
        intermolecular: IntermolecularBonded,
        blocks: Vec<MoleculeBlock>,
    ) -> Topology {
        Topology {
            include: vec![],
            atomkinds: atomkinds.into(),
            moleculekinds: moleculekinds.into(),
            system: System {
                intermolecular,
                blocks,
            },
        }
    }

    /// Create partial topology without system. Used for topology includes.
    pub fn from_file_partial(filename: impl AsRef<Path> + Clone) -> anyhow::Result<Topology> {
        let yaml = std::fs::read_to_string(filename.clone())?;
        let mut topology: Topology = serde_yaml::from_str(&yaml)?;
        for file in topology.include.iter_mut() {
            file.finalize(filename.clone());
        }
        topology.include_topologies(topology.include.clone())?;
        Ok(topology)
    }

    /// Parse a yaml file as Topology which *must* include a system.
    pub fn from_file(filename: impl AsRef<Path>) -> anyhow::Result<Topology> {
        let yaml = std::fs::read_to_string(&filename)?;
        let mut topology: Topology = serde_yaml::from_str(&yaml)?;

        if topology.system.is_empty() {
            anyhow::bail!("missing or empty field `system`");
        };
        if topology.system.blocks.is_empty() {
            anyhow::bail!("missing or empty field `blocks`");
        }

        // finalize includes
        for file in topology.include.iter_mut() {
            file.finalize(&filename);
        }

        topology.include_topologies(topology.include.clone())?;
        topology.finalize_atoms()?;
        topology.finalize_molecules()?;
        topology.finalize_blocks(&filename)?;
        topology.validate_intermolecular()?;
        topology.validate()?;
        Ok(topology)
    }

    /// Get molecule blocks of the system.
    pub fn blocks(&self) -> &[MoleculeBlock] {
        &self.system.blocks
    }

    /// Get intermolecular bonded interactions of the system.
    pub fn intermolecular(&self) -> &IntermolecularBonded {
        &self.system.intermolecular
    }

    /// Get the total number of particules in the topology.
    pub fn num_particles(&self) -> usize {
        self.system
            .blocks
            .iter()
            .map(|block| block.num_atoms(&self.moleculekinds))
            .sum()
    }

    /// Get atoms kinds of the topology.
    pub fn atomkinds(&self) -> &[AtomKind] {
        &self.atomkinds
    }

    /// Get atoms kinds of the topology.
    pub fn atomkinds_mut(&mut self) -> &mut [AtomKind] {
        &mut self.atomkinds
    }

    /// Get molecule kinds of the topology.
    pub fn moleculekinds(&self) -> &[MoleculeKind] {
        &self.moleculekinds
    }

    /// Add an atom kind to the topology.
    pub fn add_atomkind(&mut self, atom: AtomKind) {
        self.atomkinds.push(atom)
    }

    /// Add a molecule kind to the topology.
    pub fn add_moleculekind(&mut self, molecule: MoleculeKind) {
        self.moleculekinds.push(molecule)
    }

    /// Find atom with given name.
    pub fn find_atom(&self, name: &str) -> Option<&AtomKind> {
        self.atomkinds().iter().find(|a| a.name() == name)
    }

    /// Find molecule with given name.
    pub fn find_molecule(&self, name: &str) -> Option<&MoleculeKind> {
        self.moleculekinds().iter().find(|r| r.name() == name)
    }

    /// Add atom kinds into a topology. In case an AtomKind with the same name already
    /// exists in the Topology, it is NOT overwritten and a warning is raised.
    pub(crate) fn include_atomkinds(&mut self, atoms: &[AtomKind]) {
        for atom in atoms.iter() {
            if self.atomkinds().iter().any(|x| x.name() == atom.name()) {
                log::warn!(
                    "Atom kind '{}' redefinition in included topology.",
                    atom.name()
                )
            } else {
                self.add_atomkind(atom.clone());
            }
        }
    }

    /// Add molecule kinds into a toplogy. In case a MoleculeKind with the same name
    /// already exists in the Topology, it is NOT overwritten and a warning is raised.
    pub(crate) fn include_moleculekinds(&mut self, molecules: Vec<MoleculeKind>) {
        for molecule in molecules.into_iter() {
            if self
                .moleculekinds()
                .iter()
                .any(|x| x.name() == molecule.name())
            {
                log::warn!(
                    "Molecule kind '{}' redefinition in included topology.",
                    molecule.name()
                )
            } else {
                self.add_moleculekind(molecule);
            }
        }
    }

    /// Read additional topologies into topology.
    ///
    /// ## Parameters
    /// - `parent_path` path to the directory containing the parent topology file
    /// - `topologies` paths to the topologies to be included (absolute or relative to the `parent_path`)
    pub(crate) fn include_topologies(
        &mut self,
        topologies: Vec<InputPath>,
    ) -> Result<(), anyhow::Error> {
        for file in topologies.iter() {
            let included_top = Topology::from_file_partial(file.path().unwrap())?;
            self.include_atomkinds(&included_top.atomkinds);
            self.include_moleculekinds(included_top.moleculekinds);
        }

        Ok(())
    }

    /// Create groups and populate target Context-implementing structure with particles.
    pub(crate) fn insert_groups(
        &self,
        context: &mut impl Context,
        structure: Option<impl AsRef<Path>>,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<()> {
        // current index of the coordinate in the external structure file to use
        let mut curr_start = 0;

        let positions = match structure {
            Some(x) => Some(positions_from_structure_file(&x, Some(context.cell()))?),
            None => None,
        };

        // create groups
        for block in self.blocks() {
            if block.insert_policy().is_some() {
                block.insert_block(context, &[], rng)?;
            } else {
                let Some(ref positions) = positions else {
                    anyhow::bail!("block requires structure that wasn't provided")
                };
                let atoms_in_block = block.num_atoms(self.moleculekinds());
                let positions = match positions.get(curr_start..(curr_start + atoms_in_block)) {
                    None => anyhow::bail!(
                        "external structure does not match topology - not enough coordinates"
                    ),
                    Some(pos) => pos,
                };
                block.insert_block(context, positions, rng)?;
                curr_start += atoms_in_block;
            }
        }

        // check that all coordinates from the structure file have been used
        match positions {
            Some(positions) if positions.len() != curr_start => {
                anyhow::bail!("structure does not match topology - too many coordinates")
            }
            _ => (),
        }
        Ok(())
    }

    /// Set ids for atom kinds in the topology and make sure that the atom names are unique.
    fn finalize_atoms(&mut self) -> anyhow::Result<()> {
        self.atomkinds
            .iter_mut()
            .enumerate()
            .for_each(|(i, atom): (usize, &mut AtomKind)| {
                atom.set_id(i);
            });

        if self.atomkinds.iter().map(|a| a.name()).all_unique() {
            Ok(())
        } else {
            anyhow::bail!("atoms have non-unique names")
        }
    }

    /// Set ids for molecule kinds in the topology, validate the molecules and
    /// set indices of atom kinds forming each molecule.
    fn finalize_molecules(&mut self) -> anyhow::Result<()> {
        for (i, molecule) in self.moleculekinds.iter_mut().enumerate() {
            // set atom names
            if molecule.atom_names().is_empty() {
                molecule.empty_atom_names();
            }

            // validate the molecule
            molecule.validate()?;

            // set index
            molecule.set_id(i);

            // set atom indices
            let indices = molecule
                .atoms()
                .iter()
                .map(|atom| {
                    self.atomkinds
                        .iter()
                        .position(|x| x.name() == atom)
                        .ok_or_else(|| anyhow::Error::msg("undefined atom kind in a molecule"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            molecule.set_atom_indices(indices);

            // expand exclusions (must be done after validation - the bonds must be valid)
            molecule.generate_exclusions();
        }

        // check that all molecule names are unique
        if self
            .moleculekinds
            .iter()
            .duplicates_by(|m| m.name())
            .count()
            .eq(&0)
        {
            Ok(())
        } else {
            anyhow::bail!("molecules have non-unique names")
        }
    }

    /// Set molecule indices for blocks and validate them.
    fn finalize_blocks(&mut self, filename: impl AsRef<Path> + Clone) -> anyhow::Result<()> {
        for block in self.system.blocks.iter_mut() {
            block.finalize(filename.clone())?;

            let index = self
                .moleculekinds
                .iter()
                .position(|x| x.name() == block.molecule())
                .ok_or(anyhow::Error::msg("undefined molecule kind in a block"))?;
            block.set_molecule_id(index);

            // check that if positions are provided manually, they are consistent with the topology
            if let Some(InsertionPolicy::Manual(positions)) = block.insert_policy() {
                if positions.len() != block.num_atoms(&self.moleculekinds) {
                    anyhow::bail!(
                        "the number of manually provided positions does not match the number of atoms",
                    );
                }
            }
        }

        Ok(())
    }

    /// Validate intermolecular bonded interactions.
    fn validate_intermolecular(&mut self) -> anyhow::Result<()> {
        let num_particles = self.num_particles();

        #[inline(always)]
        fn check_intermolecular_items<T: Indexed>(
            items: &[T],
            num_particles: usize,
            error_msg: &'static str,
        ) -> anyhow::Result<()> {
            if !items.iter().all(|item| item.lower(num_particles)) {
                anyhow::bail!(error_msg);
            }

            Ok(())
        }
        check_intermolecular_items(
            &self.system.intermolecular.bonds,
            num_particles,
            "intermolecular bond between undefined atoms",
        )?;
        check_intermolecular_items(
            &self.system.intermolecular.torsions,
            num_particles,
            "intermolecular torsion between undefined atoms",
        )?;
        check_intermolecular_items(
            &self.system.intermolecular.dihedrals,
            num_particles,
            "intermolecular dihedral between undefined atoms",
        )?;

        Ok(())
    }
}
