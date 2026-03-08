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

//! Implementation of the bonded interactions.

#[cfg(feature = "gpu")]
pub(crate) mod kernel;

use crate::{
    group::Group,
    topology::{block::BlockActivationStatus, Topology},
    Change, Context,
};

use super::{EnergyChange, EnergyTerm};

/// Energy term for computing intramolecular bonded interactions.
#[derive(Debug, Clone, Default)]
pub struct IntramolecularBonded {}

impl EnergyChange for IntramolecularBonded {
    /// Compute the energy associated with the intramolecular
    /// bonded interactions relevant to some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(_, _) => self.all_groups(context),
            Change::None => 0.0,
            // TODO! optimization: not all bonds have to be recalculated if a single particle inside a group changes
            Change::SingleGroup(id, gc) if gc.internal_change() => {
                self.one_group(context, &context.groups()[*id])
            }
            Change::SingleGroup(..) => 0.0,
            Change::Groups(groups) => self.multiple_groups(
                context,
                &groups
                    .iter()
                    .filter_map(|(index, change)| change.internal_change().then_some(*index))
                    .collect::<Vec<usize>>(),
            ),
        }
    }
}

impl IntramolecularBonded {
    /// Calculate energy of all active bonded interactions of the specified groups.
    #[inline(always)]
    fn multiple_groups(&self, context: &impl Context, groups: &[usize]) -> f64 {
        groups
            .iter()
            .map(|&id| self.one_group(context, &context.groups()[id]))
            .sum()
    }

    /// Calculate energy of all active bonded interactions of all groups.
    #[inline(always)]
    fn all_groups(&self, context: &impl Context) -> f64 {
        context
            .groups()
            .iter()
            .map(|group| self.one_group(context, group))
            .sum()
    }

    /// Calculate energy of all active bonded interactions of target group.
    fn one_group(&self, context: &impl Context, group: &Group) -> f64 {
        let topology = context.topology_ref();
        let molecule = &topology.moleculekinds()[group.molecule()];

        if !molecule.has_bonded_potentials() {
            return 0.0;
        }

        let bond_energy: f64 = molecule
            .bonds()
            .iter()
            .map(|bond| bond.energy(context, group))
            .sum();

        let torsion_energy: f64 = molecule
            .torsions()
            .iter()
            .map(|torsion| torsion.energy(context, group))
            .sum();

        let dihedral_energy: f64 = molecule
            .dihedrals()
            .iter()
            .map(|dihedral| dihedral.energy(context, group))
            .sum();

        bond_energy + torsion_energy + dihedral_energy
    }
}

impl From<IntramolecularBonded> for EnergyTerm {
    fn from(term: IntramolecularBonded) -> Self {
        Self::IntramolecularBonded(term)
    }
}

/// Energy term for computing intermolecular bonded interactions.
#[derive(Debug, Clone)]
pub struct IntermolecularBonded {
    /// Stores whether each particle of the system is active (true) or inactive (false).
    particles_status: Vec<bool>,
    /// Backup for undo on MC reject
    backup: Option<Vec<bool>>,
}

impl EnergyChange for IntermolecularBonded {
    /// Compute the energy associated with the intermolecular
    /// bonded interactions relevant to some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::None => 0.0,
            _ => {
                let intermolecular = context.topology_ref().intermolecular();
                let bond_energy: f64 = intermolecular
                    .bonds()
                    .iter()
                    .map(|bond| bond.energy_intermolecular(context, self))
                    .sum();

                let torsion_energy: f64 = intermolecular
                    .torsions()
                    .iter()
                    .map(|torsion| torsion.energy_intermolecular(context, self))
                    .sum();

                let dihedral_energy: f64 = intermolecular
                    .dihedrals()
                    .iter()
                    .map(|dihedral| dihedral.energy_intermolecular(context, self))
                    .sum();

                bond_energy + torsion_energy + dihedral_energy
            }
        }
    }
}

impl IntermolecularBonded {
    /// Create a new IntermolecularBonded energy term from Topology.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(topology: &Topology) -> EnergyTerm {
        let particles_status: Vec<bool> = topology
            .blocks()
            .iter()
            .flat_map(|block| {
                let molecule = &topology.moleculekinds()[block.molecule_index()];
                let num_atoms = molecule.atoms().len();
                match block.active() {
                    BlockActivationStatus::All => vec![true; block.num_molecules() * num_atoms],
                    BlockActivationStatus::Partial(x) => {
                        let mut status = vec![true; x * num_atoms];
                        status.extend(vec![false; (block.num_molecules() - x) * num_atoms]);
                        status
                    }
                }
            })
            .collect();

        EnergyTerm::IntermolecularBonded(Self {
            particles_status,
            backup: None,
        })
    }

    /// Update the energy term. The update is needed if at least one particle was activated or deactivated.
    //
    // TODO:
    // Currently this updates the entire group upon any change in the size of the group.
    // However, `change` contains information about the number of (de)activated particles in the group.
    // We should probably use this information to only update the relevant part of the group.
    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::SingleGroup(i, gc) if gc.is_resize() => {
                self.update_status_one_group(&context.groups()[*i])
            }
            Change::Groups(groups) => self.update_status_multiple_groups(
                context,
                &groups
                    .iter()
                    .filter_map(|(idx, gc)| gc.is_resize().then_some(*idx))
                    .collect::<Vec<usize>>(),
            ),
            Change::Everything => self.update_status_all(context),
            Change::SingleGroup(_, _) | Change::None | Change::Volume { .. } => (),
        }

        Ok(())
    }

    /// Check whether the particle with the provided absolute index is active.
    pub(crate) fn is_active(&self, abs_index: usize) -> bool {
        self.particles_status[abs_index]
    }

    /// Update the status of particles from a single group.
    fn update_status_one_group(&mut self, group: &Group) {
        group
            .iter_active()
            .for_each(|x| self.particles_status[x] = true);
        group
            .iter_inactive()
            .for_each(|x| self.particles_status[x] = false);
    }

    /// Update the status of particles from multiple groups.
    fn update_status_multiple_groups(&mut self, context: &impl Context, groups: &[usize]) {
        groups
            .iter()
            .map(|&i| &context.groups()[i])
            .for_each(|group| self.update_status_one_group(group));
    }

    /// Update the status of particles from all groups.
    fn update_status_all(&mut self, context: &impl Context) {
        context
            .groups()
            .iter()
            .for_each(|group| self.update_status_one_group(group));
    }
}

impl From<IntermolecularBonded> for EnergyTerm {
    fn from(term: IntermolecularBonded) -> Self {
        Self::IntermolecularBonded(term)
    }
}

impl IntermolecularBonded {
    /// Only save when `update()` would actually modify state.
    pub(super) fn save_backup(&mut self, change: &Change) {
        let dominated = matches!(change, Change::Everything)
            || matches!(change, Change::SingleGroup(_, gc) if gc.is_resize())
            || matches!(change, Change::Groups(v) if v.iter().any(|(_, gc)| gc.is_resize()));

        if dominated {
            assert!(self.backup.is_none(), "backup already exists");
            self.backup = Some(self.particles_status.clone());
        }
    }

    pub(super) fn undo(&mut self) {
        if let Some(backup) = self.backup.take() {
            self.particles_status = backup;
        }
    }

    pub(super) fn discard_backup(&mut self) {
        self.backup = None;
    }
}

#[cfg(test)]
mod tests;
