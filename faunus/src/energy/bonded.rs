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

use crate::{
    group::Group,
    topology::{block::BlockActivationStatus, Topology, TopologyLike},
    Change, Context, GroupChange, SyncFrom,
};

use super::{EnergyChange, EnergyTerm};

/// Energy term for computing intramolecular bonded interactions.
#[derive(Debug, Clone)]
pub struct IntramolecularBonded {}

impl EnergyChange for IntramolecularBonded {
    /// Compute the energy associated with the intramolecular
    /// bonded interactions relevant to some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(_, _) => self.all_groups(context),
            Change::None | Change::SingleGroup(_, GroupChange::RigidBody) => 0.0,
            // TODO! optimization: not all bonds have to be recalculated if a single particle inside a group changes
            Change::SingleGroup(id, change) => match change {
                GroupChange::None | GroupChange::RigidBody => 0.0,
                _ => self.one_group(context, &context.groups()[*id]),
            },
            Change::Groups(groups) => self.multiple_groups(
                context,
                &groups
                    .iter()
                    .filter_map(|(index, change)| match change {
                        GroupChange::None | GroupChange::RigidBody => None,
                        _ => Some(*index),
                    })
                    .collect::<Vec<usize>>(),
            ),
        }
    }
}

impl IntramolecularBonded {
    /// Create a new IntramolecularBonded energy term.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new() -> EnergyTerm {
        EnergyTerm::IntramolecularBonded(IntramolecularBonded {})
    }

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
        let molecule = &topology.molecules()[group.molecule()];

        molecule
            .bonds()
            .iter()
            .map(|bond| bond.energy(context, group))
            .sum::<f64>()
            + molecule
                .torsions()
                .iter()
                .map(|torsion| torsion.energy(context, group))
                .sum::<f64>()
            + molecule
                .dihedrals()
                .iter()
                .map(|dihedral| dihedral.energy(context, group))
                .sum::<f64>()
    }
}

impl SyncFrom for IntramolecularBonded {
    fn sync_from(&mut self, _other: &IntramolecularBonded, _change: &Change) -> anyhow::Result<()> {
        // nothing to synchronize
        Ok(())
    }
}

/// Energy term for computing intermolecular bonded interactions.
#[derive(Debug, Clone)]
pub struct IntermolecularBonded {
    /// Stores whether each particle of the system is active (true) or inactive (false).
    particles_status: Vec<bool>,
}

impl EnergyChange for IntermolecularBonded {
    /// Compute the energy associated with the intermolecular
    /// bonded interactions relevant to some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::None => 0.0,
            _ => {
                let topology = context.topology_ref();
                let intermolecular = topology.intermolecular();

                intermolecular
                    .bonds()
                    .iter()
                    .map(|bond| bond.energy_intermolecular(context, self))
                    .sum::<f64>()
                    + intermolecular
                        .torsions()
                        .iter()
                        .map(|torsion| torsion.energy_intermolecular(context, self))
                        .sum::<f64>()
                    + intermolecular
                        .dihedrals()
                        .iter()
                        .map(|dihedral| dihedral.energy_intermolecular(context, self))
                        .sum::<f64>()
            }
        }
    }
}

impl IntermolecularBonded {
    /// Create a new IntermolecularBonded energy term from Topology.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(topology: &Topology) -> EnergyTerm {
        let mut particles_status = Vec::new();

        topology.blocks().iter().for_each(|block| {
            let molecule = &topology.molecules()[block.molecule_index()];

            match block.active() {
                // add n_molecules * n_atoms active particles
                BlockActivationStatus::All => {
                    particles_status.extend(vec![true; block.number() * molecule.atoms().len()])
                }

                // add n_active_molecules * n_atoms active particles and
                // (n_molecules - n_active_molecules) * n_atoms inactive particles
                BlockActivationStatus::Partial(x) => {
                    particles_status.extend(vec![true; x * molecule.atoms().len()]);
                    particles_status
                        .extend(vec![false; (block.number() - x) * molecule.atoms().len()]);
                }
            }
        });

        EnergyTerm::IntermolecularBonded(IntermolecularBonded { particles_status })
    }

    /// Update the energy term. The update is needed if at least one particle was activated or deactivated.
    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::SingleGroup(i, GroupChange::Resize(_)) => {
                self.update_status_one_group(&context.groups()[*i])
            }
            Change::SingleGroup(_, _) => (),
            Change::Groups(groups) => self.update_status_multiple_groups(
                context,
                &groups
                    .iter()
                    .filter_map(|x| match x.1 {
                        // filter out groups which were not resized
                        GroupChange::Resize(_) => Some(x.0),
                        _ => None,
                    })
                    .collect::<Vec<usize>>(),
            ),
            Change::Everything => self.update_status_all(context),
            Change::None | Change::Volume { .. } => (),
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
            .iter_active()
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

impl SyncFrom for IntermolecularBonded {
    fn sync_from(&mut self, other: &IntermolecularBonded, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything | Change::Groups(_) | Change::SingleGroup(_, _) => {
                self.particles_status = other.particles_status.clone()
            }
            Change::None | Change::Volume(_, _) => (),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use float_cmp::assert_approx_eq;

    use crate::{
        cell::Cuboid, energy::Hamiltonian, group::GroupCollection, montecarlo::NewOld,
        platform::reference::ReferencePlatform, WithTopology,
    };

    use super::*;

    /// Get intramolecular bonded structure for testing.
    fn get_intramolecular_bonded() -> (ReferencePlatform, IntramolecularBonded) {
        let topology = Topology::from_file("tests/files/bonded_interactions.yaml").unwrap();

        let mut rng = rand::thread_rng();
        let system = ReferencePlatform::from_raw_parts(
            Rc::new(topology),
            Box::new(Cuboid::cubic(20.0)),
            RefCell::new(Hamiltonian::default()),
            None::<&str>,
            &mut rng,
        )
        .unwrap();

        let bonded = match IntramolecularBonded::new() {
            EnergyTerm::IntramolecularBonded(e) => e,
            _ => panic!("IntramolecularBonded not constructed."),
        };

        (system, bonded)
    }

    #[test]
    fn test_intramolecular_one_group() {
        let (system, bonded) = get_intramolecular_bonded();
        let expected = 1559328.708422025;

        assert_approx_eq!(
            f64,
            bonded.one_group(&system, &system.groups()[0]),
            expected
        );
        assert_approx_eq!(
            f64,
            bonded.one_group(&system, &system.groups()[1]),
            expected
        );
    }

    #[test]
    fn test_intramolecular_multiple_groups() {
        let (system, bonded) = get_intramolecular_bonded();
        let expected = 2.0 * 1559328.708422025;

        assert_approx_eq!(f64, bonded.multiple_groups(&system, &[0, 1]), expected);
    }

    #[test]
    fn test_intramolecular_all_groups() {
        let (system, bonded) = get_intramolecular_bonded();
        let expected = 2.0 * 1559328.708422025;

        assert_approx_eq!(f64, bonded.all_groups(&system), expected);
    }

    #[test]
    fn test_intramolecular_energy() {
        let (system, bonded) = get_intramolecular_bonded();

        // no change
        let change = Change::None;
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change everything
        let change = Change::Everything;
        let expected = bonded.all_groups(&system);
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        // change volume
        let change = Change::Volume(
            crate::cell::VolumeScalePolicy::Isotropic,
            NewOld {
                old: 104.0,
                new: 108.0,
            },
        );
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        // single group with no change
        let change = Change::SingleGroup(1, GroupChange::None);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // multiple groups with no change
        let change = Change::Groups(vec![(0, GroupChange::None), (1, GroupChange::None)]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change single rigid group
        let change = Change::SingleGroup(1, GroupChange::RigidBody);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change multiple rigid groups
        let change = Change::Groups(vec![
            (0, GroupChange::RigidBody),
            (1, GroupChange::RigidBody),
        ]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change several particles within a single group
        let change = Change::SingleGroup(1, GroupChange::PartialUpdate(vec![1, 2]));
        let expected = bonded.one_group(&system, &system.groups()[1]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        // change several particles in multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![1])),
            (1, GroupChange::PartialUpdate(vec![0, 2])),
        ]);
        let expected = bonded.multiple_groups(&system, &[0, 1]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);
    }
}
