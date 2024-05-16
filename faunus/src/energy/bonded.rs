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

use crate::{group::Group, topology::TopologyLike, Change, Context};

#[derive(Debug, Clone)]
pub struct IntramolecularBonded {}

impl IntramolecularBonded {
    /// Compute the energy change associated with the intramolecular bonded interactions due to a change in the system.
    pub(super) fn energy_change(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(_, _) => self.all_groups(context),
            Change::None => 0.0,
            Change::SingleGroup(id, _) => self.one_group(context, &context.groups()[*id]),
            Change::Groups(groups) => {
                self.multiple_groups(context, &groups.iter().map(|x| x.0).collect::<Vec<usize>>())
            }
        }
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

#[derive(Debug, Clone)]
pub struct IntermolecularBonded {
    /// Stores whether each particle of the system is active (1) or inactive (0).
    particle_status: Vec<bool>,
}

impl IntermolecularBonded {
    pub(super) fn energy_change(&self, context: &impl Context, change: &Change) -> f64 {
        todo!()
    }
}

/*impl SyncFrom for IntermolecularBonds {
    todo!()
}*/
