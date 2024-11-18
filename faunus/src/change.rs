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

use serde::{Deserialize, Serialize};

use crate::{cell, group::GroupSize, montecarlo::NewOld};

/// Describes a change in the system. This can for example be used to
/// describe a change in the volume of the system, or a change in the
/// number of particles in a group.
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub enum Change {
    /// Everything has changed
    Everything,
    /// The volume has changed (policy, new/old volume)
    Volume(cell::VolumeScalePolicy, NewOld<f64>),
    /// Some groups have changed vector of (group index, group change)
    Groups(Vec<(usize, GroupChange)>),
    /// A single group has changed
    SingleGroup(usize, GroupChange),
    /// No change
    #[default]
    None,
}

/// Description of a change to a single group of particles
///
/// Defines a change to a group of particles, e.g. a rigid body update,
/// adding or removing particles, etc. It is used in connection with Monte Carlo
/// moves to communicate an update to e.g. the Hamiltonian.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GroupChange {
    /// Rigid body update where *all* particles are e.g. rotated or translated with *no* internal energy change
    RigidBody,
    /// Update by relative indices, assuming that the internal energy changes (relative indices)
    PartialUpdate(Vec<usize>),
    /// Resize group
    Resize(GroupSize),
    /// The identity of a set of particles has changed (relative indices)
    UpdateIdentity(Vec<usize>),
    /// Nothing has changed
    None,
}

impl GroupChange {
    pub const fn internal_change(&self) -> bool {
        !matches!(self, GroupChange::RigidBody)
    }
}
