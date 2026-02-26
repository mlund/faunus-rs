// Copyright 2023-2026 Mikael Lund
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

//! Save and load simulation state for checkpointing and resuming simulations.

use crate::{cell::Cell, group::GroupSize, Particle};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Saved state of a single group.
///
/// The `molecule` and `capacity` fields are immutable in the running simulation
/// but saved here so that `load_state` can detect topology mismatches early
/// instead of silently producing wrong energies.
#[derive(Debug, Serialize, Deserialize)]
pub struct GroupState {
    pub molecule: usize,
    pub capacity: usize,
    pub size: GroupSize,
}

/// Checkpoint of the simulation state.
///
/// Only runtime-mutable quantities are saved. Topology, hamiltonian, analyses,
/// and moves are rebuilt from the YAML input file on resume, keeping the state
/// file small and avoiding serialization of trait objects.
#[derive(Debug, Serialize, Deserialize)]
pub struct State {
    pub particles: Vec<Particle>,
    pub cell: Cell,
    pub groups: Vec<GroupState>,
    pub step: usize,
}

impl State {
    /// Load a state from a YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let yaml = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read state file {:?}", path.as_ref()))?;
        Ok(serde_yaml::from_str(&yaml)?)
    }

    /// Save the state to a YAML file.
    pub fn to_file(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path.as_ref(), yaml)
            .with_context(|| format!("Failed to write state file {:?}", path.as_ref()))
    }
}
