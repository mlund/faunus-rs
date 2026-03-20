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

use crate::{
    cell::Cell,
    context::WithCell,
    group::{GroupCollection, GroupSize},
    Particle, UnitQuaternion,
};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Saved state of a single group (serde detail, not part of the public API).
///
/// Immutable topology fields (`molecule`, `capacity`) are stored to detect
/// mismatches early on load, rather than silently producing wrong energies.
#[derive(Debug, Serialize, Deserialize)]
struct GroupState {
    molecule: usize,
    capacity: usize,
    size: GroupSize,
    /// Rigid-body orientation needed by LD and 6D tabulated energies.
    /// Defaults to identity for backward compatibility with pre-quaternion state files.
    #[serde(default = "UnitQuaternion::identity")]
    quaternion: UnitQuaternion,
}

/// Checkpoint of the simulation state.
///
/// Only runtime-mutable quantities are saved. Topology, hamiltonian, analyses,
/// and moves are rebuilt from the YAML input file on resume, keeping the state
/// file small and avoiding serialization of trait objects.
#[derive(Debug, Serialize, Deserialize)]
pub struct State {
    particles: Vec<Particle>,
    cell: Cell,
    groups: Vec<GroupState>,
    step: usize,
}

impl State {
    /// Capture the current simulation state for checkpointing.
    pub fn save(context: &(impl GroupCollection + WithCell), step: usize) -> Self {
        State {
            particles: context.get_all_particles(),
            cell: context.cell().clone(),
            groups: context
                .groups()
                .iter()
                .map(|g| GroupState {
                    molecule: g.molecule(),
                    capacity: g.capacity(),
                    size: g.size(),
                    quaternion: *g.quaternion(),
                })
                .collect(),
            step,
        }
    }

    /// Restore state into a context. Returns the saved step counter.
    ///
    /// Validates topology compatibility before modifying any state,
    /// so a mismatched state file is rejected cleanly.
    pub fn load(self, context: &mut impl crate::Context) -> anyhow::Result<usize> {
        let num_particles = context.num_particles();
        let num_groups = context.groups().len();

        if self.particles.len() != num_particles {
            anyhow::bail!(
                "Particle count mismatch: state has {}, context has {}",
                self.particles.len(),
                num_particles
            );
        }
        if self.groups.len() != num_groups {
            anyhow::bail!(
                "Group count mismatch: state has {}, context has {}",
                self.groups.len(),
                num_groups
            );
        }

        // Warn about atom_id changes (expected after atom swap reactions)
        for (i, state_p) in self.particles.iter().enumerate() {
            let ctx_id = context.particle(i).atom_id;
            if state_p.atom_id != ctx_id {
                log::warn!(
                    "Particle {} atom_id differs: state has {}, topology has {} (atom swap?)",
                    i,
                    state_p.atom_id,
                    ctx_id
                );
            }
        }

        // Catch molecule reordering or resized molecule definitions
        for (i, (gs, group)) in self
            .groups
            .iter()
            .zip(context.groups().iter())
            .enumerate()
        {
            if gs.molecule != group.molecule() {
                anyhow::bail!(
                    "Group {} molecule mismatch: state has {}, topology has {}",
                    i,
                    gs.molecule,
                    group.molecule()
                );
            }
            if gs.capacity != group.capacity() {
                anyhow::bail!(
                    "Group {} capacity mismatch: state has {}, topology has {}",
                    i,
                    gs.capacity,
                    group.capacity()
                );
            }
        }

        *context.cell_mut() = self.cell;
        let sizes: Vec<_> = self.groups.iter().map(|gs| gs.size).collect();
        let quaternions: Vec<_> = self.groups.iter().map(|gs| gs.quaternion).collect();
        context.apply_particles_and_groups(&self.particles, &sizes, &quaternions)?;
        context.update(&crate::Change::Everything)?;

        log::info!("Restored simulation state");
        Ok(self.step)
    }

    /// Override the step counter (e.g. reset to 0 after umbrella drive phase).
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

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
