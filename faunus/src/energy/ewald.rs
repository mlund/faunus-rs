// Copyright 2025 Mikael Lund
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

//! Ewald reciprocal-space energy term.
//!
//! Wraps `coulomb::reciprocal::EwaldReciprocal` to provide the k-space
//! contribution to the electrostatic energy in Monte Carlo simulations.
//! Uses the Nymand–Linse O(M·N_k) partial structure factor update for
//! single-group moves instead of O(N·N_k) full rebuilds.

use crate::cell::Shape;
use crate::change::GroupChange;
use crate::{Change, Context};
use interatomic::coulomb::reciprocal::{EwaldPolicy, EwaldReciprocal};
use interatomic::coulomb::DebyeLength;
use serde::{Deserialize, Serialize};

use super::EnergyTerm;

/// Ewald reciprocal-space energy with backup/undo for MC moves.
#[derive(Clone)]
pub struct EwaldReciprocalEnergy {
    ewald: EwaldReciprocal,
    /// Cached reciprocal + self energy (kJ/mol)
    cached_energy: f64,
    /// Charges indexed by global particle index (includes zeros)
    charges: Vec<f64>,
    /// Electric prefactor: e²/(4πε₀ε_r) in kJ/mol·Å
    prefactor: f64,
    backup: Option<EwaldBackup>,
    /// Reusable position buffers to avoid per-update allocations.
    pos_buf: (Vec<f64>, Vec<f64>, Vec<f64>),
}

impl std::fmt::Debug for EwaldReciprocalEnergy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EwaldReciprocalEnergy")
            .field("cached_energy", &self.cached_energy)
            .field("prefactor", &self.prefactor)
            .field("num_charges", &self.charges.len())
            .finish()
    }
}

#[derive(Clone)]
struct EwaldBackup {
    ewald: EwaldReciprocal,
    cached_energy: f64,
    /// Old positions of affected particles: (global_index, [x, y, z])
    old_positions: Vec<(usize, [f64; 3])>,
}

/// YAML configuration for Ewald summation (both real-space and reciprocal).
///
/// When present, a matching real-space Ewald pair potential is automatically
/// injected into the nonbonded defaults (before splining), and the reciprocal-space
/// term is added as a separate energy contribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwaldBuilder {
    /// Real-space cutoff (Å)
    pub cutoff: f64,
    /// Target relative accuracy (e.g. 1e-5)
    #[serde(default = "default_accuracy")]
    pub accuracy: f64,
    /// Ewald policy: PBC (default) or IPBC
    #[serde(default)]
    pub policy: EwaldPolicy,
}

const fn default_accuracy() -> f64 {
    1e-5
}

impl EwaldReciprocalEnergy {
    /// Create from builder, context, and medium.
    pub fn new(
        builder: &EwaldBuilder,
        context: &impl Context,
        medium: &interatomic::coulomb::Medium,
    ) -> anyhow::Result<Self> {
        let box_length = Self::box_length_from_context(context)?;
        let kappa = medium.debye_length().map(|d| 1.0 / d);
        let ewald = EwaldReciprocal::new(box_length, builder.cutoff, builder.accuracy, kappa)
            .with_policy(builder.policy);
        let prefactor = interatomic::coulomb::TO_CHEMISTRY_UNIT / medium.permittivity();

        log::info!(
            "Ewald reciprocal ({:?}): α={:.4}, n_max={}, k-vectors={}",
            builder.policy,
            ewald.alpha(),
            ewald.n_max(),
            ewald.num_k_vectors()
        );

        let charges = Self::extract_charges(context);
        let n = charges.len();
        let mut term = Self {
            ewald,
            cached_energy: 0.0,
            charges,
            prefactor,
            backup: None,
            pos_buf: (vec![0.0; n], vec![0.0; n], vec![0.0; n]),
        };
        term.full_update(context);
        Ok(term)
    }

    fn box_length_from_context(context: &impl Context) -> anyhow::Result<[f64; 3]> {
        let bb = context
            .cell()
            .bounding_box()
            .ok_or_else(|| anyhow::anyhow!("Ewald requires a cuboid cell with finite volume"))?;
        Ok([bb.x, bb.y, bb.z])
    }

    /// Re-extract charges and resize position buffers to match.
    fn refresh_charges(&mut self, context: &impl Context) {
        self.charges = Self::extract_charges(context);
        let n = self.charges.len();
        self.pos_buf.0.resize(n, 0.0);
        self.pos_buf.1.resize(n, 0.0);
        self.pos_buf.2.resize(n, 0.0);
    }

    fn extract_charges(context: &impl Context) -> Vec<f64> {
        let topology = context.topology_ref();
        let atomkinds = topology.atomkinds();
        let n = context.groups().iter().map(|g| g.capacity()).sum();
        let mut charges = vec![0.0; n];
        for group in context.groups() {
            for i in group.iter_active() {
                charges[i] = atomkinds[context.get_atomkind(i)].charge();
            }
        }
        charges
    }

    /// Full recompute of structure factors and cached energy.
    fn full_update(&mut self, context: &impl Context) {
        self.fill_positions(context);
        let (x, y, z) = &self.pos_buf;
        self.ewald.update_all(x, y, z, &self.charges, None);
        self.update_cached_energy();
    }

    /// Full recompute with new box dimensions (volume change).
    fn full_update_with_box(&mut self, context: &impl Context) -> anyhow::Result<()> {
        let box_length = Self::box_length_from_context(context)?;
        self.fill_positions(context);
        let (x, y, z) = &self.pos_buf;
        self.ewald
            .update_all(x, y, z, &self.charges, Some(box_length));
        self.update_cached_energy();
        Ok(())
    }

    fn update_cached_energy(&mut self) {
        self.cached_energy =
            self.prefactor * (self.ewald.energy() + self.ewald.self_energy(&self.charges));
    }

    /// Fill pre-allocated position buffers from the current context.
    fn fill_positions(&mut self, context: &impl Context) {
        let (x, y, z) = &mut self.pos_buf;
        x.iter_mut().for_each(|v| *v = 0.0);
        y.iter_mut().for_each(|v| *v = 0.0);
        z.iter_mut().for_each(|v| *v = 0.0);
        for group in context.groups() {
            for i in group.iter_active() {
                let pos = context.position(i);
                x[i] = pos.x;
                y[i] = pos.y;
                z[i] = pos.z;
            }
        }
    }

    /// Compute energy relevant to a change.
    pub fn energy(&self, _context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::None => 0.0,
            _ => self.cached_energy,
        }
    }

    /// Update internal state after a system change.
    ///
    /// For single-group moves, uses O(M·N_k) incremental structure factor
    /// updates via `update_particle` (Nymand & Linse, JCP 112, 6152, 2000).
    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::None => {}
            Change::Volume(..) | Change::Everything => {
                self.full_update_with_box(context)?;
            }
            Change::Groups(changes) => {
                if changes
                    .iter()
                    .any(|(_, gc)| matches!(gc, GroupChange::Resize(_)))
                {
                    self.refresh_charges(context);
                }
                self.full_update(context);
            }
            Change::SingleGroup(gi, gc) => match gc {
                GroupChange::Resize(_) => {
                    self.refresh_charges(context);
                    self.full_update(context);
                }
                GroupChange::None => {}
                _ => {
                    self.incremental_update(context, *gi, gc);
                }
            },
        }
        Ok(())
    }

    /// Resolve affected global particle indices from a group change.
    fn affected_indices(group: &crate::group::Group, gc: &GroupChange) -> Vec<usize> {
        match gc {
            GroupChange::RigidBody => group.iter_active().collect(),
            GroupChange::PartialUpdate(rel) | GroupChange::UpdateIdentity(rel) => {
                let offset = group.iter_active().next().unwrap_or(0);
                rel.iter().map(|&ri| offset + ri).collect()
            }
            GroupChange::None | GroupChange::Resize(_) => Vec::new(),
        }
    }

    /// O(M·N_k) incremental update for a single-group move.
    ///
    /// Uses old positions from backup and new positions from context to
    /// call `update_particle` for each affected charged particle.
    fn incremental_update(
        &mut self,
        context: &impl Context,
        group_index: usize,
        group_change: &GroupChange,
    ) {
        let backup = match self.backup.as_ref() {
            Some(b) if !b.old_positions.is_empty() => b,
            _ => {
                self.full_update(context);
                return;
            }
        };

        let group = &context.groups()[group_index];
        let affected = Self::affected_indices(group, group_change);

        for &idx in &affected {
            let charge = self.charges[idx];
            if charge == 0.0 {
                continue;
            }
            // Linear scan is efficient for the typical small number of affected particles
            if let Some(&(_, old)) = backup.old_positions.iter().find(|(i, _)| *i == idx) {
                let new_pos = context.position(idx);
                let new = [new_pos.x, new_pos.y, new_pos.z];
                self.ewald.update_particle(charge, old, new);
            }
        }
        self.update_cached_energy();
    }

    /// Save state for later undo. Context has OLD positions (called before move).
    pub(super) fn save_backup(&mut self, change: &Change, context: &impl Context) {
        let old_positions = self.collect_affected_positions(change, context);
        self.backup = Some(EwaldBackup {
            ewald: self.ewald.clone(),
            cached_energy: self.cached_energy,
            old_positions,
        });
    }

    /// Collect positions of particles that will be affected by the change.
    fn collect_affected_positions(
        &self,
        change: &Change,
        context: &impl Context,
    ) -> Vec<(usize, [f64; 3])> {
        match change {
            Change::SingleGroup(gi, gc) => {
                let group = &context.groups()[*gi];
                Self::affected_indices(group, gc)
                    .into_iter()
                    .filter(|&i| self.charges[i] != 0.0)
                    .map(|i| {
                        let pos = context.position(i);
                        (i, [pos.x, pos.y, pos.z])
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// Restore from backup (reject path).
    pub(super) fn undo(&mut self) {
        if let Some(backup) = self.backup.take() {
            self.ewald = backup.ewald;
            self.cached_energy = backup.cached_energy;
        }
    }

    /// Drop backup (accept path).
    pub(super) fn discard_backup(&mut self) {
        self.backup = None;
    }
}

impl From<EwaldReciprocalEnergy> for EnergyTerm {
    fn from(ewald: EwaldReciprocalEnergy) -> Self {
        Self::EwaldReciprocal(Box::new(ewald))
    }
}
