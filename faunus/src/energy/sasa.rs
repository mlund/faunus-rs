// Copyright 2024 Mikael Lund
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

//! # Energy due to solvent-accessible surface area and atomic-level surface energy densities.

use crate::{Change, Context, GroupChange};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use voronota_ltr::{Ball, PeriodicBox, UpdateableTessellation};

#[derive(Clone, Builder)]
#[builder(derive(Deserialize, Serialize, Debug))]
#[builder_struct_attr(serde(deny_unknown_fields))]
pub struct SasaEnergy {
    probe_radius: f64,
    #[builder(default = None, setter(strip_option))]
    energy_offset: Option<f64>,
    #[builder(default = "false")]
    offset_from_first: bool,
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    balls: Vec<Ball>,
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    tensions: Vec<f64>,
    /// Incremental tessellation avoids full recomputation when only a subset of atoms move.
    #[builder_field_attr(serde(skip))]
    #[builder(setter(skip), default = "UpdateableTessellation::with_backup()")]
    tess: UpdateableTessellation,
    /// Flat ball array is partitioned by group; ranges enable O(1) group→ball lookup.
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    group_ball_ranges: Vec<Range<usize>>,
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    periodic_box: Option<PeriodicBox>,
    /// Inline backup buffers reused across MC steps to avoid per-step allocation.
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    backup_ids: Vec<usize>,
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    backup_balls: Vec<Ball>,
    /// Full-state backup for resize/everything changes that alter the ball count.
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    backup_tensions: Vec<f64>,
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    backup_ranges: Vec<Range<usize>>,
    #[builder_field_attr(serde(skip))]
    #[builder(default = "false")]
    has_backup: bool,
    #[builder_field_attr(serde(skip))]
    #[builder(default = "false")]
    is_full_backup: bool,
}

impl std::fmt::Debug for SasaEnergy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SasaEnergy")
            .field("probe_radius", &self.probe_radius)
            .field("energy_offset", &self.energy_offset)
            .field("num_balls", &self.balls.len())
            .finish()
    }
}

impl SasaEnergy {
    /// Sum of γ_i × A_i over all particles (without offset).
    fn raw_energy(&self) -> f64 {
        self.tess
            .result()
            .cells
            .iter()
            .map(|cell| self.tensions[cell.index] * cell.sas_area)
            .sum()
    }

    pub fn energy(&self, _context: &impl Context, _change: &Change) -> f64 {
        self.raw_energy() + self.energy_offset.unwrap_or(0.0)
    }

    /// Free function to avoid borrowing `self`, allowing callers to pass `&mut self.backup_ids`.
    fn append_ball_ids(
        ranges: &[Range<usize>],
        k: usize,
        group_change: &GroupChange,
        out: &mut Vec<usize>,
    ) {
        match group_change {
            GroupChange::PartialUpdate(relative_indices) => {
                let offset = ranges[k].start;
                out.extend(relative_indices.iter().map(|&i| offset + i));
            }
            _ => out.extend(ranges[k].clone()),
        }
    }

    /// Copy positions from context into `self.balls` for balls in group `k`.
    fn sync_balls(&mut self, context: &impl Context, k: usize, ball_ids: &[usize]) {
        let group = &context.groups()[k];
        let range_start = self.group_ball_ranges[k].start;
        for &ball_id in ball_ids {
            // iter_active() returns Range<usize>, so nth() is O(1)
            let atom_index = group.iter_active().nth(ball_id - range_start).unwrap();
            let pos = context.position(atom_index);
            self.balls[ball_id].x = pos.x;
            self.balls[ball_id].y = pos.y;
            self.balls[ball_id].z = pos.z;
        }
    }

    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match change {
            // GCMC resize changes the ball count, invalidating all ranges and the tessellation.
            Change::SingleGroup(_, gc) if gc.is_resize() => self.rebuild_all(context),
            Change::SingleGroup(k, gc) => {
                let mut changed = Vec::new();
                Self::append_ball_ids(&self.group_ball_ranges, *k, gc, &mut changed);
                self.sync_balls(context, *k, &changed);
                self.tess.update_with_changed(&self.balls, &changed);
            }
            Change::Groups(changes) if changes.iter().any(|(_, gc)| gc.is_resize()) => {
                self.rebuild_all(context);
            }
            Change::Groups(changes) => {
                let mut all_changed = Vec::new();
                for (k, gc) in changes {
                    let start = all_changed.len();
                    Self::append_ball_ids(&self.group_ball_ranges, *k, gc, &mut all_changed);
                    self.sync_balls(context, *k, &all_changed[start..]);
                }
                self.tess.update_with_changed(&self.balls, &all_changed);
            }
            Change::Everything | Change::Volume(..) => self.rebuild_all(context),
            Change::None => {}
        }
        Ok(())
    }

    fn rebuild_all(&mut self, context: &impl Context) {
        self.periodic_box = super::make_periodic_box(context.cell());
        let topology = context.topology();
        let atomkinds = topology.atomkinds();

        self.balls.clear();
        self.tensions.clear();
        self.group_ball_ranges.clear();

        for group in context.groups() {
            let start = self.balls.len();
            for i in group.iter_active() {
                let pos = context.position(i);
                let ak = &atomkinds[context.atom_kind(i)];
                let radius = ak.sigma().map(|s| s / 2.0).unwrap_or(0.0);
                self.balls.push(Ball::new(pos.x, pos.y, pos.z, radius));
                self.tensions.push(ak.gamma().unwrap_or(0.0));
            }
            self.group_ball_ranges.push(start..self.balls.len());
        }

        self.tess
            .init(&self.balls, self.probe_radius, self.periodic_box.as_ref());

        if self.offset_from_first && self.energy_offset.is_none() && !self.balls.is_empty() {
            let energy = self.raw_energy();
            self.energy_offset = Some(-energy);
            log::info!(
                "SASA energy offset set from first configuration = {:.2} kJ/mol",
                -energy
            );
        }
    }
}

impl SasaEnergy {
    /// Save ball positions for later undo. The tessellation snapshots its own state
    /// automatically inside each `update_with_changed` / `init` call.
    pub(super) fn save_backup(&mut self, change: &Change) {
        assert!(!self.has_backup, "backup already exists");
        self.backup_ids.clear();
        self.backup_balls.clear();

        let needs_full = match change {
            Change::SingleGroup(_, gc) => gc.is_resize(),
            Change::Groups(changes) => changes.iter().any(|(_, gc)| gc.is_resize()),
            Change::Everything | Change::Volume(..) => true,
            Change::None => false,
        };

        if needs_full {
            // Resize changes ball count — must snapshot entire state
            self.backup_balls.extend_from_slice(&self.balls);
            self.backup_tensions.clear();
            self.backup_tensions.extend_from_slice(&self.tensions);
            self.backup_ranges.clear();
            self.backup_ranges
                .extend_from_slice(&self.group_ball_ranges);
            self.is_full_backup = true;
        } else {
            match change {
                Change::SingleGroup(k, gc) => {
                    Self::append_ball_ids(&self.group_ball_ranges, *k, gc, &mut self.backup_ids);
                }
                Change::Groups(changes) => {
                    for (k, gc) in changes {
                        Self::append_ball_ids(
                            &self.group_ball_ranges,
                            *k,
                            gc,
                            &mut self.backup_ids,
                        );
                    }
                }
                _ => {}
            }
            self.backup_balls
                .extend(self.backup_ids.iter().map(|&i| self.balls[i]));
            self.is_full_backup = false;
        }
        self.has_backup = true;
    }

    pub(super) fn undo(&mut self) {
        assert!(self.has_backup, "undo called without backup");
        if self.is_full_backup {
            // Swap is O(1) — the old (post-resize) state lands in the backup buffers
            // and will be overwritten on the next save_backup call.
            std::mem::swap(&mut self.balls, &mut self.backup_balls);
            std::mem::swap(&mut self.tensions, &mut self.backup_tensions);
            std::mem::swap(&mut self.group_ball_ranges, &mut self.backup_ranges);
        } else {
            for (&ball_id, &saved) in self.backup_ids.iter().zip(self.backup_balls.iter()) {
                self.balls[ball_id] = saved;
            }
        }
        self.tess.restore();
        self.has_backup = false;
    }

    pub(super) fn discard_backup(&mut self) {
        self.has_backup = false;
    }

    pub(super) fn to_yaml(&self) -> serde_yml::Value {
        let mut map = serde_yml::Mapping::new();
        map.insert("probe_radius".into(), self.probe_radius.into());
        if let Some(offset) = self.energy_offset {
            map.insert("energy_offset".into(), offset.into());
        }
        serde_yml::Value::Mapping(map)
    }
}

#[cfg(test)]
mod tests_sasaenergy {
    use crate::{backend::Backend, energy::EnergyChange, WithHamiltonian};
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_sasa() {
        let context = Backend::new(
            "tests/files/sasa_interactions.yaml",
            None,
            &mut rand::thread_rng(),
        )
        .unwrap();

        let energy = context
            .hamiltonian()
            .energy(&context, &crate::Change::Everything);

        assert_approx_eq!(f64, energy, 248.32404971035157);
    }

    #[test]
    fn test_sasa_two_molecules() {
        let context = Backend::new(
            "tests/files/sasa_two_molecules.yaml",
            None,
            &mut rand::thread_rng(),
        )
        .unwrap();

        let energy = context
            .hamiltonian()
            .energy(&context, &crate::Change::Everything);

        assert_approx_eq!(f64, energy, 426.557_354_952_765_3);
    }

    #[test]
    fn test_sasa_offset() {
        let context = Backend::new(
            "tests/files/sasa_interactions_offset.yaml",
            None,
            &mut rand::thread_rng(),
        )
        .unwrap();

        let energy = context
            .hamiltonian()
            .energy(&context, &crate::Change::Everything);

        assert_approx_eq!(f64, energy, 0.0);
    }
}
