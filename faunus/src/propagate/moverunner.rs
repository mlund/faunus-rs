// Copyright 2023-2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

use super::MoveProposal;
use crate::{
    energy::EnergyChange,
    montecarlo::{AcceptanceCriterion, MoveStatistics, NewOld},
    Context,
};
use core::fmt;
use rand::RngCore;

/// Wrapper that owns bookkeeping (statistics, weight, repeat) and delegates proposal logic.
pub struct MoveRunner<T: Context> {
    /// Send-bound required for Gibbs ensemble scoped threads
    inner: Box<dyn MoveProposal<T> + Send>,
    statistics: MoveStatistics,
    weight: f64,
    repeat: usize,
}

impl<T: Context> fmt::Debug for MoveRunner<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoveRunner")
            .field("inner", &self.inner)
            .field("statistics", &self.statistics)
            .field("weight", &self.weight)
            .field("repeat", &self.repeat)
            .finish()
    }
}

impl<T: Context> MoveRunner<T> {
    pub fn new(inner: Box<dyn MoveProposal<T> + Send>, weight: f64, repeat: usize) -> Self {
        Self {
            inner,
            statistics: MoveStatistics::default(),
            weight,
            repeat,
        }
    }

    pub const fn weight(&self) -> f64 {
        self.weight
    }

    #[allow(dead_code)] // read by tests
    pub const fn repeat(&self) -> usize {
        self.repeat
    }

    pub const fn statistics(&self) -> &MoveStatistics {
        &self.statistics
    }

    /// Perform the move: propose, evaluate energy, accept/reject. Repeats as configured.
    pub fn do_move(
        &mut self,
        context: &mut T,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut dyn RngCore,
    ) -> anyhow::Result<()> {
        for _ in 0..self.repeat {
            // The system may have changed since the move was built, or since its last trial.
            self.inner.revalidate(context)?;

            let Some(proposed) = self.inner.propose_move(context, rng) else {
                // Move couldn't be proposed (e.g. no feasible reaction) — count as rejected
                self.statistics.reject();
                continue;
            };

            let old_energy = context.hamiltonian().energy(context, proposed.change());
            // Save energy term backups while context still has old positions
            context.save_energy_backups(proposed.change());
            proposed.apply_with_backup(context)?;
            context.update(proposed.change())?;
            let new_energy = context.hamiltonian().energy(context, proposed.change());
            // Roll the trial back to the pre-move state before surfacing an unphysical energy, so the
            // context stays consistent (and the backup disarmed) even if the caller handles the error.
            if let Err(err) = crate::montecarlo::ensure_physical_energy(new_energy) {
                context.undo()?;
                return Err(err);
            }

            let energy = NewOld::<f64>::from(new_energy, old_energy);
            let bias = self.inner.bias(proposed.change(), &energy);

            let accepted = criterion.accept(energy, bias, thermal_energy, rng);
            if accepted {
                self.statistics
                    .accept(energy.difference(), proposed.displacement().clone());
                context.discard_backup();
            } else {
                self.statistics.reject();
                context.undo()?;
            }
            self.inner.on_trial_outcome(context, accepted);
        }

        *step += self.inner.step_by();
        Ok(())
    }

    /// Serialize to YAML, merging bookkeeping fields into the inner move's tagged output.
    pub fn to_yaml(&self) -> Option<yaml_serde::Value> {
        let tagged = self.inner.to_yaml()?;
        if let yaml_serde::Value::Tagged(mut tagged_value) = tagged {
            if let yaml_serde::Value::Mapping(ref mut map) = tagged_value.value {
                map.insert("weight".into(), self.weight.into());
                map.insert("repeat".into(), self.repeat.into());
                let mut stats = yaml_serde::to_value(&self.statistics).ok()?;
                if let yaml_serde::Value::Mapping(ref mut smap) = stats {
                    smap.insert(
                        "acceptance_ratio".into(),
                        self.statistics.acceptance_ratio().into(),
                    );
                }
                map.insert("statistics".into(), stats);
            }
            Some(yaml_serde::Value::Tagged(tagged_value))
        } else {
            Some(tagged)
        }
    }
}
