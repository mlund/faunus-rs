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

//! Monte Carlo moves and MD propagators.

mod builder;
mod langevin;
mod moveproposal;
mod moverunner;
#[cfg(test)]
mod tests;

pub use builder::MoveBuilder;
pub use langevin::{LangevinConfig, LangevinRunner};
pub(crate) use moveproposal::{default_repeat, default_weight, tagged_yaml};
pub use moveproposal::{Displacement, MoveProposal, MoveTarget, ProposedMove};
pub use moverunner::MoveRunner;

use crate::{
    analysis::{AnalysisCollection, Analyze},
    energy::EnergyChange,
    montecarlo::AcceptanceCriterion,
    Context,
};
use builder::{PropagateBuilder, Seed, SelectionStrategy};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::Rng;
use std::path::Path;

/// A collection of moves with a selection strategy and repeat count.
#[derive(Debug)]
pub struct MoveCollection<T: Context> {
    strategy: SelectionStrategy,
    repeat: usize,
    moves: Vec<MoveRunner<T>>,
    elapsed: std::time::Duration,
}

impl<T: Context> MoveCollection<T> {
    fn new(strategy: SelectionStrategy, repeat: usize, moves: Vec<MoveRunner<T>>) -> Self {
        Self {
            strategy,
            repeat,
            moves,
            elapsed: std::time::Duration::default(),
        }
    }

    pub(crate) fn propagate(
        &mut self,
        context: &mut T,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut impl Rng,
        analyses: &mut AnalysisCollection<T>,
    ) -> anyhow::Result<()> {
        for _ in 0..self.repeat {
            match self.strategy {
                SelectionStrategy::Stochastic => {
                    let selected = self.moves.choose_weighted_mut(rng, |mv| mv.weight())?;
                    selected.do_move(context, criterion, thermal_energy, step, rng)?;
                    analyses.sample(context, *step)?;
                }
                SelectionStrategy::Deterministic => {
                    for mv in self.moves.iter_mut() {
                        mv.do_move(context, criterion, thermal_energy, step, rng)?;
                        analyses.sample(context, *step)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn moves(&self) -> &[MoveRunner<T>] {
        &self.moves
    }

    pub const fn repeat(&self) -> usize {
        self.repeat
    }

    fn to_yaml(&self) -> serde_yaml::Value {
        let tag = match self.strategy {
            SelectionStrategy::Stochastic => "Stochastic",
            SelectionStrategy::Deterministic => "Deterministic",
        };
        let mut map = serde_yaml::Mapping::new();
        map.insert("repeat".into(), self.repeat.into());
        map.insert(
            "elapsed_seconds".into(),
            serde_yaml::Value::Number(serde_yaml::Number::from(self.elapsed.as_secs_f64())),
        );
        let moves: Vec<_> = self.moves.iter().filter_map(|m| m.to_yaml()).collect();
        map.insert("moves".into(), serde_yaml::Value::Sequence(moves));
        serde_yaml::Value::Tagged(Box::new(serde_yaml::value::TaggedValue {
            tag: serde_yaml::value::Tag::new(tag),
            value: serde_yaml::Value::Mapping(map),
        }))
    }
}

/// A single block in the propagation loop: either MC moves or Langevin dynamics.
#[derive(Debug)]
pub enum PropagationBlock<T: Context> {
    MonteCarlo(MoveCollection<T>),
    LangevinDynamics(Box<LangevinRunner>),
}

impl<T: Context> PropagationBlock<T> {
    /// Cumulative wall-clock time spent in this block.
    pub fn elapsed(&self) -> std::time::Duration {
        match self {
            Self::MonteCarlo(mc) => mc.elapsed,
            Self::LangevinDynamics(ld) => ld.elapsed,
        }
    }

    pub(crate) fn propagate(
        &mut self,
        context: &mut T,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut impl Rng,
        analyses: &mut AnalysisCollection<T>,
    ) -> anyhow::Result<()> {
        let t0 = std::time::Instant::now();
        let result = match self {
            Self::MonteCarlo(mc) => {
                mc.propagate(context, criterion, thermal_energy, step, rng, analyses)
            }
            Self::LangevinDynamics(ld) => {
                let energy_before = context
                    .hamiltonian()
                    .energy(context, &crate::Change::Everything);
                ld.propagate(context)?;
                let energy_after = context
                    .hamiltonian()
                    .energy(context, &crate::Change::Everything);
                ld.energy_change_sum += energy_after - energy_before;
                // Sample analyses after positions are written back to context
                analyses.sample(context, *step)?;
                *step += ld.config.steps;
                Ok(())
            }
        };
        match self {
            Self::MonteCarlo(mc) => mc.elapsed += t0.elapsed(),
            Self::LangevinDynamics(ld) => ld.elapsed += t0.elapsed(),
        }
        result
    }

    /// Access the MC moves in this block (empty slice for LD blocks).
    pub fn moves(&self) -> &[MoveRunner<T>] {
        match self {
            Self::MonteCarlo(mc) => mc.moves(),
            Self::LangevinDynamics(_) => &[],
        }
    }

    pub fn repeat(&self) -> usize {
        match self {
            Self::MonteCarlo(mc) => mc.repeat(),
            Self::LangevinDynamics(ld) => ld.config.steps,
        }
    }

    fn to_yaml(&self) -> serde_yaml::Value {
        match self {
            Self::MonteCarlo(mc) => mc.to_yaml(),
            Self::LangevinDynamics(ld) => ld.to_yaml(),
        }
    }
}

/// Specifies how many moves should be performed,
/// what moves can be performed and how they should be selected.
#[derive(Debug)]
pub struct Propagate<T: Context> {
    max_repeats: usize,
    current_repeat: usize,
    seed: Seed,
    rng: Option<StdRng>,
    blocks: Vec<PropagationBlock<T>>,
    criterion: AcceptanceCriterion,
}

impl<T: Context> Propagate<T> {
    /// Perform one 'propagate' cycle.
    ///
    /// Returns `true` if the simulation should continue, `false` if finished.
    pub fn propagate(
        &mut self,
        context: &mut T,
        thermal_energy: f64,
        step: &mut usize,
        analyses: &mut AnalysisCollection<T>,
    ) -> anyhow::Result<bool> {
        if self.current_repeat >= self.max_repeats {
            return Ok(false);
        }

        for block in self.blocks.iter_mut() {
            block.propagate(
                context,
                &self.criterion,
                thermal_energy,
                step,
                self.rng
                    .as_mut()
                    .expect("Random number generator should already be seeded."),
                analyses,
            )?;
        }

        self.current_repeat += 1;
        Ok(true)
    }

    /// Build a `Propagate<T>` from an input YAML file.
    pub fn from_file(filename: impl AsRef<Path>, context: &T) -> anyhow::Result<Self> {
        let yaml = std::fs::read_to_string(filename)?;
        let full: serde_yaml::Value = serde_yaml::from_str(&yaml)?;

        let current = full
            .get("propagate")
            .ok_or_else(|| anyhow::anyhow!("Could not find `propagate` in the YAML file."))?;

        let builder: PropagateBuilder =
            serde_yaml::from_value(current.clone()).map_err(anyhow::Error::msg)?;

        let blocks = builder
            .move_collections
            .into_iter()
            .map(|c| c.build(context))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let rng = match builder.seed {
            Seed::Hardware => Some(rand::SeedableRng::from_entropy()),
            Seed::Fixed(x) => Some(rand::SeedableRng::seed_from_u64(x as u64)),
        };

        Ok(Self {
            max_repeats: builder.max_repeats,
            current_repeat: 0,
            seed: builder.seed,
            rng,
            blocks,
            criterion: builder.criterion,
        })
    }

    pub fn blocks(&self) -> &[PropagationBlock<T>] {
        &self.blocks
    }

    /// Total accumulated energy change from all propagation blocks.
    pub fn energy_change_sum(&self) -> f64 {
        self.blocks
            .iter()
            .map(|b| match b {
                PropagationBlock::MonteCarlo(mc) => mc
                    .moves()
                    .iter()
                    .map(|m| m.statistics().energy_change_sum)
                    .sum(),
                PropagationBlock::LangevinDynamics(ld) => ld.energy_change_sum,
            })
            .sum()
    }

    pub const fn max_repeats(&self) -> usize {
        self.max_repeats
    }

    /// Replace the internal RNG with one seeded from `seed`.
    /// Used to give each Gibbs box a unique seed.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Some(rand::SeedableRng::seed_from_u64(seed));
    }

    /// Serialize the propagate state to a YAML value.
    pub fn to_yaml(&self) -> serde_yaml::Value {
        let mut map = serde_yaml::Mapping::new();
        map.insert("repeat".into(), self.max_repeats.into());
        map.insert(
            "seed".into(),
            serde_yaml::to_value(&self.seed).unwrap_or_default(),
        );
        let collections: Vec<_> = self.blocks.iter().map(|b| b.to_yaml()).collect();
        map.insert(
            "collections".into(),
            serde_yaml::Value::Sequence(collections),
        );
        map.insert(
            "criterion".into(),
            serde_yaml::to_value(self.criterion).unwrap_or_default(),
        );
        serde_yaml::Value::Mapping(map)
    }
}

/// Parse the optional `propagate.gibbs` section from an input YAML file.
pub fn gibbs_config_from_file(
    filename: impl AsRef<Path>,
) -> anyhow::Result<Option<crate::montecarlo::gibbs::GibbsConfig>> {
    let yaml = std::fs::read_to_string(filename)?;
    let full: serde_yaml::Value = serde_yaml::from_str(&yaml)?;
    let Some(gibbs_value) = full.get("propagate").and_then(|p| p.get("gibbs")) else {
        return Ok(None);
    };
    let config = serde_yaml::from_value(gibbs_value.clone())?;
    Ok(Some(config))
}
