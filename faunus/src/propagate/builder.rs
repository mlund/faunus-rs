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

#[cfg(feature = "gpu")]
use super::langevin::{LangevinConfig, LangevinRunner};
use super::{
    moveproposal::default_repeat, moverunner::MoveRunner, MoveCollection, MoveProposal,
    PropagationBlock,
};
use crate::{montecarlo::AcceptanceCriterion, Context};
use serde::{Deserialize, Serialize};

/// Shared contract for MC moves: all have `weight`, `repeat`, and `finalize`.
/// Enforces at compile time what was previously an implicit field-name convention
/// relied upon by a macro.
pub(crate) trait BuildableMove<T: Context>:
    MoveProposal<T> + Send + Sized + 'static
{
    /// `thermal_energy` is the system RT in **kJ/mol**, the same quantity the acceptance
    /// criterion divides by; a move that biases the acceptance must use it, not a
    /// temperature of its own.
    fn finalize(&mut self, context: &T, thermal_energy: f64) -> anyhow::Result<()>;
    fn weight(&self) -> f64;
    fn repeat(&self) -> usize;

    /// `thermal_energy` is the system RT in kJ/mol.
    fn into_runner(mut self, context: &T, thermal_energy: f64) -> anyhow::Result<MoveRunner<T>> {
        self.finalize(context, thermal_energy)?;
        let (w, r) = (self.weight(), self.repeat());
        Ok(MoveRunner::new(Box::new(self), w, r))
    }
}

/// All possible supported moves (used for YAML deserialization).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MoveBuilder {
    TranslateMolecule(crate::montecarlo::TranslateMolecule),
    TranslateAtom(Box<crate::montecarlo::TranslateAtom>),
    RotateMolecule(crate::montecarlo::RotateMolecule),
    VolumeMove(crate::montecarlo::VolumeMove),
    PivotMove(crate::montecarlo::PivotMove),
    CrankshaftMove(crate::montecarlo::CrankshaftMove),
    SpeciationMove(crate::montecarlo::SpeciationMove),
    ClusterMove(crate::montecarlo::ClusterMove),
}

impl MoveBuilder {
    /// Finalize and validate the inner move, then wrap it in a `MoveRunner`.
    ///
    /// `thermal_energy` is the system RT in kJ/mol.
    pub fn build<T: Context>(
        self,
        context: &T,
        thermal_energy: f64,
    ) -> anyhow::Result<MoveRunner<T>> {
        match self {
            Self::TranslateMolecule(m) => m.into_runner(context, thermal_energy),
            Self::TranslateAtom(m) => (*m).into_runner(context, thermal_energy),
            Self::RotateMolecule(m) => m.into_runner(context, thermal_energy),
            Self::VolumeMove(m) => m.into_runner(context, thermal_energy),
            Self::PivotMove(m) => m.into_runner(context, thermal_energy),
            Self::CrankshaftMove(m) => m.into_runner(context, thermal_energy),
            Self::SpeciationMove(m) => m.into_runner(context, thermal_energy),
            Self::ClusterMove(m) => m.into_runner(context, thermal_energy),
        }
    }
}

/// Shared builder for both stochastic and deterministic move collections.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CollectionBuilder {
    #[serde(rename = "cycles")]
    #[serde(default = "default_repeat")]
    pub(super) repeat: usize,
    #[serde(default)]
    pub(super) moves: Vec<MoveBuilder>,
}

impl CollectionBuilder {
    fn build_moves<T: Context>(
        self,
        context: &T,
        thermal_energy: f64,
    ) -> anyhow::Result<(usize, Vec<MoveRunner<T>>)> {
        let moves = self
            .moves
            .into_iter()
            .map(|m| m.build(context, thermal_energy))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok((self.repeat, moves))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum MoveCollectionBuilder {
    Stochastic(CollectionBuilder),
    Deterministic(CollectionBuilder),
    #[cfg(feature = "gpu")]
    LangevinDynamics(LangevinConfig),
}

impl MoveCollectionBuilder {
    pub(super) fn build<T: Context>(
        self,
        context: &T,
        thermal_energy: f64,
    ) -> anyhow::Result<PropagationBlock<T>> {
        let (strategy, builder) = match self {
            #[cfg(feature = "gpu")]
            Self::LangevinDynamics(config) => {
                return Ok(PropagationBlock::LangevinDynamics(Box::new(
                    LangevinRunner::new(config),
                )));
            }
            Self::Stochastic(b) => (SelectionStrategy::Stochastic, b),
            Self::Deterministic(b) => (SelectionStrategy::Deterministic, b),
        };
        let (repeat, moves) = builder.build_moves(context, thermal_energy)?;
        Ok(PropagationBlock::MonteCarlo(MoveCollection::new(
            strategy, repeat, moves,
        )))
    }
}

/// Non-generic builder for deserialization; produces `Propagate<T>` via `build()`.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(super) struct PropagateBuilder {
    #[serde(rename = "steps")]
    #[serde(default = "default_repeat")]
    pub(super) max_repeats: usize,
    #[serde(default)]
    pub(super) seed: Seed,
    #[serde(default)]
    #[serde(rename = "collections")]
    pub(super) move_collections: Vec<MoveCollectionBuilder>,
    #[serde(default)]
    pub(super) criterion: AcceptanceCriterion,
    /// Present so `deny_unknown_fields` accepts the `gibbs` key; parsed separately.
    #[serde(default)]
    #[allow(dead_code)]
    gibbs: Option<yaml_serde::Value>,
}

/// Seed used for selecting stochastic moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) enum Seed {
    #[default]
    Hardware,
    Fixed(usize),
}

impl Seed {
    /// The master generator for a run. Every other stream is derived from this
    /// one so that a single `seed` reproduces the whole simulation.
    pub(crate) fn build_rng(self) -> rand::rngs::StdRng {
        match self {
            Self::Hardware => rand::SeedableRng::from_entropy(),
            Self::Fixed(x) => rand::SeedableRng::seed_from_u64(x as u64),
        }
    }
}

/// How moves in a collection are selected during propagation.
#[derive(Clone, Copy, Debug)]
pub(super) enum SelectionStrategy {
    /// One move chosen per iteration via weighted random sampling.
    Stochastic,
    /// All moves executed in order each iteration.
    Deterministic,
}
