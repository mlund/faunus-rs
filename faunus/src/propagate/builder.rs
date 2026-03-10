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
    moveproposal::default_repeat, moverunner::MoveRunner, MoveCollection, PropagationBlock,
};
use crate::{montecarlo::AcceptanceCriterion, Context};
use serde::{Deserialize, Serialize};

/// All possible supported moves (used for YAML deserialization).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MoveBuilder {
    TranslateMolecule(crate::montecarlo::TranslateMolecule),
    TranslateAtom(crate::montecarlo::TranslateAtom),
    RotateMolecule(crate::montecarlo::RotateMolecule),
    VolumeMove(crate::montecarlo::VolumeMove),
    PivotMove(crate::montecarlo::PivotMove),
    CrankshaftMove(crate::montecarlo::CrankshaftMove),
    SpeciationMove(crate::montecarlo::SpeciationMove),
}

impl MoveBuilder {
    /// Finalize and validate the inner move, then wrap it in a `MoveRunner`.
    pub fn build<T: Context>(self, context: &T) -> anyhow::Result<MoveRunner<T>> {
        macro_rules! build_move {
            ($m:expr) => {{
                let mut m = $m;
                m.finalize(context)?;
                let (w, r) = (m.weight, m.repeat);
                MoveRunner::new(Box::new(m), w, r)
            }};
        }
        Ok(match self {
            Self::TranslateMolecule(m) => build_move!(m),
            Self::TranslateAtom(m) => build_move!(m),
            Self::RotateMolecule(m) => build_move!(m),
            Self::VolumeMove(m) => build_move!(m),
            Self::PivotMove(m) => build_move!(m),
            Self::CrankshaftMove(m) => build_move!(m),
            Self::SpeciationMove(m) => build_move!(m),
        })
    }
}

/// Shared builder for both stochastic and deterministic move collections.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CollectionBuilder {
    #[serde(default = "default_repeat")]
    pub(super) repeat: usize,
    #[serde(default)]
    pub(super) moves: Vec<MoveBuilder>,
}

impl CollectionBuilder {
    fn build_moves<T: Context>(self, context: &T) -> anyhow::Result<(usize, Vec<MoveRunner<T>>)> {
        let moves = self
            .moves
            .into_iter()
            .map(|m| m.build(context))
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
    pub(super) fn build<T: Context>(self, context: &T) -> anyhow::Result<PropagationBlock<T>> {
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
        let (repeat, moves) = builder.build_moves(context)?;
        Ok(PropagationBlock::MonteCarlo(MoveCollection::new(
            strategy, repeat, moves,
        )))
    }
}

/// Non-generic builder for deserialization; produces `Propagate<T>` via `build()`.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(super) struct PropagateBuilder {
    #[serde(rename = "repeat")]
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
    gibbs: Option<serde_yaml::Value>,
}

/// Seed used for selecting stochastic moves.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) enum Seed {
    #[default]
    Hardware,
    Fixed(usize),
}

/// How moves in a collection are selected during propagation.
#[derive(Clone, Copy, Debug)]
pub(super) enum SelectionStrategy {
    /// One move chosen per iteration via weighted random sampling.
    Stochastic,
    /// All moves executed in order each iteration.
    Deterministic,
}
