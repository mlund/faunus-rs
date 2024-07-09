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

//! Reading Monte Carlo moves and MD propagators.

use crate::{
    montecarlo::{Bias, Frequency, MoveStatistics, NewOld},
    Change, Context, Info,
};
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};

/// Collection of moves and their properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveCollection {
    selection: MovesSelection,
    repeat: Option<usize>,
    moves: Vec<Move>,
}

/// All possible supported moves.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Move {
    TranslateMolecule(crate::montecarlo::TranslateMolecule),
}

/// The method for selecting moves from the collection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum MovesSelection {
    Stochastic { seed: Seed },
    Deterministic,
}

/// Seed used for selecting stochastic moves.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub(crate) enum Seed {
    #[default]
    Hardware,
    Fixed(usize),
}

impl Move {
    /// Perform a move on given `context`.
    pub fn do_move(
        &mut self,
        context: &mut impl Context,
        step: &mut usize,
        rng: &mut ThreadRng,
    ) -> Option<Change> {
        match self {
            Move::TranslateMolecule(x) => x.do_move(context, step, rng),
        }
    }

    /// Moves may generate optional bias that should be added to the trial energy
    /// when determining the acceptance probability.
    /// It can also be used to force acceptance of a move in e.g. hybrid MD/MC schemes.
    /// By default, this returns `Bias::None`.
    #[allow(unused_variables)]
    pub fn bias(&self, change: &Change, energies: &NewOld<f64>) -> Bias {
        Bias::None
    }
    /// Get statistics for the move.
    pub fn statistics(&self) -> &MoveStatistics {
        match self {
            Move::TranslateMolecule(x) => x.statistics(),
        }
    }

    /// Get mutable statistics for the move.
    pub fn statistics_mut(&mut self) -> &mut MoveStatistics {
        match self {
            Move::TranslateMolecule(x) => x.statistics_mut(),
        }
    }

    /// Called when the move is accepted.
    ///
    /// This will update the statistics.
    #[allow(unused_variables)]
    pub fn accepted(&mut self, change: &Change, energy_change: f64) {
        self.statistics_mut().accept(energy_change);
    }

    /// Called when the move is rejected.
    ///
    /// This will update the statistics.
    #[allow(unused_variables)]
    pub fn rejected(&mut self, change: &Change) {
        self.statistics_mut().reject();
    }

    /// Get the frequency of the move.
    pub fn frequency(&self) -> Frequency {
        match self {
            Move::TranslateMolecule(x) => x.frequency(),
        }
    }

    /// Validate and finalize the move.
    fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        match self {
            Move::TranslateMolecule(x) => x.finalize(context),
        }
    }
}

impl Info for Move {
    fn short_name(&self) -> Option<&'static str> {
        match self {
            Move::TranslateMolecule(x) => x.short_name(),
        }
    }

    fn long_name(&self) -> Option<&'static str> {
        match self {
            Move::TranslateMolecule(x) => x.long_name(),
        }
    }
}

impl MoveCollection {
    /// Get the collection of moves from a yaml string.
    /// This method also requires a Context structure to validate and finalize the moves.
    pub fn from_string(string: &str, context: &impl Context) -> anyhow::Result<MoveCollection> {
        let mut collection: MoveCollection = serde_yaml::from_str(string)?;

        // validate and finalize all moves
        collection
            .moves
            .iter_mut()
            .try_for_each(|x| x.finalize(context))?;

        Ok(collection)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn propagators_collection_parse() {
        let string = "selection: !Stochastic {seed: !Fixed 748732827}
moves: [!TranslateMolecule {molecule: \"MOL2\", max_displacement: 0.3, frequency: !Probability 0.5}]
";

        let collection: MoveCollection = serde_yaml::from_str(string).unwrap();
        println!("{:?}", collection);
    }
}
