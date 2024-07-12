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
    analysis::{AnalysisCollection, Analyze},
    energy::EnergyChange,
    montecarlo::{AcceptanceCriterion, Bias, MoveStatistics, NewOld},
    Change, Context, Info,
};
use core::fmt::Debug;
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};

/// All possible supported moves.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Move {
    TranslateMolecule(crate::montecarlo::TranslateMolecule),
}

/// Specifies how many moves should be performed,
/// what moves can be performed and how they should be selected.
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Propagate {
    #[serde(rename = "repeat")]
    #[serde(default)]
    max_repeats: usize,
    #[serde(skip)]
    current_repeat: usize,
    #[serde(default)]
    seed: Seed,
    #[serde(skip)]
    rng: ThreadRng,
    #[serde(default)]
    move_collections: Vec<MoveCollection>,
}

impl Propagate {
    /// Perform one 'propagate' cycle.
    ///
    /// ## Returns
    /// - `true` if the cycle was performed successfully and the simulation should continue.
    /// - `false` if the simulation is finished.
    /// - Error if some issue occured.
    pub fn propagate<C: Context>(
        &mut self,
        context: &mut NewOld<C>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        analyses: &mut AnalysisCollection<C>,
    ) -> anyhow::Result<bool> {
        if self.current_repeat >= self.max_repeats {
            return Ok(false);
        }

        for collection in self.move_collections.iter_mut() {
            collection.propagate(
                context,
                criterion,
                thermal_energy,
                step,
                &mut self.rng,
                analyses,
            )?;
        }

        Ok(true)
    }
}

/// Collection of moves and their properties.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveCollection {
    /// The way moves should be selected from the collection.
    selection: MovesSelection,
    /// How many moves should be selected per one propagate cycle.
    #[serde(default = "default_repeat")]
    repeat: usize,
    /// List of moves.
    moves: Vec<Move>,
}

/// Default value of `repeat` for the `MoveCollection`.
fn default_repeat() -> usize {
    1
}

impl MoveCollection {
    /// Select move from the `MoveCollection` and perform it. Repeat if requested.
    pub(crate) fn propagate<C: Context>(
        &mut self,
        context: &mut NewOld<C>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut ThreadRng,
        analyses: &mut AnalysisCollection<C>,
    ) -> anyhow::Result<()> {
        for _ in 0..self.repeat {
            match self.selection {
                MovesSelection::Stochastic => self.propagate_stochastic(
                    context,
                    criterion,
                    thermal_energy,
                    step,
                    rng,
                    analyses,
                ),
                MovesSelection::Deterministic => self.propagate_deterministic(
                    context,
                    criterion,
                    thermal_energy,
                    step,
                    rng,
                    analyses,
                ),
            }?
        }

        Ok(())
    }

    /// Attempt to perform selected moves of the collection.
    fn propagate_stochastic<C: Context>(
        &mut self,
        context: &mut NewOld<C>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut ThreadRng,
        analyses: &mut AnalysisCollection<C>,
    ) -> anyhow::Result<()> {
        for _ in 0..self.repeat {
            let selected = self.moves.choose_weighted_mut(rng, |mv| mv.weight())?;
            selected.do_move(context, criterion, thermal_energy, step, rng)?;

            // perform analyses
            match analyses.sample(&context.old, *step) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Attempt to perform all moves of the collection.
    fn propagate_deterministic<C: Context>(
        &mut self,
        context: &mut NewOld<C>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut ThreadRng,
        analyses: &mut AnalysisCollection<C>,
    ) -> anyhow::Result<()> {
        for _ in 0..self.repeat {
            for mv in self.moves.iter_mut() {
                mv.do_move(context, criterion, thermal_energy, step, rng)?;

                // perform analyses
                match analyses.sample(&context.old, *step) {
                    Ok(_) => (),
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(())
    }
}

/// The method for selecting moves from the collection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum MovesSelection {
    Stochastic,
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
    /// Attempts to perform the move.
    /// Consists of proposing the move, accepting/rejecting it and updating the context.
    /// This process is repeated N times, depending on the characteristics of the move.
    fn do_move(
        &mut self,
        context: &mut NewOld<impl Context>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<()> {
        for _ in 0..self.repeat() {
            let change = self
                .propose_move(&mut context.new, step, rng)
                .ok_or(anyhow::anyhow!("Could not propose a move."))?;
            context.new.update(&change)?;

            let energy = NewOld::<f64>::from(
                context.new.hamiltonian().energy(&context.new, &change),
                context.old.hamiltonian().energy(&context.old, &change),
            );
            let bias = self.bias(&change, &energy);

            if criterion.accept(energy, bias, thermal_energy, rng) {
                self.accepted(&change, energy.difference());
                context.old.sync_from(&context.new, &change)?;
            } else {
                self.rejected(&change);
                context.new.sync_from(&context.old, &change)?;
            }
        }

        *step += self.step_by();

        Ok(())
    }

    /// Propose a move on the given `context`.
    /// This modifies the context and returns the proposed change.
    fn propose_move(
        &mut self,
        context: &mut impl Context,
        step: &mut usize,
        rng: &mut ThreadRng,
    ) -> Option<Change> {
        match self {
            Move::TranslateMolecule(x) => x.propose_move(context, step, rng),
        }
    }

    /// Moves may generate optional bias that should be added to the trial energy
    /// when determining the acceptance probability.
    /// It can also be used to force acceptance of a move in e.g. hybrid MD/MC schemes.
    /// By default, this returns `Bias::None`.
    #[allow(unused_variables)]
    fn bias(&self, change: &Change, energies: &NewOld<f64>) -> Bias {
        Bias::None
    }
    /// Get statistics for the move.
    fn statistics(&self) -> &MoveStatistics {
        match self {
            Move::TranslateMolecule(x) => x.statistics(),
        }
    }

    /// Get mutable statistics for the move.
    fn statistics_mut(&mut self) -> &mut MoveStatistics {
        match self {
            Move::TranslateMolecule(x) => x.statistics_mut(),
        }
    }

    /// Called when the move is accepted.
    ///
    /// This will update the statistics.
    #[allow(unused_variables)]
    fn accepted(&mut self, change: &Change, energy_change: f64) {
        self.statistics_mut().accept(energy_change);
    }

    /// Called when the move is rejected.
    ///
    /// This will update the statistics.
    #[allow(unused_variables)]
    fn rejected(&mut self, change: &Change) {
        self.statistics_mut().reject();
    }

    /// Get the weight of the move.
    fn weight(&self) -> f64 {
        match self {
            Move::TranslateMolecule(x) => x.weight(),
        }
    }

    /// Validate and finalize the move.
    fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        match self {
            Move::TranslateMolecule(x) => x.finalize(context),
        }
    }

    /// How many times the move should be repeated upon selection.
    fn repeat(&self) -> usize {
        match self {
            Move::TranslateMolecule(_) => 1,
        }
    }

    /// The number of steps to move forward after attempting the move.
    fn step_by(&self) -> usize {
        match self {
            Move::TranslateMolecule(_) => 1,
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
