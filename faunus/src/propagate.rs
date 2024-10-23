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

use crate::{
    analysis::{AnalysisCollection, Analyze},
    energy::EnergyChange,
    montecarlo::{AcceptanceCriterion, Bias, MoveStatistics, NewOld},
    Change, Context, Info,
};
use core::fmt::Debug;
use rand::Rng;
use rand::{prelude::SliceRandom, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Specifies how many moves should be performed,
/// what moves can be performed and how they should be selected.
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Propagate {
    #[serde(rename = "repeat")]
    #[serde(default = "default_repeat")]
    /// Number of times the `Propagate` block should be performed.
    max_repeats: usize,
    #[serde(skip)]
    /// Current repeat.
    current_repeat: usize,
    #[serde(default)]
    /// Seed for the random number generator.
    seed: Seed,
    #[serde(skip)]
    /// Random number generator used for selecting the moves.
    rng: Option<StdRng>,
    #[serde(default)]
    #[serde(rename = "collections")]
    /// Collections of moves to be performed.
    move_collections: Vec<MoveCollection>,
    /// Acceptance criterion.
    #[serde(default)]
    criterion: AcceptanceCriterion,
}

/// Default value of `repeat` for various structures.
pub(crate) fn default_repeat() -> usize {
    1
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

    /// Get the `Propagate` structure from input yaml file.
    /// This also requires `Context` which is used to validate and finalize the individual moves.
    pub fn from_file(
        filename: impl AsRef<Path>,
        context: &impl Context,
    ) -> anyhow::Result<Propagate> {
        let yaml = std::fs::read_to_string(filename)?;
        let full: serde_yaml::Value = serde_yaml::from_str(&yaml)?;

        let mut current = &full;
        for key in ["propagate"] {
            current = match current.get(key) {
                Some(x) => x,
                None => anyhow::bail!("Could not find `{}` in the YAML file.", key),
            }
        }

        let mut propagate: Propagate =
            serde_yaml::from_value(current.clone()).map_err(anyhow::Error::msg)?;

        // finalize and validate the collections
        propagate
            .move_collections
            .iter_mut()
            .try_for_each(|x| x.finalize(context))?;

        // seed the random number generator
        match propagate.seed {
            Seed::Hardware => propagate.rng = Some(rand::SeedableRng::from_entropy()),
            Seed::Fixed(x) => propagate.rng = Some(rand::SeedableRng::seed_from_u64(x as u64)),
        }

        Ok(propagate)
    }

    pub fn get_collections(&self) -> &[MoveCollection] {
        &self.move_collections
    }
}

/// Collection of moves that should be stochastically selected in the simulation.
/// Only one move is selected per repeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StochasticCollection {
    /// How many moves should be selected per one propagate cycle.
    #[serde(default = "default_repeat")]
    repeat: usize,
    /// List of moves.
    #[serde(default)]
    moves: Vec<Move>,
}

impl StochasticCollection {
    /// Attempt to perform selected moves of the collection.
    fn propagate<C: Context>(
        &mut self,
        context: &mut NewOld<C>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut impl Rng,
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
}

/// Collection of moves that are deterministally selected during the simulation.
/// Multiple moves can be selected per repeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeterministicCollection {
    /// How many times the moves should be performed per one propagate cycle.
    #[serde(default = "default_repeat")]
    repeat: usize,
    /// List of moves.
    #[serde(default)]
    moves: Vec<Move>,
}

impl DeterministicCollection {
    /// Attempt to perform all moves of the collection.
    fn propagate<C: Context>(
        &mut self,
        context: &mut NewOld<C>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut impl Rng,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MoveCollection {
    Stochastic(StochasticCollection),
    Deterministic(DeterministicCollection),
}

impl MoveCollection {
    /// Select move from the `MoveCollection` and perform it. Repeat if requested.
    pub(crate) fn propagate<C: Context>(
        &mut self,
        context: &mut NewOld<C>,
        criterion: &AcceptanceCriterion,
        thermal_energy: f64,
        step: &mut usize,
        rng: &mut impl Rng,
        analyses: &mut AnalysisCollection<C>,
    ) -> anyhow::Result<()> {
        match self {
            Self::Stochastic(x) => {
                x.propagate(context, criterion, thermal_energy, step, rng, analyses)
            }
            Self::Deterministic(x) => {
                x.propagate(context, criterion, thermal_energy, step, rng, analyses)
            }
        }
    }

    /// Get mutable reference to the moves of the collection.
    pub(crate) fn get_moves_mut(&mut self) -> &mut [Move] {
        match self {
            Self::Stochastic(x) => &mut x.moves,
            Self::Deterministic(x) => &mut x.moves,
        }
    }

    /// Get immutable reference to the moves of the collection.
    pub fn get_moves(&self) -> &[Move] {
        match self {
            Self::Stochastic(x) => &x.moves,
            Self::Deterministic(x) => &x.moves,
        }
    }

    /// Finalize and validate moves of a collection.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.get_moves_mut()
            .iter_mut()
            .try_for_each(|x| x.finalize(context))?;

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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) enum Seed {
    #[default]
    Hardware,
    Fixed(usize),
}

/// All possible supported moves.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Move {
    TranslateMolecule(crate::montecarlo::TranslateMolecule),
    TranslateAtom(crate::montecarlo::TranslateAtom),
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
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        for _ in 0..self.repeat() {
            let change = self
                .propose_move(&mut context.new, rng)
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

        // repeated moves only increase the step counter by 1
        *step += self.step_by();

        Ok(())
    }

    /// Propose a move on the given `context`.
    /// This modifies the context and returns the proposed change.
    fn propose_move(&mut self, context: &mut impl Context, rng: &mut impl Rng) -> Option<Change> {
        match self {
            Move::TranslateMolecule(x) => x.propose_move(context, rng),
            Move::TranslateAtom(x) => x.propose_move(context, rng),
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
    #[allow(dead_code)]
    pub fn get_statistics(&self) -> &MoveStatistics {
        match self {
            Move::TranslateMolecule(x) => x.get_statistics(),
            Move::TranslateAtom(x) => x.get_statistics(),
        }
    }

    /// Get mutable statistics for the move.
    pub(crate) fn get_statistics_mut(&mut self) -> &mut MoveStatistics {
        match self {
            Move::TranslateMolecule(x) => x.get_statistics_mut(),
            Move::TranslateAtom(x) => x.get_statistics_mut(),
        }
    }

    /// Called when the move is accepted.
    ///
    /// This will update the statistics.
    #[allow(unused_variables)]
    fn accepted(&mut self, change: &Change, energy_change: f64) {
        self.get_statistics_mut().accept(energy_change);
    }

    /// Called when the move is rejected.
    ///
    /// This will update the statistics.
    #[allow(unused_variables)]
    fn rejected(&mut self, change: &Change) {
        self.get_statistics_mut().reject();
    }

    /// Get the weight of the move.
    pub fn weight(&self) -> f64 {
        match self {
            Move::TranslateMolecule(x) => x.weight(),
            Move::TranslateAtom(x) => x.weight(),
        }
    }

    /// Validate and finalize the move.
    fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        match self {
            Move::TranslateMolecule(x) => x.finalize(context),
            Move::TranslateAtom(x) => x.finalize(context),
        }
    }

    /// How many times the move should be repeated upon selection.
    pub fn repeat(&self) -> usize {
        match self {
            Move::TranslateMolecule(x) => x.repeat(),
            Move::TranslateAtom(x) => x.repeat(),
        }
    }

    /// The number of steps to move forward after attempting the move.
    pub fn step_by(&self) -> usize {
        match self {
            Move::TranslateMolecule(_) => 1,
            Move::TranslateAtom(_) => 1,
        }
    }
}

impl Info for Move {
    fn short_name(&self) -> Option<&'static str> {
        match self {
            Move::TranslateMolecule(x) => x.short_name(),
            Move::TranslateAtom(x) => x.short_name(),
        }
    }

    fn long_name(&self) -> Option<&'static str> {
        match self {
            Move::TranslateMolecule(x) => x.long_name(),
            Move::TranslateAtom(x) => x.long_name(),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::platform::reference::ReferencePlatform;

    use super::*;

    #[test]
    fn seed_parse() {
        let string = "!Fixed 49786352";
        let seed: Seed = serde_yaml::from_str(string).unwrap();
        matches!(seed, Seed::Fixed(49786352));

        let string = "Hardware";
        let seed: Seed = serde_yaml::from_str(string).unwrap();
        matches!(seed, Seed::Hardware);
    }

    #[test]
    fn stochastic_parse() {
        let string = "repeat: 20
moves:
   - !TranslateMolecule { molecule: Water, dp: 0.4, weight: 1.0 }
   - !TranslateMolecule { molecule: Protein, dp: 0.6, weight: 2.0 }
   - !TranslateMolecule { molecule: Lipid, dp: 0.5, weight: 0.5 }";

        let collection: StochasticCollection = serde_yaml::from_str(string).unwrap();
        assert_eq!(collection.repeat, 20);
        assert_eq!(collection.moves.len(), 3);
    }

    #[test]
    fn deterministic_parse() {
        // TODO: replace with some actual deterministic moves
        let string = "repeat: 10
moves:
   - !TranslateMolecule { molecule: Water, dp: 0.4, weight: 1.0 }
   - !TranslateMolecule { molecule: Protein, dp: 0.6, weight: 2.0 }
   - !TranslateMolecule { molecule: Lipid, dp: 0.5, weight: 0.5 }";

        let collection: StochasticCollection = serde_yaml::from_str(string).unwrap();
        assert_eq!(collection.repeat, 10);
        assert_eq!(collection.moves.len(), 3);
    }

    #[test]
    fn propagate_parse() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();
        let propagate = Propagate::from_file("tests/files/topology_pass.yaml", &context).unwrap();

        assert_eq!(propagate.max_repeats, 10000);
        assert_eq!(propagate.seed, Seed::Hardware);
        assert_eq!(propagate.current_repeat, 0);
        assert_eq!(propagate.criterion, AcceptanceCriterion::MetropolisHastings);
        assert_eq!(propagate.move_collections.len(), 2);
        let stochastic_collection = match propagate.move_collections[0].clone() {
            MoveCollection::Stochastic(x) => x,
            _ => panic!("Invalid Move Collection variant."),
        };

        assert_eq!(stochastic_collection.repeat, 1);
        assert_eq!(stochastic_collection.moves.len(), 3);
        let stochastic_move_1 = stochastic_collection.moves[0].clone();
        assert_eq!(stochastic_move_1.repeat(), 2);
        assert_eq!(stochastic_move_1.weight(), 0.5);
        let stochastic_move_2 = stochastic_collection.moves[1].clone();
        assert_eq!(stochastic_move_2.repeat(), 1);
        assert_eq!(stochastic_move_2.weight(), 1.0);

        let deterministic_collection = match propagate.move_collections[1].clone() {
            MoveCollection::Deterministic(x) => x,
            _ => panic!("Invalid Move Collection variant."),
        };

        assert_eq!(deterministic_collection.repeat, 5);
        assert_eq!(deterministic_collection.moves.len(), 1);
    }

    #[test]
    fn propagate_parse_fail() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_invalid_propagate.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        assert!(
            Propagate::from_file("tests/files/topology_invalid_propagate.yaml", &context).is_err()
        );
    }

    #[test]
    fn propagate_translate_atom_parse_fail1() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_invalid_translate_atom1.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        assert!(Propagate::from_file(
            "tests/files/topology_invalid_translate_atom1.yaml",
            &context
        )
        .is_err());
    }

    #[test]
    fn propagate_translate_atom_parse_fail2() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_invalid_translate_atom2.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        assert!(Propagate::from_file(
            "tests/files/topology_invalid_translate_atom2.yaml",
            &context
        )
        .is_err());
    }

    #[test]
    fn propagate_translate_atom_parse_fail3() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_invalid_translate_atom3.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        assert!(Propagate::from_file(
            "tests/files/topology_invalid_translate_atom3.yaml",
            &context
        )
        .is_err());
    }
}
