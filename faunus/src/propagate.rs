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
    transform::Transform,
    Change, Context, Info, Point,
};
use core::fmt::{self, Debug};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default value of `repeat` for various structures.
pub(crate) const fn default_repeat() -> usize {
    1
}

/// Target for a proposed Monte Carlo move.
#[derive(Clone, Debug)]
pub enum MoveTarget {
    /// Apply to a single group.
    Group(usize),
    /// Apply to the entire system.
    System,
}

/// A fully described but unapplied Monte Carlo move.
#[derive(Clone, Debug)]
pub struct ProposedMove {
    pub change: Change,
    pub displacement: Displacement,
    pub transform: Transform,
    pub target: MoveTarget,
}

impl ProposedMove {
    /// Apply the transform to the context, saving backup for undo.
    pub fn apply_with_backup(&self, context: &mut impl Context) -> anyhow::Result<()> {
        match self.target {
            MoveTarget::Group(i) => self.transform.on_group_with_backup(i, context),
            MoveTarget::System => self.transform.on_system_with_backup(context),
        }
    }
}

/// Narrow trait for the unique logic of each Monte Carlo move.
pub trait MoveProposal<T: Context>: Debug + Info {
    /// Describe a move without applying it; context is read-only.
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove>;

    /// Optional bias added to the trial energy for acceptance.
    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> Bias {
        Bias::None
    }

    /// Number of steps to advance after attempting the move.
    fn step_by(&self) -> usize {
        1
    }

    /// Serialize the move-specific fields to a tagged YAML value.
    fn to_yaml(&self) -> Option<serde_yaml::Value>;
}

/// Wrap a serializable value in a YAML tag.
pub(crate) fn tagged_yaml(tag: &str, value: &impl Serialize) -> Option<serde_yaml::Value> {
    let value = serde_yaml::to_value(value).ok()?;
    Some(serde_yaml::Value::Tagged(Box::new(
        serde_yaml::value::TaggedValue {
            tag: serde_yaml::value::Tag::new(tag),
            value,
        },
    )))
}

/// Wrapper that owns bookkeeping (statistics, weight, repeat) and delegates proposal logic.
pub struct MoveRunner<T: Context> {
    inner: Box<dyn MoveProposal<T>>,
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
    pub fn new(inner: Box<dyn MoveProposal<T>>, weight: f64, repeat: usize) -> Self {
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
            let proposed = self
                .inner
                .propose_move(context, rng)
                .ok_or_else(|| anyhow::anyhow!("Could not propose a move."))?;

            let old_energy = context.hamiltonian().energy(context, &proposed.change);
            proposed.apply_with_backup(context)?;
            context.update_with_backup(&proposed.change)?;
            let new_energy = context.hamiltonian().energy(context, &proposed.change);

            let energy = NewOld::<f64>::from(new_energy, old_energy);
            let bias = self.inner.bias(&proposed.change, &energy);

            if criterion.accept(energy, bias, thermal_energy, rng) {
                self.statistics
                    .accept(energy.difference(), proposed.displacement);
                context.discard_backup();
            } else {
                self.statistics.reject();
                context.undo()?;
            }
        }

        *step += self.inner.step_by();
        Ok(())
    }

    /// Serialize to YAML, merging bookkeeping fields into the inner move's tagged output.
    pub fn to_yaml(&self) -> Option<serde_yaml::Value> {
        let tagged = self.inner.to_yaml()?;
        if let serde_yaml::Value::Tagged(mut tagged_value) = tagged {
            if let serde_yaml::Value::Mapping(ref mut map) = tagged_value.value {
                map.insert("weight".into(), self.weight.into());
                map.insert("repeat".into(), self.repeat.into());
                map.insert(
                    "statistics".into(),
                    serde_yaml::to_value(&self.statistics).ok()?,
                );
            }
            Some(serde_yaml::Value::Tagged(tagged_value))
        } else {
            Some(tagged)
        }
    }
}

/// Enum used to store the extent of displacement of a move.
///
/// This is used for collecting statistics about for far moves change
/// the system. Used to track mean squared displacements.
#[derive(Clone, Debug)]
pub enum Displacement {
    /// Displacement vector; typically due to a translation
    Distance(Point),
    /// Angular displacement; typically due to a rotation
    Angle(f64),
    /// Displacement vector and angular displacement; typically due to a rototranslational move
    AngleDistance(f64, Point),
    /// A custom displacement
    Custom(f64),
    /// Zero displacement - typically used for rejected moves
    Zero,
    /// No displacement appropriate
    None,
}

impl TryFrom<Displacement> for f64 {
    type Error = &'static str;
    fn try_from(value: Displacement) -> Result<Self, Self::Error> {
        match value {
            Displacement::Distance(x) => Ok(x.norm()),
            Displacement::Angle(x) => Ok(x),
            Displacement::Custom(x) => Ok(x),
            Displacement::Zero => Ok(0.0),
            _ => Err("Cannot convert displacement to floating point number"),
        }
    }
}

/// All possible supported moves (used for YAML deserialization).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MoveBuilder {
    TranslateMolecule(crate::montecarlo::TranslateMolecule),
    TranslateAtom(crate::montecarlo::TranslateAtom),
    RotateMolecule(crate::montecarlo::RotateMolecule),
    VolumeMove(crate::montecarlo::VolumeMove),
    PivotMove(crate::montecarlo::PivotMove),
    CrankshaftMove(crate::montecarlo::CrankshaftMove),
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
        })
    }
}

/// Shared builder for both stochastic and deterministic move collections.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CollectionBuilder {
    #[serde(default = "default_repeat")]
    repeat: usize,
    #[serde(default)]
    moves: Vec<MoveBuilder>,
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
enum MoveCollectionBuilder {
    Stochastic(CollectionBuilder),
    Deterministic(CollectionBuilder),
}

impl MoveCollectionBuilder {
    fn build<T: Context>(self, context: &T) -> anyhow::Result<MoveCollection<T>> {
        let (strategy, builder) = match self {
            Self::Stochastic(b) => (SelectionStrategy::Stochastic, b),
            Self::Deterministic(b) => (SelectionStrategy::Deterministic, b),
        };
        let (repeat, moves) = builder.build_moves(context)?;
        Ok(MoveCollection {
            strategy,
            repeat,
            moves,
        })
    }
}

/// Non-generic builder for deserialization; produces `Propagate<T>` via `build()`.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct PropagateBuilder {
    #[serde(rename = "repeat")]
    #[serde(default = "default_repeat")]
    max_repeats: usize,
    #[serde(default)]
    seed: Seed,
    #[serde(default)]
    #[serde(rename = "collections")]
    move_collections: Vec<MoveCollectionBuilder>,
    #[serde(default)]
    criterion: AcceptanceCriterion,
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
enum SelectionStrategy {
    /// One move chosen per iteration via weighted random sampling.
    Stochastic,
    /// All moves executed in order each iteration.
    Deterministic,
}

/// A collection of moves with a selection strategy and repeat count.
#[derive(Debug)]
pub struct MoveCollection<T: Context> {
    strategy: SelectionStrategy,
    repeat: usize,
    moves: Vec<MoveRunner<T>>,
}

impl<T: Context> MoveCollection<T> {
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
        let moves: Vec<_> = self.moves.iter().filter_map(|m| m.to_yaml()).collect();
        map.insert("moves".into(), serde_yaml::Value::Sequence(moves));
        serde_yaml::Value::Tagged(Box::new(serde_yaml::value::TaggedValue {
            tag: serde_yaml::value::Tag::new(tag),
            value: serde_yaml::Value::Mapping(map),
        }))
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
    move_collections: Vec<MoveCollection<T>>,
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

    /// Build a `Propagate<T>` from an input YAML file.
    pub fn from_file(filename: impl AsRef<Path>, context: &T) -> anyhow::Result<Self> {
        let yaml = std::fs::read_to_string(filename)?;
        let full: serde_yaml::Value = serde_yaml::from_str(&yaml)?;

        let current = full
            .get("propagate")
            .ok_or_else(|| anyhow::anyhow!("Could not find `propagate` in the YAML file."))?;

        let builder: PropagateBuilder =
            serde_yaml::from_value(current.clone()).map_err(anyhow::Error::msg)?;

        let move_collections = builder
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
            move_collections,
            criterion: builder.criterion,
        })
    }

    pub fn collections(&self) -> &[MoveCollection<T>] {
        &self.move_collections
    }

    pub const fn max_repeats(&self) -> usize {
        self.max_repeats
    }

    /// Serialize the propagate state to a YAML value.
    pub fn to_yaml(&self) -> serde_yaml::Value {
        let mut map = serde_yaml::Mapping::new();
        map.insert("repeat".into(), self.max_repeats.into());
        map.insert(
            "seed".into(),
            serde_yaml::to_value(&self.seed).unwrap_or_default(),
        );
        let collections: Vec<_> = self.move_collections.iter().map(|c| c.to_yaml()).collect();
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

#[cfg(all(test, feature = "chemfiles"))]
mod tests {

    use crate::platform::reference::ReferencePlatform;

    use super::*;

    #[test]
    fn seed_parse() {
        let string = "!Fixed 49786352";
        let seed: Seed = serde_yaml::from_str(string).unwrap();
        assert!(matches!(seed, Seed::Fixed(49786352)));

        let string = "Hardware";
        let seed: Seed = serde_yaml::from_str(string).unwrap();
        assert!(matches!(seed, Seed::Hardware));
    }

    #[test]
    fn stochastic_parse() {
        let string = "repeat: 20
moves:
   - !TranslateMolecule { molecule: Water, dp: 0.4, weight: 1.0 }
   - !TranslateMolecule { molecule: Protein, dp: 0.6, weight: 2.0 }
   - !TranslateMolecule { molecule: Lipid, dp: 0.5, weight: 0.5 }";

        let collection: CollectionBuilder = serde_yaml::from_str(string).unwrap();
        assert_eq!(collection.repeat, 20);
        assert_eq!(collection.moves.len(), 3);
    }

    #[test]
    fn deterministic_parse() {
        let string = "repeat: 10
moves:
   - !TranslateMolecule { molecule: Water, dp: 0.4, weight: 1.0 }
   - !TranslateMolecule { molecule: Protein, dp: 0.6, weight: 2.0 }
   - !TranslateMolecule { molecule: Lipid, dp: 0.5, weight: 0.5 }";

        let collection: CollectionBuilder = serde_yaml::from_str(string).unwrap();
        assert_eq!(collection.repeat, 10);
        assert_eq!(collection.moves.len(), 3);
    }

    #[test]
    fn propagate_parse() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();
        let propagate = Propagate::from_file("tests/files/topology_pass.yaml", &context).unwrap();

        assert_eq!(propagate.max_repeats, 10000);
        assert_eq!(propagate.seed, Seed::Hardware);
        assert_eq!(propagate.current_repeat, 0);
        assert_eq!(propagate.criterion, AcceptanceCriterion::MetropolisHastings);
        assert_eq!(propagate.move_collections.len(), 2);

        let stochastic = &propagate.move_collections[0];
        assert_eq!(stochastic.moves().len(), 3);
        assert_eq!(stochastic.moves()[0].repeat(), 2);
        assert_eq!(stochastic.moves()[0].weight(), 0.5);
        assert_eq!(stochastic.moves()[1].repeat(), 1);
        assert_eq!(stochastic.moves()[1].weight(), 1.0);

        let deterministic = &propagate.move_collections[1];
        assert_eq!(deterministic.repeat(), 5);
        assert_eq!(deterministic.moves().len(), 1);
    }

    #[test]
    fn propagate_parse_fail() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_invalid_propagate.yaml",
            Some(Path::new("tests/files/structure.xyz")),
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
            Some(Path::new("tests/files/structure.xyz")),
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
            Some(Path::new("tests/files/structure.xyz")),
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
            Some(Path::new("tests/files/structure.xyz")),
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
