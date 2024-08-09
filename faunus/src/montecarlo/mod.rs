// Copyright 2023 Mikael Lund
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

//! # Support for Monte Carlo sampling

use crate::analysis::{AnalysisCollection, Analyze};
use crate::propagate::Propagate;
use crate::{time::Timer, Context};
use average::Mean;
use log;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{cmp::Ordering, ops::Neg};

mod translate;

pub use translate::*;

/// Custom bias to be added to the energy after a given move
///
/// Some moves may need to add additional bias not captured by the Hamiltonian.
#[derive(Clone, Copy, Debug)]
pub enum Bias {
    /// Custom bias to be added to the energy
    Energy(f64),
    /// Force acceptance of the move regardless of energy change
    ForceAccept,
    /// No bias
    None,
}

/// Named helper struct to handle `new`, `old` pairs.
///
/// Used e.g. for data before and after a Monte Carlo move
/// and reduces risk mixing up the order or old and new values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NewOld<T> {
    pub new: T,
    pub old: T,
}

impl<T> NewOld<T> {
    pub fn from(new: T, old: T) -> Self {
        Self { new, old }
    }
}

impl NewOld<usize> {
    /// Difference `new - old` as a signed integer
    pub fn difference(&self) -> i32 {
        self.new as i32 - self.old as i32
    }
}

impl NewOld<f64> {
    /// Difference `new - old`
    pub fn difference(&self) -> f64 {
        self.new - self.old
    }
}

impl Copy for NewOld<usize> {}
impl Copy for NewOld<f64> {}

/// # Helper class to keep track of accepted and rejected moves
///
/// It is optionally possible to let this class keep track of a single mean square displacement
/// which can be useful for many Monte Carlo moves.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct MoveStatistics {
    /// Number of trial moves
    pub num_trials: usize,
    /// Number of accepted moves
    pub num_accepted: usize,
    /// Mean square displacement of some quantity (optional)
    pub mean_square_displacement: Option<Mean>,
    /// Timer that measures the time spent in the move
    #[serde(skip_deserializing)]
    pub timer: Timer,
    /// Custom statistics and information (only serialized)
    #[serde(skip_deserializing)]
    pub info: HashMap<String, crate::topology::Value>,
    /// Sum of energy changes due to this move
    pub energy_change_sum: f64,
}

impl MoveStatistics {
    /// Register an accepted move and increment counters
    pub fn accept(&mut self, energy_change: f64) {
        self.num_trials += 1;
        self.num_accepted += 1;
        self.energy_change_sum += energy_change;
    }

    /// Register a rejected move and increment counters
    pub fn reject(&mut self) {
        self.num_trials += 1;
    }

    /// Acceptance ratio
    pub fn acceptance_ratio(&self) -> f64 {
        self.num_accepted as f64 / self.num_trials as f64
    }
}

/// All possible acceptance criteria for Monte Carlo moves
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Copy)]
pub enum AcceptanceCriterion {
    #[default]
    #[serde(alias = "Metropolis")]
    /// Metropolis-Hastings acceptance criterion
    /// More information: <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>
    MetropolisHastings,
    /// Energy minimization acceptance criterion
    /// This will always accept a move if the new energy is lower than the old energy.
    Minimize,
}

impl AcceptanceCriterion {
    /// Acceptance criterion based on an old and new energy.
    ///
    /// The energies are normalized by the given thermal energy, _kT_,.
    pub(crate) fn accept(
        &self,
        energy: NewOld<f64>,
        bias: Bias,
        thermal_energy: f64,
        rng: &mut impl Rng,
    ) -> bool {
        match self {
            AcceptanceCriterion::MetropolisHastings => {
                // useful for hard-sphere systems where initial configurations may overlap
                if energy.old.is_infinite() && energy.new.is_finite() {
                    log::trace!("Accepting infinite -> finite energy change");
                    return true;
                }
                // always accept if negative infinity
                if energy.new.is_infinite() && energy.new.is_sign_negative() {
                    return true;
                }

                let du = energy.difference()
                    + match bias {
                        Bias::Energy(bias) => bias,
                        Bias::None => 0.0,
                        Bias::ForceAccept => return true,
                    };
                let p = f64::min(1.0, f64::exp(-du / thermal_energy));
                rng.gen::<f64>() < p
            }

            AcceptanceCriterion::Minimize => {
                if energy.old.is_infinite() && energy.new.is_finite() {
                    return true;
                }
                energy.difference()
                    + match bias {
                        Bias::Energy(bias) => bias,
                        Bias::None => 0.0,
                        Bias::ForceAccept => return true,
                    }
                    <= 0.0
            }
        }
    }
}

/// # Monte Carlo simulation instance
///
/// This maintains two [`Context`]s, one for the current state and one for the new state, as
/// well as a [`Propagate`] section specifying what moves to perform and how often.
/// Selected moves are performed in the new context. If the move is accepted, the new context
/// is synced to the old context. If the move is rejected, the new context is discarded.
///
/// The MarkovChain can be converted into an `Iterator` where each iteration corresponds to one 'propagate' cycle.
#[derive(Debug)]
pub struct MarkovChain<T: Context> {
    /// Description of moves to perform.
    propagate: Propagate,
    /// Pair of contexts, one for the current state and one for the new state.
    context: NewOld<T>,
    /// Current step.
    step: usize,
    /// Thermal energy - must be same unit as energy.
    thermal_energy: f64,
    /// Collection of analyses to perform during the simulation.
    analyses: AnalysisCollection<T>,
}

impl<T: Context> MarkovChain<T> {
    pub fn iter(&mut self) -> MarkovChainIterator<T> {
        MarkovChainIterator { markov: self }
    }
}

/// Iterator over MarkovChain.
/// Necessary if we want to access MarkovChain after the iteration is finished.
#[derive(Debug)]
pub struct MarkovChainIterator<'a, T: Context> {
    markov: &'a mut MarkovChain<T>,
}

impl<'a, T: Context> Iterator for MarkovChainIterator<'a, T> {
    type Item = anyhow::Result<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.markov.propagate.propagate(
            &mut self.markov.context,
            self.markov.thermal_energy,
            &mut self.markov.step,
            &mut self.markov.analyses,
        ) {
            Err(e) => Some(Err(e)),
            Ok(true) => Some(Ok(self.markov.step)),
            Ok(false) => None,
        }
    }
}

impl<T: Context + 'static> MarkovChain<T> {
    pub fn new(context: T, propagate: Propagate, thermal_energy: f64) -> Self {
        Self {
            context: NewOld::from(context.clone(), context),
            thermal_energy,
            step: 0,
            propagate,
            analyses: AnalysisCollection::default(),
        }
    }
    /// Set the thermal energy, _kT_.
    ///
    /// This is used to normalize the energy change when determining the acceptance probability.
    /// Must match the unit of the energy.
    pub fn set_thermal_energy(&mut self, thermal_energy: f64) {
        self.thermal_energy = thermal_energy;
    }
    /// Append an analysis to the back of the collection.
    pub fn add_analysis(&mut self, analysis: Box<dyn Analyze<T>>) {
        self.analyses.push(analysis)
    }
}

/// Entropy bias due to a change in number of particles
///
/// See:
/// - <https://en.wikipedia.org/wiki/Entropy_(statistical_thermodynamics)#Entropy_of_mixing>
/// - <https://doi.org/10/fqcpg3>
///
/// # Examples
/// ~~~
/// use faunus::montecarlo::*;
/// let vol = NewOld::from(1.0, 1.0);
/// assert_eq!(entropy_bias(NewOld::from(0, 0), vol.clone()), 0.0);
/// assert_eq!(entropy_bias(NewOld::from(2, 1), vol.clone()), f64::ln(2.0));
/// assert_eq!(entropy_bias(NewOld::from(1, 2), vol.clone()), f64::ln(0.5));
/// ~~~
///
/// Note that the volume unit should match so that n/V matches the unit of the chemical potential
pub fn entropy_bias(n: NewOld<usize>, volume: NewOld<f64>) -> f64 {
    let dn = n.difference();
    match dn.cmp(&0) {
        Ordering::Equal => {
            if volume.difference().abs() > f64::EPSILON {
                unimplemented!("Entropy bias currently cannot be used for volume changes")
            }
            0.0
        }
        Ordering::Greater => (0..dn)
            .map(|i| f64::ln(f64::from(n.old as i32 + i + 1) / volume.new))
            .sum::<f64>(),
        Ordering::Less => (0..-dn)
            .map(|i| f64::ln(f64::from(n.old as i32 - i) / volume.old))
            .sum::<f64>()
            .neg(),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::platform::reference::ReferencePlatform;
    use float_cmp::assert_approx_eq;

    #[test]
    fn translate_molecules_simulation() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/translate_molecules_simulation.yaml",
            None::<String>,
            &mut rng,
        )
        .unwrap();

        let propagate =
            Propagate::from_file("tests/files/translate_molecules_simulation.yaml", &context)
                .unwrap();

        let mut markov_chain = MarkovChain::new(context, propagate, 1.0);

        for step in markov_chain.iter() {
            step.unwrap();
        }

        let move1_stats =
            markov_chain.propagate.get_collections()[0].get_moves()[0].get_statistics();

        assert_eq!(move1_stats.num_trials, 73);
        assert_eq!(move1_stats.num_accepted, 71);
        assert_approx_eq!(f64, move1_stats.energy_change_sum, -3.3952572353350177);

        let move2_stats =
            markov_chain.propagate.get_collections()[0].get_moves()[1].get_statistics();

        assert_eq!(move2_stats.num_trials, 81);
        assert_eq!(move2_stats.num_accepted, 79);
        assert_approx_eq!(f64, move2_stats.energy_change_sum, -1.1611869334060376);

        let move3_stats =
            markov_chain.propagate.get_collections()[0].get_moves()[2].get_statistics();

        assert_eq!(move3_stats.num_trials, 0);
        assert_eq!(move3_stats.num_accepted, 0);
        assert_approx_eq!(f64, move3_stats.energy_change_sum, 0.0);

        let move4_stats =
            markov_chain.propagate.get_collections()[1].get_moves()[0].get_statistics();

        assert_eq!(move4_stats.num_trials, 100);
        assert_eq!(move4_stats.num_accepted, 94);
        assert_approx_eq!(f64, move4_stats.energy_change_sum, -61.739122509342266);

        let move5_stats =
            markov_chain.propagate.get_collections()[2].get_moves()[0].get_statistics();

        assert_eq!(move5_stats.num_trials, 500);
        assert_eq!(move5_stats.num_accepted, 466);
        assert_approx_eq!(f64, move5_stats.energy_change_sum, -515.1334649717064);

        println!("{:?}", markov_chain.context.new.particles());

        for context in [&markov_chain.context.new, &markov_chain.context.old] {
            let p1 = &context.particles()[0];
            let p2 = &context.particles()[1];
            let p3 = &context.particles()[2];

            assert_approx_eq!(f64, p1.pos.x, p2.pos.x + 1.0, epsilon = 0.0000001);
            assert_approx_eq!(f64, p1.pos.x, p3.pos.x + 1.0, epsilon = 0.0000001);
            assert_approx_eq!(f64, p2.pos.x, p3.pos.x, epsilon = 0.0000001);

            assert_approx_eq!(f64, p1.pos.y, p2.pos.y, epsilon = 0.0000001);
            assert_approx_eq!(f64, p1.pos.y + 1.0, p3.pos.y, epsilon = 0.0000001);
            assert_approx_eq!(f64, p2.pos.y + 1.0, p3.pos.y, epsilon = 0.0000001);

            assert_approx_eq!(f64, p1.pos.z, p2.pos.z, epsilon = 0.0000001);
            assert_approx_eq!(f64, p1.pos.z, p3.pos.z, epsilon = 0.0000001);
            assert_approx_eq!(f64, p2.pos.z, p3.pos.z, epsilon = 0.0000001);

            let p4 = &context.particles()[3];
            let p5 = &context.particles()[4];
            let p6 = &context.particles()[5];

            assert_approx_eq!(f64, p4.pos.x + 1.0, p5.pos.x, epsilon = 0.0000001);
            assert_approx_eq!(f64, p4.pos.x + 1.0, p6.pos.x, epsilon = 0.0000001);
            assert_approx_eq!(f64, p5.pos.x, p6.pos.x, epsilon = 0.0000001);

            assert_approx_eq!(f64, p4.pos.y, p5.pos.y, epsilon = 0.0000001);
            assert_approx_eq!(f64, p4.pos.y, p6.pos.y, epsilon = 0.0000001);
            assert_approx_eq!(f64, p5.pos.y, p6.pos.y, epsilon = 0.0000001);

            assert_approx_eq!(f64, p4.pos.z, p5.pos.z, epsilon = 0.0000001);
            assert_approx_eq!(f64, p4.pos.z, p6.pos.z + 1.0, epsilon = 0.0000001);
            assert_approx_eq!(f64, p5.pos.z, p6.pos.z + 1.0, epsilon = 0.0000001);

            let p7 = &context.particles()[6];
            let p8 = &context.particles()[7];
            let p9 = &context.particles()[8];

            assert_approx_eq!(f64, p7.pos.x, p8.pos.x, epsilon = 0.0000001);
            assert_approx_eq!(f64, p7.pos.x, p9.pos.x, epsilon = 0.0000001);
            assert_approx_eq!(f64, p8.pos.x, p9.pos.x, epsilon = 0.0000001);

            assert_approx_eq!(f64, p7.pos.y, p8.pos.y + 1.0, epsilon = 0.0000001);
            assert_approx_eq!(f64, p7.pos.y, p9.pos.y + 2.0, epsilon = 0.0000001);
            assert_approx_eq!(f64, p8.pos.y, p9.pos.y + 1.0, epsilon = 0.0000001);

            assert_approx_eq!(f64, p7.pos.z, p8.pos.z, epsilon = 0.0000001);
            assert_approx_eq!(f64, p7.pos.z, p9.pos.z, epsilon = 0.0000001);
            assert_approx_eq!(f64, p8.pos.z, p9.pos.z, epsilon = 0.0000001);
        }
    }
}
