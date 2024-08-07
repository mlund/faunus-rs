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
use std::iter::FusedIterator;
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
/// The chain implements `Iterator` where each iteration corresponds to one 'propagate' cycle.
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

impl<T: Context + 'static> Iterator for MarkovChain<T> {
    type Item = anyhow::Result<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.propagate.propagate(
            &mut self.context,
            self.thermal_energy,
            &mut self.step,
            &mut self.analyses,
        ) {
            Err(e) => Some(Err(e)),
            Ok(true) => Some(Ok(self.step)),
            Ok(false) => None,
        }
    }
}

impl<T: Context + 'static> ExactSizeIterator for MarkovChain<T> {}
impl<T: Context + 'static> FusedIterator for MarkovChain<T> {}

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
