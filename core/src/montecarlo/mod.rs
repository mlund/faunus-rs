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

use crate::energy::EnergyTerm;
use crate::{time::Timer, Change, Context, Info, SyncFromAny, MOLAR_GAS_CONSTANT};
use average::Mean;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, ops::Neg};

/// Custom bias to be added to the energy after a given move
///
/// Some moves may need to add additional bias not captured by the Hamiltonian.
#[derive(Clone, Copy, Debug)]
pub enum Bias {
    /// Custom bias to be added to the energy
    Energy(f64),
    /// Force acceptance of the move regardless of energy change
    ForceAccept,
}

/// Named helper struct to handle `new`, `old` pairs.
///
/// Used e.g. for data before and after a Monte Carlo move
/// and reduces risk mixing up the order or old and new values.
#[derive(Debug, Clone, Default)]
pub struct NewOld<T: core::fmt::Debug> {
    pub new: T,
    pub old: T,
}

impl<T: core::fmt::Debug> NewOld<T> {
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
    pub info: serde_json::Map<String, serde_json::Value>,
}

impl MoveStatistics {
    /// Register an accepted move and increment counters
    pub fn accept(&mut self) {
        self.num_trials += 1;
        self.num_accepted += 1;
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

/// Frequency of a Monte Carlo move or a measurement
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Frequency {
    /// Every `n` steps
    Every(usize),
    /// With probability `p` regardless of number of affected molecules or atoms
    Probability(f64),
    /// With probability `p` for each affected molecule or atom
    Sweep(f64),
    /// Once at step `n`
    Once(usize),
    /// Once at the very last step
    End,
}

impl Frequency {
    /// Check if action, typically a move or analysis, should be performed at given step
    pub fn should_perform(&self, step: usize, rng: &mut ThreadRng) -> bool {
        match self {
            Frequency::Every(n) => step % n == 0,
            Frequency::Probability(p) => rng.gen::<f64>() < *p,
            Frequency::Once(n) => step == *n,
            _ => unimplemented!("Unsupported frequency policy"),
        }
    }
}

/// Interface for acceptance criterion for Monte Carlo moves
trait AcceptanceCriterion {
    /// Acceptance criterion based on an old and new energy and a temperature (J/mol and Kelvin)
    fn accept(energies: NewOld<f64>, temperature: f64, rng: &mut ThreadRng) -> bool;
}

/// Metropolis-Hastings acceptance criterion
///
/// More information: <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>
#[derive(Clone, Debug, Default, Serialize, Deserialize, Copy)]
pub struct MetropolisHastings {}

impl AcceptanceCriterion for MetropolisHastings {
    fn accept(energies: NewOld<f64>, temperature: f64, rng: &mut ThreadRng) -> bool {
        // useful for hard-sphere systems where initial configurations may overlap
        if energies.old.is_infinite() && energies.new.is_finite() {
            return true;
        }
        // always accept if negative infinity
        if energies.new.is_infinite() && energies.new.is_sign_negative() {
            return true;
        }

        let thermal_energy = MOLAR_GAS_CONSTANT * temperature;
        let acceptance_probability =
            f64::min(1.0, f64::exp(-energies.difference() / thermal_energy));
        rng.gen::<f64>() < acceptance_probability
    }
}

/// Energy minimization acceptance criterion
///
/// This will always accept a move if the new energy is lower than the old energy.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Copy)]
pub struct Minimize {}

impl AcceptanceCriterion for Minimize {
    fn accept(energies: NewOld<f64>, _temperature: f64, _rng: &mut ThreadRng) -> bool {
        if energies.old.is_infinite() && energies.new.is_finite() {
            return true;
        }
        energies.difference() <= 0.0
    }
}

pub trait Move: Info + std::fmt::Debug + SyncFromAny {
    /// Perform a move on given `context`.
    fn do_move(&mut self) -> anyhow::Result<Change>;

    /// Moves may generate optional bias that should be added to the trial energy
    /// when determining the acceptance probability.
    /// It can also be used to force acceptance of a move in e.g. hybrid MD/MC schemes.
    fn bias(&self, _change: &Change, _energies: NewOld<f64>) -> Option<Bias> {
        None
    }

    /// Get statistics for the move
    fn statistics(&self) -> &MoveStatistics;

    /// Get mutable statistics for the move
    fn statistics_mut(&mut self) -> &mut MoveStatistics;

    /// Called when the move is accepted
    ///
    /// This will update the statistics.
    /// Often re-implemented to perform additional actions.
    fn accepted(&mut self, _change: &Change) {
        self.statistics_mut().accept();
    }

    /// Called when the move is rejected
    ///
    /// This will update the statistics.
    /// Often re-implemented to perform additional actions.
    fn rejected(&mut self, _change: &Change) {
        self.statistics_mut().reject();
    }

    /// Get the move frequency
    fn frequency(&self) -> Frequency;
}

/// Collection of moves
#[derive(Default, Debug)]
pub struct MoveCollection {
    moves: Vec<Box<dyn Move>>,
}

impl MoveCollection {
    pub fn push(&mut self, m: impl Move + 'static) {
        self.moves.push(Box::new(m));
    }
    pub fn random_move(&mut self, rng: &mut ThreadRng) -> Option<&mut Box<dyn Move>> {
        self.moves.iter_mut().choose(rng)
    }
}

/// # Monte Carlo simulation instance
///
/// This maintains two [`Context`]s, one for the current state and one for the new state, as
/// well as a list of moves to perform.
/// Moves are picked randomly and performed in the new context. If the move is accepted, the
/// new context is synced to the old context. If the move is rejected, the new context is
/// discarded.
pub struct Simulation<T: Context> {
    /// List of moves to perform
    moves: MoveCollection,
    /// Pair of contexts, one for the current state and one for the new state
    context: NewOld<T>,
}

impl<T: Context> Simulation<T> {
    pub fn do_move(&mut self, temperature: f64, rng: &mut ThreadRng) {
        if let Some(m) = self.moves.random_move(rng) {
            let change = m.do_move().unwrap();
            let energy = NewOld::<f64>::from(
                self.context.new.hamiltonian().energy_change(&change),
                self.context.old.hamiltonian().energy_change(&change),
            );
            let acceptance = match m.bias(&change, energy.clone()) {
                Some(Bias::ForceAccept) => true,
                Some(Bias::Energy(bias)) => {
                    let mut biased_energy = energy.clone();
                    biased_energy.new += bias;
                    MetropolisHastings::accept(biased_energy, temperature, rng)
                }
                None => MetropolisHastings::accept(energy.clone(), temperature, rng),
            };
            if acceptance {
                m.accepted(&change);
                todo!("Sync new context to old context")
                //self.context.new.sync_from(&self._context.old);
            } else {
                m.rejected(&change);
                todo!("Sync old context to new context")
                //self.context.new.sync_from(&self._context.old);
            }
        }
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
