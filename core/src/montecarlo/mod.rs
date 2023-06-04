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

use crate::{time::Timer, Change, Context, Info, SyncFromAny, MOLAR_GAS_CONSTANT};
use average::Mean;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, ops::Neg};

/// Helper to handle old and new values, e.g. before and after a Monte Carlo move
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OldNew<T: core::fmt::Debug> {
    old: T,
    new: T,
}

impl OldNew<usize> {
    /// Difference `new - old` as a signed integer
    pub fn delta(&self) -> i32 {
        self.new as i32 - self.old as i32
    }
}

impl OldNew<f64> {
    /// Difference `new - old`
    pub fn delta(&self) -> f64 {
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
    pub timer: Timer,
    /// Custom statistics and information
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

/// Interface for acceptance criterion for Monte Carlo moves
trait AcceptanceCriterion {
    /// Acceptance criterion based on an old and new energy and a temperature (J/mol and Kelvin)
    fn accept(old_energy: f64, new_energy: f64, temperature: f64, rng: &mut ThreadRng) -> bool;
}

/// Metropolis-Hastings acceptance criterion
///
/// More information: <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>
#[derive(Clone, Debug, Default, Serialize, Deserialize, Copy)]
pub struct MetropolisHastings {}

impl AcceptanceCriterion for MetropolisHastings {
    fn accept(old_energy: f64, new_energy: f64, temperature: f64, rng: &mut ThreadRng) -> bool {
        // useful for hard-sphere systems where initial configurations may overlap
        if old_energy.is_infinite() && new_energy.is_finite() {
            return true;
        }
        // always accept if negative infinity
        if new_energy.is_infinite() && new_energy.is_sign_negative() {
            return true;
        }

        let energy_change = new_energy - old_energy;
        let thermal_energy = MOLAR_GAS_CONSTANT * temperature;
        let acceptance_probability = f64::min(1.0, f64::exp(energy_change / thermal_energy));
        rng.gen::<f64>() < acceptance_probability
    }
}

/// Energy minimization acceptance criterion
///
/// This will always accept a move if the new energy is lower than the old energy.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Copy)]
pub struct Minimize {}

impl AcceptanceCriterion for Minimize {
    fn accept(old_energy: f64, new_energy: f64, _temperature: f64, _rng: &mut ThreadRng) -> bool {
        if old_energy.is_infinite() && new_energy.is_finite() {
            return true;
        }
        new_energy <= old_energy
    }
}

pub trait Move<T: Context>: Info + std::fmt::Debug + SyncFromAny {
    /// Try moving in the given `context` and return the change.
    fn do_move(&mut self, context: &mut T) -> Result<Change, anyhow::Error>;

    /// Get statistics for the move
    fn statistics(&self) -> &MoveStatistics;

    /// Get mutable statistics for the move
    fn statistics_mut(&mut self) -> &mut MoveStatistics;

    /// Called when the move is accepted
    ///
    /// This will update the statistics.
    /// Can be re-implemented to perform additional actions.
    fn accepted(&mut self, _change: Change) {
        self.statistics_mut().accept();
    }

    /// Called when the move is rejected
    ///
    /// This will update the statistics.
    /// Can be re-implemented to perform additional actions.
    fn rejected(&mut self, _change: Change) {
        self.statistics_mut().reject();
    }

    /// Get the move frequency
    fn frequency(&self) -> Frequency;
}

/// # Monte Carlo simulation instance
///
/// This maintains two [`Context`]s, one for the current state and one for the new state, as
/// well as a list of moves to perform.
/// Moves are picked randomly and performed in the new context. If the move is accepted, the
/// new context is synced to the old context. If the move is rejected, the new context is
/// discarded.
pub struct Simulation<T: Context> {
    _moves: Vec<Box<dyn Move<T>>>,
    /// Currently accepted state
    _old_context: T,
    /// All moves are performed in this context and, if accepted, synced to `old_context`
    _new_context: T,
}

/// Entropy contribution due to a change in number of particles
///
/// Note that the volume unit should match so that n/V matches the unit of the chemical potential
pub fn entropy_bias(n: OldNew<usize>, volume: OldNew<f64>) -> f64 {
    let dn = n.delta();
    match dn.cmp(&0) {
        Ordering::Equal => 0.0,
        Ordering::Greater => (0..dn)
            .map(|i| f64::ln(f64::from(i + 1 + n.old as i32) / volume.new))
            .sum::<f64>(),
        Ordering::Less => (0..-dn)
            .map(|i| f64::ln(f64::from(i - n.old as i32) / volume.old))
            .sum::<f64>()
            .neg(),
    }
}
