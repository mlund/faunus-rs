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
use crate::{time::Timer, Change, Context, Info};
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
    /// Acceptance criterion based on an old and new energy.
    ///
    /// The energies are normalized by the given thermal energy, _kT_,.
    fn accept(
        &self,
        energies: NewOld<f64>,
        bias: Bias,
        thermal_energy: f64,
        rng: &mut ThreadRng,
    ) -> bool;
}

impl Default for Box<dyn AcceptanceCriterion> {
    fn default() -> Self {
        Box::<MetropolisHastings>::default()
    }
}

impl std::fmt::Debug for Box<dyn AcceptanceCriterion> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceptanceCriterion").finish()
    }
}

/// Metropolis-Hastings acceptance criterion
///
/// More information: <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>
#[derive(Clone, Debug, Default, Serialize, Deserialize, Copy)]
pub struct MetropolisHastings {}

impl AcceptanceCriterion for MetropolisHastings {
    fn accept(
        &self,
        energy: NewOld<f64>,
        bias: Bias,
        thermal_energy: f64,
        rng: &mut ThreadRng,
    ) -> bool {
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
}

/// Energy minimization acceptance criterion
///
/// This will always accept a move if the new energy is lower than the old energy.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Copy)]
pub struct Minimize {}

impl AcceptanceCriterion for Minimize {
    #[allow(unused_variables)]
    fn accept(
        &self,
        energy: NewOld<f64>,
        bias: Bias,
        thermal_energy: f64,
        rng: &mut ThreadRng,
    ) -> bool {
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

pub trait Move<T>: Info + std::fmt::Debug
where
    T: Context,
{
    /// Perform a move on given `context`.
    fn do_move(&mut self, context: &mut T, rng: &mut ThreadRng) -> Option<Change>;

    /// Moves may generate optional bias that should be added to the trial energy
    /// when determining the acceptance probability.
    /// It can also be used to force acceptance of a move in e.g. hybrid MD/MC schemes.
    /// By default, this returns `Bias::None`.
    #[allow(unused_variables)]
    fn bias(&self, change: &Change, energies: &NewOld<f64>) -> Bias {
        Bias::None
    }

    /// Get statistics for the move
    fn statistics(&self) -> &MoveStatistics;

    /// Get mutable statistics for the move
    fn statistics_mut(&mut self) -> &mut MoveStatistics;

    /// Called when the move is accepted
    ///
    /// This will update the statistics.
    /// Often re-implemented to perform additional actions.
    #[allow(unused_variables)]
    fn accepted(&mut self, change: &Change, energy_change: f64) {
        self.statistics_mut().accept(energy_change);
    }

    /// Called when the move is rejected
    ///
    /// This will update the statistics.
    /// Often re-implemented to perform additional actions.
    #[allow(unused_variables)]
    fn rejected(&mut self, change: &Change) {
        self.statistics_mut().reject();
    }

    /// Get the move frequency
    fn frequency(&self) -> Frequency;
}

/// Collection of moves
///
/// # Todo
/// - `choose` should respect `frequency`
/// - Should implement serialize, see e.g.
/// <https://stackoverflow.com/questions/50021897/how-to-implement-serdeserialize-for-a-boxed-trait-object>
#[derive(Debug)]
pub struct MoveCollection<T: Context> {
    moves: Vec<Box<dyn Move<T>>>,
}

impl<T: Context> Default for MoveCollection<T> {
    fn default() -> Self {
        Self { moves: Vec::new() }
    }
}

impl<T: Context> MoveCollection<T> {
    /// Appends a move to the back of the collection.
    pub fn push(&mut self, m: impl Move<T> + 'static) {
        self.moves.push(Box::new(m));
    }
    /// Picks a random move from the collection.
    pub fn choose(&mut self, rng: &mut ThreadRng) -> Option<&mut Box<dyn Move<T>>> {
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
///
/// The chain implements `Iterator` where each iteration corresponds to a single Monte Carlo step.
#[derive(Default, Debug)]
pub struct MarkovChain<T: Context> {
    /// List of moves to perform
    moves: MoveCollection<T>,
    /// Pair of contexts, one for the current state and one for the new state
    context: NewOld<T>,
    /// Current step
    step: usize,
    /// Maximum number of steps
    max_steps: usize,
    /// Random number engine
    rng: ThreadRng,
    /// Thermal energy - must be same unit as energy
    thermal_energy: f64,
    /// Acceptance policy
    criterion: Box<dyn AcceptanceCriterion>,
}

impl<T: Context + 'static> Iterator for MarkovChain<T> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.max_steps {
            return None;
        }
        self.do_move();
        self.step += 1;
        Some(self.step)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.max_steps - self.step;
        (size, Some(size))
    }
}

impl<T: Context + 'static> ExactSizeIterator for MarkovChain<T> {}
impl<T: Context + 'static> FusedIterator for MarkovChain<T> {}

impl<T: Context + 'static> MarkovChain<T> {
    pub fn new(context: T, max_steps: usize, thermal_energy: f64) -> Self {
        Self {
            context: NewOld::from(context.clone(), context),
            max_steps,
            thermal_energy,
            step: 0,
            rng: rand::thread_rng(),
            moves: MoveCollection::default(),
            criterion: Box::<MetropolisHastings>::default(),
        }
    }
    /// Set the thermal energy, _kT_.
    ///
    /// This is used to normalize the energy change when determining the acceptance probability.
    /// Must match the unit of the energy.
    pub fn set_thermal_energy(&mut self, thermal_energy: f64) {
        self.thermal_energy = thermal_energy;
    }
    /// Set random number generator
    pub fn set_rng(&mut self, rng: ThreadRng) {
        self.rng = rng;
    }
    /// Append a move to the back of the collection.
    pub fn add_move(&mut self, m: impl Move<T> + 'static) {
        self.moves.push(m);
    }

    fn do_move(&mut self) {
        if let Some(mv) = self.moves.choose(&mut self.rng) {
            let change = mv.do_move(&mut self.context.new, &mut self.rng).unwrap();
            self.context.new.update(&change).unwrap();
            let energy = NewOld::<f64>::from(
                self.context
                    .new
                    .hamiltonian()
                    .energy_change(&self.context.new, &change),
                self.context
                    .old
                    .hamiltonian()
                    .energy_change(&self.context.old, &change),
            );
            let bias = mv.bias(&change, &energy);
            if self
                .criterion
                .accept(energy, bias, self.thermal_energy, &mut self.rng)
            {
                mv.accepted(&change, energy.difference());
                self.context
                    .old
                    .sync_from(&self.context.new, &change)
                    .unwrap();
            } else {
                mv.rejected(&change);
                self.context
                    .new
                    .sync_from(&self.context.old, &change)
                    .unwrap();
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
