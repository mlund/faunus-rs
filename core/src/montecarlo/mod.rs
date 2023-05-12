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

//! Support for Monte Carlo moves.

use crate::cite::Citation;
use crate::time::Timer;
use crate::Change;
use crate::Context;
use serde::{Deserialize, Serialize};

/// Helper class to keep track of accepted and rejected moves
/// It is optionally possible to let this class keep track of a single mean square displacement
/// which can be useful for many Monte Carlo moves.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct MoveStatistics {
    /// Number of trial moves
    pub num_trials: usize,
    /// Number of accepted moves
    pub num_accepted: usize,
    /// Mean square displacement of some quantity (optional)
    pub mean_square_displacement: Option<f64>,
    /// Timer that measures the time spent in the move
    timer: Timer,
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

pub trait Move<T: Context>: Citation + std::fmt::Debug {
    /// Make a trial move in the given `context` and return an object
    /// describing the change.
    fn do_move(&mut self, context: &mut T) -> Result<Change, anyhow::Error>;

    /// Get statistics for the move
    fn statistics(&self) -> &MoveStatistics;

    /// Get mutable statistics for the move
    fn statistics_mut(&mut self) -> &mut MoveStatistics;
}

pub struct MonteCarlo<T: Context> {
    moves: Vec<Box<dyn Move<T>>>,
    /// Currently accepted state
    old_context: T,
    /// All moves are performed in this context and, if accepted, synced to `old_context`
    new_context: T,
}
