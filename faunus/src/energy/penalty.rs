// Copyright 2025 Mikael Lund
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

//! Flat-histogram bias energy term.
//!
//! Maps collective variable(s) to a bin in the shared [`FlatHistogramState`],
//! returning `ln g(bin) × kT` as a bias energy. Out-of-range CV values
//! produce infinite energy for early rejection.

use crate::collective_variable::CollectiveVariable;
use crate::flat_histogram::FlatHistogramState;
use crate::{Change, Context};
use std::sync::{Arc, RwLock};

/// Flat-histogram bias that enters the Hamiltonian as `ln g(CV) × kT`.
///
/// Placed at the front of the Hamiltonian so out-of-range CVs trigger
/// early rejection before expensive nonbonded terms are evaluated.
#[derive(Debug, Clone)]
pub struct Penalty {
    cv: CollectiveVariable,
    cv2: Option<CollectiveVariable>,
    state: Arc<RwLock<FlatHistogramState>>,
    thermal_energy: f64,
}

impl Penalty {
    /// Create a new penalty term.
    ///
    /// For 1D, pass `cv2 = None`. For 2D, supply both CVs.
    pub fn new(
        cv: CollectiveVariable,
        cv2: Option<CollectiveVariable>,
        state: Arc<RwLock<FlatHistogramState>>,
        thermal_energy: f64,
    ) -> Self {
        Self {
            cv,
            cv2,
            state,
            thermal_energy,
        }
    }

    /// Compute bias energy: `ln_g(bin) * kT`, or infinity if out of range.
    pub fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        if matches!(change, Change::None) {
            return 0.0;
        }
        let cv = self.eval_cv(context);
        let state = self.state.read().expect("poisoned lock");
        match state.bin_index(&cv) {
            Some(b) => state.ln_g(b) * self.thermal_energy,
            None => f64::INFINITY,
        }
    }

    /// Evaluate CV(s) and update the shared histogram + density of states.
    pub fn update(&self, context: &impl Context) {
        let cv = self.eval_cv(context);
        let mut state = self.state.write().expect("poisoned lock");
        if let Some(bin) = state.bin_index(&cv) {
            state.update(bin);
        }
    }

    /// Evaluate CV value(s) into a slice suitable for `bin_index`.
    fn eval_cv(&self, context: &impl Context) -> [f64; 2] {
        let v1 = self.cv.evaluate(context);
        let v2 = self.cv2.as_ref().map_or(0.0, |cv| cv.evaluate(context));
        [v1, v2]
    }
}
