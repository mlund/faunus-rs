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

use crate::collective_variable::{CollectiveVariable, CvKindBuilder};
use crate::flat_histogram::{FlatHistogramState, GridDim};
use crate::Change;
use crate::ObserveContext;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Once, PoisonError, RwLock};

/// Recover the guard from a poisoned shared-state lock, warning once so a run does not
/// silently proceed on bias data that a panicking walker may have left inconsistent.
fn recover_poisoned<T>(poisoned: PoisonError<T>) -> T {
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "penalty bias lock was poisoned by a panicking walker; recovering and continuing, \
             but shared bias state may be inconsistent — treat results with caution"
        );
    });
    poisoned.into_inner()
}

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
    pub(crate) fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        if matches!(change, Change::None) {
            return 0.0;
        }
        let cv = self.eval_cv(context);
        // Recover rather than propagate a poisoned lock: in a multi-walker run one thread
        // panicking must not cascade a panic into every sibling's next bias evaluation.
        let state = self.state.read().unwrap_or_else(recover_poisoned);
        match state.bin_index(&cv) {
            Some(b) => state.ln_g(b) * self.thermal_energy,
            None => f64::INFINITY,
        }
    }

    /// Evaluate CV(s) and update the shared histogram + density of states.
    // Only the `cli`-gated umbrella / Wang-Landau drivers use this.
    #[cfg(feature = "cli")]
    pub(crate) fn update(&self, context: &impl ObserveContext) {
        let cv = self.eval_cv(context);
        let mut state = self.state.write().unwrap_or_else(recover_poisoned);
        if let Some(bin) = state.bin_index(&cv) {
            state.update(bin);
        }
    }

    /// Access the shared flat-histogram state (for reweighting diagnostics).
    pub fn state(&self) -> &Arc<RwLock<FlatHistogramState>> {
        &self.state
    }

    /// Evaluate CV value(s) into a slice suitable for `bin_index`.
    fn eval_cv(&self, context: &impl ObserveContext) -> [f64; 2] {
        let v1 = self.cv.evaluate(context);
        let v2 = self.cv2.as_ref().map_or(0.0, |cv| cv.evaluate(context));
        [v1, v2]
    }
}

/// Builder for deserializing a static `Penalty` from a `!Penalty` energy-list entry.
///
/// Loads a converged `FlatHistogramState` from a checkpoint file and uses the
/// stored `ln g` as a fixed bias. Useful for production runs after Wang-Landau
/// convergence, where the bias flattens the free energy surface and observables
/// are recovered via reweighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PenaltyBuilder {
    /// Path to a `FlatHistogramState` checkpoint (e.g. `wl_states/histogram.yaml`).
    pub file: PathBuf,
    /// Primary collective variable. The grid (range and resolution) comes from
    /// the checkpoint, so only the CV kind is specified here.
    pub coordinate: Box<dyn CvKindBuilder>,
    /// Optional second CV for 2D penalty surfaces.
    pub coordinate2: Option<Box<dyn CvKindBuilder>>,
}

/// Reject a config whose CV count disagrees with the checkpoint's grid dimensionality.
///
/// The checkpoint is authoritative: a mismatch would drive the wrong number of CVs into
/// `bin_index`, which silently bins at `cv[1] = 0.0` (2-D grid, 1-D config) or discards
/// the second CV (1-D grid, 2-D config) rather than erroring. Nothing downstream catches
/// it, so validate here.
fn check_dimensionality(
    grid_is_2d: bool,
    has_cv2: bool,
    file: &std::path::Path,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        grid_is_2d == has_cv2,
        "penalty checkpoint '{}' is a {}-D grid but the config supplies {} collective variable(s); \
         provide `coordinate2` for a 2-D grid and omit it for a 1-D grid",
        file.display(),
        if grid_is_2d { 2 } else { 1 },
        if has_cv2 { 2 } else { 1 },
    );
    Ok(())
}

impl PenaltyBuilder {
    /// Build a static [`Penalty`] by loading the checkpoint and resolving CVs.
    pub fn build(
        &self,
        context: &impl ObserveContext,
        thermal_energy: f64,
    ) -> anyhow::Result<Penalty> {
        let state = FlatHistogramState::from_file(&self.file)?;
        let grid_is_2d = matches!(state.dim(), GridDim::Two { .. });
        check_dimensionality(grid_is_2d, self.coordinate2.is_some(), &self.file)?;
        log::info!(
            "Loaded penalty from '{}': {} bins, Δg={:.1} kT",
            self.file.display(),
            state.dim().num_bins(),
            state.ln_g_range(),
        );
        let cv = self.coordinate.build_cv(context)?;
        let cv2 = self
            .coordinate2
            .as_ref()
            .map(|b| b.build_cv(context))
            .transpose()?;
        Ok(Penalty::new(
            cv,
            cv2,
            Arc::new(RwLock::new(state)),
            thermal_energy,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_dimensionality_is_accepted() {
        let file = std::path::Path::new("histogram.yaml");
        assert!(check_dimensionality(false, false, file).is_ok()); // 1-D grid, 1 CV
        assert!(check_dimensionality(true, true, file).is_ok()); // 2-D grid, 2 CVs
    }

    #[test]
    fn mismatched_dimensionality_is_rejected() {
        let file = std::path::Path::new("histogram.yaml");
        // 2-D grid driven by a 1-D config: would silently bin at cv[1] = 0.0.
        assert!(check_dimensionality(true, false, file).is_err());
        // 1-D grid with a superfluous coordinate2: would discard the second CV.
        assert!(check_dimensionality(false, true, file).is_err());
    }
}
