//! Reweighting support for biased sampling.
//!
//! After Wang-Landau convergence the `Penalty` energy term flattens the free
//! energy surface. Raw ensemble averages are incorrect; observables must be
//! reweighted by `w = exp(-ln g(bin))`.

use crate::energy::penalty::Penalty;
use crate::{Change, Context};

/// Source of per-frame reweighting factors.
pub(crate) enum WeightSource {
    /// No reweighting (w = 1).
    Uniform,
    /// WL unbiasing: w = exp(-penalty_energy / kT).
    Penalty {
        penalty: Penalty,
        inv_thermal_energy: f64,
    },
}

impl WeightSource {
    /// Compute the reweighting factor for the current configuration.
    pub fn weight(&self, context: &impl Context) -> f64 {
        match self {
            Self::Uniform => 1.0,
            Self::Penalty {
                penalty,
                inv_thermal_energy,
            } => {
                // Penalty::energy returns ln_g(bin) * kT; dividing by kT gives ln_g
                let ln_g = penalty.energy(context, &Change::Everything) * inv_thermal_energy;
                if !ln_g.is_finite() {
                    // Out-of-range CV → infinite penalty → frame contributes nothing
                    0.0
                } else {
                    // Clamp to avoid exp overflow in deep free energy minima
                    (-ln_g.max(f64::MIN_EXP as f64)).exp()
                }
            }
        }
    }
}
