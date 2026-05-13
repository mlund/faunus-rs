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

//! Widom perturbation accumulator with log-sum-exp numerics.
//!
//! Shared by [`super::VirtualTranslate`] and [`super::VirtualVolumeMove`]
//! to compute free energy from `<exp(-dU/kT)>` without overflow.

use crate::auxiliary::BlockAverage;

/// Numerically stable accumulator for the Widom exponential average `<exp(-dU/kT)>`.
///
/// Uses the log-sum-exp trick (cf. `duello::diffusion::zwanzig`) so that
/// arbitrarily large `|dU|` values are handled without overflow or sample
/// skipping. Supports weighted samples for reweighting biased trajectories.
///
/// Free energy: `F = -ln(<exp(-dU/kT)>)` in units of kT.
#[derive(Clone, Debug, Default)]
pub(crate) struct WidomAccumulator {
    /// ln(Σ w_i · exp(-dU_i - shift)), where shift keeps exponents ≤ 0.
    log_sum: f64,
    /// max(-dU_i) seen so far.
    shift: f64,
    /// Σ w_i
    sum_weights: f64,
    /// Number of samples in the current block (reset by end_block).
    count: u64,
    /// Overall log-sum-exp state mirroring the block fields but never reset.
    total_log_sum: f64,
    total_shift: f64,
    total_sum_weights: f64,
    /// Total samples across all blocks (never reset).
    total_count: u64,
    /// Between-block error estimation for the free energy.
    free_energy: BlockAverage,
}

impl WidomAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sample to the running log-sum-exp average.
    ///
    /// `energy_change` is dU in kT; `weight` is the reweighting factor
    /// (1.0 for unbiased runs, `exp(-ln_g)` for Wang-Landau reweighting).
    pub fn collect(&mut self, energy_change: f64, weight: f64) {
        if weight == 0.0 {
            return;
        }
        let x = -energy_change; // we accumulate exp(x) = exp(-dU)

        Self::accumulate(
            x,
            weight,
            &mut self.count,
            &mut self.shift,
            &mut self.log_sum,
            &mut self.sum_weights,
        );
        Self::accumulate(
            x,
            weight,
            &mut self.total_count,
            &mut self.total_shift,
            &mut self.total_log_sum,
            &mut self.total_sum_weights,
        );
    }

    fn accumulate(
        x: f64,
        weight: f64,
        count: &mut u64,
        shift: &mut f64,
        log_sum: &mut f64,
        sum_weights: &mut f64,
    ) {
        if *count == 0 {
            *shift = x;
            *log_sum = weight.ln();
        } else if x > *shift {
            // Rescale existing sum so all exponents remain ≤ 0, preventing overflow
            *log_sum += *shift - x;
            *shift = x;
            *log_sum = ln_add_exp(*log_sum, weight.ln());
        } else {
            *log_sum = ln_add_exp(*log_sum, (x - *shift) + weight.ln());
        }
        *sum_weights += weight;
        *count += 1;
    }

    /// Mean free energy `-ln(<exp(-dU/kT)>)` over all collected samples in units of kT.
    pub fn mean_free_energy(&self) -> f64 {
        if self.total_count == 0 || self.total_sum_weights <= 0.0 {
            return f64::INFINITY;
        }
        // F = -ln(Σ w_i exp(-dU_i) / Σ w_i)
        //   = -(shift + log_sum - ln(sum_weights))
        -(self.total_shift + self.total_log_sum - self.total_sum_weights.ln())
    }

    /// Whether any samples have been collected.
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Total number of samples collected across all blocks.
    pub fn len(&self) -> usize {
        self.total_count as usize
    }

    fn block_mean_free_energy(&self) -> f64 {
        if self.count == 0 || self.sum_weights <= 0.0 {
            return f64::INFINITY;
        }
        -(self.shift + self.log_sum - self.sum_weights.ln())
    }

    /// Finalize the current block: push its free energy into the block average,
    /// then reset the within-block accumulator for the next block.
    pub fn end_block(&mut self) {
        if self.count > 0 {
            self.free_energy.add(self.block_mean_free_energy());
            self.log_sum = 0.0;
            self.shift = 0.0;
            self.sum_weights = 0.0;
            self.count = 0;
        }
    }

    /// Standard error of the mean free energy across blocks.
    pub fn free_energy_error(&self) -> f64 {
        self.free_energy.error()
    }

    /// Sample standard deviation of the free energy across blocks.
    pub fn free_energy_stddev(&self) -> f64 {
        self.free_energy.stddev()
    }

    /// Number of completed blocks.
    pub fn n_blocks(&self) -> u64 {
        self.free_energy.n()
    }
}

/// Compute `ln(exp(a) + exp(b))` without overflow.
fn ln_add_exp(a: f64, b: f64) -> f64 {
    let (max, min) = if a >= b { (a, b) } else { (b, a) };
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max + (min - max).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn single_sample_zero_energy() {
        let mut acc = WidomAccumulator::new();
        acc.collect(0.0, 1.0);
        // exp(0) = 1, mean = 1, free_energy = -ln(1) = 0
        assert_approx_eq!(f64, acc.mean_free_energy(), 0.0);
        assert_eq!(acc.len(), 1);
    }

    #[test]
    fn single_sample_positive_energy() {
        let mut acc = WidomAccumulator::new();
        acc.collect(2.0, 1.0);
        // exp(-2) → free_energy = -ln(exp(-2)) = 2.0
        assert_approx_eq!(f64, acc.mean_free_energy(), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn two_samples_unit_weight() {
        let mut acc = WidomAccumulator::new();
        acc.collect(0.0, 1.0); // exp(0) = 1
        acc.collect(1.0, 1.0); // exp(-1) ≈ 0.3679
        let expected = -((1.0 + (-1.0_f64).exp()) / 2.0).ln();
        assert_approx_eq!(f64, acc.mean_free_energy(), expected, epsilon = 1e-12);
    }

    #[test]
    fn empty_returns_infinity() {
        let acc = WidomAccumulator::new();
        assert!(acc.mean_free_energy().is_infinite());
        assert!(acc.is_empty());
    }

    #[test]
    fn extreme_negative_energy_no_overflow() {
        let mut acc = WidomAccumulator::new();
        // Very negative dU → exp(-dU) is huge, but log-sum-exp handles it
        acc.collect(-1000.0, 1.0);
        assert_approx_eq!(f64, acc.mean_free_energy(), -1000.0, epsilon = 1e-10);
    }

    #[test]
    fn extreme_positive_energy_no_underflow() {
        let mut acc = WidomAccumulator::new();
        // Very positive dU → exp(-dU) ≈ 0, free_energy ≈ dU
        acc.collect(1000.0, 1.0);
        assert_approx_eq!(f64, acc.mean_free_energy(), 1000.0, epsilon = 1e-10);
    }

    #[test]
    fn weighted_samples() {
        let mut acc = WidomAccumulator::new();
        // Weight 2 on dU=0 (exp=1), weight 1 on dU=1 (exp=e^-1)
        // Weighted mean = (2*1 + 1*exp(-1)) / 3
        acc.collect(0.0, 2.0);
        acc.collect(1.0, 1.0);
        let expected = -((2.0 + (-1.0_f64).exp()) / 3.0).ln();
        assert_approx_eq!(f64, acc.mean_free_energy(), expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_weight_ignored() {
        let mut acc = WidomAccumulator::new();
        acc.collect(0.0, 0.0);
        assert!(acc.is_empty());
    }

    #[test]
    fn block_averaging() {
        let mut acc = WidomAccumulator::new();

        // Block 1: dU=0 → free_energy = 0
        acc.collect(0.0, 1.0);
        acc.end_block();

        // Block 2: dU=2 → free_energy = 2
        acc.collect(2.0, 1.0);
        acc.end_block();

        // Block mean ≈ 1.0, error > 0
        assert_approx_eq!(f64, acc.free_energy.mean(), 1.0, epsilon = 1e-12);
        assert!(acc.free_energy_error() > 0.0);

        // Total accumulator is never reset: 2 samples, overall mean = -ln((1 + exp(-2)) / 2)
        assert_eq!(acc.len(), 2);
        let expected = -((1.0 + (-2.0_f64).exp()) / 2.0).ln();
        assert_approx_eq!(f64, acc.mean_free_energy(), expected, epsilon = 1e-12);
    }

    #[test]
    fn ln_add_exp_basic() {
        assert_approx_eq!(f64, ln_add_exp(0.0, 0.0), (2.0_f64).ln());
        assert_approx_eq!(f64, ln_add_exp(100.0, 0.0), 100.0, epsilon = 1e-10);
        assert_approx_eq!(f64, ln_add_exp(0.0, 100.0), 100.0, epsilon = 1e-10);
    }
}
