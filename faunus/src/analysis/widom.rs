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
use std::num::NonZeroUsize;

/// Single log-sum-exp accumulator state: tracks `ln(Σ w_i · exp(x_i - shift))`
/// with `shift = max(x_i)` to keep all exponents ≤ 0.
#[derive(Clone, Debug, Default)]
struct LogSumExp {
    log_sum: f64,
    shift: f64,
    sum_weights: f64,
    count: u64,
}

impl LogSumExp {
    /// Add a `(x = -dU, weight)` sample, rescaling on a new maximum to avoid overflow.
    fn accumulate(&mut self, x: f64, weight: f64) {
        if self.count == 0 {
            self.shift = x;
            self.log_sum = weight.ln();
        } else if x > self.shift {
            self.log_sum += self.shift - x;
            self.shift = x;
            self.log_sum = ln_add_exp(self.log_sum, weight.ln());
        } else {
            self.log_sum = ln_add_exp(self.log_sum, (x - self.shift) + weight.ln());
        }
        self.sum_weights += weight;
        self.count += 1;
    }

    /// `F = -ln(Σ w·exp(-dU) / Σ w)`. `+∞` for an empty state.
    fn free_energy(&self) -> f64 {
        if self.count == 0 || self.sum_weights <= 0.0 {
            f64::INFINITY
        } else {
            -(self.shift + self.log_sum - self.sum_weights.ln())
        }
    }
}

/// Numerically stable accumulator for the Widom exponential average `<exp(-dU/kT)>`.
///
/// Uses the log-sum-exp trick (cf. `duello::diffusion::zwanzig`) so that
/// arbitrarily large `|dU|` values are handled without overflow or sample
/// skipping. Supports weighted samples for reweighting biased trajectories.
///
/// Free energy: `F = -ln(<exp(-dU/kT)>)` in units of kT.
///
/// When constructed via [`with_block_size`](Self::with_block_size), `collect`
/// auto-finalizes a block every `block_size` samples so the consumer does not
/// have to track block boundaries.
#[derive(Clone, Debug, Default)]
pub(crate) struct WidomAccumulator {
    /// Samples in the current (open) block; reset by [`end_block`](Self::end_block).
    open: LogSumExp,
    /// Running aggregate of all closed blocks, each folded in as a single
    /// `(weight = block.sum_weights, x = -block.free_energy)` point. This
    /// is mathematically equivalent to a per-sample running total at far
    /// lower cost: `collect` updates only `open`; `closed` advances once
    /// per block boundary.
    closed: LogSumExp,
    /// Raw count of `collect` calls, kept separately because `closed.count`
    /// counts blocks (merged points), not original samples.
    n_samples: u64,
    /// Mean ± SEM of per-block free energies for block-error estimation.
    free_energy: BlockAverage,
    block_size: Option<NonZeroUsize>,
}

impl WidomAccumulator {
    /// Enable automatic block segmentation: `collect` finalizes a block
    /// every `block_size` samples. `NonZeroUsize` makes a zero block size
    /// unrepresentable — otherwise every sample would become its own block
    /// and skew SEM/stddev.
    pub fn with_block_size(mut self, block_size: NonZeroUsize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// Add a sample to the running log-sum-exp average.
    ///
    /// `energy_change` is dU in kT; `weight` is the reweighting factor
    /// (1.0 for unbiased runs, `exp(-ln_g)` for Wang-Landau reweighting).
    pub fn collect(&mut self, energy_change: f64, weight: f64) {
        if weight == 0.0 {
            return;
        }
        self.open.accumulate(-energy_change, weight);
        self.n_samples += 1;
        if let Some(n) = self.block_size {
            if self.open.count >= n.get() as u64 {
                self.end_block();
            }
        }
    }

    /// Mean free energy `-ln(<exp(-dU/kT)>)` over all collected samples in kT.
    pub fn mean_free_energy(&self) -> f64 {
        if self.n_samples == 0 {
            return f64::INFINITY;
        }
        if self.open.count == 0 {
            return self.closed.free_energy();
        }
        // Merge the open block into a transient copy of `closed` as one
        // (weight, x) point; mathematically exact since
        // `Σ_i w_i exp(-dU_i) = W_open · exp(-F_open) + closed contributions`.
        let mut combined = self.closed.clone();
        combined.accumulate(-self.open.free_energy(), self.open.sum_weights);
        combined.free_energy()
    }

    /// Whether any samples have been collected.
    pub fn is_empty(&self) -> bool {
        self.n_samples == 0
    }

    /// Total samples collected across all blocks.
    pub fn len(&self) -> usize {
        self.n_samples as usize
    }

    /// Finalize the current block: push its free energy into the block
    /// average and fold it into the running `closed` aggregate, then reset
    /// the within-block state for the next block.
    pub fn end_block(&mut self) {
        if self.open.count == 0 {
            return;
        }
        let f_open = self.open.free_energy();
        self.free_energy.add(f_open);
        self.closed.accumulate(-f_open, self.open.sum_weights);
        self.open = LogSumExp::default();
    }

    /// Borrow the per-block free-energy aggregator. Callers can read
    /// `mean()`, `error()`, `stddev()`, `n()`, or scale to a derived
    /// quantity via `&BlockAverage * scale` → `BlockSummary`.
    pub fn free_energy(&self) -> &BlockAverage {
        &self.free_energy
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
        let mut acc = WidomAccumulator::default();
        acc.collect(0.0, 1.0);
        // exp(0) = 1, mean = 1, free_energy = -ln(1) = 0
        assert_approx_eq!(f64, acc.mean_free_energy(), 0.0);
        assert_eq!(acc.len(), 1);
    }

    #[test]
    fn single_sample_positive_energy() {
        let mut acc = WidomAccumulator::default();
        acc.collect(2.0, 1.0);
        // exp(-2) → free_energy = -ln(exp(-2)) = 2.0
        assert_approx_eq!(f64, acc.mean_free_energy(), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn two_samples_unit_weight() {
        let mut acc = WidomAccumulator::default();
        acc.collect(0.0, 1.0); // exp(0) = 1
        acc.collect(1.0, 1.0); // exp(-1) ≈ 0.3679
        let expected = -((1.0 + (-1.0_f64).exp()) / 2.0).ln();
        assert_approx_eq!(f64, acc.mean_free_energy(), expected, epsilon = 1e-12);
    }

    #[test]
    fn empty_returns_infinity() {
        let acc = WidomAccumulator::default();
        assert!(acc.mean_free_energy().is_infinite());
        assert!(acc.is_empty());
    }

    #[test]
    fn extreme_negative_energy_no_overflow() {
        let mut acc = WidomAccumulator::default();
        // Very negative dU → exp(-dU) is huge, but log-sum-exp handles it
        acc.collect(-1000.0, 1.0);
        assert_approx_eq!(f64, acc.mean_free_energy(), -1000.0, epsilon = 1e-10);
    }

    #[test]
    fn extreme_positive_energy_no_underflow() {
        let mut acc = WidomAccumulator::default();
        // Very positive dU → exp(-dU) ≈ 0, free_energy ≈ dU
        acc.collect(1000.0, 1.0);
        assert_approx_eq!(f64, acc.mean_free_energy(), 1000.0, epsilon = 1e-10);
    }

    #[test]
    fn weighted_samples() {
        let mut acc = WidomAccumulator::default();
        // Weight 2 on dU=0 (exp=1), weight 1 on dU=1 (exp=e^-1)
        // Weighted mean = (2*1 + 1*exp(-1)) / 3
        acc.collect(0.0, 2.0);
        acc.collect(1.0, 1.0);
        let expected = -((2.0 + (-1.0_f64).exp()) / 3.0).ln();
        assert_approx_eq!(f64, acc.mean_free_energy(), expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_weight_ignored() {
        let mut acc = WidomAccumulator::default();
        acc.collect(0.0, 0.0);
        assert!(acc.is_empty());
    }

    #[test]
    fn block_averaging() {
        let mut acc = WidomAccumulator::default();

        // Block 1: dU=0 → free_energy = 0
        acc.collect(0.0, 1.0);
        acc.end_block();

        // Block 2: dU=2 → free_energy = 2
        acc.collect(2.0, 1.0);
        acc.end_block();

        // Block mean ≈ 1.0, error > 0
        assert_approx_eq!(f64, acc.free_energy().mean(), 1.0, epsilon = 1e-12);
        assert!(acc.free_energy().stddev() > 0.0);

        // Total accumulator is never reset: 2 samples, overall mean = -ln((1 + exp(-2)) / 2)
        assert_eq!(acc.len(), 2);
        let expected = -((1.0 + (-2.0_f64).exp()) / 2.0).ln();
        assert_approx_eq!(f64, acc.mean_free_energy(), expected, epsilon = 1e-12);
    }

    #[test]
    fn with_block_size_auto_finalizes() {
        let n = NonZeroUsize::new(2).unwrap();
        let mut acc = WidomAccumulator::default().with_block_size(n);
        // Block 1: free_energy = 0
        acc.collect(0.0, 1.0);
        acc.collect(0.0, 1.0);
        assert_eq!(acc.free_energy().n(), 1);
        // Block 2: free_energy = 2
        acc.collect(2.0, 1.0);
        acc.collect(2.0, 1.0);
        assert_eq!(acc.free_energy().n(), 2);
        // Total accumulator still reflects all 4 samples.
        assert_eq!(acc.len(), 4);
        // Distinct per-block free energies (0, 2) → between-block variance > 0.
        assert!(acc.free_energy().stddev() > 0.0);
    }

    #[test]
    fn ln_add_exp_basic() {
        assert_approx_eq!(f64, ln_add_exp(0.0, 0.0), (2.0_f64).ln());
        assert_approx_eq!(f64, ln_add_exp(100.0, 0.0), 100.0, epsilon = 1e-10);
        assert_approx_eq!(f64, ln_add_exp(0.0, 100.0), 100.0, epsilon = 1e-10);
    }
}
