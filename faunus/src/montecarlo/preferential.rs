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

//! Preferential sampling for distance-biased atom selection.
//!
//! Biases particle selection toward a reference molecule using distance-dependent weights,
//! with a corresponding acceptance correction to maintain detailed balance.
//! See [Owicki & Scheraga, 1977](https://doi.org/10.1016/0009-2614(77)85051-3)
//! and [Allen & Tildesley, 2017](https://doi.org/10.1093/oso/9780198803195.001.0001) §9.3.1, eqns 9.42–9.44.

use crate::cell::BoundaryConditions;
use crate::{Context, Point};
use average::{Estimate, Mean};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// How to measure distance between an atom and the reference group.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Surface-to-surface distance to the reference bounding sphere.
    /// `d = max(0, |pos - cm_ref| - R_ref)`
    #[default]
    BoundingSphere,
    /// Distance to the reference mass center.
    MassCenter,
}

fn default_exponent() -> f64 {
    2.0
}

fn default_offset() -> f64 {
    1.0
}

/// Distance-biased atom selection with detailed-balance correction.
///
/// Biases selection toward a reference group using weight `W'(r) = (r + offset)^{-ν}`,
/// where `r` is measured by the configured [`DistanceMetric`]. The acceptance criterion
/// must include `ln(W_new / W_old)` to maintain detailed balance
/// ([Allen & Tildesley, 2017](https://doi.org/10.1093/oso/9780198803195.001.0001), eqn 9.44).
///
/// Weights are cached and incrementally updated: only the previously moved atom's
/// weight is recomputed between repeat iterations, giving O(1) amortized cost.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferentialSampling {
    /// Reference molecule name (resolved to id during finalize)
    reference: String,
    #[serde(skip)]
    reference_id: Option<usize>,
    /// Distance metric
    #[serde(default)]
    metric: DistanceMetric,
    /// Exponent ν in W'(r) = (r + offset)^{-ν}
    #[serde(default = "default_exponent")]
    exponent: f64,
    /// Offset to avoid singularity at r=0 (Angstrom)
    #[serde(default = "default_offset")]
    offset: f64,
    /// Per-candidate weights, rebuilt lazily
    #[serde(skip)]
    weights: Vec<f64>,
    /// Sum of weights Σ W'(r_j)
    #[serde(skip)]
    w_total: f64,
    /// Stored ln(W_new / W_old) for acceptance correction
    #[serde(skip)]
    ln_bias: f64,
    /// Index into candidates array whose weight needs refresh after last move
    #[serde(skip)]
    dirty_index: Option<usize>,
    /// Cumulative sum of ln(W_new / W_old) over all proposed moves
    #[serde(skip_deserializing)]
    sum_bias: f64,
    /// Running mean of |ln(W_new / W_old)| — diagnostic for bias correction magnitude
    #[serde(skip_deserializing)]
    mean_bias: Mean,
}

impl PreferentialSampling {
    /// Resolve reference molecule name to id.
    pub fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.reference_id = Some(super::find_molecule_id(
            context,
            &self.reference,
            "PreferentialSampling",
        )?);
        Ok(())
    }

    /// Weight function: `W'(r) = (r + offset)^{-exponent}`
    fn weight(&self, r: f64) -> f64 {
        (r + self.offset).powf(-self.exponent)
    }

    /// Distance from a point to the reference, given its mass center and bounding radius.
    fn distance_with_metric(
        &self,
        pos: &Point,
        ref_cm: &Point,
        ref_radius: f64,
        context: &impl Context,
    ) -> f64 {
        let dist = context.cell().distance(pos, ref_cm).norm();
        match self.metric {
            DistanceMetric::BoundingSphere => (dist - ref_radius).max(0.0),
            DistanceMetric::MassCenter => dist,
        }
    }

    /// Look up the reference group's mass center and bounding radius.
    fn reference_geometry(&self, context: &impl Context) -> (Point, f64) {
        let ref_id = self.reference_id.expect("not finalized");
        let groups = context.groups();
        let ref_group = groups
            .iter()
            .find(|g| g.molecule() == ref_id && !g.is_empty())
            .expect("reference group should exist and be non-empty");
        let cm = *ref_group
            .mass_center()
            .expect("reference group should have mass center");
        let radius = ref_group.bounding_radius().unwrap_or(0.0);
        (cm, radius)
    }

    /// Rebuild all weights from scratch.
    fn rebuild_weights(&mut self, candidates: &[usize], context: &impl Context) {
        let (ref_cm, ref_radius) = self.reference_geometry(context);
        self.weights = candidates
            .iter()
            .map(|&atom| {
                let pos = context.position(atom);
                let r = self.distance_with_metric(&pos, &ref_cm, ref_radius, context);
                self.weight(r)
            })
            .collect();
        self.w_total = self.weights.iter().sum();
        self.dirty_index = None;
    }

    /// Select a candidate atom weighted by distance to the reference.
    ///
    /// On first call, builds the full weight vector. On subsequent calls within
    /// the same repeat block, only updates the previously moved atom's weight.
    pub fn weighted_select(
        &mut self,
        context: &impl Context,
        candidates: &[usize],
        rng: &mut dyn RngCore,
    ) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        // Incremental update: only recompute the dirty atom's weight
        if let Some(dirty) = self.dirty_index {
            if dirty < self.weights.len() && self.weights.len() == candidates.len() {
                let (ref_cm, ref_radius) = self.reference_geometry(context);
                let pos = context.position(candidates[dirty]);
                let r = self.distance_with_metric(&pos, &ref_cm, ref_radius, context);
                let new_w = self.weight(r);
                self.w_total += new_w - self.weights[dirty];
                self.weights[dirty] = new_w;
                self.dirty_index = None;
            } else {
                self.rebuild_weights(candidates, context);
            }
        } else if self.weights.len() != candidates.len() {
            self.rebuild_weights(candidates, context);
        }

        // Weighted random selection via cumulative distribution
        let threshold = rng.r#gen::<f64>() * self.w_total;
        let mut cumulative = 0.0;
        for (i, &w) in self.weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                self.dirty_index = Some(i);
                return Some(candidates[i]);
            }
        }
        // Rounding fallback
        let last = candidates.len() - 1;
        self.dirty_index = Some(last);
        Some(candidates[last])
    }

    /// Compute the acceptance bias for a moved atom.
    ///
    /// Called after atom selection but before the move is applied.
    /// Uses the known displacement to predict the new position.
    pub fn compute_bias(&mut self, old_pos: &Point, new_pos: &Point, context: &impl Context) {
        let (ref_cm, ref_radius) = self.reference_geometry(context);
        let r_old = self.distance_with_metric(old_pos, &ref_cm, ref_radius, context);
        let r_new = self.distance_with_metric(new_pos, &ref_cm, ref_radius, context);
        let w_atom_old = self.weight(r_old);
        let w_atom_new = self.weight(r_new);
        let w_new = self.w_total - w_atom_old + w_atom_new;
        self.ln_bias = (w_new / self.w_total).ln();
        self.sum_bias += self.ln_bias;
        self.mean_bias.add(self.ln_bias.abs());
    }

    /// Returns the stored dimensionless bias `ln(W_new / W_old)`.
    pub fn ln_bias(&self) -> f64 {
        self.ln_bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    fn make_sampler(exponent: f64, offset: f64) -> PreferentialSampling {
        PreferentialSampling {
            reference: String::new(),
            reference_id: None,
            metric: DistanceMetric::BoundingSphere,
            exponent,
            offset,
            weights: Vec::new(),
            w_total: 0.0,
            ln_bias: 0.0,
            dirty_index: None,
            sum_bias: 0.0,
            mean_bias: Mean::new(),
        }
    }

    #[test]
    fn weight_function_values() {
        let ps = make_sampler(2.0, 1.0);
        assert_approx_eq!(f64, ps.weight(0.0), 1.0, epsilon = 1e-15); // (0+1)^-2
        assert_approx_eq!(f64, ps.weight(1.0), 0.25, epsilon = 1e-15); // (1+1)^-2
        assert_approx_eq!(f64, ps.weight(2.0), 1.0 / 9.0, epsilon = 1e-15); // (2+1)^-2
    }

    #[test]
    fn weight_with_custom_params() {
        let ps = make_sampler(3.0, 0.5);
        let expected = (2.5_f64 + 0.5).powf(-3.0);
        assert_approx_eq!(f64, ps.weight(2.5), expected, epsilon = 1e-15);
    }

    /// Verify the exact bias formula: ln((W - w(r_old) + w(r_new)) / W)
    ///
    /// Three atoms at distances 2, 5, 10 from reference with ν=2, offset=1:
    ///   w(2) = (3)^{-2} = 1/9
    ///   w(5) = (6)^{-2} = 1/36
    ///   w(10) = (11)^{-2} = 1/121
    ///   W = 1/9 + 1/36 + 1/121 = 484/4356 + 121/4356 + 36/4356 = 641/4356
    ///
    /// Atom 0 moves from distance 2 → 8:
    ///   w(8) = (9)^{-2} = 1/81
    ///   W_new = W - 1/9 + 1/81 = 641/4356 - 484/4356 + 4356/352836
    ///         = (641 - 484)/4356 + 1/81 = 157/4356 + 1/81
    ///   ln_bias = ln(W_new / W)
    #[test]
    fn bias_formula_exact() {
        let mut ps = make_sampler(2.0, 1.0);
        let distances = [2.0, 5.0, 10.0];
        let weights: Vec<f64> = distances.iter().map(|&r| ps.weight(r)).collect();

        assert_approx_eq!(f64, weights[0], 1.0 / 9.0, epsilon = 1e-15);
        assert_approx_eq!(f64, weights[1], 1.0 / 36.0, epsilon = 1e-15);
        assert_approx_eq!(f64, weights[2], 1.0 / 121.0, epsilon = 1e-15);

        ps.weights = weights.clone();
        ps.w_total = weights.iter().sum();

        let w_old = ps.w_total;
        let w_new = w_old - ps.weight(2.0) + ps.weight(8.0);
        let expected_bias = (w_new / w_old).ln();

        // Atom moves away from reference → W decreases → bias < 0
        // → exp(-bias) > 1 → move is easier to accept (correct: we WANT
        // to accept moves away, to counter the selection bias toward close atoms)
        assert!(expected_bias < 0.0);

        // Verify numerical value: W = 1/9 + 1/36 + 1/121 ≈ 0.14720
        // W_new = 1/81 + 1/36 + 1/121 ≈ 0.04819
        // ln(W_new/W) ≈ ln(0.04819/0.14720) ≈ -1.1155
        let w_exact = 1.0 / 9.0 + 1.0 / 36.0 + 1.0 / 121.0;
        let w_new_exact = 1.0 / 81.0 + 1.0 / 36.0 + 1.0 / 121.0;
        let expected_exact = f64::ln(w_new_exact / w_exact);
        assert_approx_eq!(f64, expected_bias, expected_exact, epsilon = 1e-14);
    }

    /// Moving an atom to the same distance produces zero bias.
    #[test]
    fn same_distance_gives_zero_bias() {
        let mut ps = make_sampler(2.0, 1.0);
        let weights = vec![ps.weight(3.0), ps.weight(7.0)];
        ps.weights = weights.clone();
        ps.w_total = weights.iter().sum();

        // Atom 0 moves from distance 3 → distance 3 (same)
        let w_new = ps.w_total - ps.weight(3.0) + ps.weight(3.0);
        let ln_bias = (w_new / ps.w_total).ln();
        assert_approx_eq!(f64, ln_bias, 0.0, epsilon = 1e-15);
    }

    /// Selection probability must equal w_i / W for each candidate.
    /// With fixed seed, verify the exact selected index.
    #[test]
    fn selection_probabilities() {
        let ps = make_sampler(2.0, 1.0);
        let distances = [1.0, 5.0, 20.0];
        let weights: Vec<f64> = distances.iter().map(|&r| ps.weight(r)).collect();
        let w_total: f64 = weights.iter().sum();

        // Analytical selection probabilities
        let p: Vec<f64> = weights.iter().map(|w| w / w_total).collect();
        // w(1) = 1/4, w(5) = 1/36, w(20) = 1/441
        // p(0) ≈ 0.25 / 0.280 ≈ 0.893 — closest atom dominates
        assert!(p[0] > 0.85);
        assert!(p[1] < 0.12);
        assert!(p[2] < 0.02);

        // Empirical check: count selections over many draws
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut counts = [0u64; 3];
        let n = 100_000;
        for _ in 0..n {
            let threshold = rng.r#gen::<f64>() * w_total;
            let mut cumulative = 0.0;
            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if cumulative >= threshold {
                    counts[i] += 1;
                    break;
                }
            }
        }

        // Empirical frequencies should match analytical probabilities within ~1%
        for (i, &count) in counts.iter().enumerate() {
            let empirical = count as f64 / n as f64;
            assert!(
                (empirical - p[i]).abs() < 0.01,
                "atom {i}: empirical={empirical:.4}, expected={:.4}",
                p[i]
            );
        }
    }

    /// Bias is antisymmetric: moving closer gives opposite sign to moving farther.
    #[test]
    fn bias_antisymmetry() {
        let ps = make_sampler(2.0, 1.0);
        let w5 = ps.weight(5.0);
        let w10 = ps.weight(10.0);
        let w15 = ps.weight(15.0);

        // Two atoms at distances 5 and 15; move atom 0 from 5 → 10
        let w_total = w5 + w15;
        let w_new_fwd = w_total - w5 + w10;
        let bias_fwd = (w_new_fwd / w_total).ln();

        // Move atom 0 from 10 → 5 (reverse)
        let w_total_rev = w10 + w15;
        let w_new_rev = w_total_rev - w10 + w5;
        let bias_rev = (w_new_rev / w_total_rev).ln();

        // Moving away: bias < 0 (easier accept). Moving closer: bias > 0 (harder accept).
        assert!(bias_fwd < 0.0);
        assert!(bias_rev > 0.0);
    }
}
