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

//! Rotational diffusion analysis via the quaternion covariance matrix.
//!
//! Estimates the anisotropic rotational diffusion tensor from group quaternions
//! using the method of [Favro (1960)](https://doi.org/10.1103/PhysRev.119.53)
//! and [Holtbrügge & Schäfer (2025)](https://doi.org/10.1101/2025.05.27.656261).

use super::{Analyze, Frequency};
use crate::auxiliary::{ColumnWriter, MappingExt};
use crate::selection::{Selection, SelectionCache};
use crate::Context;
use anyhow::Result;
use average::{Estimate, Variance};
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

const fn default_max_lag() -> usize {
    1000
}

/// Number of upper-triangle elements in a 3×3 symmetric matrix.
const UPPER_TRIANGLE: usize = 6;

/// Geometric ratio for log-spaced lag selection (~13 lags per decade).
const LOG_LAG_RATIO: f64 = 1.2;

/// Minimum Tr(Q̃) at max_lag to consider the correlation converged.
/// The plateau is ¾ = 0.75; this requires reaching ~20% of it.
const TRACE_CONVERGENCE_THRESHOLD: f64 = 0.15;

/// YAML builder for [`RotationalDiffusion`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationalDiffusionBuilder {
    pub selection: Selection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    pub frequency: Frequency,
    #[serde(default = "default_max_lag")]
    pub max_lag: usize,
}

impl RotationalDiffusionBuilder {
    pub fn build(&self, context: &impl Context) -> Result<RotationalDiffusion> {
        let topology = context.topology_ref();
        let groups = context.groups();
        let group_indices = self.selection.resolve_groups(topology, groups);
        if group_indices.is_empty() {
            anyhow::bail!(
                "RotationalDiffusion: selection '{}' matched no groups",
                self.selection.source()
            );
        }

        for &gi in &group_indices {
            let mol_id = groups[gi].molecule();
            if topology.moleculekinds()[mol_id].atomic() {
                anyhow::bail!(
                    "RotationalDiffusion: selection '{}' matched atomic group {} (molecule '{}'); \
                     only molecular groups with rigid-body orientation are supported",
                    self.selection.source(),
                    gi,
                    topology.moleculekinds()[mol_id].name()
                );
            }
        }

        anyhow::ensure!(self.max_lag > 0, "RotationalDiffusion: max_lag must be > 0");

        let log_lags = log_spaced_lags(self.max_lag);
        let writer = if let Some(path) = &self.file {
            let mut headers: Vec<String> = vec!["step".into()];
            for &lag in &log_lags {
                headers.push(format!("q1q1_lag{lag}"));
                headers.push(format!("q2q2_lag{lag}"));
                headers.push(format!("q3q3_lag{lag}"));
            }
            let header_refs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
            Some(ColumnWriter::open(path, &header_refs)?)
        } else {
            None
        };

        let lag_to_index: HashMap<usize, usize> = log_lags
            .iter()
            .enumerate()
            .map(|(idx, &lag)| (lag, idx))
            .collect();
        let num_lags = log_lags.len();

        Ok(RotationalDiffusion {
            selection: self.selection.clone(),
            frequency: self.frequency,
            num_samples: 0,
            snapshots: HashMap::new(),
            covariance: (0..num_lags)
                .map(|_| std::array::from_fn(|_| Variance::new()))
                .collect(),
            lag_to_index,
            group_cache: SelectionCache::default(),
            writer,
            log_lags,
        })
    }
}

/// Rotational diffusion analysis via the quaternion covariance matrix.
///
/// Computes the 3×3 quaternion covariance matrix Q̃(τ) from group orientations,
/// averaging over all molecules matching the selection. The covariance at each
/// lag τ contains six independent rotational correlation functions that encode
/// the anisotropic diffusion tensor.
///
/// See [Favro (1960)](https://doi.org/10.1103/PhysRev.119.53) and
/// [Holtbrügge & Schäfer (2025)](https://doi.org/10.1101/2025.05.27.656261).
#[derive(Debug)]
pub struct RotationalDiffusion {
    selection: Selection,
    frequency: Frequency,
    num_samples: usize,
    /// Per-group quaternion ring buffers, keyed by group index.
    snapshots: HashMap<usize, VecDeque<crate::UnitQuaternion>>,
    /// Q̃_ij(τ) and Ṽ_ij(τ) accumulators at log-spaced lags, shared across all molecules.
    /// Indexed by position in `log_lags`, not by lag value directly.
    covariance: Vec<[Variance; UPPER_TRIANGLE]>,
    /// Reverse lookup: lag value → index into `covariance`. Only log-spaced lags have entries.
    lag_to_index: HashMap<usize, usize>,
    group_cache: SelectionCache,
    #[debug(skip)]
    writer: Option<ColumnWriter>,
    log_lags: Vec<usize>,
}

impl RotationalDiffusion {
    fn max_lag(&self) -> usize {
        *self.log_lags.last().unwrap_or(&0)
    }

    /// Iterate over log-spaced lags that have accumulated data.
    fn valid_lags(&self) -> impl Iterator<Item = (usize, &[Variance; UPPER_TRIANGLE])> {
        self.log_lags.iter().filter_map(|&lag| {
            let idx = *self.lag_to_index.get(&lag)?;
            let c = &self.covariance[idx];
            if c[0].is_empty() {
                return None;
            }
            Some((lag, c))
        })
    }
}

/// Generate log-spaced lag indices from 1 to max_lag (inclusive, deduplicated).
fn log_spaced_lags(max_lag: usize) -> Vec<usize> {
    if max_lag == 0 {
        return vec![];
    }
    let mut lags = Vec::new();
    let mut value = 1.0_f64;
    while (value as usize) <= max_lag {
        let lag = value as usize;
        if lags.last() != Some(&lag) {
            lags.push(lag);
        }
        value *= LOG_LAG_RATIO;
    }
    if lags.last() != Some(&max_lag) {
        lags.push(max_lag);
    }
    lags
}

/// Compute time-dependent diffusion coefficients from Q̃ eigenvalues.
///
/// Implements Holtbrügge Eq. 11 using the Favro model. Eigenvalues must be
/// sorted ascending (Q_11 ≤ Q_22 ≤ Q_33). Returns sorted `[D_x, D_y, D_z]`
/// in rad²/snapshot, or `None` if the argument to ln is non-positive.
fn diffusion_from_eigenvalues(eigenvalues: &[f64; 3], lag: usize) -> Option<[f64; 3]> {
    let tau = lag as f64;
    let [q11, q22, q33] = *eigenvalues;

    // Eq. 11: D_i = 1/(2τ) ln[(1-2Q_j-2Q_k) / ((1-2Q_i-2Q_j)(1-2Q_i-2Q_k))]
    // Cyclic permutations: (i,j,k) = (x,y,z), (y,z,x), (z,x,y)
    let permutations = [(q11, q22, q33), (q22, q33, q11), (q33, q11, q22)];
    let mut d = [0.0; 3];
    for (idx, &(qi, qj, qk)) in permutations.iter().enumerate() {
        let numerator = 1.0 - 2.0 * qj - 2.0 * qk;
        let denom = (1.0 - 2.0 * qi - 2.0 * qj) * (1.0 - 2.0 * qi - 2.0 * qk);
        let arg = numerator / denom;
        if arg <= 0.0 {
            return None;
        }
        d[idx] = arg.ln() / (2.0 * tau);
    }
    d.sort_unstable_by(|a, b| a.total_cmp(b));
    Some(d)
}

/// Build a 3×3 symmetric matrix from upper-triangle values.
fn matrix_from_upper_triangle(vals: &[f64; UPPER_TRIANGLE]) -> nalgebra::Matrix3<f64> {
    nalgebra::Matrix3::new(
        vals[0], vals[1], vals[2], vals[1], vals[3], vals[4], vals[2], vals[4], vals[5],
    )
}

/// Accumulate the upper triangle of the quaternion vector outer product.
fn accumulate_covariance(accum: &mut [Variance; UPPER_TRIANGLE], v: &nalgebra::Vector3<f64>) {
    let (x, y, z) = (v.x, v.y, v.z);
    accum[0].add(x * x);
    accum[1].add(x * y);
    accum[2].add(x * z);
    accum[3].add(y * y);
    accum[4].add(y * z);
    accum[5].add(z * z);
}

impl crate::Info for RotationalDiffusion {
    fn short_name(&self) -> Option<&'static str> {
        Some("rotational_diffusion")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Rotational diffusion via quaternion covariance")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("doi:10.1103/PhysRev.119.53")
    }
}

impl<T: Context> Analyze<T> for RotationalDiffusion {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, step: usize, _weight: f64) -> Result<()> {
        let gen = context.group_lists_generation();
        // Must copy: the borrow from `get_or_resolve` conflicts with `self.snapshots` mutation
        let group_indices = self
            .group_cache
            .get_or_resolve(gen, || context.resolve_groups_live(&self.selection))
            .to_vec();

        self.snapshots.retain(|gi, _| group_indices.contains(gi));

        let max_lag = self.max_lag();
        for &gi in &group_indices {
            let q = *context.groups()[gi].quaternion();
            let buf = self
                .snapshots
                .entry(gi)
                .or_insert_with(|| VecDeque::with_capacity(max_lag + 1));
            if buf.len() > max_lag {
                buf.pop_front();
            }
            buf.push_back(q);
        }

        for &gi in &group_indices {
            let buf = &self.snapshots[&gi];
            let n = buf.len();
            if n < 2 {
                continue;
            }
            let q_current_inv = buf[n - 1].inverse();
            for (&lag, &cov_idx) in &self.lag_to_index {
                if lag >= n {
                    continue;
                }
                let q_past = buf[n - 1 - lag];
                // Body-frame reorientation (Holtbrügge Eq. 5)
                let q_rel = q_past * q_current_inv;
                accumulate_covariance(&mut self.covariance[cov_idx], &q_rel.vector().into_owned());
            }
        }

        if let Some(ref mut writer) = self.writer {
            let mut cells: Vec<String> = Vec::with_capacity(1 + self.log_lags.len() * 3);
            cells.push(step.to_string());
            for &lag in &self.log_lags {
                if let Some(c) = self
                    .lag_to_index
                    .get(&lag)
                    .and_then(|&i| self.covariance.get(i))
                {
                    cells.push(format!("{:.6e}", c[0].mean()));
                    cells.push(format!("{:.6e}", c[3].mean()));
                    cells.push(format!("{:.6e}", c[5].mean()));
                } else {
                    cells.extend(["nan".into(), "nan".into(), "nan".into()]);
                }
            }
            let refs: Vec<&dyn std::fmt::Display> =
                cells.iter().map(|s| s as &dyn std::fmt::Display).collect();
            writer.write_row(&refs)?;
        }

        self.num_samples += 1;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        if self.num_samples == 0 {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.num_samples)?;

        let mut cov_list = Vec::new();
        for (lag, c) in self.valid_lags() {
            let mut entry = serde_yml::Mapping::new();
            entry.try_insert("lag", lag)?;
            entry.try_insert("q1q1", c[0].mean())?;
            entry.try_insert("q1q2", c[1].mean())?;
            entry.try_insert("q1q3", c[2].mean())?;
            entry.try_insert("q2q2", c[3].mean())?;
            entry.try_insert("q2q3", c[4].mean())?;
            entry.try_insert("q3q3", c[5].mean())?;
            entry.try_insert("var_q1q1", c[0].sample_variance())?;
            entry.try_insert("var_q2q2", c[3].sample_variance())?;
            entry.try_insert("var_q3q3", c[5].sample_variance())?;
            entry.try_insert("samples", c[0].len())?;
            cov_list.push(serde_yml::Value::Mapping(entry));
        }
        map.insert("covariance".into(), serde_yml::Value::Sequence(cov_list));

        let mut d_list = Vec::new();
        for (lag, c) in self.valid_lags() {
            let means: [f64; UPPER_TRIANGLE] = std::array::from_fn(|i| c[i].mean());
            let mut eigenvalues = matrix_from_upper_triangle(&means).symmetric_eigenvalues();
            eigenvalues
                .as_mut_slice()
                .sort_unstable_by(|a, b| a.total_cmp(b));
            let evals = [eigenvalues[0], eigenvalues[1], eigenvalues[2]];

            if let Some([dx, dy, dz]) = diffusion_from_eigenvalues(&evals, lag) {
                let mut entry = serde_yml::Mapping::new();
                entry.try_insert("lag", lag)?;
                entry.try_insert("Dx", dx)?;
                entry.try_insert("Dy", dy)?;
                entry.try_insert("Dz", dz)?;
                // Anisotropy Δ = 2D_z/(D_x+D_y) (Holtbrügge Eq. 17)
                let dxy_sum = dx + dy;
                if dxy_sum > 0.0 {
                    entry.try_insert("anisotropy", 2.0 * dz / dxy_sum)?;
                }
                // Rhombicity ρ = 3/2(D_y-D_x)/(D_z - ½(D_x+D_y)) (Holtbrügge Eq. 18)
                let denom = dz - 0.5 * dxy_sum;
                if denom.abs() > 1e-30 {
                    entry.try_insert("rhombicity", 1.5 * (dy - dx) / denom)?;
                }
                d_list.push(serde_yml::Value::Mapping(entry));
            }
        }
        if !d_list.is_empty() {
            map.insert(
                "diffusion_coefficients".into(),
                serde_yml::Value::Sequence(d_list),
            );
        }

        // Isotropic fit of Tr(Q̃(τ)) = ¾(1 - exp(-2Dτ)) over all valid lags
        let (fit_lags, fit_trace): (Vec<f64>, Vec<f64>) = self
            .valid_lags()
            .map(|(lag, c)| (lag as f64, c[0].mean() + c[3].mean() + c[5].mean()))
            .unzip();

        if let Some(d_iso) = crate::auxiliary::fit_isotropic_d_rot(&fit_lags, &fit_trace) {
            map.try_insert("d_rot_isotropic", d_iso)?;
        }
        if let Some(&tr) = fit_trace.last() {
            if tr < TRACE_CONVERGENCE_THRESHOLD {
                map.try_insert(
                    "warning",
                    "Tr(Q) below convergence threshold at max_lag; consider increasing max_lag",
                )?;
            }
        }

        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnitQuaternion;
    use float_cmp::assert_approx_eq;

    #[test]
    fn log_spaced_lags_basic() {
        let lags = log_spaced_lags(100);
        assert_eq!(*lags.first().unwrap(), 1);
        assert_eq!(*lags.last().unwrap(), 100);
        for w in lags.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn identity_quaternions_give_zero_covariance() {
        let mut covariance: Vec<[Variance; UPPER_TRIANGLE]> = (0..5)
            .map(|_| std::array::from_fn(|_| Variance::new()))
            .collect();

        let q = UnitQuaternion::identity();
        let buf: Vec<UnitQuaternion> = vec![q; 6];

        for lag in 1..=5 {
            for i in 0..buf.len() - lag {
                let q_rel = buf[i] * buf[i + lag].inverse();
                accumulate_covariance(&mut covariance[lag - 1], &q_rel.vector().into_owned());
            }
        }

        for lag_idx in 0..5 {
            assert_approx_eq!(f64, covariance[lag_idx][0].mean(), 0.0, epsilon = 1e-14);
            assert_approx_eq!(f64, covariance[lag_idx][3].mean(), 0.0, epsilon = 1e-14);
            assert_approx_eq!(f64, covariance[lag_idx][5].mean(), 0.0, epsilon = 1e-14);
            assert_approx_eq!(
                f64,
                covariance[lag_idx][0].sample_variance(),
                0.0,
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn known_rotation_gives_expected_covariance() {
        let angle = 0.1;
        let axis = nalgebra::UnitVector3::new_normalize(nalgebra::Vector3::new(0.0, 0.0, 1.0));
        let q_step = UnitQuaternion::from_axis_angle(&axis, angle);

        let n = 20;
        let mut quaternions = Vec::with_capacity(n);
        let mut q = UnitQuaternion::identity();
        for _ in 0..n {
            quaternions.push(q);
            q = q_step * q;
        }

        let mut q1q1 = Variance::new();
        let mut q2q2 = Variance::new();
        let mut q3q3 = Variance::new();
        for i in 0..n - 1 {
            let q_rel = quaternions[i] * quaternions[i + 1].inverse();
            let v = q_rel.vector();
            q1q1.add(v.x * v.x);
            q2q2.add(v.y * v.y);
            q3q3.add(v.z * v.z);
        }

        // Rotation about z by angle θ: q_rel = (cos(θ/2), 0, 0, sin(θ/2))
        let expected_q3q3 = (angle / 2.0).sin().powi(2);
        assert_approx_eq!(f64, q1q1.mean(), 0.0, epsilon = 1e-14);
        assert_approx_eq!(f64, q2q2.mean(), 0.0, epsilon = 1e-14);
        assert_approx_eq!(f64, q3q3.mean(), expected_q3q3, epsilon = 1e-10);
        assert_approx_eq!(f64, q3q3.sample_variance(), 0.0, epsilon = 1e-20);
    }

    #[test]
    fn matrix_from_upper_triangle_is_symmetric() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = matrix_from_upper_triangle(&vals);
        assert_approx_eq!(f64, m[(0, 1)], m[(1, 0)]);
        assert_approx_eq!(f64, m[(0, 2)], m[(2, 0)]);
        assert_approx_eq!(f64, m[(1, 2)], m[(2, 1)]);
    }

    #[test]
    fn random_rotation_matches_favro_isotropic() {
        use crate::transform::random_quaternion;
        use rand::prelude::*;

        let max_angle = 0.15;
        let n_molecules = 200;
        let n_steps = 800;
        let max_lag = 50;

        let mut rng = StdRng::seed_from_u64(42);

        let mut trajectories = Vec::with_capacity(n_molecules);
        for _ in 0..n_molecules {
            let mut q = crate::UnitQuaternion::identity();
            let mut traj = Vec::with_capacity(n_steps);
            traj.push(q);
            for _ in 1..n_steps {
                let (q_step, _angle) = random_quaternion(&mut rng, max_angle);
                q = q_step * q;
                traj.push(q);
            }
            trajectories.push(traj);
        }

        let mut covariance: Vec<[Variance; UPPER_TRIANGLE]> = (0..max_lag)
            .map(|_| std::array::from_fn(|_| Variance::new()))
            .collect();

        for traj in &trajectories {
            for lag in 1..=max_lag {
                for t in 0..traj.len() - lag {
                    let q_rel = traj[t] * traj[t + lag].inverse();
                    accumulate_covariance(&mut covariance[lag - 1], &q_rel.vector().into_owned());
                }
            }
        }

        // Trace should increase monotonically toward 0.75
        let mut prev_trace = 0.0;
        for lag_idx in 0..max_lag {
            let trace = covariance[lag_idx][0].mean()
                + covariance[lag_idx][3].mean()
                + covariance[lag_idx][5].mean();
            assert!(trace >= prev_trace - 1e-6, "trace should increase with lag");
            assert!(trace <= 0.76, "trace should not exceed 3/4");
            prev_trace = trace;
        }

        // Off-diagonals should be near zero (isotropic)
        for lag in [1, 5, 10, 20, 40] {
            let lag_idx = lag - 1;
            let off_diag = covariance[lag_idx][1].mean().abs()
                + covariance[lag_idx][2].mean().abs()
                + covariance[lag_idx][4].mean().abs();
            assert!(
                off_diag < 0.01,
                "lag={lag}: off-diagonal sum={off_diag:.6}, should be ~0"
            );
        }

        // Diagonal elements should be approximately equal (isotropic)
        for lag in [5, 10, 20] {
            let lag_idx = lag - 1;
            let q11 = covariance[lag_idx][0].mean();
            let q22 = covariance[lag_idx][3].mean();
            let q33 = covariance[lag_idx][5].mean();
            let mean_diag = (q11 + q22 + q33) / 3.0;
            for &qii in &[q11, q22, q33] {
                let rel_diff = (qii - mean_diag).abs() / mean_diag.max(1e-10);
                assert!(
                    rel_diff < 0.15,
                    "lag={lag}: diagonal elements should be similar for isotropic diffusion"
                );
            }
        }

        // Fitted model should reproduce the observed trace
        let fit_lags: Vec<f64> = (1..=max_lag).map(|i| i as f64).collect();
        let fit_trace: Vec<f64> = (0..max_lag)
            .map(|i| covariance[i][0].mean() + covariance[i][3].mean() + covariance[i][5].mean())
            .collect();
        let d_fit = crate::auxiliary::fit_isotropic_d_rot(&fit_lags, &fit_trace).unwrap();
        assert!(d_fit > 0.0, "fitted D should be positive");

        for (i, (&tau, &tr_obs)) in fit_lags.iter().zip(fit_trace.iter()).enumerate() {
            if i % 10 != 0 {
                continue;
            }
            let tr_model = 0.75 * (1.0 - (-2.0 * d_fit * tau).exp());
            let abs_err = (tr_model - tr_obs).abs();
            assert!(
                abs_err < 0.02,
                "lag={tau}: model={tr_model:.6}, observed={tr_obs:.6}, abs_err={abs_err:.4}"
            );
        }
    }

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
selection: "molecule Water"
frequency: !Every 100
max_lag: 500
"#;
        let builder: RotationalDiffusionBuilder = serde_yml::from_str(yaml).unwrap();
        assert_eq!(builder.max_lag, 500);
        assert!(builder.file.is_none());
    }

    #[test]
    fn deserialize_builder_with_defaults() {
        let yaml = r#"
selection: "molecule Water"
frequency: !Every 100
"#;
        let builder: RotationalDiffusionBuilder = serde_yml::from_str(yaml).unwrap();
        assert_eq!(builder.max_lag, 1000);
    }
}
