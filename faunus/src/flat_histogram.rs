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

//! Shared state for flat-histogram free-energy methods.
//!
//! Stores the density of states estimate `ln g(CV)`, visit histogram, and
//! modification factor `ln f` for Wang-Landau and related algorithms.
//! Designed for thread-safe sharing via `Arc<RwLock<FlatHistogramState>>`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 1D or 2D uniform grid metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GridDim {
    One {
        n: usize,
        min: f64,
        width: f64,
    },
    Two {
        n: [usize; 2],
        min: [f64; 2],
        width: [f64; 2],
    },
}

impl GridDim {
    /// Total number of bins across all dimensions.
    pub fn num_bins(&self) -> usize {
        match self {
            Self::One { n, .. } => *n,
            Self::Two { n, .. } => n[0] * n[1],
        }
    }

    /// Build a 1D grid from range and resolution.
    pub fn new_1d(min: f64, max: f64, resolution: f64) -> Result<Self> {
        anyhow::ensure!(
            max > min && resolution > 0.0,
            "require max > min and resolution > 0"
        );
        let n = ((max - min) / resolution) as usize;
        anyhow::ensure!(n > 0, "range too small for given resolution");
        Ok(Self::One {
            n,
            min,
            width: resolution,
        })
    }

    /// Build a 2D grid from ranges and resolutions.
    pub fn new_2d(min: [f64; 2], max: [f64; 2], resolution: [f64; 2]) -> Result<Self> {
        let n0 = ((max[0] - min[0]) / resolution[0]) as usize;
        let n1 = ((max[1] - min[1]) / resolution[1]) as usize;
        anyhow::ensure!(n0 > 0 && n1 > 0, "range too small for given resolution");
        Ok(Self::Two {
            n: [n0, n1],
            min,
            width: resolution,
        })
    }
}

/// Shared mutable state for Wang-Landau flat-histogram sampling.
///
/// See [Chevallier & Cazals](https://doi.org/10.1016/j.jcp.2020.109366)
/// for the exponential→1/t convergence scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatHistogramState {
    dim: GridDim,
    ln_g: Vec<f64>,
    histogram: Vec<f64>,
    ln_f: f64,
    total_updates: u64,
    num_flatness: u32,
    flatness_threshold: f64,
    min_flatness: u32,
    min_ln_f: f64,
    use_1_over_t: bool,
    converged: bool,
    /// Number of concurrent walkers; `update()` applies `ln_f / num_walkers`.
    #[serde(default = "default_num_walkers")]
    num_walkers: u32,
}

fn default_num_walkers() -> u32 {
    1
}

impl FlatHistogramState {
    /// Create a new flat-histogram state with all bins initialized to zero.
    pub fn new(
        dim: GridDim,
        flatness_threshold: f64,
        min_flatness: u32,
        min_ln_f: f64,
        ln_f_initial: f64,
        num_walkers: u32,
    ) -> Self {
        let num_bins = dim.num_bins();
        Self {
            dim,
            ln_g: vec![0.0; num_bins],
            histogram: vec![0.0; num_bins],
            ln_f: ln_f_initial,
            total_updates: 0,
            num_flatness: 0,
            flatness_threshold,
            min_flatness,
            min_ln_f,
            use_1_over_t: false,
            converged: false,
            num_walkers,
        }
    }

    /// Map CV value(s) to a bin index. Pass one value for 1D, two for 2D.
    pub fn bin_index(&self, cv: &[f64]) -> Option<usize> {
        match &self.dim {
            GridDim::One { n, min, width } => {
                let v = cv[0];
                if v < *min {
                    return None;
                }
                let i = ((v - min) / width) as usize;
                (i < *n).then_some(i)
            }
            GridDim::Two { n, min, width } => {
                if cv[0] < min[0] || cv[1] < min[1] {
                    return None;
                }
                let i0 = ((cv[0] - min[0]) / width[0]) as usize;
                let i1 = ((cv[1] - min[1]) / width[1]) as usize;
                if i0 < n[0] && i1 < n[1] {
                    Some(i0 * n[1] + i1)
                } else {
                    None
                }
            }
        }
    }

    /// Increment histogram and update density of states for a visited bin.
    ///
    /// The `ln_g` increment is scaled by `1/num_walkers` so the effective
    /// modification rate is independent of the number of concurrent walkers.
    /// In 1/t mode, `total_updates` is also scaled so the convergence rate
    /// is consistent regardless of walker count.
    pub fn update(&mut self, bin: usize) {
        self.histogram[bin] += 1.0;
        self.total_updates += 1;
        if self.use_1_over_t {
            // Scale by num_walkers so 10 walkers don't converge 10× faster
            let effective_t = self.total_updates as f64 / self.num_walkers as f64;
            self.ln_f = 1.0 / (effective_t + 1.0);
        }
        self.ln_g[bin] += self.ln_f / self.num_walkers as f64;
    }

    /// Check histogram flatness; if flat, reduce `ln_f` and reset histogram.
    /// Returns `true` if the histogram was flat and `ln_f` was reduced.
    pub fn check_and_reduce(&mut self) -> bool {
        if self.converged {
            return false;
        }
        let ratio = self.flatness_ratio();
        if ratio < self.flatness_threshold {
            return false;
        }

        self.num_flatness += 1;
        self.histogram.fill(0.0);

        if !self.use_1_over_t && self.num_flatness >= self.min_flatness {
            self.use_1_over_t = true;
            log::info!(
                "Switching to 1/t regime after {} flatness checks",
                self.num_flatness
            );
        } else if !self.use_1_over_t {
            // Exponential regime: halve ln_f
            self.ln_f /= 2.0;
        }

        if self.ln_f < self.min_ln_f {
            self.converged = true;
        }
        true
    }

    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Read `ln g` for a bin (dimensionless, in units of kT).
    pub fn ln_g(&self, bin: usize) -> f64 {
        self.ln_g[bin]
    }

    /// Current modification factor.
    pub fn ln_f(&self) -> f64 {
        self.ln_f
    }

    /// Number of flatness checks passed.
    pub fn num_flatness(&self) -> u32 {
        self.num_flatness
    }

    /// Required flatness checks before switching to 1/t regime.
    pub fn min_flatness(&self) -> u32 {
        self.min_flatness
    }

    /// Flatness ratio: min(H) / mean(H) over ALL bins.
    ///
    /// Unvisited bins have count 0, making the ratio 0 until every bin
    /// has been visited — this prevents premature flatness detection.
    pub fn flatness_ratio(&self) -> f64 {
        let mean = self.histogram.iter().sum::<f64>() / self.histogram.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }
        self.histogram.iter().copied().fold(f64::INFINITY, f64::min) / mean
    }

    /// Range of ln g values (max - min), representing the free energy span in kT.
    pub fn ln_g_range(&self) -> f64 {
        let min = self.ln_g.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self.ln_g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    /// Grid dimension metadata.
    pub fn dim(&self) -> &GridDim {
        &self.dim
    }

    /// Write free energy surface to CSV (dispatches on grid dimensionality).
    pub fn write_free_energy(&self, path: &Path) -> Result<()> {
        use crate::auxiliary::ColumnWriter;
        let ln_g_min = self.ln_g.iter().copied().fold(f64::INFINITY, f64::min);
        match &self.dim {
            GridDim::One { n, min, width } => {
                let mut w = ColumnWriter::open(path, &["cv", "free_energy_kT"])?;
                for i in 0..*n {
                    let cv = min + (i as f64 + 0.5) * width;
                    let fe = self.ln_g[i] - ln_g_min;
                    w.write_row(&[&format!("{cv:.4}"), &format!("{fe:.6}")])?;
                }
            }
            GridDim::Two { n, min, width } => {
                let mut w = ColumnWriter::open(path, &["cv1", "cv2", "free_energy_kT"])?;
                for i0 in 0..n[0] {
                    let cv1 = min[0] + (i0 as f64 + 0.5) * width[0];
                    for i1 in 0..n[1] {
                        let cv2 = min[1] + (i1 as f64 + 0.5) * width[1];
                        let fe = self.ln_g[i0 * n[1] + i1] - ln_g_min;
                        w.write_row(&[
                            &format!("{cv1:.4}"),
                            &format!("{cv2:.4}"),
                            &format!("{fe:.6}"),
                        ])?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Serialize to a YAML checkpoint file.
    pub fn to_file(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_yml::to_writer(file, self)?;
        Ok(())
    }

    /// Deserialize from a YAML checkpoint file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let state: Self = serde_yml::from_reader(file)?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bin_index_1d() {
        let state = FlatHistogramState::new(
            GridDim::new_1d(0.0, 10.0, 1.0).unwrap(),
            0.8,
            20,
            1e-6,
            1.0,
            1,
        );
        assert_eq!(state.bin_index(&[0.5]), Some(0));
        assert_eq!(state.bin_index(&[9.5]), Some(9));
        assert_eq!(state.bin_index(&[-0.1]), None);
        assert_eq!(state.bin_index(&[10.0]), None);
    }

    #[test]
    fn bin_index_2d() {
        let state = FlatHistogramState::new(
            GridDim::new_2d([0.0, 0.0], [4.0, 6.0], [2.0, 3.0]).unwrap(),
            0.8,
            20,
            1e-6,
            1.0,
            1,
        );
        // 2x2 grid, row-major: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        assert_eq!(state.bin_index(&[0.5, 0.5]), Some(0));
        assert_eq!(state.bin_index(&[0.5, 3.5]), Some(1));
        assert_eq!(state.bin_index(&[2.5, 0.5]), Some(2));
        assert_eq!(state.bin_index(&[2.5, 3.5]), Some(3));
        assert_eq!(state.bin_index(&[4.0, 0.5]), None);
    }

    #[test]
    fn update_increments_histogram_and_ln_g() {
        let mut state = FlatHistogramState::new(
            GridDim::new_1d(0.0, 10.0, 1.0).unwrap(),
            0.8,
            20,
            1e-6,
            1.0,
            1,
        );
        state.update(3);
        assert_relative_eq!(state.histogram[3], 1.0);
        assert_relative_eq!(state.ln_g[3], 1.0);

        state.update(3);
        assert_relative_eq!(state.histogram[3], 2.0);
        assert_relative_eq!(state.ln_g[3], 2.0);
    }

    #[test]
    fn flatness_ratio_calculation() {
        let mut state = FlatHistogramState::new(
            GridDim::new_1d(0.0, 4.0, 1.0).unwrap(),
            0.8,
            20,
            1e-6,
            1.0,
            1,
        );
        // Empty histogram → ratio = 0
        assert_relative_eq!(state.flatness_ratio(), 0.0);

        // All bins equal → ratio = 1
        state.histogram = vec![10.0, 10.0, 10.0, 10.0];
        assert_relative_eq!(state.flatness_ratio(), 1.0);

        // One unvisited bin → ratio = 0 (min=0)
        state.histogram = vec![10.0, 10.0, 10.0, 0.0];
        assert_relative_eq!(state.flatness_ratio(), 0.0);

        // Uneven: min=5, mean=10 → ratio = 0.5
        state.histogram = vec![5.0, 10.0, 10.0, 15.0];
        assert_relative_eq!(state.flatness_ratio(), 0.5);
    }

    #[test]
    fn check_and_reduce_exponential() {
        let mut state = FlatHistogramState::new(
            GridDim::new_1d(0.0, 4.0, 1.0).unwrap(),
            0.8,
            20,
            1e-6,
            1.0,
            1,
        );
        // Not flat → no reduction (min/mean = 1/10 = 0.1 < 0.8)
        state.histogram = vec![1.0, 10.0, 10.0, 10.0];
        assert!(!state.check_and_reduce());
        assert_relative_eq!(state.ln_f, 1.0);

        // Flat → halve ln_f
        state.histogram = vec![10.0, 10.0, 10.0, 10.0];
        assert!(state.check_and_reduce());
        assert_relative_eq!(state.ln_f, 0.5);
        assert_eq!(state.num_flatness, 1);
        // Histogram reset
        assert_relative_eq!(state.histogram.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn switch_to_1_over_t() {
        let mut state = FlatHistogramState::new(
            GridDim::new_1d(0.0, 4.0, 1.0).unwrap(),
            0.8,
            2,
            1e-6,
            1.0,
            1,
        );
        // Two flatness checks → switch to 1/t
        state.histogram = vec![10.0; 4];
        state.check_and_reduce(); // num_flatness=1, ln_f=0.5
        assert!(!state.use_1_over_t);

        state.histogram = vec![10.0; 4];
        state.check_and_reduce(); // num_flatness=2, switch to 1/t
        assert!(state.use_1_over_t);
    }

    #[test]
    fn convergence_detection() {
        let mut state = FlatHistogramState::new(
            GridDim::new_1d(0.0, 4.0, 1.0).unwrap(),
            0.8,
            20,
            0.1,
            0.05,
            1,
        );
        // ln_f starts at 0.05 < min_ln_f=0.1 → converged on first flat check
        state.histogram = vec![10.0; 4];
        state.check_and_reduce();
        assert!(state.is_converged());
    }

    #[test]
    fn checkpoint_roundtrip() {
        let mut state = FlatHistogramState::new(
            GridDim::new_1d(0.0, 10.0, 1.0).unwrap(),
            0.8,
            20,
            1e-6,
            1.0,
            1,
        );
        state.update(3);
        state.update(7);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.yaml");
        state.to_file(&path).unwrap();

        let loaded = FlatHistogramState::from_file(&path).unwrap();
        assert_relative_eq!(loaded.ln_g[3], state.ln_g[3]);
        assert_relative_eq!(loaded.ln_g[7], state.ln_g[7]);
        assert_eq!(loaded.total_updates, state.total_updates);
    }
}
