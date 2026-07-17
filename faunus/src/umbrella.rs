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

//! Multi-walker umbrella sampling with overlap-ratio free-energy stitching.
//!
//! Runs N hard-wall constrained windows in parallel, with an optional harmonic
//! drive phase to pull each walker into its target window. Adjacent-window
//! overlap fractions are used to stitch a PMF via
//! ΔF = −kT ln(f_i / f_{i+1}).

use crate::{
    analysis::{self, AnalysisCollection},
    auxiliary::ColumnWriter,
    backend::Backend,
    collective_variable::{CvKindBuilder, Finite, ForceConstant, Interval},
    context::WithHamiltonianMut,
    energy::{ConstrainBuilder, EnergyTerm, Restraint},
    histogram::Histogram,
    montecarlo::MarkovChain,
    propagate::Propagate,
    simulation,
    state::State,
};
use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::Rng;
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// YAML configuration
// ---------------------------------------------------------------------------

/// Top-level `umbrella:` section in input YAML.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UmbrellaConfig {
    cv: Box<dyn CvKindBuilder>,
    windows: WindowGrid,
    drive: DriveConfig,
}

/// Window layout parameters.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WindowGrid {
    range: (f64, f64),
    width: f64,
    spacing: f64,
    /// PMF histogram bin width (default: 1.0)
    #[serde(rename = "resolution", default = "default_bin_width")]
    bin_width: f64,
}

fn default_bin_width() -> f64 {
    1.0
}

/// Drive-phase parameters (harmonic bias to pull walker into window).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DriveConfig {
    force_constant: ForceConstant,
}

impl WindowGrid {
    /// Compute window centers: c_i = range[0] + width/2 + spacing * i.
    fn centers(&self) -> Vec<f64> {
        let first = self.range.0 + self.width / 2.0;
        let last = self.range.1 - self.width / 2.0;
        // Tolerance avoids missing the last center due to floating-point drift
        std::iter::successors(Some(first), |&c| {
            let next = c + self.spacing;
            (next <= last + 1e-10).then_some(next)
        })
        .collect()
    }

    /// Half-width for each window.
    fn half_width(&self) -> f64 {
        self.width / 2.0
    }

    /// Reject geometrically invalid or grid-misaligned window layouts before sampling.
    ///
    /// A non-positive `spacing` makes `centers()` loop forever, and both `width` and
    /// `spacing` must be integer multiples of `bin_width` so each window's local histogram
    /// maps onto the global PMF grid without rounding drift (`build_pmf`) or a dropped
    /// partial trailing bin (`Histogram::new`).
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.range.0 < self.range.1,
            "umbrella windows: range start ({}) must be less than end ({})",
            self.range.0,
            self.range.1
        );
        anyhow::ensure!(self.width > 0.0, "umbrella windows: width must be positive");
        anyhow::ensure!(
            self.spacing > 0.0,
            "umbrella windows: spacing must be positive"
        );
        anyhow::ensure!(
            self.bin_width > 0.0,
            "umbrella windows: resolution must be positive"
        );
        // Overlap-ratio stitching needs adjacent windows to share CV range; spacing ≥ width
        // leaves them touching or disjoint, which would only surface as a stitch error after
        // the whole (potentially long) run.
        anyhow::ensure!(
            self.spacing < self.width,
            "umbrella windows: spacing ({}) must be less than width ({}) so adjacent windows overlap",
            self.spacing,
            self.width
        );
        let is_multiple_of_bin = |x: f64| {
            let ratio = x / self.bin_width;
            (ratio - ratio.round()).abs() < 1e-9
        };
        anyhow::ensure!(
            is_multiple_of_bin(self.width),
            "umbrella windows: width ({}) must be an integer multiple of resolution ({})",
            self.width,
            self.bin_width
        );
        anyhow::ensure!(
            is_multiple_of_bin(self.spacing),
            "umbrella windows: spacing ({}) must be an integer multiple of resolution ({})",
            self.spacing,
            self.bin_width
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Per-window result
// ---------------------------------------------------------------------------

struct WindowResult {
    lo: f64,
    hi: f64,
    histogram: Histogram,
}

// ---------------------------------------------------------------------------
// Overlap-ratio stitching
// ---------------------------------------------------------------------------

/// Compute free-energy offsets between adjacent windows from overlap fractions.
/// Returns per-window offset in kJ/mol (first window = 0).
///
/// Offsets are cumulative, so a single broken link (no overlap or no samples in the overlap
/// band) would poison every downstream window. Rather than silently truncate the PMF there,
/// fail loudly and name the offending pair.
fn overlap_ratio_offsets(results: &[WindowResult], kt: f64) -> Result<Vec<f64>> {
    let mut offsets = vec![0.0_f64];
    for (i, pair) in results.windows(2).enumerate() {
        let (cur, next) = (&pair[0], &pair[1]);
        // Overlap region = intersection of adjacent windows
        let overlap_lo = next.lo;
        let overlap_hi = cur.hi;
        anyhow::ensure!(
            overlap_lo < overlap_hi,
            "Umbrella windows {i} and {} do not overlap ({overlap_lo:.2}..{overlap_hi:.2}); \
             cannot stitch a continuous PMF — increase window width or decrease spacing.",
            i + 1
        );
        let f_i = cur.histogram.fraction_in_range(overlap_lo, overlap_hi);
        let f_next = next.histogram.fraction_in_range(overlap_lo, overlap_hi);
        anyhow::ensure!(
            f_i > 0.0 && f_next > 0.0,
            "Umbrella windows {i} and {} have no samples in their overlap region \
             ({overlap_lo:.2}..{overlap_hi:.2}, f_i={f_i:.4}, f_next={f_next:.4}); \
             increase sampling (propagate.repeat) or window overlap.",
            i + 1
        );
        let delta = -kt * (f_i / f_next).ln();
        offsets.push(offsets.last().unwrap() + delta);
    }
    Ok(offsets)
}

/// PMF point: bin center, free energy, sampling error, and inter-window spread.
struct PmfPoint {
    cv: f64,
    f: f64,
    stderr: f64,
    spread: f64,
}

/// Build fine-grained PMF from per-window histograms and overlap-ratio offsets.
/// Within each window: F(r) = −kT ln ρ(r) + C_i.
///
/// Each contributing window carries the Poisson counting error σ_j = kT/√n_j, so its
/// inverse-variance weight is ∝ n_j — the count weighting used for the mean. The error of that
/// weighted mean is therefore kT/√(Σ_j n_j), shrinking monotonically with sampling. The spread of
/// the windows about the mean is reported separately: it diagnoses stitching quality and vanishes
/// when the windows agree, so it must not be mistaken for an error bar (issue #76).
fn build_pmf(results: &[WindowResult], kt: f64, bin_width: f64) -> Result<Vec<PmfPoint>> {
    if results.is_empty() {
        return Ok(Vec::new());
    }
    let offsets = overlap_ratio_offsets(results, kt)?;

    // Global bin grid spanning all windows
    let global_lo = results.iter().map(|w| w.lo).reduce(f64::min).unwrap();
    let global_hi = results.iter().map(|w| w.hi).reduce(f64::max).unwrap();
    let n_bins = ((global_hi - global_lo) / bin_width).ceil() as usize;

    // Per-bin: collect (f_local, weight) from each contributing window so we
    // can compute both a weighted mean and a weighted standard error
    let mut bin_entries: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_bins];

    for (win, &offset) in results.iter().zip(&offsets) {
        let n_total = win.histogram.total_count();
        if n_total == 0.0 {
            continue;
        }
        // Map local histogram bins to global grid
        let bin_offset = ((win.lo - global_lo) / bin_width).round() as usize;
        for i in 0..win.histogram.num_bins() {
            let count = win.histogram.count(i);
            if count == 0.0 {
                continue;
            }
            let global_idx = bin_offset + i;
            if global_idx < n_bins {
                let density = count / (n_total * bin_width);
                let f_local = -kt * density.ln() + offset;
                bin_entries[global_idx].push((f_local, count));
            }
        }
    }

    // Weighted mean, sampling error, and inter-window spread per bin
    let mut pmf: Vec<PmfPoint> = (0..n_bins)
        .filter(|i| !bin_entries[*i].is_empty())
        .map(|i| {
            let cv = global_lo + (i as f64 + 0.5) * bin_width;
            let entries = &bin_entries[i];
            let w_total: f64 = entries.iter().map(|&(_, w)| w).sum();
            let mean = entries.iter().map(|&(f, w)| f * w).sum::<f64>() / w_total;

            // Poisson counting error of the count-weighted mean: kT/√(Σ n_j)
            let stderr = kt / w_total.sqrt();

            // Weighted spread of the contributing windows about the mean. A lone window has
            // nothing to disagree with; short-circuit it so round-off in `mean` cannot leak a
            // spurious ~1e-16 spread into the output.
            let spread = if entries.len() > 1 {
                (entries
                    .iter()
                    .map(|&(f, w)| w * (f - mean).powi(2))
                    .sum::<f64>()
                    / w_total)
                    .sqrt()
            } else {
                0.0
            };

            PmfPoint {
                cv,
                f: mean,
                stderr,
                spread,
            }
        })
        .collect();

    if let Some(f_min) = pmf.iter().map(|p| p.f).reduce(f64::min) {
        for p in &mut pmf {
            p.f -= f_min;
        }
    }
    Ok(pmf)
}

// ---------------------------------------------------------------------------
// Window state file path
// ---------------------------------------------------------------------------

/// Per-window directory holding everything for one window: state, output,
/// histogram, and analysis (`Trajectory`, `Energy`, RDF, …) files. Isolating
/// each window keeps parallel workers from racing on shared paths (issue #32).
fn window_output_dir(state_dir: &Path, index: usize) -> PathBuf {
    state_dir.join(format!("window{index}"))
}

fn window_state_path(state_dir: &Path, index: usize) -> PathBuf {
    window_output_dir(state_dir, index).join("state.yaml")
}

fn window_output_path(state_dir: &Path, index: usize) -> PathBuf {
    window_output_dir(state_dir, index).join("output.yaml")
}

fn window_cv_path(state_dir: &Path, index: usize) -> PathBuf {
    window_output_dir(state_dir, index).join("histogram.yaml")
}

// ---------------------------------------------------------------------------
// Per-window worker
// ---------------------------------------------------------------------------

/// Parameters shared across all windows, passed by reference from the main thread.
struct WindowParams<'a> {
    input: &'a Path,
    state_dir: &'a Path,
    half_width: f64,
    bin_width: f64,
    drive_k: ForceConstant,
    cv_builder: &'a (dyn CvKindBuilder + 'static),
    medium: &'a interatomic::coulomb::Medium,
    multi_progress: &'a MultiProgress,
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix} [{bar:30.cyan/dim}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("━╸─")
}

/// Run a single umbrella window: drive → production → collect CV samples.
fn run_window(
    params: &WindowParams<'_>,
    mut context: Backend,
    window_index: usize,
    center: f64,
    window_seed: u64,
    pb: &ProgressBar,
) -> Result<WindowResult> {
    use interatomic::coulomb::Temperature;

    let lo = center - params.half_width;
    let hi = center + params.half_width;
    let rt = crate::R_IN_KJ_PER_MOL * params.medium.temperature();
    let input = params.input;

    let state_path = window_state_path(params.state_dir, window_index);

    // All per-window files nest under this directory; create it before the
    // drive phase, which writes state.yaml ahead of the analysis outputs.
    std::fs::create_dir_all(window_output_dir(params.state_dir, window_index))?;

    // Skip the drive phase on restart — the walker is already equilibrated in its window
    let had_state = state_path.exists();
    if had_state {
        let state = State::from_file(&state_path)?;
        // MarkovChain.load_state handles topology validation and position/group restoration
        let propagate = Propagate::from_file(input, &context, rt)?;
        let analyses = AnalysisCollection::default();
        let mut mc = MarkovChain::new(context, propagate, rt, analyses)?;
        mc.load_state(state)?;
        context = mc.into_context();
        params.multi_progress.println(format!(
            "Window {window_index}: restored state from {}",
            state_path.display()
        ))?;
    }

    let window = Interval::new(lo, hi)?;

    pb.set_style(progress_style());

    // Drive phase: harmonic-only bias steers the walker from its initial
    // configuration toward the window center. No hard-wall here — the walker
    // must be free to traverse intermediate CV values to reach the target.
    if !had_state {
        let harmonic_builder = ConstrainBuilder {
            cv: dyn_clone::clone_box(params.cv_builder),
            restraint: Restraint::Harmonic {
                force_constant: params.drive_k,
                equilibrium: Finite::new(center)?,
            },
        };
        let harmonic_term = EnergyTerm::from(harmonic_builder.build(&context)?);
        context.hamiltonian_mut().push_front(harmonic_term);

        let propagate = Propagate::from_file(input, &context, rt)?;
        let drive_len = propagate.max_repeats() as u64;
        let analyses = AnalysisCollection::default();
        let mut mc = MarkovChain::new(context, propagate, rt, analyses)?;

        let cv_check = params.cv_builder.build_cv(mc.context())?;
        pb.set_length(drive_len);
        pb.set_prefix(format!("W{window_index:>2} drive"));
        let mut reached = false;
        for i in 0..drive_len as usize {
            mc.run_n_steps(1)?;
            pb.set_position(i as u64 + 1);
            let val = cv_check.evaluate(mc.context());
            if val >= lo && val <= hi {
                params.multi_progress.println(format!(
                    "Window {window_index}: drive reached CV={val:.2} at step {i} ✅"
                ))?;
                reached = true;
                break;
            }
        }
        if !reached {
            let val = cv_check.evaluate(mc.context());
            anyhow::bail!(
                "Window {window_index}: drive exhausted {drive_len} sweeps without \
                 reaching [{lo:.1}, {hi:.1}] (CV={val:.2}). Increase propagate.repeat \
                 or drive.force_constant."
            );
        }

        // Persist post-drive state so restarts skip directly to production
        mc.save_state().with_step(0).to_file(&state_path)?;

        context = mc.into_context();

        // Remove harmonic bias — it would distort the equilibrium distribution
        // and invalidate BAR overlap fractions during production
        context.hamiltonian_mut().pop_front();
        params.multi_progress.println(format!(
            "Window {window_index}: saved state to {}",
            state_path.display()
        ))?;
    }

    // Hard-wall at front so Hamiltonian short-circuits on out-of-window configs
    // before evaluating expensive nonbonded terms
    let hard_wall_builder = ConstrainBuilder {
        cv: dyn_clone::clone_box(params.cv_builder),
        restraint: Restraint::Between(window),
    };
    let hard_wall_term = EnergyTerm::from(hard_wall_builder.build(&context)?);
    context.hamiltonian_mut().push_front(hard_wall_term);

    // Persisted histogram lets us extend sampling across restarts without
    // re-running already completed production steps
    let cv_path = window_cv_path(params.state_dir, window_index);
    let mut histogram = Histogram::new(lo, hi, params.bin_width)?;
    if cv_path.exists() {
        let loaded: Histogram = crate::auxiliary::read_yaml_file(&cv_path)?;
        // A persisted histogram binned on a stale grid (edited width/bin_width) would map
        // its counts to the wrong CV; reject the mismatch instead of silently corrupting.
        loaded
            .ensure_shape(lo, params.bin_width, histogram.num_bins())
            .with_context(|| {
                format!(
                    "Window {window_index}: persisted histogram '{}' does not match the \
                     current window geometry — delete the state directory to restart, or \
                     restore the original input",
                    cv_path.display()
                )
            })?;
        params.multi_progress.println(format!(
            "Window {window_index}: extending from {} existing counts",
            loaded.total_count() as u64
        ))?;
        histogram = loaded;
    }

    // Production phase: hard-wall only, collect CV samples.
    // Each run appends max_repeats new samples so users can extend sampling
    // by simply rerunning without editing the input file.
    let mut propagate = Propagate::from_file(input, &context, rt)?;
    // Per-window seed decorrelates the sampled trajectories across windows (the drive phase
    // is throwaway equilibration, so only production needs the independent stream).
    propagate.reseed(window_seed);
    let max_repeats = propagate.max_repeats();
    // Block ensures `mc` (which owns `context`) is dropped before we serialize
    // the histogram and return the result.
    {
        let out_dir = window_output_dir(params.state_dir, window_index);
        let analyses =
            analysis::from_file_creating_dir(input, &context, Some(params.medium), &out_dir)?;
        let mut mc = MarkovChain::new(context, propagate, rt, analyses)?;

        let cv = params.cv_builder.build_cv(mc.context())?;
        let initial_energy = mc.system_energy();

        pb.set_length(max_repeats as u64);
        pb.set_position(0);
        pb.set_prefix(format!("W{window_index:>2} prod "));
        pb.set_message("");
        // Sample CV after each propagation cycle; run_n_steps(1) avoids the
        // borrow conflict between mc.iter() and mc.context()
        for i in 0..max_repeats {
            mc.run_n_steps(1)?;
            histogram.add(cv.evaluate(mc.context()));
            pb.set_position(i as u64 + 1);
        }
        pb.finish_and_clear();

        mc.finalize_analyses()?;

        // Write per-window output
        let output_path = window_output_path(params.state_dir, window_index);
        let mut yaml_output = std::fs::File::create(&output_path)
            .with_context(|| format!("Cannot create '{}'", output_path.display()))?;
        simulation::write_yaml(params.medium, &mut yaml_output, Some("medium"))?;
        let final_energy = mc.system_energy();
        let drift = mc.energy_drift(initial_energy);
        let energy_summary = std::collections::BTreeMap::from([
            ("initial".to_string(), initial_energy),
            ("final".to_string(), final_energy),
            ("drift".to_string(), drift),
        ]);
        simulation::write_yaml(&energy_summary, &mut yaml_output, Some("energy_change"))?;
        simulation::write_yaml(
            &mc.propagation().to_yaml(),
            &mut yaml_output,
            Some("propagate"),
        )?;
        let analysis_yaml = analysis::analyses_to_yaml(mc.analyses());
        if !analysis_yaml.is_empty() {
            simulation::write_yaml(&analysis_yaml, &mut yaml_output, Some("analysis"))?;
        }

        // Save final state
        let state = mc.save_state();
        state.to_file(&state_path)?;
    }

    crate::auxiliary::write_yaml_file(&cv_path, &histogram)?;

    params.multi_progress.println(format!(
        "Window {window_index} (center={center:.1}): {} counts",
        histogram.total_count() as u64,
    ))?;

    Ok(WindowResult { lo, hi, histogram })
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse the `umbrella:` section from a YAML input file.
fn parse_config(input: &Path) -> Result<UmbrellaConfig> {
    crate::auxiliary::parse_yaml_section(input, "umbrella")
}

/// Run multi-walker umbrella sampling with BAR stitching.
pub fn run(
    input: &Path,
    state_dir: &Path,
    common_state: Option<&Path>,
    output: &Path,
    max_threads: usize,
) -> Result<()> {
    use interatomic::coulomb::Temperature;

    let config = parse_config(input)?;
    config.windows.validate()?;
    let medium = crate::backend::get_medium(input)?;
    let rt = crate::R_IN_KJ_PER_MOL * medium.temperature();

    let centers = config.windows.centers();
    let half_width = config.windows.half_width();
    let n_windows = centers.len();

    if n_windows == 0 {
        anyhow::bail!("No umbrella windows generated — check range/width/spacing");
    }

    let multi_progress = MultiProgress::new();

    multi_progress.suspend(|| {
        log::info!(
            "Umbrella sampling: {n_windows} windows, width={:.1}, spacing={:.1}",
            config.windows.width,
            config.windows.spacing
        );
        log::info!(
            "Window centers: {:?}",
            centers
                .iter()
                .map(|c| format!("{c:.1}"))
                .collect::<Vec<_>>()
        );
    });

    std::fs::create_dir_all(state_dir).map_err(|e| {
        anyhow::anyhow!(
            "Cannot create state directory '{}': {e}. \
             Note: use --state-dir for the directory and --state for a common state file.",
            state_dir.display()
        )
    })?;

    // Build once; cloned per window to avoid redundant YAML parsing + Hamiltonian construction.
    let mut base_context = Backend::new(
        input,
        None,
        &mut crate::propagate::setup_rng_from_file(input)?,
    )?;

    // Load common state file to seed all windows that don't yet have a per-window state
    if let Some(state_path) = common_state {
        let state = State::from_file(state_path)?;
        let propagate = Propagate::from_file(input, &base_context, rt)?;
        let analyses = AnalysisCollection::default();
        let mut mc = MarkovChain::new(base_context, propagate, rt, analyses)?;
        mc.load_state(state)?;
        base_context = mc.into_context();
        log::info!("Loaded common state from {}", state_path.display());
    }

    // Give every window an independent RNG stream. All windows clone the same seeded base
    // context, so without a per-window reseed a fixed seed makes every walker draw the
    // identical sequence and propose correlated moves. Pre-drawn from a seed RNG (order is
    // deterministic under `-j 1`), mirroring the per-box reseed in Gibbs sampling.
    let window_seeds: Vec<u64> = {
        let mut seed_rng = crate::propagate::setup_rng_from_file(input)?;
        (0..n_windows).map(|_| seed_rng.gen()).collect()
    };

    let n_threads = crate::auxiliary::resolve_thread_count(max_threads);

    // Process in batches so N_windows > N_cores doesn't over-subscribe the machine
    let mut all_results: Vec<WindowResult> = Vec::with_capacity(n_windows);

    for batch_start in (0..n_windows).step_by(n_threads) {
        let batch_end = (batch_start + n_threads).min(n_windows);
        let batch_indices: Vec<usize> = (batch_start..batch_end).collect();

        multi_progress.suspend(|| {
            log::info!(
                "Running windows {batch_start}..{} ({} threads)",
                batch_end - 1,
                batch_indices.len()
            );
        });

        let params = WindowParams {
            input,
            state_dir,
            half_width,
            bin_width: config.windows.bin_width,
            drive_k: config.drive.force_constant,
            cv_builder: config.cv.as_ref(),
            medium: &medium,
            multi_progress: &multi_progress,
        };

        // Pre-create all bars for this batch so MultiProgress has a stable set of lines
        let bars: Vec<ProgressBar> = batch_indices
            .iter()
            .map(|&i| {
                let pb = multi_progress.add(ProgressBar::new(0));
                pb.set_style(progress_style());
                pb.set_prefix(format!("W{i:>2} wait "));
                pb
            })
            .collect();

        let batch_results: Vec<Result<WindowResult>> = std::thread::scope(|s| {
            let handles: Vec<_> = batch_indices
                .iter()
                .zip(&bars)
                .map(|(&i, pb)| {
                    let center = centers[i];
                    let ctx = base_context.clone();
                    let seed = window_seeds[i];
                    let p = &params;
                    s.spawn(move || run_window(p, ctx, i, center, seed, pb))
                })
                .collect();

            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("window thread panicked"))?
                })
                .collect()
        });

        // Remove all batch bars after threads complete
        for pb in &bars {
            pb.finish_and_clear();
            multi_progress.remove(pb);
        }

        all_results.extend(batch_results.into_iter().collect::<Result<Vec<_>>>()?);
    }

    // Histogram + overlap-ratio stitching into fine-grained PMF
    let pmf = build_pmf(&all_results, rt, config.windows.bin_width)?;

    // Write PMF output
    let mut writer = ColumnWriter::open(output, &["cv", "pmf_kT", "stderr_kT", "spread_kT"])?;
    for p in &pmf {
        // Scientific notation for the error columns: fixed-point truncates a small but finite
        // error to a literal zero, which breaks 1/σ-weighted fits downstream (issue #76)
        writer.write_row(&[
            &format!("{:.4}", p.cv),
            &format!("{:.6}", p.f / rt),
            &format!("{:.6e}", p.stderr / rt),
            &format!("{:.6e}", p.spread / rt),
        ])?;
    }
    log::info!("Wrote PMF ({} bins) to {}", pmf.len(), output.display());

    // The error bars measure sampling only, so a broken stitch no longer widens them (issue #76).
    // Surface it here instead — otherwise poor overlap is visible only to whoever opens the CSV.
    if let Some(worst) = pmf
        .iter()
        .max_by(|a, b| a.spread.total_cmp(&b.spread))
        .filter(|p| p.spread > rt)
    {
        log::warn!(
            "Windows overlapping cv={:.2} disagree by {:.1} RT — more than thermal energy. \
             The stitched PMF is unreliable there; increase window overlap or sampling. \
             See the 'spread_kT' column of {}.",
            worst.cv,
            worst.spread / rt,
            output.display()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_grid_bin_width_key_is_resolution() {
        // #123: umbrella's PMF bin width unified on `resolution`; `bin_width` is a clean break.
        let ok = "range: [0.0, 10.0]\nwidth: 4.0\nspacing: 2.0\nresolution: 0.5";
        let grid: WindowGrid = serde_yml::from_str(ok).unwrap();
        assert!((grid.bin_width - 0.5).abs() < 1e-12);
        let old = "range: [0.0, 10.0]\nwidth: 4.0\nspacing: 2.0\nbin_width: 0.5";
        assert!(serde_yml::from_str::<WindowGrid>(old).is_err());
    }

    #[test]
    fn window_grid_centers() {
        let grid = WindowGrid {
            range: (25.0, 150.0),
            width: 10.0,
            spacing: 8.0,
            bin_width: 1.0,
        };
        let centers = grid.centers();
        assert!((centers[0] - 30.0).abs() < 1e-10);
        assert!((centers[1] - 38.0).abs() < 1e-10);
        let last = *centers.last().unwrap();
        // Last center must be ≤ range[1] - width/2 = 145
        assert!(last <= 145.0 + 1e-10);
    }

    #[test]
    fn window_grid_single_window() {
        let grid = WindowGrid {
            range: (0.0, 10.0),
            width: 10.0,
            spacing: 10.0,
            bin_width: 1.0,
        };
        let centers = grid.centers();
        assert_eq!(centers.len(), 1);
        assert!((centers[0] - 5.0).abs() < 1e-10);
    }

    fn grid(range: (f64, f64), width: f64, spacing: f64, bin_width: f64) -> WindowGrid {
        WindowGrid {
            range,
            width,
            spacing,
            bin_width,
        }
    }

    #[test]
    fn window_grid_validate_accepts_aligned_geometry() {
        // The umbrella_1d fixture geometry: width and spacing are multiples of bin_width.
        assert!(grid((-9.0, 9.0), 6.0, 3.0, 0.5).validate().is_ok());
    }

    #[test]
    fn window_grid_validate_rejects_nonpositive_spacing() {
        assert!(grid((0.0, 10.0), 6.0, 0.0, 1.0).validate().is_err());
        assert!(grid((0.0, 10.0), 6.0, -3.0, 1.0).validate().is_err());
    }

    #[test]
    fn window_grid_validate_rejects_unaligned_width_or_spacing() {
        // width 5 not a multiple of bin_width 2.0 (5/2 = 2.5)
        assert!(grid((0.0, 10.0), 5.0, 2.0, 2.0).validate().is_err());
        // spacing 2.5 not a multiple of bin_width 1.0
        assert!(grid((0.0, 10.0), 6.0, 2.5, 1.0).validate().is_err());
    }

    #[test]
    fn window_grid_validate_rejects_inverted_range() {
        assert!(grid((10.0, 0.0), 6.0, 3.0, 1.0).validate().is_err());
    }

    #[test]
    fn window_grid_validate_rejects_nonoverlapping_spacing() {
        // spacing == width → adjacent windows only touch; spacing > width → disjoint.
        assert!(grid((0.0, 30.0), 6.0, 6.0, 1.0).validate().is_err());
        assert!(grid((0.0, 30.0), 6.0, 8.0, 1.0).validate().is_err());
    }

    /// Helper to build a `WindowResult` from raw samples.
    fn window_result(lo: f64, hi: f64, bin_width: f64, samples: &[f64]) -> WindowResult {
        let mut histogram = Histogram::new(lo, hi, bin_width).unwrap();
        for &v in samples {
            histogram.add(v);
        }
        WindowResult { lo, hi, histogram }
    }

    #[test]
    fn overlap_ratio_offsets_symmetric() {
        // Two windows with equal overlap fractions → ΔF = 0
        let results = vec![
            window_result(0.0, 10.0, 1.0, &[3.0, 5.0, 7.0, 8.5, 9.0]),
            window_result(8.0, 18.0, 1.0, &[8.5, 9.0, 11.0, 13.0, 15.0]),
        ];
        let offsets = overlap_ratio_offsets(&results, 1.0).unwrap();
        assert_eq!(offsets.len(), 2);
        assert!((offsets[0] - 0.0).abs() < 1e-10);
        // f_i = 2/5 (8.5, 9.0 in [8,10]), f_next = 2/5 (8.5, 9.0 in [8,10])
        // ΔF = -kT * ln(1) = 0
        assert!((offsets[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn overlap_ratio_offsets_known_ratio() {
        // f_i = 1/4, f_next = 3/4 → ΔF = -ln(1/3) = ln(3)
        let results = vec![
            window_result(0.0, 10.0, 1.0, &[2.0, 4.0, 6.0, 9.0]),
            window_result(8.0, 18.0, 1.0, &[8.5, 9.0, 9.5, 15.0]),
        ];
        let offsets = overlap_ratio_offsets(&results, 1.0).unwrap();
        let expected = -(1.0_f64 / 3.0).ln(); // ln(3)
        assert!((offsets[1] - expected).abs() < 1e-10);
    }

    #[test]
    fn overlap_ratio_offsets_rejects_nonoverlapping_windows() {
        // Windows [0,8] and [10,18] share no CV range → cannot be stitched.
        let results = vec![
            window_result(0.0, 8.0, 1.0, &[1.0, 3.0, 5.0, 7.0]),
            window_result(10.0, 18.0, 1.0, &[11.0, 13.0, 15.0, 17.0]),
        ];
        let err = overlap_ratio_offsets(&results, 1.0).unwrap_err();
        assert!(err.to_string().contains("do not overlap"), "{err}");
    }

    #[test]
    fn overlap_ratio_offsets_rejects_empty_overlap() {
        // Windows overlap in [8,10] but neither samples there → no stitch possible.
        let results = vec![
            window_result(0.0, 10.0, 1.0, &[1.0, 3.0, 5.0]),
            window_result(8.0, 18.0, 1.0, &[13.0, 15.0, 17.0]),
        ];
        let err = overlap_ratio_offsets(&results, 1.0).unwrap_err();
        assert!(err.to_string().contains("no samples"), "{err}");
    }

    /// Repeat `value` `count` times — builds histograms with exact per-bin counts.
    fn repeat(value: f64, count: usize) -> Vec<f64> {
        vec![value; count]
    }

    #[test]
    fn build_pmf_stderr_is_poisson_over_summed_counts() {
        let kt = 2.0;
        // Bin [8,9) is covered by both windows with 40 and 60 counts; bin [2,3) by one with 5.
        let mut samples_a = repeat(8.5, 40);
        samples_a.extend(repeat(2.5, 5));
        let mut samples_b = repeat(8.5, 60);
        samples_b.extend(repeat(15.5, 5));
        let results = vec![
            window_result(0.0, 10.0, 1.0, &samples_a),
            window_result(8.0, 18.0, 1.0, &samples_b),
        ];
        let pmf = build_pmf(&results, kt, 1.0).unwrap();
        let at = |cv: f64| pmf.iter().find(|p| (p.cv - cv).abs() < 1e-9).unwrap();

        // Two windows: kT/√(40 + 60)
        assert!((at(8.5).stderr - kt / 100.0_f64.sqrt()).abs() < 1e-12);
        // Single window: kT/√5, unchanged from the pre-#76 isolated-bin estimate
        assert!((at(2.5).stderr - kt / 5.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn build_pmf_stderr_survives_perfect_window_agreement() {
        // The issue-#76 scenario: two windows whose aligned free energies coincide in the overlap.
        // The spread vanishes there, but the error bar must not.
        let n = 10000;
        let samples_a: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 10.0).collect();
        let samples_b: Vec<f64> = (0..n).map(|i| 8.0 + i as f64 / n as f64 * 10.0).collect();
        let results = vec![
            window_result(0.0, 10.0, 1.0, &samples_a),
            window_result(8.0, 18.0, 1.0, &samples_b),
        ];
        let pmf = build_pmf(&results, 1.0, 1.0).unwrap();
        let overlap = pmf.iter().find(|p| (p.cv - 8.5).abs() < 1e-9).unwrap();

        assert!(overlap.spread < 1e-9, "spread={}", overlap.spread);
        assert!(overlap.stderr > 0.0);
        assert!(pmf.iter().all(|p| p.stderr > 0.0));
        // More counts must mean a smaller error: the two-window bin beats a one-window bin
        let single = pmf.iter().find(|p| (p.cv - 4.5).abs() < 1e-9).unwrap();
        assert!(overlap.stderr < single.stderr);
    }

    #[test]
    fn build_pmf_spread_flags_disagreeing_windows() {
        // Window B's density profile differs from A's across the overlap, so the two stitched
        // estimates disagree bin-by-bin even though their overlap fractions align on average.
        let mut samples_a = repeat(8.5, 50);
        samples_a.extend(repeat(9.5, 50));
        samples_a.extend(repeat(2.5, 100));
        let mut samples_b = repeat(8.5, 20);
        samples_b.extend(repeat(9.5, 80));
        samples_b.extend(repeat(15.5, 100));
        let results = vec![
            window_result(0.0, 10.0, 1.0, &samples_a),
            window_result(8.0, 18.0, 1.0, &samples_b),
        ];
        let pmf = build_pmf(&results, 1.0, 1.0).unwrap();
        let at = |cv: f64| pmf.iter().find(|p| (p.cv - cv).abs() < 1e-9).unwrap();

        assert!(at(8.5).spread > 0.1, "spread={}", at(8.5).spread);
        // A bin only one window reaches has nothing to disagree with
        assert_eq!(at(2.5).spread, 0.0);
    }

    #[test]
    fn build_pmf_produces_bins() {
        // Uniform samples in two overlapping windows → flat PMF
        let n = 10000;
        let samples_a: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 10.0).collect();
        let samples_b: Vec<f64> = (0..n).map(|i| 8.0 + i as f64 / n as f64 * 10.0).collect();
        let results = vec![
            window_result(0.0, 10.0, 1.0, &samples_a),
            window_result(8.0, 18.0, 1.0, &samples_b),
        ];
        let pmf = build_pmf(&results, 1.0, 1.0).unwrap();
        // Should have bins from 0 to 18
        assert!(pmf.len() > 2);
        // PMF should be approximately flat (uniform → constant −kT ln ρ)
        let max_f = pmf.iter().map(|p| p.f).fold(0.0_f64, f64::max);
        assert!(max_f < 0.5, "PMF not flat for uniform data: max={max_f}");
    }
}
