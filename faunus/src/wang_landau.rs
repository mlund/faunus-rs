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

//! Wang-Landau flat-histogram sampling with multi-walker support.
//!
//! Iteratively estimates the density of states `g(CV)` in collective variable
//! space, producing a free energy surface without predefined windows or force
//! constants. Multiple walkers share a single histogram and bias estimate.
//!
//! See [Chevallier & Cazals](https://doi.org/10.1016/j.jcp.2020.109366).

use crate::{
    analysis::{self, AnalysisCollection},
    backend::Backend,
    collective_variable::CollectiveVariableBuilder,
    energy::{EnergyTerm, Penalty},
    flat_histogram::{FlatHistogramState, GridDim},
    montecarlo::MarkovChain,
    propagate::Propagate,
    state::State,
    WithHamiltonian,
};
use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

// ---------------------------------------------------------------------------
// YAML configuration
// ---------------------------------------------------------------------------

/// Top-level `wang_landau:` section in input YAML.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WangLandauConfig {
    coordinate: CollectiveVariableBuilder,
    coordinate2: Option<CollectiveVariableBuilder>,
    #[serde(default = "default_ln_f_initial")]
    ln_f_initial: f64,
    #[serde(default = "default_flatness_threshold")]
    flatness_threshold: f64,
    #[serde(default = "default_min_flatness")]
    min_flatness: u32,
    #[serde(default = "default_min_ln_f")]
    min_ln_f: f64,
    #[serde(default = "default_steps_per_check")]
    steps_per_check: usize,
}

fn default_ln_f_initial() -> f64 {
    1.0
}
fn default_flatness_threshold() -> f64 {
    0.8
}
fn default_min_flatness() -> u32 {
    20
}
fn default_min_ln_f() -> f64 {
    1e-6
}
fn default_steps_per_check() -> usize {
    10000
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

fn parse_config(input: &Path) -> Result<WangLandauConfig> {
    crate::auxiliary::parse_yaml_section(input, "wang_landau")
}

// ---------------------------------------------------------------------------
// State directory paths
// ---------------------------------------------------------------------------

fn walker_state_path(state_dir: &Path, index: usize) -> PathBuf {
    state_dir.join(format!("walker{index}_state.yaml"))
}

fn histogram_path(state_dir: &Path) -> PathBuf {
    state_dir.join("histogram.yaml")
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run Wang-Landau flat-histogram sampling.
pub fn run(input: &Path, state_dir: &Path, output: &Path, max_threads: usize) -> Result<()> {
    use interatomic::coulomb::Temperature;

    let config = parse_config(input)?;
    let medium = crate::backend::get_medium(input)?;
    let rt = crate::R_IN_KJ_PER_MOL * medium.temperature();

    // Build grid dimensions from CV range/resolution
    let resolution1 = config.coordinate.resolution.unwrap_or(1.0);
    let (min1, max1) = config.coordinate.range;

    let dim = if let Some(ref cv2_builder) = config.coordinate2 {
        let resolution2 = cv2_builder.resolution.unwrap_or(1.0);
        let (min2, max2) = cv2_builder.range;
        GridDim::new_2d([min1, min2], [max1, max2], [resolution1, resolution2])?
    } else {
        GridDim::new_1d(min1, max1, resolution1)?
    };

    log::info!(
        "Wang-Landau: {} bins, ln_f_initial={}, flatness_threshold={}, min_flatness={}",
        dim.num_bins(),
        config.ln_f_initial,
        config.flatness_threshold,
        config.min_flatness,
    );

    let n_threads = crate::auxiliary::resolve_thread_count(max_threads);

    std::fs::create_dir_all(state_dir)?;

    // Restore or create shared state
    let hist_path = histogram_path(state_dir);
    let shared_state = if hist_path.exists() {
        let state = FlatHistogramState::from_file(&hist_path)?;
        log::info!(
            "Restored checkpoint: ln_f={:.2e}, flatness_checks={}",
            state.ln_f(),
            state.num_flatness()
        );
        Arc::new(RwLock::new(state))
    } else {
        Arc::new(RwLock::new(FlatHistogramState::new(
            dim,
            config.flatness_threshold,
            config.min_flatness,
            config.min_ln_f,
            config.ln_f_initial,
            n_threads as u32,
        )))
    };

    // Build base context once
    let base_context = Backend::new(input, None, &mut rand::thread_rng())?;

    // Verify CVs resolve against the base context (catches selection errors early)
    config.coordinate.build(&base_context)?;
    if let Some(ref cv2) = config.coordinate2 {
        cv2.build(&base_context)?;
    }

    log::info!(
        "Spawning {n_threads} walker(s), {} steps per check",
        config.steps_per_check
    );

    let multi_progress = MultiProgress::new();

    // Spawn long-lived walkers that loop internally until converged.
    // Flatness checks happen on the main thread between walker batches.
    let pb = multi_progress.add(ProgressBar::new(100));
    pb.set_style(
        ProgressStyle::with_template("[{bar:20.cyan/dim}] {pos}% {msg}")
            .unwrap()
            .progress_chars("━╸─"),
    );

    let barrier = std::sync::Barrier::new(n_threads);

    let errors: Vec<Result<()>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_threads)
            .map(|i| {
                let mut ctx = base_context.clone();
                let state_ref = Arc::clone(&shared_state);
                let input_ref = input;
                let state_dir_ref = state_dir;
                let cv_builder = &config.coordinate;
                let cv2_builder = config.coordinate2.as_ref();
                let steps = config.steps_per_check;
                let medium_ref = &medium;
                let hist_path_ref = &hist_path;
                let barrier_ref = &barrier;
                let pb_ref = &pb;

                s.spawn(move || -> Result<()> {
                    let walker_path = walker_state_path(state_dir_ref, i);
                    if walker_path.exists() {
                        let ws = State::from_file(&walker_path)?;
                        let propagate = Propagate::from_file(input_ref, &ctx)?;
                        let analyses = AnalysisCollection::default();
                        let mut mc = MarkovChain::new(ctx, propagate, rt, analyses)?;
                        mc.load_state(ws)?;
                        ctx = mc.into_context();
                    }

                    // Clone penalty before moving into Hamiltonian — the clone
                    // is used for histogram updates outside the energy evaluation path
                    let cv = cv_builder.build(&ctx)?;
                    let cv2 = cv2_builder.map(|b| b.build(&ctx)).transpose()?;
                    let penalty = Penalty::new(cv, cv2, state_ref.clone(), rt);
                    let penalty_for_update = penalty.clone();
                    ctx.hamiltonian_mut().push_front(EnergyTerm::from(penalty));

                    // WL controls its own loop; disable the YAML repeat limit
                    let mut propagate = Propagate::from_file(input_ref, &ctx)?;
                    propagate.set_unlimited_repeats();
                    let analyses = analysis::from_file(input_ref, &ctx, Some(medium_ref))?;
                    let mut mc = MarkovChain::new(ctx, propagate, rt, analyses)?;

                    let mut iteration = 0u32;
                    loop {
                        // Each step: MC trial → penalty energy (read lock) → accept/reject
                        // → histogram update (write lock). Per-step update is required
                        // for WL correctness: the bias must change between consecutive trials.
                        for _ in 0..steps {
                            mc.run_n_steps(1)?;
                            penalty_for_update.update(mc.context());
                        }

                        // Double barrier: all walkers finish their batch before walker 0
                        // checks flatness, then all wait for the check to complete.
                        // This prevents racing between histogram writes and the
                        // flatness check which resets the histogram.
                        barrier_ref.wait();
                        if i == 0 {
                            let mut state = state_ref.write().expect("poisoned lock");
                            let range = state.ln_g_range();
                            state.check_and_reduce();
                            let ln_f = state.ln_f();
                            // Progress on log scale: ln_f shrinks from initial to target
                            let progress = {
                                let total = config.ln_f_initial.ln() - config.min_ln_f.ln();
                                let done =
                                    config.ln_f_initial.ln() - ln_f.max(config.min_ln_f).ln();
                                (done / total * 100.0).clamp(0.0, 100.0)
                            };
                            pb_ref.set_position(progress as u64);
                            let regime = if state.num_flatness() >= config.min_flatness {
                                "1/t"
                            } else {
                                "exp"
                            };
                            pb_ref.set_message(format!(
                                "Δg={range:.1} kT ln_f={ln_f:.1e} [{regime} {}/{}]",
                                state.num_flatness(),
                                config.min_flatness,
                            ));
                            // Non-fatal: checkpoint loss is recoverable by re-running
                            let _ = state.to_file(hist_path_ref);
                        }
                        barrier_ref.wait();

                        if state_ref.read().expect("poisoned lock").is_converged() {
                            if i == 0 {
                                log::info!("Converged after {} iterations", iteration);
                            }
                            break;
                        }
                        iteration += 1;
                    }

                    // Remove penalty before saving so the state file is independent
                    // of the WL bias and can be used as a starting point for other runs
                    mc.context_mut().hamiltonian_mut().pop_front();
                    mc.save_state().to_file(&walker_path)?;
                    Ok(())
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("walker thread panicked"))?
            })
            .collect()
    });

    pb.finish_and_clear();
    for result in errors {
        result?;
    }

    // Write free energy surface
    let state = shared_state.read().expect("poisoned lock");
    state.write_free_energy(output)?;
    log::info!(
        "Wrote free energy ({} bins) to {}",
        state.dim().num_bins(),
        output.display()
    );

    Ok(())
}
