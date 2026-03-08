// Copyright 2023 Mikael Lund
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

use crate::{
    analysis,
    montecarlo::{gibbs::GibbsEnsemble, MarkovChain},
    platform::soa::SoaPlatform,
    simulation::{self, box_prefixed_path, write_yaml, Simulation},
    Context, WithCell,
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::ProgressBar;
use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run Monte Carlo simulation
    #[clap(arg_required_else_help = true)]
    Run {
        /// Input file in YAML format
        #[clap(long, short = 'i')]
        input: PathBuf,
        /// State file for saving and restoring simulation state
        #[clap(long, short = 's')]
        state: Option<PathBuf>,
    },
}

#[derive(Parser)]
#[clap(version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    pub command: Commands,

    /// Verbose output. See more with e.g. RUST_LOG=Trace
    #[clap(long, short = 'v', action)]
    pub verbose: bool,
    /// Output file in YAML format
    #[clap(long, short = 'o', default_value = "output.yaml")]
    pub output: PathBuf,
}

pub fn do_main() -> Result<()> {
    let args = Args::parse();
    if std::env::var(DEFAULT_FILTER_ENV).is_err() {
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe {
            std::env::set_var(
                DEFAULT_FILTER_ENV,
                if args.verbose {
                    "Debug"
                } else {
                    "Info,cubecl_wgpu=warn"
                },
            )
        };
    }
    pretty_env_logger::init();

    match args.command {
        Commands::Run { input, state } => {
            let (sim, medium) = Simulation::from_file(&input, state.as_deref())?;
            match sim {
                Simulation::SingleBox(mc) => {
                    let mut yaml_output = std::fs::File::create(&args.output)?;
                    run_single_box(mc, &medium, state.as_deref(), &mut yaml_output)?;
                }
                Simulation::Gibbs(ensemble) => {
                    run_gibbs(*ensemble, &medium, state.as_deref(), &args.output)?;
                }
            }
        }
    }
    Ok(())
}

/// Write per-box YAML output: medium, blocks, cell, energy, propagate, analysis.
/// Returns `(final_energy, drift)`.
fn write_mc_output<T: Context + WithCell<SimCell = crate::cell::Cell> + 'static>(
    mc: &MarkovChain<T>,
    medium: &interatomic::coulomb::Medium,
    initial_energy: f64,
    output: &mut std::fs::File,
) -> Result<(f64, f64)> {
    write_yaml(medium, output, Some("medium"))?;
    write_yaml(&mc.context().topology().blocks(), output, Some("blocks"))?;
    write_yaml(&mc.context().cell(), output, Some("cell"))?;

    let final_energy = mc.system_energy();
    let drift = mc.energy_drift(initial_energy);
    let energy_summary = std::collections::BTreeMap::from([
        ("initial".to_string(), initial_energy),
        ("final".to_string(), final_energy),
        ("drift".to_string(), drift),
    ]);
    write_yaml(&energy_summary, output, Some("energy_change"))?;
    let energy_info = mc.context().hamiltonian().info_to_yaml();
    if let Some(map) = energy_info.as_mapping() {
        if !map.is_empty() {
            write_yaml(&energy_info, output, Some("energy"))?;
        }
    }
    write_yaml(
        &mc.context().hamiltonian().timing_to_yaml(),
        output,
        Some("energy_timers"),
    )?;
    write_yaml(&mc.propagation().to_yaml(), output, Some("propagate"))?;

    let analysis_yaml = analysis::analyses_to_yaml(mc.analyses());
    if !analysis_yaml.is_empty() {
        write_yaml(&analysis_yaml, output, Some("analysis"))?;
    }
    Ok((final_energy, drift))
}

fn run_single_box<T: Context + WithCell<SimCell = crate::cell::Cell> + 'static>(
    mut mc: MarkovChain<T>,
    medium: &interatomic::coulomb::Medium,
    state: Option<&std::path::Path>,
    yaml_output: &mut std::fs::File,
) -> Result<()> {
    let thermal_energy = simulation::thermal_energy(medium);
    log::info!("{}", medium);
    log::info!("Thermal energy: {thermal_energy:.2} kJ/mol");

    let initial_energy = mc.system_energy();
    log::info!("Initial energy = {initial_energy:.2} kJ/mol");

    let pb = ProgressBar::new(mc.propagation().max_repeats() as u64);
    for (i, step) in mc.iter().enumerate() {
        step?;
        pb.set_position(i as u64 + 1);
    }
    pb.finish();

    mc.finalize_analyses()?;

    let (final_energy, drift) = write_mc_output(&mc, medium, initial_energy, yaml_output)?;

    let relative_drift = if final_energy.abs() > f64::EPSILON {
        drift / final_energy.abs()
    } else {
        drift
    };
    log::info!("Final energy = {final_energy:.2} kJ/mol");
    if relative_drift > 1e-9 {
        log::warn!("Energy drift: {drift:.2e} kJ/mol (relative: {relative_drift:.2e}) ⚠️");
    } else {
        log::info!("Energy drift: {drift:.2e} kJ/mol (relative: {relative_drift:.2e}) ✅");
    }

    if let Some(state_path) = state {
        mc.save_state().to_file(state_path)?;
        log::info!("Saved simulation state to {}", state_path.display());
    }

    Ok(())
}

fn run_gibbs(
    mut ensemble: GibbsEnsemble<SoaPlatform>,
    medium: &interatomic::coulomb::Medium,
    state: Option<&std::path::Path>,
    output_path: &std::path::Path,
) -> Result<()> {
    let thermal_energy = simulation::thermal_energy(medium);
    log::info!("{}", medium);
    log::info!("Thermal energy: {thermal_energy:.2} kJ/mol");

    let initial_energies: [f64; 2] = std::array::from_fn(|i| {
        let e = ensemble.boxes()[i].system_energy();
        log::info!("Box {i}: initial energy = {e:.2} kJ/mol");
        e
    });

    let pb = ProgressBar::new(ensemble.max_sweeps() as u64);
    let max_sweeps = ensemble.max_sweeps();

    // run with progress reporting
    for sweep in 0..max_sweeps {
        // parallel intra-box + sequential inter-box (one sweep)
        let intra_steps = ensemble.intra_steps();
        let [b0, b1] = ensemble.boxes_mut();
        std::thread::scope(|s| -> Result<()> {
            let h0 = s.spawn(|| b0.run_n_steps(intra_steps));
            let h1 = s.spawn(|| b1.run_n_steps(intra_steps));
            h0.join()
                .map_err(|_| anyhow::anyhow!("box 0 thread panicked"))??;
            h1.join()
                .map_err(|_| anyhow::anyhow!("box 1 thread panicked"))??;
            Ok(())
        })?;

        ensemble.perform_inter_moves()?;

        pb.set_position(sweep as u64 + 1);
    }
    pb.finish();

    ensemble.finalize_analyses()?;

    for (i, (mc, &init_e)) in ensemble.boxes().iter().zip(&initial_energies).enumerate() {
        let box_output = box_prefixed_path(output_path, i);
        let mut f = std::fs::File::create(&box_output)?;
        let (final_energy, drift) = write_mc_output(mc, medium, init_e, &mut f)?;
        log::info!("Box {i}: final energy = {final_energy:.2} kJ/mol, drift = {drift:.2e}");
    }

    // write inter-box move summary to the main output file
    let mut main_output = std::fs::File::create(output_path)?;
    write_yaml(medium, &mut main_output, Some("medium"))?;
    let inter_yaml = ensemble.inter_moves_to_yaml();
    if !inter_yaml.is_empty() {
        write_yaml(&inter_yaml, &mut main_output, Some("gibbs_moves"))?;
    }

    // per-box state files
    if let Some(state_path) = state {
        for (i, mc) in ensemble.boxes().iter().enumerate() {
            let box_state = box_prefixed_path(state_path, i);
            mc.save_state().to_file(&box_state)?;
            log::info!("Saved box {i} state to {}", box_state.display());
        }
    }

    Ok(())
}
