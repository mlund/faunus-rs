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
    analysis::{self, AnalysisCollectionExt, Analyze, Frequency},
    backend::Backend,
    energy::EnergyChange,
    group::GroupCollection,
    montecarlo::{gibbs::GibbsEnsemble, MarkovChain},
    simulation::{self, box_prefixed_path, write_yaml, Simulation},
    topology::io::frame_state::{self, FrameStateReader},
    Change, Context, Particle, ParticleSystem, Point, WithHamiltonian,
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::ProgressBar;
use interatomic::coulomb::Temperature;
use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use std::path::PathBuf;

/// Thermal energy kT in kJ/mol.
fn thermal_energy(medium: &interatomic::coulomb::Medium) -> f64 {
    crate::R_IN_KJ_PER_MOL * medium.temperature()
}

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
    /// Replay a trajectory through a (possibly different) Hamiltonian
    #[clap(arg_required_else_help = true)]
    Rerun {
        /// Input file in YAML format (Hamiltonian + analysis config)
        #[clap(long, short = 'i')]
        input: PathBuf,
        /// XTC trajectory file
        #[clap(long)]
        traj: PathBuf,
        /// Frame state file (default: derived from --traj by replacing extension with .aux)
        #[clap(long)]
        aux: Option<PathBuf>,
    },
    /// Multi-walker umbrella sampling with free-energy stitching
    #[clap(arg_required_else_help = true)]
    Umbrella {
        /// Input file in YAML format
        #[clap(long, short = 'i')]
        input: PathBuf,
        /// Directory for per-window state files
        #[clap(long, short = 's', default_value = "umbrella_states")]
        state_dir: PathBuf,
        /// PMF output file
        #[clap(long, short = 'o', default_value = "pmf.csv")]
        pmf_output: PathBuf,
        /// Max parallel threads (0 = all cores)
        #[clap(long, short = 'j', default_value = "0")]
        threads: usize,
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
        Commands::Rerun { input, traj, aux } => {
            let mut yaml_output = std::fs::File::create(&args.output)?;
            run_rerun(&input, &traj, aux.as_deref(), &mut yaml_output)?;
        }
        Commands::Umbrella {
            input,
            state_dir,
            pmf_output,
            threads,
        } => {
            crate::umbrella::run(&input, &state_dir, &pmf_output, threads)?;
        }
    }
    Ok(())
}

/// Sum of charges of all active particles in the system.
fn net_charge(context: &impl ParticleSystem) -> f64 {
    let atomkinds = context.topology_ref().atomkinds();
    context
        .groups()
        .iter()
        .flat_map(|g| g.iter_active())
        .map(|i| atomkinds[context.atom_kind(i)].charge())
        .sum()
}

/// Write per-box YAML output: medium, blocks, cell, energy, propagate, analysis.
/// Returns `(final_energy, drift)`.
fn write_mc_output<T: Context + 'static>(
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

fn run_single_box<T: Context + 'static>(
    mut mc: MarkovChain<T>,
    medium: &interatomic::coulomb::Medium,
    state: Option<&std::path::Path>,
    yaml_output: &mut std::fs::File,
) -> Result<()> {
    let thermal_energy = thermal_energy(medium);
    log::info!("{}", medium);
    log::info!("Thermal energy: {thermal_energy:.2} kJ/mol");

    let initial_energy = mc.system_energy();
    log::info!("Initial energy = {initial_energy:.2} kJ/mol");
    log::info!("Net charge = {:.4e} e", net_charge(mc.context()));

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
    log::info!("Net charge = {:.4e} e", net_charge(mc.context()));
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

/// Validate that the aux file header matches the context topology.
fn validate_aux_header(header: &frame_state::FrameStateHeader, context: &Backend) -> Result<()> {
    let groups = context.groups();
    if header.n_groups as usize != groups.len() {
        anyhow::bail!(
            "Frame state group count ({}) doesn't match context ({})",
            header.n_groups,
            groups.len()
        );
    }
    if header.n_particles as usize != context.num_particles() {
        anyhow::bail!(
            "Frame state particle count ({}) doesn't match context ({})",
            header.n_particles,
            context.num_particles()
        );
    }
    for (i, (&(mol_id, capacity), group)) in header.groups.iter().zip(groups).enumerate() {
        if mol_id as usize != group.molecule() || capacity as usize != group.capacity() {
            anyhow::bail!(
                "Frame state group {i} topology mismatch: aux=({mol_id}, {capacity}), \
                 context=({}, {})",
                group.molecule(),
                group.capacity()
            );
        }
    }
    Ok(())
}

/// Load a single frame's state into the context from XTC positions + aux data.
fn load_frame_into_context(
    context: &mut Backend,
    particles: &[Particle],
    frame_state: &frame_state::FrameStateFrame,
) -> Result<()> {
    use crate::group::GroupSize;

    let sizes: Vec<GroupSize> = frame_state
        .group_sizes
        .iter()
        .zip(context.groups().iter())
        .map(|(&size, g)| GroupSize::from_count(size as usize, g.capacity()))
        .collect();

    context.apply_particles_and_groups(particles, &sizes, &frame_state.quaternions)?;
    context.update(&Change::Everything)?;
    Ok(())
}

/// Replay a trajectory through a (possibly different) Hamiltonian, running
/// analyses on each frame.
fn run_rerun(
    input: &std::path::Path,
    traj_path: &std::path::Path,
    aux_path: Option<&std::path::Path>,
    yaml_output: &mut std::fs::File,
) -> Result<()> {
    use crate::topology::io::NM_TO_ANGSTROM;

    let (mut context, mut analyses, medium) = simulation::build_context_and_analyses(input)?;
    log::info!("{}", medium);
    log::info!("Thermal energy: {:.2} kJ/mol", thermal_energy(&medium));

    // Every frame must be sampled since we can't skip frames in the XTC/aux pair
    analyses.override_frequencies(Frequency::Every(1));

    let aux = aux_path
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| frame_state::aux_path_from_traj(traj_path));
    let mut aux_reader = FrameStateReader::open(&aux)?;

    validate_aux_header(aux_reader.header(), &context)?;

    let mut xtc_reader = molly::XTCReader::open(traj_path)
        .map_err(|e| anyhow::anyhow!("Cannot open XTC file {}: {e}", traj_path.display()))?;
    let mut frame = molly::Frame::default();

    let n_particles = context.num_particles();
    let mut frame_index = 0usize;
    // Pre-allocate and reuse across frames to avoid per-frame heap allocation
    let mut particles = Vec::with_capacity(n_particles);
    let mut aux_frame = frame_state::FrameStateFrame::default();

    log::info!("Replaying trajectory: {}", traj_path.display());

    loop {
        match xtc_reader.read_frame(&mut frame) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Error reading XTC frame {frame_index}: {e}"
                ))
            }
        }

        if frame.natoms() != n_particles {
            anyhow::bail!(
                "XTC frame {frame_index} has {} atoms, expected {n_particles}",
                frame.natoms()
            );
        }

        if !aux_reader.read_frame_into(&mut aux_frame)? {
            anyhow::bail!("Aux file ended before XTC at frame {frame_index}");
        }

        // XTC stores positions in nm with corner at origin; Faunus uses Å with
        // center at origin. Column-major layout: [c0x, c0y, c0z, c1x, ...].
        let cols = frame.boxvec.to_cols_array();
        let box_half = Point::new(
            cols[0] as f64 * NM_TO_ANGSTROM * 0.5,
            cols[4] as f64 * NM_TO_ANGSTROM * 0.5,
            cols[8] as f64 * NM_TO_ANGSTROM * 0.5,
        );

        particles.clear();
        particles.extend((0..n_particles).map(|i| {
            let pos = Point::new(
                frame.positions[3 * i] as f64 * NM_TO_ANGSTROM - box_half.x,
                frame.positions[3 * i + 1] as f64 * NM_TO_ANGSTROM - box_half.y,
                frame.positions[3 * i + 2] as f64 * NM_TO_ANGSTROM - box_half.z,
            );
            Particle::new(aux_frame.atom_ids[i] as usize, pos)
        }));

        load_frame_into_context(&mut context, &particles, &aux_frame)?;
        analyses.sample(&context, frame_index)?;
        frame_index += 1;
    }

    log::info!("Processed {frame_index} frames");

    analyses.finalize(&context)?;
    analyses.write_to_disk()?;

    write_yaml(&medium, yaml_output, Some("medium"))?;

    let energy = context.hamiltonian().energy(&context, &Change::Everything);
    let energy_summary = std::collections::BTreeMap::from([
        ("last_frame_energy".to_string(), energy),
        ("frames".to_string(), frame_index as f64),
    ]);
    write_yaml(&energy_summary, yaml_output, Some("rerun"))?;

    let analysis_yaml = analysis::analyses_to_yaml(&analyses);
    if !analysis_yaml.is_empty() {
        write_yaml(&analysis_yaml, yaml_output, Some("analysis"))?;
    }

    Ok(())
}

fn run_gibbs(
    mut ensemble: GibbsEnsemble<Backend>,
    medium: &interatomic::coulomb::Medium,
    state: Option<&std::path::Path>,
    output_path: &std::path::Path,
) -> Result<()> {
    let thermal_energy = thermal_energy(medium);
    log::info!("{}", medium);
    log::info!("Thermal energy: {thermal_energy:.2} kJ/mol");

    let initial_energies: [f64; 2] = std::array::from_fn(|i| {
        let mc = &ensemble.boxes()[i];
        let e = mc.system_energy();
        let q = net_charge(mc.context());
        log::info!("Box {i}: initial energy = {e:.2} kJ/mol, net charge = {q:.4e} e");
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
        let q = net_charge(mc.context());
        log::info!("Box {i}: final energy = {final_energy:.2} kJ/mol, drift = {drift:.2e}, net charge = {q:.4e} e");
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
