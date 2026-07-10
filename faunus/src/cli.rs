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

use crate::{simulation::git_revision, Simulation};
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
        /// Common starting state for all windows (seed for the drive phase;
        /// ignored for windows that already have a per-window state)
        #[clap(long)]
        state: Option<PathBuf>,
        /// PMF output file
        #[clap(long, short = 'o', default_value = "pmf.csv")]
        pmf_output: PathBuf,
        /// Max parallel threads (0 = all cores)
        #[clap(long, short = 'j', default_value = "0")]
        threads: usize,
    },
    /// Wang-Landau flat-histogram sampling
    #[clap(arg_required_else_help = true)]
    WangLandau {
        /// Input file in YAML format
        #[clap(long, short = 'i')]
        input: PathBuf,
        /// Directory for walker and histogram state files
        #[clap(long, short = 's', default_value = "wl_states")]
        state_dir: PathBuf,
        /// Free energy output file
        #[clap(long, short = 'o', default_value = "free_energy.csv")]
        output: PathBuf,
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

/// Check that `-o` is writable before the run, so an unusable path costs nothing rather than a
/// whole simulation. Deliberately does not truncate: a run that fails or is interrupted must
/// leave any previous results intact.
fn probe_output_path(output: &std::path::Path) -> Result<()> {
    use anyhow::Context as _;
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(output)
        .with_context(|| format!("Cannot write output file '{}'", output.display()))?;
    Ok(())
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
    log::info!(
        "faunus {} (git {})",
        env!("CARGO_PKG_VERSION"),
        git_revision()
    );

    match args.command {
        Commands::Run { input, state } => {
            let mut simulation = Simulation::from_file(&input, state.as_deref())?;
            // Reject an unwritable `-o` before the run rather than after it.
            probe_output_path(&args.output)?;
            let progress = ProgressBar::new(0);
            let output = simulation.run_with_progress(&mut |step, total| {
                progress.set_length(total as u64);
                progress.set_position(step as u64);
            })?;
            progress.finish();

            output.write_to(&args.output)?;
            if let Some(state_path) = state.as_deref() {
                simulation.save_state(state_path)?;
            }
        }
        Commands::Rerun { input, traj, aux } => {
            probe_output_path(&args.output)?;
            crate::replay(&input, &traj, aux.as_deref())?.write_to(&args.output)?;
        }
        Commands::Umbrella {
            input,
            state_dir,
            state,
            pmf_output,
            threads,
        } => {
            crate::umbrella::run(&input, &state_dir, state.as_deref(), &pmf_output, threads)?;
        }
        Commands::WangLandau {
            input,
            state_dir,
            output,
            threads,
        } => {
            crate::wang_landau::run(&input, &state_dir, &output, threads)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::probe_output_path;

    /// A run that fails or is interrupted must leave the previous `-o` file intact, so the probe
    /// may create the file but must never truncate it.
    #[test]
    fn probe_output_path_creates_without_truncating() {
        let dir = tempfile::tempdir().unwrap();

        let absent = dir.path().join("new.yaml");
        probe_output_path(&absent).unwrap();
        assert!(absent.exists(), "probe should create a missing output file");

        let existing = dir.path().join("previous.yaml");
        std::fs::write(&existing, "earlier results\n").unwrap();
        probe_output_path(&existing).unwrap();
        assert_eq!(
            std::fs::read_to_string(&existing).unwrap(),
            "earlier results\n",
            "probe must not truncate an existing output file"
        );
    }

    #[test]
    fn probe_output_path_rejects_an_unwritable_path() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("no-such-dir").join("out.yaml");
        let error = probe_output_path(&nested).unwrap_err();
        assert!(
            format!("{error:#}").contains("out.yaml"),
            "error should name the path: {error:#}"
        );
    }
}
