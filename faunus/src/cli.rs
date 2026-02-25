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
    analysis::{self},
    montecarlo::MarkovChain,
    platform::reference::{get_medium, ReferencePlatform},
    propagate::Propagate,
    WithCell, WithTopology,
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use coulomb::Temperature;
use indicatif::ProgressBar;
use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use std::{io::Write, path::PathBuf};

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run Monte Carlo simulation
    #[clap(arg_required_else_help = true)]
    Run {
        /// Input file in YAML format
        #[clap(long, short = 'i')]
        input: PathBuf,
        /// Start from previously saved state file
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
        std::env::set_var(
            DEFAULT_FILTER_ENV,
            if args.verbose { "Debug" } else { "Info" },
        );
    }
    pretty_env_logger::init();

    let mut yaml_output = std::fs::File::create(args.output)?;

    match args.command {
        Commands::Run { input, state } => {
            run(input, state, &mut yaml_output)?;
        }
    }
    Ok(())
}

/// Helper function to serialize data to an existing YAML file
fn write_yaml<T: serde::Serialize>(
    data: &T,
    output: &mut std::fs::File,
    key: Option<&str>,
) -> Result<()> {
    match key {
        Some(key) => {
            let mut wrapper = std::collections::BTreeMap::new();
            wrapper.insert(key.to_string(), data);
            let yaml = serde_yaml::to_string(&wrapper)?;
            output.write_all(yaml.as_bytes())?;
        }
        None => {
            let yaml = serde_yaml::to_string(data)?;
            output.write_all(yaml.as_bytes())?;
        }
    }
    Ok(())
}

fn run(input: PathBuf, _state: Option<PathBuf>, yaml_output: &mut std::fs::File) -> Result<()> {
    let context = ReferencePlatform::new(&input, None, &mut rand::thread_rng())?;
    let propagate = Propagate::from_file(&input, &context).unwrap();

    let medium = get_medium(&input)?;
    log::info!("{}", medium);

    const KILO_JOULE_PER_JOULE: f64 = 1e-3;
    let thermal_energy =
        physical_constants::MOLAR_GAS_CONSTANT * KILO_JOULE_PER_JOULE * medium.temperature();
    log::info!("Thermal energy: {:.2} kJ/mol", thermal_energy);

    let analysis = analysis::from_file(&input, &context)?;
    let mut markov_chain = MarkovChain::new(context.clone(), propagate, thermal_energy, analysis)?;

    write_yaml(&medium, yaml_output, Some("medium"))?;
    write_yaml(&context.cell(), yaml_output, Some("cell"))?;
    write_yaml(&context.topology().blocks(), yaml_output, Some("blocks"))?;

    // Step through the Markov chain and update progress bar
    let pb = ProgressBar::new(markov_chain.get_propagate().max_repeats() as u64);
    for step in markov_chain.iter() {
        pb.set_position(step? as u64);
    }

    // Write final state
    write_yaml(
        &markov_chain.get_propagate(),
        yaml_output,
        Some("propagate"),
    )?;

    Ok(())
}
