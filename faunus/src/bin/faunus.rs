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

use anyhow::Result;
use clap::{Parser, Subcommand};
use faunus::topology::Topology;
use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run Monte Carlo simulation
    #[clap(arg_required_else_help = true)]
    Run {
        /// Input file in YAML format
        #[clap(long, short = 'i')]
        input: PathBuf,
        /// Start from previously saved state file
        #[clap(long, short = 's')]
        state: PathBuf,
    },
}

#[derive(Parser)]
#[clap(version, about, long_about = None)]
pub struct Args {
    #[clap(subcommand)]
    pub command: Commands,

    /// Verbose output. See more with e.g. RUST_LOG=Trace
    #[clap(long, short = 'v', action)]
    pub verbose: bool,
}

fn main() {
    if let Err(err) = do_main() {
        eprintln!("Error: {}", &err);
        std::process::exit(1);
    }
}

fn do_main() -> Result<()> {
    let args = Args::parse();

    if args.verbose && std::env::var(DEFAULT_FILTER_ENV).is_err() {
        std::env::set_var(DEFAULT_FILTER_ENV, "Debug");
    }
    pretty_env_logger::init();

    match args.command {
        Commands::Run { input, state } => {
            run(input, state)?;
        }
    }

    Ok(())
}

fn run(input: PathBuf, _state: PathBuf) -> Result<()> {
    let _topology = Topology::from_file(input)?;
    Ok(())
}
