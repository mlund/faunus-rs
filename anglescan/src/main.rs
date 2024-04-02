// use super::interact::twobody::TwobodyInteraction;
use anglescan::anglescan::TwobodyAngles;
use anglescan::structure::Structure;
use clap::{Parser, Subcommand};
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan angles and tabulate energy between two rigid bodies
    Scan {
        /// Path to the AAM structure
        #[arg(short = '1', long)]
        mol1: PathBuf,
        /// Path to the AAM structure
        #[arg(short = '2', long)]
        mol2: PathBuf,
        /// Angular resolution in radians
        #[arg(short = 'r', long, default_value = "0.1")]
        resolution: f64,
        // #[arg(short = 'd', long)]
        // distance_range: DistanceInterval,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Scan {
            mol1,
            mol2,
            resolution,
        }) => {
            let ref_pos1 = Structure::from_aam_file(&mol1);
            let ref_pos2 = Structure::from_aam_file(&mol2);
            let scan = TwobodyAngles::new(resolution);
            println!("{}", scan);

            let sum: f64 = scan
                .iter()
                .map(|(q1, q2)| {
                    let mut _pos1 = ref_pos1.positions.iter().map(|p| q1 * p);
                    let mut _pos2 = ref_pos2.positions.iter().map(|p| q2 * p);
                    _pos1.nth(0).unwrap()[0] + _pos2.nth(0).unwrap()[0]
                })
                .sum();
            println!("sum: {}", sum);
        }
        None => {
            println!("No command given");
        }
    }
}
