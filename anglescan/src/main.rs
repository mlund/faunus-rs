use anglescan::anglescan::TwobodyAngles;
use anglescan::energy;
use anglescan::structure::AtomKinds;
use anglescan::structure::Structure;
use anglescan::Vector3;
use clap::{Parser, Subcommand};
use itertools_num::linspace;
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
        /// Minimum mass center distance
        #[arg(long)]
        rmin: f64,
        /// Maximum mass center distance
        #[arg(long)]
        rmax: f64,
        /// Mass center distance step
        #[arg(long)]
        dr: f64,
        /// YAML file with atom definitions
        #[arg(short = 'a', long)]
        atoms: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    let multipole = interact::multipole::Yukawa::new(30.0, Some(30.0));
    match cli.command {
        Some(Commands::Scan {
            mol1,
            mol2,
            resolution,
            rmin,
            rmax,
            dr,
            atoms,
        }) => {
            let atomkinds = AtomKinds::from_yaml(&atoms).unwrap();
            let ref_pos1 = Structure::from_aam_file(&mol1, &atomkinds);
            let ref_pos2 = Structure::from_aam_file(&mol2, &atomkinds);
            let scan = TwobodyAngles::new(resolution);
            println!("{} per distance", scan);

            assert!(rmin < rmax);
            //let pair_matrix = energy::PairMatrix::new(&atomkinds.atomlist, &multipole);
            let distances =
                linspace(rmin, rmax, ((rmax - rmin) / dr) as usize).collect::<Vec<f64>>();
            distances.par_iter().for_each(|r| {
                let r_vec = Vector3::<f64>::new(0.0, 0.0, *r);
                let sum: f64 = scan
                    .iter()
                    .map(|(q1, q2)| {
                        let mut _pos1 = ref_pos1.positions.iter().map(|p| q1 * p);
                        let mut _pos2 = ref_pos2.positions.iter().map(|p| (q2 * p) + r_vec);
                        _pos1.nth(0).unwrap()[2] + _pos2.nth(0).unwrap()[2]
                    })
                    .sum();
                println!("distance = {:.2}, sum: {:.2}", r, sum);
            })
        }
        None => {
            println!("No command given");
        }
    }
}
