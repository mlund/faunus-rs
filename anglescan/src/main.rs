use anglescan::anglescan::TwobodyAngles;
use anglescan::energy;
use anglescan::structure::AtomKinds;
use anglescan::structure::Structure;
use anglescan::Vector3;
use clap::{Parser, Subcommand};
use faunus::electrolyte::{DebyeLength, Medium, Salt};
use itertools_num::linspace;
use rayon::prelude::*;
use std::path::PathBuf;
extern crate pretty_env_logger;
use log::{debug, info};

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
    pretty_env_logger::init();

    let cli = Cli::parse();
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
            assert!(rmin < rmax);
            let mut atomkinds = AtomKinds::from_yaml(&atoms).unwrap();
            atomkinds.set_default_epsilon(0.05 * 2.45);
            let ref_a = Structure::from_xyz(&mol1, &atomkinds);
            let ref_b = Structure::from_xyz(&mol2, &atomkinds);
            let scan = TwobodyAngles::new(resolution);
            println!("{} per distance", scan);

            let temperature = 298.15; // K
            let molarity = 0.1; // mol/l
            let medium = Medium::salt_water(temperature, Salt::SodiumChloride, molarity);
            let debye_length = medium.debye_length().unwrap();
            info!("Debye length: {:.2} angstrom", debye_length);

            let cutoff = 3.0 * debye_length;
            let multipole = interact::multipole::Coulomb::new(cutoff, Some(debye_length));
            let pair_matrix = energy::PairMatrix::new(&atomkinds.atomlist, &multipole);
            let distances =
                linspace(rmin, rmax, ((rmax - rmin) / dr) as usize).collect::<Vec<f64>>();
            distances.par_iter().for_each(|r| {
                let r_vec = Vector3::<f64>::new(0.0, 0.0, *r);
                let mut a = ref_a.clone();
                let mut b = ref_b.clone();
                let sum: f64 = scan
                    .iter()
                    .map(|(q1, q2)| {
                        a.positions = ref_a.positions.iter().map(|pos| q1 * pos).collect();
                        b.positions = ref_b
                            .positions
                            .iter()
                            .map(|pos| (q2 * pos) + r_vec)
                            .collect();
                        pair_matrix.sum_energy(&a, &b)
                    })
                    .sum();
                println!("distance = {:.2}, energy: {:.2}", r, sum);
            })
        }
        None => {
            println!("No command given");
        }
    }
}
