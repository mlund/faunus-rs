use anglescan::anglescan::TwobodyAngles;
use anglescan::energy;
use anglescan::structure::AtomKinds;
use anglescan::structure::Structure;
use anglescan::Vector3;
use clap::{Parser, Subcommand};
use faunus::electrolyte::{DebyeLength, Medium, Salt};
use indicatif::ParallelProgressIterator;
use iter_num_tools::arange;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::*;
use std::iter::Sum;
use std::ops::Add;
use std::ops::Neg;
use std::path::PathBuf;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;

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
        /// Salt molarity in mol/l
        #[arg(short = 'M', long, default_value = "0.1")]
        molarity: f64,
        /// Cutoff distance for pair-wise interactions (angstroms)
        #[arg(long, default_value = "40.0")]
        cutoff: f64,
        /// Temperature in K
        #[arg(short = 'T', long, default_value = "298.15")]
        temperature: f64,
    },
}

/// Structure to store energy samples
#[derive(Debug, Default, Clone)]
struct Sample {
    /// Number of samples
    n: u64,
    /// Thermal energy, RT in kJ/mol
    thermal_energy: f64,
    /// Boltzmann weighted energy, U * exp(-U/kT)
    mean_energy: f64,
    /// Boltzmann factored energy, exp(-U/kT)
    exp_energy: f64,
}

impl Sample {
    /// New from energy in kJ/mol and temperature in K
    pub fn new(energy: f64, temperature: f64) -> Self {
        let thermal_energy = faunus::MOLAR_GAS_CONSTANT * temperature * 1e-3; // kJ/mol
        let exp_energy = (-energy / thermal_energy).exp();
        Self {
            n: 1,
            thermal_energy,
            mean_energy: energy * exp_energy,
            exp_energy,
        }
    }
    /// Mean energy (kJ/mol)
    pub fn mean_energy(&self) -> f64 {
        self.mean_energy / self.exp_energy
    }
    /// Free energy (kJ / mol)
    pub fn free_energy(&self) -> f64 {
        (self.exp_energy / self.n as f64).ln().neg() * self.thermal_energy
    }
}

impl Sum for Sample {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Sample::default(), |sum, s| sum + s)
    }
}

impl Add for Sample {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            n: self.n + other.n,
            thermal_energy: f64::max(self.thermal_energy, other.thermal_energy),
            mean_energy: self.mean_energy + other.mean_energy,
            exp_energy: self.exp_energy + other.exp_energy,
        }
    }
}

/// Calculate energy of all two-body poses
fn do_scan(scan_command: &Commands) {
    let Commands::Scan {
        mol1,
        mol2,
        resolution,
        rmin,
        rmax,
        dr,
        atoms,
        molarity,
        cutoff,
        temperature,
    } = scan_command;
    assert!(rmin < rmax);
    let mut atomkinds = AtomKinds::from_yaml(&atoms).unwrap();
    atomkinds.set_default_epsilon(0.05 * 2.45);
    let scan = TwobodyAngles::new(*resolution);
    let medium = Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity);
    let multipole = interact::multipole::Coulomb::new(*cutoff, medium.debye_length());
    let pair_matrix = energy::PairMatrix::new(&atomkinds.atomlist, &multipole);
    let ref_a = Structure::from_xyz(&mol1, &atomkinds);
    let ref_b = Structure::from_xyz(&mol2, &atomkinds);

    debug!("{}", ref_a);
    debug!("{}", ref_b);
    info!("{} per distance", scan);
    info!("{}", medium);

    // Scan over mass center distances
    let distances: Vec<f64> = arange(*rmin..*rmax, *dr).collect::<Vec<_>>();
    distances
        .par_iter()
        .map(|r| Vector3::<f64>::new(0.0, 0.0, *r))
        .progress_count(distances.len() as u64)
        .for_each(|r| {
            let mut a = ref_a.clone();
            let mut b = ref_b.clone();
            let samples: Sample = scan // Scan over angles
                .iter()
                .map(|(q1, q2)| {
                    a.pos = ref_a.pos.iter().map(|pos| q1 * pos).collect();
                    b.pos = ref_b.pos.iter().map(|pos| (q2 * pos) + r).collect();
                    Sample::new(pair_matrix.sum_energy(&a, &b), *temperature)
                })
                .sum();
            println!(
                "R = {:.2} Å, ❬U❭/kT = {:.2}, w/kT = {:.2}",
                r.norm(),
                samples.mean_energy() / samples.thermal_energy,
                samples.free_energy() / samples.thermal_energy
            );
        })
}
fn main() {
    pretty_env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Some(cmd) => match cmd {
            Commands::Scan { .. } => do_scan(&cmd),
        },
        None => {
            println!("No command given");
        }
    }
}
