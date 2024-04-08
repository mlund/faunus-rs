use anglescan::anglescan::TwobodyAngles;
use anglescan::structure::{AtomKinds, Structure};
use anglescan::{energy, Vector3};
use clap::{Parser, Subcommand};
use faunus::electrolyte::{DebyeLength, Medium, Salt};
use indicatif::ParallelProgressIterator;
use iter_num_tools::arange;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rgb::RGB8;
use std::io::Write;
use std::iter::Sum;
use std::ops::{Add, Neg};
use std::path::PathBuf;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
extern crate flate2;
use flate2::write::GzEncoder;
use flate2::Compression;
use textplots::{Chart, ColorPlot, Shape};

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
        /// Path to XYZ file
        #[arg(short = '1', long)]
        mol1: PathBuf,
        /// Path to XYZ file
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
        /// YAML file with atom definitions (names, charges, etc.)
        #[arg(short = 'a', long)]
        atoms: PathBuf,
        /// 1:1 salt molarity in mol/l
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
    let mut atomkinds = AtomKinds::from_yaml(atoms).unwrap();
    atomkinds.set_default_epsilon(1.0 * 2.45);
    let scan = TwobodyAngles::new(*resolution);
    let medium = Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity);
    let multipole = interact::multipole::Coulomb::new(*cutoff, medium.debye_length());
    let pair_matrix = energy::PairMatrix::new(&atomkinds.atomlist, &multipole);
    let ref_a = Structure::from_xyz(mol1, &atomkinds);
    let ref_b = Structure::from_xyz(mol2, &atomkinds);

    debug!("{}", ref_a);
    debug!("{}", ref_b);
    info!("{} per distance", scan);
    info!("{}", medium);

    // File with F(R) and U(R)
    let mut pmf_file = std::fs::File::create("pmf.dat").unwrap();
    let mut pmf_data = Vec::<(f32, f32)>::new();
    let mut mean_energy_data = Vec::<(f32, f32)>::new();
    writeln!(pmf_file, "# R/Å F/kT U/kT").unwrap();

    // Scan over mass center distances
    let distances: Vec<f64> = arange(*rmin..*rmax, *dr).collect::<Vec<_>>();
    info!(
        "Scanning COM range [{:.1}, {:.1}) in {:.1} Å steps",
        rmin, rmax, dr
    );
    distances
        .par_iter()
        .progress_count(distances.len() as u64)
        .map(|r| Vector3::<f64>::new(0.0, 0.0, *r))
        .map(|r| {
            let mut a = ref_a.clone();
            let mut b = ref_b.clone();
            let file = std::fs::File::create(format!("R_{:.1}.dat.gz", r.norm())).unwrap();
            let mut encoder = GzEncoder::new(file, Compression::default());
            let sample = scan // Scan over angles
                .iter()
                .map(|(q1, q2)| {
                    a.pos = ref_a.pos.iter().map(|pos| q1 * pos).collect();
                    b.pos = ref_b.pos.iter().map(|pos| (q2 * pos) + r).collect();
                    let energy = pair_matrix.sum_energy(&a, &b);
                    writeln!(
                        encoder,
                        "{:.2} {:.2} {:.2} {:?} {:?} {:.2}",
                        r.x,
                        r.y,
                        r.z,
                        q1.coords.as_slice(),
                        q2.coords.as_slice(),
                        energy
                    )
                    .unwrap();
                    Sample::new(energy, *temperature)
                })
                .sum::<Sample>();
            (r, sample)
        })
        .collect::<Vec<_>>()
        .iter()
        .for_each(|(r, sample)| {
            let mean_energy = sample.mean_energy() / sample.thermal_energy;
            let free_energy = sample.free_energy() / sample.thermal_energy;
            pmf_data.push((r.norm() as f32, free_energy as f32));
            mean_energy_data.push((r.norm() as f32, mean_energy as f32));
            writeln!(
                pmf_file,
                "{:.2} {:.2} {:.2}",
                r.norm(),
                free_energy,
                mean_energy
            )
            .unwrap();
        });
    info!("Plot: free energy (yellow) and energy (red) along mass center separation. In units of kT and angstroms.");
    if log::max_level() >= log::Level::Info {
        const YELLOW: RGB8 = RGB8::new(255, 255, 0);
        const RED: RGB8 = RGB8::new(255, 0, 0);
        Chart::new(100, 50, *rmin as f32, *rmax as f32)
            .linecolorplot(&Shape::Lines(&mean_energy_data), RED)
            .linecolorplot(&Shape::Lines(&pmf_data), YELLOW)
            .nice();
    }
}
fn main() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
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
