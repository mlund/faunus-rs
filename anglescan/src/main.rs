use anglescan::{
    anglescan::TwobodyAngles,
    energy,
    structure::{AtomKinds, Structure},
    Sample, Vector3,
};
use clap::{Parser, Subcommand};
use faunus::electrolyte::{DebyeLength, Medium, Salt};
use indicatif::ParallelProgressIterator;
use iter_num_tools::arange;
use nu_ansi_term::Color::{Red, Yellow};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rgb::RGB8;
use std::io::Write;
use std::path::PathBuf;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
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

    // Scan over mass center distances
    let distances: Vec<f64> = arange(*rmin..*rmax, *dr).collect::<Vec<_>>();
    info!(
        "Scanning COM range [{:.1}, {:.1}) in {:.1} Å steps",
        rmin, rmax, dr
    );
    let com_scan = distances
        .par_iter()
        .progress_count(distances.len() as u64)
        .map(|r| Vector3::<f64>::new(0.0, 0.0, *r))
        .map(|r| {
            let sample = scan.sample_all_angles(&ref_a, &ref_b, &pair_matrix, &r, *temperature);
            (r, sample)
        })
        .collect::<Vec<_>>();

    report_pmf(com_scan.as_slice(), &PathBuf::from("pmf.dat"));
}

/// Write PMF and mean energy as a function of mass center separation to file
fn report_pmf(samples: &[(Vector3<f64>, Sample)], path: &PathBuf) {
    // File with F(R) and U(R)
    let mut pmf_file = std::fs::File::create(path).unwrap();
    let mut pmf_data = Vec::<(f32, f32)>::new();
    let mut mean_energy_data = Vec::<(f32, f32)>::new();
    writeln!(pmf_file, "# R/Å F/kT U/kT").unwrap();
    samples.iter().for_each(|(r, sample)| {
        let mean_energy = sample.mean_energy() / sample.thermal_energy;
        let free_energy = sample.free_energy() / sample.thermal_energy;
        pmf_data.push((r.norm() as f32, free_energy as f32));
        mean_energy_data.push((r.norm() as f32, mean_energy as f32));
        writeln!(pmf_file, "{:.2} {:.2} {:.2}", *r, free_energy, mean_energy).unwrap();
    });
    info!(
        "Plot: {} and {} along mass center separation. In units of kT and angstroms.",
        Yellow.paint("free energy"),
        Red.paint("mean energy")
    );
    if log::max_level() >= log::Level::Info {
        const YELLOW: RGB8 = RGB8::new(255, 255, 0);
        const RED: RGB8 = RGB8::new(255, 0, 0);
        let rmin = samples.first().unwrap().0.norm() as f32;
        let rmax = samples.last().unwrap().0.norm() as f32;
        Chart::new(100, 50, rmin, rmax)
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
