use anglescan::{
    energy,
    structure::{AtomKinds, Structure},
    Sample, TwobodyAngles, Vector3,
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use coulomb::{DebyeLength, Medium, Salt};
use indicatif::ParallelProgressIterator;
use nu_ansi_term::Color::{Red, Yellow};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rgb::RGB8;
use std::{io::Write, path::PathBuf};
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
        /// Path to first XYZ file
        #[arg(short = '1', long)]
        mol1: PathBuf,
        /// Path to second XYZ file
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
fn do_scan(scan_command: &Commands) -> Result<()> {
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

    let mut atomkinds = AtomKinds::from_yaml(atoms)?;
    atomkinds.set_missing_epsilon(2.479);

    let scan = TwobodyAngles::new(*resolution).unwrap();
    let medium = Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity);
    let multipole = coulomb::pairwise::Plain::new(*cutoff, medium.debye_length());
    let pair_matrix = energy::PairMatrix::new(&atomkinds.atomlist, &multipole);
    let ref_a = Structure::from_xyz(mol1, &atomkinds);
    let ref_b = Structure::from_xyz(mol2, &atomkinds);

    info!("{} per distance", scan);
    info!("{}", medium);

    // Scan over mass center distances
    let distances: Vec<f64> = iter_num_tools::arange(*rmin..*rmax, *dr).collect::<Vec<_>>();
    info!(
        "Scanning COM range [{:.1}, {:.1}) in {:.1} Å steps 🐾",
        rmin, rmax, dr
    );
    let com_scan = distances
        .par_iter()
        .progress_count(distances.len() as u64)
        .map(|r| {
            let r_vec = Vector3::<f64>::new(0.0, 0.0, *r);
            let sample = scan
                .sample_all_angles(&ref_a, &ref_b, &pair_matrix, &r_vec, *temperature)
                .unwrap();
            (r_vec, sample)
        })
        .collect::<Vec<_>>();

    report_pmf(com_scan.as_slice(), &PathBuf::from("pmf.dat"));
    Ok(())
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
        writeln!(
            pmf_file,
            "{:.2} {:.2} {:.2}",
            r.norm(),
            free_energy,
            mean_energy
        )
        .unwrap();
    });
    info!(
        "Plot: {} and {} along mass center separation. In units of kT and angstroms.",
        Yellow.bold().paint("free energy"),
        Red.bold().paint("mean energy")
    );
    if log::max_level() >= log::Level::Info {
        const YELLOW: RGB8 = RGB8::new(255, 255, 0);
        const RED: RGB8 = RGB8::new(255, 0, 0);
        let rmin = mean_energy_data.first().unwrap().0;
        let rmax = mean_energy_data.last().unwrap().0;
        Chart::new(100, 50, rmin, rmax)
            .linecolorplot(&Shape::Lines(&mean_energy_data), RED)
            .linecolorplot(&Shape::Lines(&pmf_data), YELLOW)
            .nice();
    }
}

fn do_main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Some(cmd) => match cmd {
            Commands::Scan { .. } => do_scan(&cmd)?,
        },
        None => {
            anyhow::bail!("No command given");
        }
    };
    Ok(())
}

fn main() {
    if let Err(err) = do_main() {
        eprintln!("Error: {}", &err);
        std::process::exit(1);
    }
}
