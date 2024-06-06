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
use std::f64::consts::PI;
use std::{io::Write, path::PathBuf};
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
use iter_num_tools::arange;
use textplots::{Chart, ColorPlot, Shape};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Potential {
        /// Path to first XYZ file
        #[arg(short = '1', long)]
        mol1: PathBuf,
        /// Angular resolution in radians
        #[arg(short = 'r', long, default_value = "0.1")]
        resolution: f64,
        /// Radius around center of mass to scan to calc. potentil (angstroms)
        #[arg(long)]
        radius: f64,
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
fn do_scan(cmd: &Commands) -> Result<()> {
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
    } = cmd
    else {
        anyhow::bail!("Unknown command");
    };
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

// Calculate electric potential at points on a sphere around a molecule
fn do_potential(cmd: &Commands) -> Result<()> {
    let Commands::Potential {
        mol1,
        resolution,
        radius,
        atoms,
        molarity,
        cutoff,
        temperature,
    } = cmd
    else {
        panic!("Unexpected command");
    };
    let mut atomkinds = AtomKinds::from_yaml(atoms)?;
    atomkinds.set_missing_epsilon(2.479);

    let medium = Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity);
    let multipole = coulomb::pairwise::Plain::new(*cutoff, medium.debye_length());
    let structure = Structure::from_xyz(mol1, &atomkinds);

    let n_points = (4.0 * PI / resolution.powi(2)).round() as usize;
    let vertices = anglescan::make_icosphere_vertices(n_points)?;
    let resolution = (4.0 * PI / vertices.len() as f64).sqrt();
    log::info!(
        "Requested {} points on a sphere; got {} -> new resolution = {:.2}",
        n_points,
        vertices.len(),
        resolution
    );

    let mut icotable = anglescan::IcoSphereTable::from_min_points(n_points)?;
    let mut pqr_file = std::fs::File::create("potential.pqr")?;
    let mut pot_vertices_file = std::fs::File::create("pot_at_vertices.dat")?;

    // Calculate electric potential at vertices scaled by radius
    for (vertex, data) in std::iter::zip(icotable.vertices.clone(), icotable.vertex_data_mut()) {
        *data = energy::electric_potential(&structure, &vertex.scale(*radius), &multipole);
        let (_r, theta, phi) = to_spherical(&vertex);
        writeln!(pot_vertices_file, "{:.3} {:.3} {:.4}", theta, phi, *data)?;
        pqr_write_atom(&mut pqr_file, 1, &vertex.scale(*radius), *data, 2.0)?;
    }

    let mut pot_angles_file = std::fs::File::create("pot_at_angles.dat")?;
    for theta in arange(0.0001..PI, resolution) {
        for phi in arange(0.0001..2.0 * PI, resolution / 2.0) {
            let point = &to_cartesian(1.0, theta, phi);
            let potential = icotable.barycentric_interpolation(point);
            let exact_potential =
                energy::electric_potential(&structure, &point.scale(*radius), &multipole);
            let rel_err = (potential - exact_potential) / exact_potential;
            let abs_err = (potential - exact_potential).abs();
            if abs_err > 0.05 {
                println!(
                    "Potential at theta={:.3} phi={:.3} is {:.4} (exact: {:.4}) abs. error {:.4}",
                    theta, phi, potential, exact_potential, abs_err
                );
                let face = icotable.nearest_face(point);
                let bary = icotable.barycentric(point, &face);
                println!("Face: {:?} Barycentric: {:?}", face, bary);
                println!("");
            }
            writeln!(
                pot_angles_file,
                "{:.3} {:.3} {:.4} {:.4} {:.4}",
                theta, phi, potential, exact_potential, rel_err
            )?;
        }
    }

    Ok(())
}

fn pqr_write_atom(
    stream: &mut impl std::io::Write,
    atom_id: usize,
    pos: &Vector3<f64>,
    charge: f64,
    radius: f64,
) -> Result<()> {
    writeln!(
        stream,
        "ATOM  {:5} {:4.4} {:4.3}{:5}    {:8.3} {:8.3} {:8.3} {:.3} {:.3}",
        atom_id, "A", "AAA", 1, pos.x, pos.y, pos.z, charge, radius
    )?;
    Ok(())
}

/// Converts Cartesian coordinates to spherical coordinates (r, theta, phi)
/// where:
/// - r is the radius
/// - theta is the polar angle (0..pi)
/// - phi is the azimuthal angle (0..2pi)
fn to_spherical(cartesian: &Vector3<f64>) -> (f64, f64, f64) {
    let r = cartesian.norm();
    let theta = (cartesian.z / r).acos();
    let phi = cartesian.y.atan2(cartesian.x);
    if phi < 0.0 {
        (r, theta, phi + 2.0 * PI)
    } else {
        (r, theta, phi)
    }
}

/// Converts spherical coordinates (r, theta, phi) to Cartesian coordinates
/// where:
/// - r is the radius
/// - theta is the polar angle (0..pi)
/// - phi is the azimuthal angle (0..2pi)
fn to_cartesian(r: f64, theta: f64, phi: f64) -> Vector3<f64> {
    let (theta_sin, theta_cos) = theta.sin_cos();
    let (phi_sin, phi_cos) = phi.sin_cos();
    Vector3::new(
        r * theta_sin * phi_cos,
        r * theta_sin * phi_sin,
        r * theta_cos,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_spherical_cartesian_conversion() {
        const ANGLE_TOL: f64 = 1e-6;
        // Skip theta = 0 as phi is undefined
        for theta in arange(0.00001..PI, 0.01) {
            for phi in arange(0.0..2.0 * PI, 0.01) {
                let cartesian = to_cartesian(1.0, theta, phi);
                let (_, theta_converted, phi_converted) = to_spherical(&cartesian);
                assert_relative_eq!(theta, theta_converted, epsilon = ANGLE_TOL);
                assert_relative_eq!(phi, phi_converted, epsilon = ANGLE_TOL);
            }
        }
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
            Commands::Potential { .. } => do_potential(&cmd)?,
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
