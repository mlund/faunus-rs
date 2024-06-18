use anglescan::{
    energy,
    icotable::{IcoTable, Table6D},
    structure::{AtomKinds, Structure},
    to_cartesian, to_spherical, Sample, TwobodyAngles, UnitQuaternion, Vector3,
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use coulomb::{DebyeLength, Medium, Salt};
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use nu_ansi_term::Color::{Red, Yellow};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rgb::RGB8;
use std::{f64::consts::PI, io::Write, ops::Neg, path::PathBuf};
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
use iter_num_tools::arange;
use rand::Rng;
use textplots::{Chart, ColorPlot, Shape};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Dipole {
        /// Path to first XYZ file
        #[arg(short = 'o', long)]
        output: PathBuf,
        /// Dipole moment strength
        #[arg(short = 'm')]
        mu: f64,
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
    },

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
        /// Use icosphere table
        #[arg(long, default_value = "true")]
        icotable: bool,
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
        icotable,
    } = cmd
    else {
        anyhow::bail!("Unknown command");
    };
    assert!(rmin < rmax);

    let mut atomkinds = AtomKinds::from_yaml(atoms)?;
    atomkinds.set_missing_epsilon(2.479);

    let medium = Medium::salt_water(*temperature, Salt::SodiumChloride, *molarity);
    let multipole = coulomb::pairwise::Plain::new(*cutoff, medium.debye_length());
    let pair_matrix = energy::PairMatrix::new(&atomkinds.atomlist, &multipole);
    let ref_a = Structure::from_xyz(mol1, &atomkinds);
    let ref_b = Structure::from_xyz(mol2, &atomkinds);

    info!("{}", medium);

    // Scan over mass center distances
    let distances = iter_num_tools::arange(*rmin..*rmax, *dr).collect_vec();
    info!(
        "Scanning COM range [{:.1}, {:.1}) in {:.1} Å steps 🐾",
        rmin, rmax, dr
    );
    if *icotable {
        do_icoscan(
            *rmin,
            *rmax,
            *dr,
            *resolution,
            ref_a,
            ref_b,
            pair_matrix,
            temperature,
        )
    } else {
        do_anglescan(
            distances,
            *resolution,
            ref_a,
            ref_b,
            pair_matrix,
            temperature,
        )
    }
}

fn do_icoscan(
    rmin: f64,
    rmax: f64,
    dr: f64,
    angle_resolution: f64,
    _ref_a: Structure,
    _ref_b: Structure,
    _pair_matrix: energy::PairMatrix,
    _temperature: &f64,
) -> std::result::Result<(), anyhow::Error> {
    let distances = iter_num_tools::arange(rmin..rmax, dr).collect_vec();

    let _table = Table6D::from_resolution(rmin, rmax, dr, angle_resolution)?;
    for r in distances {
        let _r_vec = Vector3::new(0.0, 0.0, r);
    }
    Ok(())
}

fn do_anglescan(
    distances: Vec<f64>,
    angle_resolution: f64,
    ref_a: Structure,
    ref_b: Structure,
    pair_matrix: energy::PairMatrix,
    temperature: &f64,
) -> std::result::Result<(), anyhow::Error> {
    let scan = TwobodyAngles::from_resolution(angle_resolution).unwrap();
    info!("{} per distance", scan);
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

fn do_dipole(cmd: &Commands) -> Result<()> {
    let Commands::Dipole {
        output,
        mu,
        resolution,
        rmin,
        rmax,
        dr,
    } = cmd
    else {
        panic!("Unexpected command");
    };
    let distances: Vec<f64> = iter_num_tools::arange(*rmin..*rmax, *dr).collect();
    let n_points = (4.0 * PI / resolution.powi(2)).round() as usize;
    let mut icotable = IcoTable::<f64>::from_min_points(n_points, 0.0)?;
    let resolution = (4.0 * PI / icotable.vertices.len() as f64).sqrt();
    log::info!(
        "Requested {} points on a sphere; got {} -> new resolution = {:.3}",
        n_points,
        icotable.vertices.len(),
        resolution
    );

    let mut dipole_file = std::fs::File::create(output)?;
    writeln!(dipole_file, "# R/Å w_vertex w_exact w_interpolated")?;

    let charge = 1.0;
    let bjerrum_len = 7.0;

    // for each ion-dipole separation, calculate the partition function and free energy
    for radius in distances {
        // exact exp. energy at a given point, exp(-βu)
        let exact_exp_energy = |p: &Vector3<f64>| {
            let (_r, theta, _phi) = to_spherical(p);
            let field = bjerrum_len * charge / radius.powi(2);
            let u = field * mu * theta.cos();
            (-u).exp()
        };
        icotable.set_vertex_data(exact_exp_energy);

        // Q summed from exact data at each vertex
        let partition_function = icotable.vertex_data().sum::<f64>() / icotable.len() as f64;

        // analytical solution to angular average of exp(-βu)
        let field = -bjerrum_len * charge / radius.powi(2);
        let exact_free_energy = ((field * mu).sinh() / (field * mu)).ln().neg();

        // rotations to apply to vertices of a new icosphere used for sampling interpolated points
        let mut rng = rand::thread_rng();
        let quaternions: Vec<UnitQuaternion<f64>> = (0..20)
            .map(|_| {
                let point: Vector3<f64> = faunus::transform::random_unit_vector(&mut rng);
                UnitQuaternion::<f64>::from_axis_angle(
                    &nalgebra::Unit::new_normalize(point),
                    rng.gen_range(0.0..PI),
                )
            })
            .collect();

        // Sample interpolated points using a randomly rotate icospheres
        let mut rotated_icosphere = IcoTable::<f64>::from_min_points(1000, 0.0)?;
        let mut partition_func_interpolated = 0.0;

        for q in &quaternions {
            rotated_icosphere.transform_vertex_positions(|v| q.transform_vector(v));
            partition_func_interpolated += rotated_icosphere
                .vertices
                .iter()
                .map(|v| icotable.interpolate(&v.pos))
                .sum::<f64>()
                / rotated_icosphere.len() as f64;
        }
        partition_func_interpolated /= quaternions.len() as f64;

        writeln!(
            dipole_file,
            "{:.5} {:.5} {:.5} {:.5}",
            radius,
            partition_function.ln().neg(),
            exact_free_energy,
            partition_func_interpolated.ln().neg(),
        )?;
    }
    Ok(())
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

    let mut icotable = IcoTable::<f64>::from_min_points(n_points, 0.0)?;
    icotable
        .set_vertex_data(|v| energy::electric_potential(&structure, &v.scale(*radius), &multipole));

    std::fs::File::create("pot_at_vertices.dat")?.write_fmt(format_args!("{}", icotable))?;

    icotable.save_vmd("triangles.vmd", Some(*radius))?;

    // Make PQR file illustrating the electric potential at each vertex
    let mut pqr_file = std::fs::File::create("potential.pqr")?;
    for (vertex, data) in std::iter::zip(&icotable.vertices, icotable.vertex_data()) {
        pqr_write_atom(&mut pqr_file, 1, &vertex.pos.scale(*radius), *data, 2.0)?;
    }

    // Compare interpolated and exact potential linearly in angular space
    let mut pot_angles_file = std::fs::File::create("pot_at_angles.dat")?;
    let mut pqr_file = std::fs::File::create("potential_angles.pqr")?;
    writeln!(pot_angles_file, "# theta phi interpolated exact relerr")?;
    for theta in arange(0.0001..PI, resolution) {
        for phi in arange(0.0001..2.0 * PI, resolution) {
            let point = &to_cartesian(1.0, theta, phi);
            let interpolated = icotable.interpolate(point);
            let exact = energy::electric_potential(&structure, &point.scale(*radius), &multipole);
            pqr_write_atom(&mut pqr_file, 1, &point.scale(*radius), exact, 2.0)?;
            let rel_err = (interpolated - exact) / exact;
            let abs_err = (interpolated - exact).abs();
            if abs_err > 0.05 {
                log::debug!(
                    "Potential at theta={:.3} phi={:.3} is {:.4} (exact: {:.4}) abs. error {:.4}",
                    theta,
                    phi,
                    interpolated,
                    exact,
                    abs_err
                );
                let face = icotable.nearest_face(point);
                let bary = icotable.naive_barycentric(point, &face);
                log::debug!("Face: {:?} Barycentric: {:?}\n", face, bary);
            }
            writeln!(
                pot_angles_file,
                "{:.3} {:.3} {:.4} {:.4} {:.4}",
                theta, phi, interpolated, exact, rel_err
            )?;
        }
    }
    Ok(())
}

/// From single ATOM record in PQR file stream
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

fn do_main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    pretty_env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Some(cmd) => match cmd {
            Commands::Dipole { .. } => do_dipole(&cmd)?,
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
