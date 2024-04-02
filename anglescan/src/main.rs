// use super::interact::twobody::TwobodyInteraction;
use anglescan::anglescan::TwobodyAngles;
use chemfiles::Frame;
use clap::{Parser, Subcommand};
use nalgebra::Vector3;
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

#[derive(Debug, Default)]
pub struct AminoAcidModelRecord {
    pub name: String,
    pub pos: Vector3<f64>,
    pub charge: f64,
    pub mass: f64,
    pub radius: f64,
}

// Ancient AAM file format from Faunus
impl AminoAcidModelRecord {
    // create from space-separated text record (name, _, x, y, z, charge, mass, radius)
    pub fn from_line(text: &str) -> Self {
        let mut parts = text.split_whitespace();
        assert!(parts.clone().count() == 8);
        let name = parts.next().unwrap();
        parts.next(); // skip the second field
        let pos = Vector3::new(
            parts.next().unwrap().parse().unwrap(),
            parts.next().unwrap().parse().unwrap(),
            parts.next().unwrap().parse().unwrap(),
        );
        let charge: f64 = parts.next().unwrap().parse().unwrap();
        let mass: f64 = parts.next().unwrap().parse().unwrap();
        let radius: f64 = parts.next().unwrap().parse().unwrap();
        Self {
            name: name.to_string(),
            pos,
            charge,
            mass,
            radius,
        }
    }
}

/// Molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug)]
pub struct Structure {
    pub positions: Vec<Vector3<f64>>,
    pub masses: Vec<f64>,
    pub charges: Vec<f64>,
    pub radii: Vec<f64>,
}

impl Structure {
    /// Constructs a new structure from a Faunus AAM file
    pub fn from_aam_file(path: &PathBuf) -> Self {
        let aam: Vec<AminoAcidModelRecord> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .skip(1) // skip header
            .map(|line| AminoAcidModelRecord::from_line(line))
            .collect();
        let mut structure = Self {
            positions: aam.iter().map(|i| i.pos).collect(),
            masses: aam.iter().map(|i| i.mass).collect(),
            charges: aam.iter().map(|i| i.charge).collect(),
            radii: aam.iter().map(|i| i.radius).collect(),
        };
        let center = structure.mass_center();
        structure.translate(&-center); // translate to 0,0,0
        structure
    }

    /// Constructs a new structure from a chemfiles trajectory file
    pub fn from_chemfiles(path: &PathBuf) -> Self {
        let mut traj = chemfiles::Trajectory::open(path, 'r').unwrap();
        let mut frame = Frame::new();
        traj.read(&mut frame).unwrap();
        let masses = frame
            .iter_atoms()
            .map(|atom| atom.mass())
            .collect::<Vec<f64>>();
        let positions = frame
            .positions()
            .iter()
            .map(to_vector3)
            .collect::<Vec<Vector3<f64>>>();
        let mut structure = Self {
            positions,
            masses,
            charges: vec![0.0; frame.size()],
            radii: vec![0.0; frame.size()],
        };
        let center = structure.mass_center();
        structure.translate(&-center);
        structure
    }

    /// Returns the center of mass of the structure
    pub fn mass_center(&self) -> Vector3<f64> {
        let total_mass: f64 = self.masses.iter().sum();
        self.positions
            .iter()
            .zip(&self.masses)
            .map(|(pos, mass)| pos.scale(*mass))
            .fold(Vector3::<f64>::zeros(), |sum, i| sum + i)
            / total_mass
    }
    /// Translates the structure by a displacement vector
    pub fn translate(&mut self, displacement: &Vector3<f64>) {
        self.positions
            .iter_mut()
            .for_each(|pos| *pos += displacement);
    }
}

/// Converts a slice of f64 to a nalgebra Vector3
fn to_vector3(pos: &[f64; 3]) -> Vector3<f64> {
    Vector3::<f64>::new(pos[0], pos[1], pos[2])
}

struct DistanceInterval {
    min: f64,
    max: f64,
    step: f64,
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
