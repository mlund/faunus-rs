use crate::Vector3;
use chemfiles::Frame;
use std::path::PathBuf;

/// Ancient AAM file format from Faunus
#[derive(Debug, Default)]
pub struct AminoAcidModelRecord {
    pub name: String,
    pub pos: Vector3<f64>,
    pub charge: f64,
    pub mass: f64,
    pub radius: f64,
}

impl AminoAcidModelRecord {
    /// Create from space-separated text record (name, _, x, y, z, charge, mass, radius)
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

/// Ad hoc molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug)]
pub struct Structure {
    /// Particle positions
    pub positions: Vec<Vector3<f64>>,
    /// Particle masses
    pub masses: Vec<f64>,
    /// Particle charges
    pub charges: Vec<f64>,
    /// Particle radii
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
