use crate::Vector3;
use anyhow::Result;
use chemfiles::Frame;
use faunus::topology::AtomKind;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize)]
pub struct AtomKinds {
    pub atomlist: Vec<AtomKind>,
    pub comment: Option<String>,
    pub version: semver::Version,
}

impl AtomKinds {
    pub fn from_yaml(path: &PathBuf) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_yaml::from_reader(file).map_err(Into::into)
    }
}

impl Into<Vec<AtomKind>> for AtomKinds {
    fn into(self) -> Vec<AtomKind> {
        self.atomlist
    }
}

// test yaml reading of atoms
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_atomkinds() {
        let path = PathBuf::from("../assets/amino-acid-model-atoms.yml");
        let atomlist = AtomKinds::from_yaml(&path).unwrap();
        assert_eq!(atomlist.atomlist.len(), 37);
        assert_eq!(atomlist.version.to_string(), "1.1.0");
        assert_eq!(atomlist.atomlist[0].name, "Na");
    }
}

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
    /// Atom kind ids
    pub atom_ids: Vec<usize>,
}

impl Structure {
    /// Constructs a new structure from a Faunus AAM file
    pub fn from_aam_file(path: &PathBuf, atomkinds: &AtomKinds) -> Self {
        let aam: Vec<AminoAcidModelRecord> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .skip(1) // skip header
            .map(|line| AminoAcidModelRecord::from_line(line))
            .collect();

        let atom_ids = aam
            .iter()
            .map(|i| {
                atomkinds
                    .atomlist
                    .iter()
                    .position(|j| j.name == i.name)
                    .expect(format!("Unknown atom name in AAM file: {}", i.name).as_str())
            })
            .collect();

        let mut structure = Self {
            positions: aam.iter().map(|i| i.pos).collect(),
            masses: aam.iter().map(|i| i.mass).collect(),
            charges: aam.iter().map(|i| i.charge).collect(),
            radii: aam.iter().map(|i| i.radius).collect(),
            atom_ids,
        };
        let center = structure.mass_center();
        structure.translate(&-center); // translate to 0,0,0
        structure
    }

    /// Constructs a new structure from a chemfiles trajectory file
    pub fn from_chemfiles(path: &PathBuf, atomkinds: &AtomKinds) -> Self {
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
        let atom_ids = frame
            .iter_atoms()
            .map(|atom| {
                atomkinds
                    .atomlist
                    .iter()
                    .position(|kind| kind.name == atom.name())
                    .expect(format!("Unknown atom name in structure file: {:?}", atom).as_str())
            })
            .collect::<Vec<usize>>();
        let mut structure = Self {
            positions,
            masses,
            charges: vec![0.0; frame.size()],
            radii: vec![0.0; frame.size()],
            atom_ids,
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
