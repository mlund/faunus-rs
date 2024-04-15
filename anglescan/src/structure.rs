use crate::Vector3;
use anyhow::Result;
use chemfiles::Frame;
use faunus::topology::AtomKind;
use itertools::Itertools;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize)]
pub struct AtomKinds {
    pub atomlist: Vec<AtomKind>,
    pub comment: Option<String>,
    pub version: semver::Version,
}

impl AtomKinds {
    /// Construct from a YAML file with an `atomlist` array
    pub fn from_yaml(path: &PathBuf) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_yaml::from_reader(file).map_err(Into::into)
    }
    /// Set sigma (Å) for atoms with `None` sigma
    pub fn set_missing_sigma(&mut self, default_sigma: f64) {
        self.atomlist
            .iter_mut()
            .filter(|i| i.sigma.is_none())
            .for_each(|i| {
                i.sigma = Some(default_sigma);
            });
    }
    /// Set epsilon (kJ/mol) for atoms with `None` epsilon.
    pub fn set_missing_epsilon(&mut self, default_epsilon: f64) {
        self.atomlist
            .iter_mut()
            .filter(|i| i.epsilon.is_none())
            .for_each(|i| {
                i.epsilon = Some(default_epsilon);
            });
    }
}

impl From<AtomKinds> for Vec<AtomKind> {
    fn from(atomkinds: AtomKinds) -> Vec<AtomKind> {
        atomkinds.atomlist
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
        let name = parts.next().unwrap().to_string();
        parts.next(); // skip the second field
        let pos = Vector3::new(
            parts.next().unwrap().parse().unwrap(),
            parts.next().unwrap().parse().unwrap(),
            parts.next().unwrap().parse().unwrap(),
        );
        let (charge, mass, radius) = parts.map(|i| i.parse().unwrap()).next_tuple().unwrap();
        Self {
            name,
            pos,
            charge,
            mass,
            radius,
        }
    }
}

/// Ad hoc molecular structure containing atoms with positions, masses, charges, and radii
#[derive(Debug, Clone)]
pub struct Structure {
    /// Particle positions
    pub pos: Vec<Vector3<f64>>,
    /// Particle masses
    pub masses: Vec<f64>,
    /// Particle charges
    pub charges: Vec<f64>,
    /// Particle radii
    pub radii: Vec<f64>,
    /// Atom kind ids
    pub atom_ids: Vec<usize>,
}

/// Parse a single line from an XYZ file
fn from_xyz_line(line: &str) -> (String, Vector3<f64>) {
    let mut parts = line.split_whitespace();
    assert_eq!(parts.clone().count(), 4); // name, x, y, z
    let name = parts.next().unwrap().to_string();
    let x: f64 = parts.next().unwrap().parse().unwrap();
    let y: f64 = parts.next().unwrap().parse().unwrap();
    let z: f64 = parts.next().unwrap().parse().unwrap();
    (name, Vector3::new(x, y, z))
}

impl Structure {
    /// Constructs a new structure from an XYZ file, centering the structure at the origin
    pub fn from_xyz(path: &PathBuf, atomkinds: &AtomKinds) -> Self {
        let nxyz: Vec<(String, Vector3<f64>)> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .skip(2) // skip header
            .map(from_xyz_line)
            .collect();

        let atom_ids = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .atomlist
                    .iter()
                    .position(|j| j.name == *name)
                    .unwrap_or_else(|| panic!("Unknown atom name in XYZ file: {}", name))
            })
            .collect();

        let masses = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .atomlist
                    .iter()
                    .find(|i| i.name == *name)
                    .unwrap()
                    .mass
            })
            .collect();

        let charges = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .atomlist
                    .iter()
                    .find(|i| i.name == *name)
                    .unwrap()
                    .charge
            })
            .collect();

        let radii = nxyz
            .iter()
            .map(|(name, _)| {
                atomkinds
                    .atomlist
                    .iter()
                    .find(|i| i.name == *name)
                    .unwrap()
                    .sigma
                    .unwrap_or(0.0)
            })
            .collect();
        let mut structure = Self {
            pos: nxyz.iter().map(|(_, pos)| *pos).collect(),
            masses,
            charges,
            radii,
            atom_ids,
        };
        let center = structure.mass_center();
        structure.translate(&-center); // translate to 0,0,0
        debug!("Read {}: {}", path.display(), structure);
        structure
    }
    /// Constructs a new structure from a Faunus AAM file
    pub fn from_aam(path: &PathBuf, atomkinds: &AtomKinds) -> Self {
        let aam: Vec<AminoAcidModelRecord> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .skip(1) // skip header
            .map(AminoAcidModelRecord::from_line)
            .collect();

        let atom_ids = aam
            .iter()
            .map(|i| {
                atomkinds
                    .atomlist
                    .iter()
                    .position(|j| j.name == i.name)
                    .unwrap_or_else(|| panic!("Unknown atom name in AAM file: {}", i.name))
            })
            .collect();

        let mut structure = Self {
            pos: aam.iter().map(|i| i.pos).collect(),
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
                    .unwrap_or_else(|| panic!("Unknown atom name in structure file: {:?}", atom))
            })
            .collect::<Vec<usize>>();
        let mut structure = Self {
            pos: positions,
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
        self.pos
            .iter()
            .zip(&self.masses)
            .map(|(pos, mass)| pos.scale(*mass))
            .fold(Vector3::<f64>::zeros(), |sum, i| sum + i)
            / total_mass
    }
    /// Translates the structure by a displacement vector
    pub fn translate(&mut self, displacement: &Vector3<f64>) {
        self.pos.iter_mut().for_each(|pos| *pos += displacement);
    }
    /// Net charge of the structure
    pub fn net_charge(&self) -> f64 {
        self.charges.iter().sum()
    }

    /// Calculates the inertia tensor of the structure
    ///
    /// The inertia tensor is computed from positions, 𝒑ᵢ,…𝒑ₙ, with
    /// respect to a reference point, 𝒑ᵣ, here the center of mass.
    ///
    /// 𝐈 = ∑ mᵢ(|𝒓ᵢ|²𝑰₃ - 𝒓ᵢ𝒓ᵢᵀ) where 𝒓ᵢ = 𝒑ᵢ - 𝒑ᵣ.
    ///
    pub fn inertia_tensor(&self) -> nalgebra::Matrix3<f64> {
        let center = self.mass_center();
        inertia_tensor(self.pos.iter(), self.masses.iter(), Some(center))
    }
}

/// Calculates the moment of inertia tensor of a set of point masses.
///
/// The inertia tensor is computed from positions, 𝒑₁,…,𝒑ₙ, with
/// respect to a reference point, 𝑪, typically the center of mass:
///
/// 𝐈 = ∑ mᵢ(|𝒓ᵢ|²𝑰₃ - 𝒓ᵢ𝒓ᵢᵀ) where 𝒓ᵢ = 𝒑ᵢ - 𝑪 and 𝑰₃ is the 3×3 identity matrix.
///
/// If no center is provided, 𝑪=(0,0,0).
///
/// # Further Reading
///
/// - <https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor>
///
pub fn inertia_tensor<'a>(
    positions: impl Iterator<Item = &'a Vector3<f64>>,
    masses: impl Iterator<Item = &'a f64>,
    center: Option<Vector3<f64>>,
) -> Matrix3<f64> {
    positions
        .zip(masses)
        .map(|(pos, mass)| {
            let r = pos - center.unwrap_or(Vector3::<f64>::zeros());
            (r.norm_squared() * Matrix3::<f64>::identity() - r * r.transpose()).scale(*mass)
        })
        .sum()
}

/// Display number of atoms, mass center etc.
impl Display for Structure {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "𝑁={}, ∑𝑞ᵢ={:.2}𝑒, ∑𝑚ᵢ={:.2}",
            self.pos.len(),
            self.net_charge(),
            self.masses.iter().sum::<f64>()
        )
    }
}

/// Converts a slice of f64 to a nalgebra Vector3
fn to_vector3(pos: &[f64; 3]) -> Vector3<f64> {
    Vector3::<f64>::new(pos[0], pos[1], pos[2])
}
