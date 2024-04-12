#[cfg(test)]
extern crate approx;
use crate::{energy::PairMatrix, structure::Structure, Sample};
use anyhow::Result;
#[cfg(test)]
use approx::assert_relative_eq;
use hexasphere::shapes::IcoSphere;
use itertools::Itertools;
use std::f64::consts::PI;
use std::fmt::Display;
use std::io::Write;

extern crate flate2;
use flate2::write::GzEncoder;
use flate2::Compression;

pub type Vector3 = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

/// Struct to exhaust all possible 6D relative orientations between two rigid bodies.
///
/// Fibonacci sphere points are used to generate rotations around the z-axis.
#[derive(Debug)]
pub struct TwobodyAngles {
    /// Rotations of the first body
    pub q1: Vec<UnitQuaternion>,
    /// Rotations of the second body
    pub q2: Vec<UnitQuaternion>,
    /// Rotations around connecting z-azis (0..2π)
    pub dihedrals: Vec<UnitQuaternion>,
}

impl Display for TwobodyAngles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n1 = self.q1.len();
        let n2 = self.q2.len();
        let n3 = self.dihedrals.len();
        write!(f, "{} x {} x {} = {} poses 💃🕺", n1, n2, n3, n1 * n2 * n3)
    }
}

impl TwobodyAngles {
    /// Generates a set of quaternions for a rigid body scan
    ///
    /// # Arguments
    /// angle_resolution: f64 - the resolution of the scan in radians
    pub fn new(angle_resolution: f64) -> Self {
        assert!(
            angle_resolution > 0.0,
            "angle_resolution must be greater than 0"
        );

        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        // let mut points = fibonacci_sphere(n_points);
        let mut points = generate_icosphere(n_points).unwrap();
        let angle_resolution = (4.0 * PI / points.len() as f64).sqrt();
        log::info!(
            "Requested {} points on a sphere; got {} -> new resolution is {:.2}",
            n_points,
            points.len(),
            angle_resolution
        );

        // Ensure that icosphere points are not on the z-axis
        let v = nalgebra::UnitVector3::new_normalize(Vector3::new(0.01, 0.01, 1.0).normalize());
        let q_bias = UnitQuaternion::from_axis_angle(&v, 0.01);
        for p in &mut points {
            *p = q_bias * (*p);
        }

        let q1 = points
            .iter()
            .map(|axis| UnitQuaternion::rotation_between(axis, &-Vector3::z_axis()).unwrap())
            .collect();

        let q2 = points
            .iter()
            .map(|axis| UnitQuaternion::rotation_between(axis, &Vector3::z_axis()).unwrap())
            .collect();

        let dihedrals = iter_num_tools::arange(0.0..2.0 * PI, angle_resolution)
            .map(|angle| UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle))
            .collect();

        Self { q1, q2, dihedrals }
    }

    /// Generates a set of quaternions for a rigid body scan.
    /// Each returned unit quaternion pair can be used to rotate two rigid bodies
    /// to exhaustively scan all possible relative orientations.
    pub fn iter(&self) -> impl Iterator<Item = (UnitQuaternion, UnitQuaternion)> + '_ {
        let dihedral_x_q2 = self
            .dihedrals
            .iter()
            .cartesian_product(self.q2.iter())
            .map(|(&d, &q2)| d * q2);
        self.q1.iter().cloned().cartesian_product(dihedral_x_q2)
    }
    /// Total length of the iterator
    pub fn len(&self) -> usize {
        self.q1.len() * self.q2.len() * self.dihedrals.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scan over all angles and write to a file
    ///
    /// This does the following:
    /// - Rotates the COM vector, r by q1
    /// - Rotates the second body by q2
    /// - Translates the second body by r
    /// - Calculates the energy between the two structures
    /// - Writes the distance and energy to a buffered file
    /// - Sum energies and partition function and return as a `Sample`
    ///
    /// # Arguments:
    /// - `ref_a: &Structure` - reference structure A
    /// - `ref_b: &Structure` - reference structure B
    /// - `pair_matrix: &PairMatrix` - pair matrix of twobody energies
    /// - `r: &Vector3` - distance vector between the two structures
    /// - `temperature: f64` - temperature in K
    pub fn sample_all_angles(
        &self,
        ref_a: &Structure,
        ref_b: &Structure,
        pair_matrix: &PairMatrix,
        r: &Vector3,
        temperature: f64,
    ) -> Sample {
        let outfile = format!("R_{:.1}.dat.gz", r.norm());
        let mut encoder = GzEncoder::new(
            std::fs::File::create(outfile).unwrap(),
            Compression::default(),
        );
        let sample = self // Scan over angles
            .iter()
            .map(|(q1, q2)| {
                let (a, b) = Self::transform_structures(ref_a, ref_b, &q1, &q2, r);
                let energy = pair_matrix.sum_energy(&a, &b);
                let com = b.mass_center();
                writeln!(
                    encoder,
                    "{:.3} {:.3} {:.3} {:.3} {:?}",
                    energy,
                    com.x,
                    com.y,
                    com.z,
                    q2.axis_angle().unwrap()
                )
                .unwrap();
                Sample::new(energy, temperature)
            })
            .sum::<Sample>();
        sample
    }

    /// Transform the two reference structures by the given quaternions and distance vector
    ///
    /// This only transforms the second reference structure by translating and rotating it,
    /// while the first reference structure is left unchanged.
    fn transform_structures(
        ref_a: &Structure,
        ref_b: &Structure,
        q1: &UnitQuaternion,
        q2: &UnitQuaternion,
        r: &Vector3, // mass center separation = (0,0,r)
    ) -> (Structure, Structure) {
        let a = ref_a.clone();
        let mut b = ref_b.clone();
        b.pos = ref_b.pos.iter().map(|pos| q2 * pos + q1 * r).collect();
        (a, b)
    }
}

/// Generates n points uniformly distributed on a unit sphere
///
/// Related information:
/// - https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
/// - https://en.wikipedia.org/wiki/Geodesic_polyhedron
/// - c++: https://github.com/caosdoar/spheres
pub fn fibonacci_sphere(n_points: usize) -> Vec<Vector3> {
    assert!(n_points > 1, "n_points must be greater than 1");
    let phi = PI * (3.0 - (5.0f64).sqrt()); // golden angle in radians
    let make_ith_point = |i: usize| -> Vector3 {
        let mut p = Vector3::zeros();
        p.y = 1.0 - 2.0 * (i as f64 / (n_points - 1) as f64); // y goes from 1 to -1
        let radius = (1.0 - p.y * p.y).sqrt(); // radius at y
        let theta = phi * i as f64; // golden angle increment
        p.x = theta.cos() * radius;
        p.z = theta.sin() * radius;
        p.normalize()
    };
    (0..n_points).map(make_ith_point).collect()
}

/// Generate an icosphere with at least `min_points` surface points
///
/// The number of _points_ on the icosphere is:
///
///    N = 10 * (n_divisions + 1)^2 + 2
///
/// with the first few values 12, 42, 92, 162, ...
/// for 0, 1, 2, 3, ... divisions.
///
/// ## Further reading
///
/// - https://danielsieger.com/blog/2021/01/03/generating-platonic-solids.html
/// - https://danielsieger.com/blog/2021/03/27/generating-spheres.html
/// - https://en.wikipedia.org/wiki/Loop_subdivision_surface
///
fn generate_icosphere(min_points: usize) -> Result<Vec<Vector3>> {
    let num_points = (0..200).map(|n_div| (10 * (n_div + 1) * (n_div + 1) + 2));
    match num_points.enumerate().find(|(_, n)| *n >= min_points) {
        Some((n_divisions, _)) => Ok(IcoSphere::new(n_divisions, |_| ())
            .raw_points()
            .iter()
            .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64).normalize())
            .collect()),
        None => {
            anyhow::bail!("too many points");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosphere() {
        let points = generate_icosphere(1).unwrap();
        assert_eq!(points.len(), 12);
        let points = generate_icosphere(10).unwrap();
        assert_eq!(points.len(), 12);
        let points = generate_icosphere(13).unwrap();
        assert_eq!(points.len(), 42);
        let points = generate_icosphere(42).unwrap();
        assert_eq!(points.len(), 42);
        let points = generate_icosphere(43).unwrap();
        assert_eq!(points.len(), 92);
        let _ = generate_icosphere(400003).is_err();

        let samples = 1000;
        let points = generate_icosphere(samples).unwrap();
        let mut center: Vector3 = Vector3::zeros();
        assert_eq!(points.len(), 1002);
        for point in points {
            assert_relative_eq!((point.norm() - 1.0).abs(), 0.0, epsilon = 1e-6);
            center += point;
        }
        assert_relative_eq!(center.norm(), 0.0, epsilon = 1e-1);
    }

    #[test]
    fn test_twobody_angles() {
        use std::f64::consts::FRAC_1_SQRT_2;
        let twobody_angles = TwobodyAngles::new(1.1);
        let n = twobody_angles.q1.len() * twobody_angles.q2.len() * twobody_angles.dihedrals.len();
        assert_eq!(n, 600);
        assert_eq!(twobody_angles.len(), n);
        assert_eq!(twobody_angles.iter().count(), n);

        let pairs = twobody_angles.iter().collect::<Vec<_>>();
        assert_relative_eq!(pairs[0].0.coords.x, FRAC_1_SQRT_2);
        assert_relative_eq!(pairs[0].0.coords.y, 0.0);
        assert_relative_eq!(pairs[0].0.coords.z, 0.0);
        assert_relative_eq!(pairs[0].0.coords.w, FRAC_1_SQRT_2);
        assert_relative_eq!(pairs[0].1.coords.x, -FRAC_1_SQRT_2);
        assert_relative_eq!(pairs[0].1.coords.y, 0.0);
        assert_relative_eq!(pairs[0].1.coords.z, 0.0);
        assert_relative_eq!(pairs[0].1.coords.w, FRAC_1_SQRT_2);
        assert_relative_eq!(pairs[n - 1].0.coords.x, -FRAC_1_SQRT_2);
        assert_relative_eq!(pairs[n - 1].0.coords.y, 0.0);
        assert_relative_eq!(pairs[n - 1].0.coords.z, 0.0);
        assert_relative_eq!(pairs[n - 1].0.coords.w, FRAC_1_SQRT_2);
        assert_relative_eq!(pairs[n - 1].1.coords.x, -0.6535804797978707);
        assert_relative_eq!(pairs[n - 1].1.coords.y, 0.2698750755945888);
        assert_relative_eq!(pairs[n - 1].1.coords.z, 0.2698750755945888);
        assert_relative_eq!(pairs[n - 1].1.coords.w, -0.6535804797978708);
        println!("{}", twobody_angles);
    }

    #[test]
    fn test_fibonacci_sphere() {
        let samples = 1000;
        let points_on_sphere = fibonacci_sphere(samples);
        let mut center: Vector3 = Vector3::zeros();
        assert_eq!(points_on_sphere.len(), samples);
        for point in points_on_sphere {
            assert_relative_eq!((point.norm() - 1.0).abs(), 0.0, epsilon = 1e-10);
            center += point;
        }
        assert_relative_eq!(center.norm(), 0.0, epsilon = 1e-1);
    }
}
