#[cfg(test)]
extern crate approx;
#[cfg(test)]
use approx::assert_relative_eq;
use itertools::Itertools;
use itertools_num::linspace;
use std::fmt::Display;

pub type Vector3 = nalgebra::Vector3<f64>;
pub type UnitQuaternion = nalgebra::UnitQuaternion<f64>;

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
        use std::f64::consts::PI;
        assert!(
            angle_resolution > 0.0,
            "angle_resolution must be greater than 0"
        );

        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        let points = fibonacci_sphere(n_points);
        let q1 = points
            .iter()
            .map(|axis| UnitQuaternion::rotation_between(axis, &Vector3::z_axis()).unwrap())
            .collect::<Vec<_>>();

        let q2 = points
            .iter()
            .map(|axis| UnitQuaternion::rotation_between(axis, &-Vector3::z_axis()).unwrap())
            .collect::<Vec<_>>();

        let n_dihedrals = (2.0 * PI / angle_resolution).round() as usize;
        let dihedrals = linspace::<f64>(0.0, 2.0 * PI, n_dihedrals)
            .map(|angle| UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle))
            .collect::<Vec<_>>();

        Self { q1, q2, dihedrals }
    }

    /// Generates a set of quaternions for a rigid body scan.
    /// Each returned unit quaternion pair can be used to rotate a rigid body
    /// to exhaustively scan all possible relative orientations.
    pub fn iter(&self) -> impl Iterator<Item = (UnitQuaternion, UnitQuaternion)> + '_ {
        let dihedral_x_q2 = self
            .dihedrals
            .iter()
            .cartesian_product(self.q2.iter())
            .map(|(&i, &j)| i * j);
        self.q1.iter().cloned().cartesian_product(dihedral_x_q2)
    }
}

// pub fn info(&self, points_on_sphere: Vec<Vector3>) {
//     let mut f = std::fs::File::create("fibonacci_points.xyz").unwrap();
//     f.write_all(format!("# Fibinacci points\n{}\n", points_on_sphere.len()).as_bytes())
//         .unwrap();
//     for point in points_on_sphere {
//         f.write_all(format!("C {}\n", point.transpose()).as_bytes())
//             .unwrap();
//     }
// }

/// Generates n points uniformly distributed on a unit sphere
///
/// Related information:
/// - https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
/// - https://en.wikipedia.org/wiki/Geodesic_polyhedron
/// - c++: https://github.com/caosdoar/spheres
pub fn fibonacci_sphere(n_samples: usize) -> Vec<Vector3> {
    assert!(n_samples > 1, "samples must be greater than 1");
    let phi = std::f64::consts::PI * (3.0 - (5.0f64).sqrt()); // golden angle in radians
    let mut unit_points_on_sphere = Vec::with_capacity(n_samples);
    for cnt in 0..n_samples {
        let mut point = Vector3::zeros();
        point.y = 1.0 - 2.0 * (cnt as f64 / (n_samples - 1) as f64); // y goes from 1 to -1
        let radius = (1.0 - point.y * point.y).sqrt(); // radius at y
        let theta = phi * cnt as f64; // golden angle increment
        point.x = theta.cos() * radius;
        point.z = theta.sin() * radius;
        unit_points_on_sphere.push(point.normalize());
    }
    unit_points_on_sphere
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twobody_angles() {
        let twobody_angles = TwobodyAngles::new(1.1);
        let n = twobody_angles.q1.len() * twobody_angles.q2.len() * twobody_angles.dihedrals.len();
        assert_eq!(n, 600);
        assert_eq!(twobody_angles.iter().count(), n);

        let pairs = twobody_angles.iter().collect::<Vec<_>>();
        assert_relative_eq!(pairs[0].0.coords.x, 0.7071067811865475);
        assert_relative_eq!(pairs[0].0.coords.y, 0.0);
        assert_relative_eq!(pairs[0].0.coords.z, 0.0);
        assert_relative_eq!(pairs[0].0.coords.w, 0.7071067811865476);
        assert_relative_eq!(pairs[0].1.coords.x, -0.7071067811865475);
        assert_relative_eq!(pairs[0].1.coords.y, 0.0);
        assert_relative_eq!(pairs[0].1.coords.z, 0.0);
        assert_relative_eq!(pairs[0].1.coords.w, 0.7071067811865476);
        assert_relative_eq!(pairs[n - 1].0.coords.x, -0.7071067811865475);
        assert_relative_eq!(pairs[n - 1].0.coords.y, 0.0);
        assert_relative_eq!(pairs[n - 1].0.coords.z, 0.0);
        assert_relative_eq!(pairs[n - 1].0.coords.w, 0.7071067811865476);
        assert_relative_eq!(pairs[n - 1].1.coords.x, -0.7071067811865475);
        assert_relative_eq!(pairs[n - 1].1.coords.y, 0.0);
        assert_relative_eq!(pairs[n - 1].1.coords.z, 0.0);
        assert_relative_eq!(pairs[n - 1].1.coords.w, -0.7071067811865476);
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
