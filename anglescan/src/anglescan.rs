#[cfg(test)]
extern crate approx;
use crate::{energy::PairMatrix, structure::Structure, Sample};
use anyhow::{Context, Result};
#[cfg(test)]
use approx::assert_relative_eq;
use hexasphere::{
    shapes::{IcoSphere, IcoSphereBase},
    AdjacencyBuilder, Subdivided,
};
use iter_num_tools::arange;
use itertools::Itertools;
use std::{f64::consts::PI, fmt::Display, io::Write, path::Path};

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
    pub fn new(angle_resolution: f64) -> Result<Self> {
        assert!(
            angle_resolution > 0.0,
            "angle_resolution must be greater than 0"
        );

        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        let mut points = make_icosphere_vertices(n_points)?;
        let angle_resolution = (4.0 * PI / points.len() as f64).sqrt();
        log::info!(
            "Requested {} points on a sphere; got {} -> new resolution = {:.2}",
            n_points,
            points.len(),
            angle_resolution
        );

        // Ensure that icosphere points are not *exactly* on the z-axis to
        // enable trouble-free alignment below; see `rotation_between()` docs
        let v = nalgebra::UnitVector3::new_normalize(Vector3::new(0.0005, 0.0005, 1.0));
        let q_bias = UnitQuaternion::from_axis_angle(&v, 0.0001);
        points.iter_mut().for_each(|p| *p = q_bias * (*p));

        // Rotation operations via unit quaternions
        let to_zaxis = |p| UnitQuaternion::rotation_between(p, &Vector3::z_axis()).unwrap();
        let to_neg_zaxis = |p| UnitQuaternion::rotation_between(p, &-Vector3::z_axis()).unwrap();
        let around_z = |angle| UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle);

        let q1 = points.iter().map(to_neg_zaxis).collect();
        let q2 = points.iter().map(to_zaxis).collect();

        let dihedrals = arange(0.0..2.0 * PI, angle_resolution)
            .map(around_z)
            .collect();

        Ok(Self { q1, q2, dihedrals })
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

    /// Opens a gz compressed file for writing
    fn open_compressed_file(outfile: impl AsRef<Path>) -> Result<GzEncoder<std::fs::File>> {
        Ok(GzEncoder::new(
            std::fs::File::create(outfile)?,
            Compression::default(),
        ))
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
    ) -> Result<Sample> {
        let mut file = Self::open_compressed_file(format!("R_{:.1}.dat.gz", r.norm()))?;
        let sample = self // Scan over angles
            .iter()
            .map(|(q1, q2)| {
                let (a, b) = Self::transform_structures(ref_a, ref_b, &q1, &q2, r);
                let energy = pair_matrix.sum_energy(&a, &b);
                let com = b.mass_center();
                writeln!(
                    file,
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
        Ok(sample)
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
        b.pos = ref_b
            .pos
            .iter()
            .map(|pos| q2.transform_vector(pos) + q1.transform_vector(r))
            .collect();
        (a, b)
    }
}

/// Generates n points uniformly distributed on a unit sphere
///
/// Related information:
/// - <https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere>
/// - <https://en.wikipedia.org/wiki/Geodesic_polyhedron>
/// - c++: <https://github.com/caosdoar/spheres>
pub fn make_fibonacci_sphere(n_points: usize) -> Vec<Vector3> {
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

/// Make icosphere with at least `min_points` surface points (vertices).
///
/// This is done by iteratively subdividing the faces of an icosahedron
/// until at least `min_points` vertices are achieved.
/// The number of vertices on the icosphere is _N_ = 10 × (_n_divisions_ + 1)² + 2
/// whereby 0, 1, 2, ... subdivisions give 12, 42, 92, ... vertices, respectively.
///
///
/// ## Further reading
///
/// - <https://en.wikipedia.org/wiki/Loop_subdivision_surface>
/// - <https://danielsieger.com/blog/2021/03/27/generating-spheres.html>
/// - <https://danielsieger.com/blog/2021/01/03/generating-platonic-solids.html>
///
/// ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Loop_Subdivision_Icosahedron.svg/300px-Loop_Subdivision_Icosahedron.svg.png)
///
pub fn make_icosphere(min_points: usize) -> Result<Subdivided<(), IcoSphereBase>> {
    let points_per_division = |n_div: usize| 10 * (n_div + 1) * (n_div + 1) + 2;
    let n_points = (0..200).map(points_per_division);

    // Number of divisions to achieve at least `min_points` vertices
    let n_divisions = n_points
        .enumerate()
        .find(|(_, n)| *n >= min_points)
        .map(|(n_div, _)| n_div)
        .context("too many vertices")?;

    Ok(IcoSphere::new(n_divisions, |_| ()))
}

/// Make icosphere vertices as 3D vectors
///
/// ## Examples
/// ~~~
/// let vertices = anglescan::make_icosphere_vertices(20).unwrap();
/// assert_eq!(vertices.len(), 42);
/// ~~~
pub fn make_icosphere_vertices(min_points: usize) -> Result<Vec<Vector3>> {
    let points = make_icosphere(min_points)?
        .raw_points()
        .iter()
        .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
        .collect();
    Ok(points)
}

// https://en.wikipedia.org/wiki/Geodesic_polyhedron
// 12 vertices will always have 5 neighbors; the rest will have 6.
pub struct IcoSphereTable {
    /// Raw icosphere structure from hexasphere crate
    _icosphere: Subdivided<(), IcoSphereBase>,
    /// Neighbor list of other vertices for each vertex
    neighbors: Vec<Vec<usize>>,
    /// 3D coordinates of the vertices
    vertices: Vec<Vector3>,
    /// All faces of the icosphere each consisting of three (sorted) vertex indices
    faces: Vec<Vec<usize>>,
    /// Data associated with each vertex
    vertex_data: Vec<f64>,
}

impl IcoSphereTable {
    pub fn from_icosphere(icosphere: Subdivided<(), IcoSphereBase>) -> Self {
        let indices = icosphere.get_all_indices();
        let mut builder = AdjacencyBuilder::new(icosphere.raw_points().len());
        builder.add_indices(indices.as_slice());
        let neighbors = builder.finish().iter().map(|i| i.to_vec()).collect();
        let vertices: Vec<Vector3> = icosphere
            .raw_points()
            .iter()
            .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();

        let faces = indices
            .chunks(3)
            .map(|c| {
                let mut v = vec![c[0] as usize, c[1] as usize, c[2] as usize];
                v.sort();
                v
            })
            .collect();

        let n_vertices = vertices.len();

        Self {
            _icosphere: icosphere,
            neighbors,
            vertices,
            faces,
            vertex_data: Vec::with_capacity(n_vertices),
        }
    }

    pub fn from_min_points(min_points: usize) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        Ok(Self::from_icosphere(icosphere))
    }

    pub fn vertex_data(&self) -> &Vec<f64> {
        &self.vertex_data
    }

    pub fn vertex_data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.vertex_data
    }

    /// Get interpolated data for an arbitrart point on the icosphere
    /// 
    /// Done by finding the nearest face and then interpolate using the three corner vertices
    pub fn get(&self, point: &Vector3) -> f64 {
        let face = self.nearest_face(point);
        let data = face.iter().map(|i| &self.vertex_data[*i]);
        let weights = face.iter().map(|i| 1.0 / (self.vertices[*i] - point).norm()).collect::<Vec<f64>>();
        let sum_weights: f64 = weights.iter().sum();
        data
            .zip(weights)
            .map(|(d, w)| *d * (w / sum_weights))
            .sum()
    }

    /// Get data for a point on the surface using barycentric interpolation
    /// https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Interpolation_on_a_triangular_unstructured_grid
    pub fn barycentric_interpolation(&self, point: &Vector3) -> f64 {
        let bary = self.barycentric(point);
        let face = self.nearest_face(point);
        bary[0] * self.vertex_data[face[0]] + bary[1] * self.vertex_data[face[1]] + bary[2] * self.vertex_data[face[2]]
    }

    /// Get Barycentric coordinate for an arbitrart point on the icosphere
    /// https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    pub fn barycentric(&self, point: &Vector3) -> Vec<f64> {
        let face = self.nearest_face(point);
        let a = self.vertices[face[0]];
        let b = self.vertices[face[1]];
        let c = self.vertices[face[2]];
        let v0 = b - a;
        let v1 = c - a;
        let v2 = point - a;
        let d00 = v0.dot(&v0);
        let d01 = v0.dot(&v1);
        let d11 = v1.dot(&v1);
        let d20 = v2.dot(&v0);
        let d21 = v2.dot(&v1);
        let denom = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        vec![1.0 - v - w, v, w]
    }

    pub fn faces(&self) -> &Vec<Vec<usize>> {
        &self.faces
    }

    /// Find nearest vertex to a given point
    ///
    /// This is brute force and has O(n) complexity. This
    /// should be updated with a more efficient algorithm that
    /// uses angular information to narrow down the search.
    pub fn nearest_vertex(&self, point: &Vector3) -> usize {
        let mut min_distance = f64::INFINITY;
        let mut nearest = 0;
        let point_hat = point.normalize();
        for (i, vertex) in self.vertices.iter().enumerate() {
            let distance = (vertex - point_hat).norm_squared();
            if distance < min_distance {
                min_distance = distance;
                nearest = i;
            }
        }
        nearest
    }

    /// Find nearest face to a given point
    ///
    /// The first nearest point is O(n) whereafter neighbor information
    /// is used to find the 2nd and 3rd nearest points which are guaranteed
    /// to define a face.
    pub fn nearest_face(&self, point: &Vector3) -> Vec<usize> {
        let nearest_vertex = self.nearest_vertex(point);
        let point_hat = point.normalize();

        let mut dist: Vec<(usize, f64)> = self.neighbors[nearest_vertex]
            .iter()
            .cloned()
            .map(|i| (i, (self.vertices[i] - point_hat).norm_squared()))
            .collect();
        dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut face: Vec<usize> = dist.iter().map(|(i, _)| i).cloned().take(2).collect();
        face.push(nearest_vertex);
        face.sort_unstable();
        assert_eq!(face.iter().unique().count(), 3);
        face
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosphere() {
        let points = make_icosphere_vertices(1).unwrap();
        assert_eq!(points.len(), 12);
        let points = make_icosphere_vertices(10).unwrap();
        assert_eq!(points.len(), 12);
        let points = make_icosphere_vertices(13).unwrap();
        assert_eq!(points.len(), 42);
        let points = make_icosphere_vertices(42).unwrap();
        assert_eq!(points.len(), 42);
        let points = make_icosphere_vertices(43).unwrap();
        assert_eq!(points.len(), 92);
        let _ = make_icosphere_vertices(400003).is_err();

        let samples = 1000;
        let points = make_icosphere_vertices(samples).unwrap();
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
        let twobody_angles = TwobodyAngles::new(1.1).unwrap();
        let n = twobody_angles.q1.len() * twobody_angles.q2.len() * twobody_angles.dihedrals.len();
        assert_eq!(n, 1008);
        assert_eq!(twobody_angles.len(), n);
        assert_eq!(twobody_angles.iter().count(), n);

        let pairs = twobody_angles.iter().collect::<Vec<_>>();
        assert_relative_eq!(pairs[0].0.coords.x, -FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[0].0.coords.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[0].0.coords.z, 0.0);
        assert_relative_eq!(pairs[0].0.coords.w, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[0].1.coords.x, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[0].1.coords.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[0].1.coords.z, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[0].1.coords.w, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].0.coords.x, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].0.coords.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[n - 1].0.coords.z, 0.0, epsilon = 1e-4);
        assert_relative_eq!(pairs[n - 1].0.coords.w, FRAC_1_SQRT_2, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.x, 0.705299, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.y, -0.050523, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.z, 0.050594, epsilon = 1e-5);
        assert_relative_eq!(pairs[n - 1].1.coords.w, -0.705294, epsilon = 1e-5);
        println!("{}", twobody_angles);
    }

    #[test]
    fn test_fibonacci_sphere() {
        let samples = 1000;
        let points_on_sphere = make_fibonacci_sphere(samples);
        let mut center: Vector3 = Vector3::zeros();
        assert_eq!(points_on_sphere.len(), samples);
        for point in points_on_sphere {
            assert_relative_eq!((point.norm() - 1.0).abs(), 0.0, epsilon = 1e-10);
            center += point;
        }
        assert_relative_eq!(center.norm(), 0.0, epsilon = 1e-1);
    }
}
