use super::anglescan::*;
use anyhow::Result;
use hexasphere::{shapes::IcoSphereBase, AdjacencyBuilder, Subdivided};
use itertools::Itertools;
use std::fmt::Debug;
use std::io::Write;
use std::path::Path;

/// Icosphere table
///
/// This is used to store data on the vertices of an icosphere.
/// It includes barycentric interpolation and nearest face search.
///
/// https://en.wikipedia.org/wiki/Geodesic_polyhedron
/// 12 vertices will always have 5 neighbors; the rest will have 6.
pub struct IcoSphereTable<T: Default + Debug + Clone> {
    /// Raw icosphere structure from hexasphere crate
    _icosphere: Subdivided<(), IcoSphereBase>,
    /// Neighbor list of other vertices for each vertex
    neighbors: Vec<Vec<usize>>,
    /// 3D coordinates of the vertices
    pub vertices: Vec<Vector3>,
    /// All faces of the icosphere each consisting of three (sorted) vertex indices
    faces: Vec<Vec<usize>>,
    /// Data associated with each vertex
    vertex_data: Vec<T>,
}

impl<T: Default + Debug + Clone> IcoSphereTable<T> {
    /// Generate table based on an existing subdivided icosaedron
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
            vertex_data: vec![T::default(); n_vertices],
        }
    }

    /// Save tabulated vertices and aqssociated data to a file
    pub fn save_table(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "# x y z θ φ data")?;
        for (vertex, data) in self.vertices.iter().zip(&self.vertex_data) {
            let (_r, theta, phi) = crate::to_spherical(vertex);
            writeln!(
                file,
                "{} {} {} {} {} {:?}",
                vertex.x, vertex.y, vertex.z, theta, phi, data
            )?;
        }
        Ok(())
    }

    /// Generate table based on a minimum number of vertices on the subdivided icosaedron
    pub fn from_min_points(min_points: usize) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        Ok(Self::from_icosphere(icosphere))
    }

    /// Set data associated with each vertex using a generator function
    pub fn set_vertex_data(&mut self, f: impl Fn(&Vector3) -> T) {
        self.vertex_data = self.vertices.iter().map(f).collect();
    }

    /// Get data associated with each vertex
    pub fn vertex_data(&self) -> &Vec<T> {
        &self.vertex_data
    }

    /// Check is a point is on a face
    pub fn is_on_face(&self, point: &Vector3, face: &[usize]) -> bool {
        let a = point - self.vertices[face[0]];
        let b = point - self.vertices[face[1]];
        let c = point - self.vertices[face[2]];
        let n = a.cross(&b);
        let n = n.normalize();
        let d = n.dot(&c);
        d.abs() < 1e-3
    }

    /// Get Barycentric coordinate for an arbitrart point on a face
    /// https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    pub fn barycentric(&self, point: &Vector3, face: &[usize]) -> Vec<f64> {
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

    /// Get list of all faces (triangles) on the icosphere
    pub fn faces(&self) -> &Vec<Vec<usize>> {
        &self.faces
    }

    /// Find nearest vertex to a given point
    ///
    /// This is brute force and has O(n) complexity. This
    /// should be updated with a more efficient algorithm that
    /// uses angular information to narrow down the search.
    ///
    /// See:
    /// - https://stackoverflow.com/questions/11947813/subdivided-icosahedron-how-to-find-the-nearest-vertex-to-an-arbitrary-point
    /// - Binary Space Partitioning: https://en.wikipedia.org/wiki/Binary_space_partitioning
    fn nearest_vertex(&self, point: &Vector3) -> usize {
        self.vertices
            .iter()
            .map(|r| (r - point).norm_squared())
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
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
    /// Save a VMD script to illustrate the icosphere
    pub fn save_vmd(&self, path: impl AsRef<Path>, scale: Option<f64>) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = scale.unwrap_or(1.0);
        writeln!(file, "draw delete all")?;
        for face in &self.faces {
            let a = &self.vertices[face[0]].scale(s);
            let b = &self.vertices[face[1]].scale(s);
            let c = &self.vertices[face[2]].scale(s);
            let color = "red";
            vmd_draw_triangle(&mut file, a, b, c, color)?;
        }
        Ok(())
    }
}

/// Get data for a point on the surface using barycentric interpolation
/// https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Interpolation_on_a_triangular_unstructured_grid
pub fn barycentric_interpolation(icotable: &IcoSphereTable<f64>, point: &Vector3) -> f64 {
    let point_norm = point.normalize();
    let face = icotable.nearest_face(&point_norm);
    let bary = icotable.barycentric(&point_norm, &face);
    bary[0] * icotable.vertex_data[face[0]]
        + bary[1] * icotable.vertex_data[face[1]]
        + bary[2] * icotable.vertex_data[face[2]]
}

/// Draw a triangle in VMD format
fn vmd_draw_triangle(
    stream: &mut impl Write,
    a: &Vector3,
    b: &Vector3,
    c: &Vector3,
    color: &str,
) -> Result<()> {
    writeln!(stream, "draw color {}", color)?;
    writeln!(
        stream,
        "draw triangle {{{:.3} {:.3} {:.3}}} {{{:.3} {:.3} {:.3}}} {{{:.3} {:.3} {:.3}}}",
        a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z
    )?;
    Ok(())
}
