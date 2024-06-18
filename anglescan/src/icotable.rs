use super::{anglescan::*, table::PaddedTable};
use anyhow::Result;
use hexasphere::{shapes::IcoSphereBase, AdjacencyBuilder, Subdivided};
use itertools::Itertools;
use nalgebra::Matrix3;
use std::{io::Write, path::Path};
use get_size::GetSize;

/// A icotable where each vertex holds an icotable of floats
pub type IcoTableOfSpheres = IcoTable<IcoTable<f64>>;

/// A 6D table for relative twobody orientations, R → 𝜔 → (𝜃𝜑) → (𝜃𝜑)
///
/// The first two dimensions are radial distances and dihedral angles.
/// The last two dimensions are polar and azimuthal angles represented via icospheres.
/// The final `f64` data is stored at vertices of the deepest icospheres.
pub type Table6D = PaddedTable<PaddedTable<IcoTableOfSpheres>>;

impl Table6D {
    pub fn from_resolution(r_min: f64, r_max: f64, dr: f64, angle_resolution: f64) -> Result<Self> {
        use core::f64::consts::PI;
        let n_points = (4.0 * PI / angle_resolution.powi(2)).round() as usize;
        let b = IcoTable::<f64>::from_min_points(n_points, 0.0)?; // B: 𝜃 and 𝜑
        let a = IcoTableOfSpheres::from_min_points(n_points, b)?; // A: 𝜃 and 𝜑
        let o = PaddedTable::<IcoTableOfSpheres>::new(0.0, 2.0 * PI, angle_resolution, a); // 𝜔
        Ok(Self::new(r_min, r_max, dr, o)) // R
    }
}

/// Represents indices of a face
pub type Face = [usize; 3];

/// Struct representing a vertex in the icosphere
#[derive(Clone, Debug, GetSize)]
pub struct Vertex<T: Clone> {
    /// 3D coordinates of the vertex on a unit sphere
    #[get_size(size = 24)]
    pub pos: Vector3,
    /// Data associated with the vertex
    pub data: T,
    /// Indices of neighboring vertices
    pub neighbors: Vec<usize>,
}

impl<T: Clone> Vertex<T> {
    /// Construct a new vertex
    pub fn new(pos: Vector3, data: T, neighbors: Vec<usize>) -> Self {
        assert!(matches!(neighbors.len(), 5 | 6));
        Self {
            pos,
            data,
            neighbors,
        }
    }
}

/// Icosphere table
///
/// This is used to store data on the vertices of an icosphere.
/// It includes barycentric interpolation and nearest face search.
///
/// https://en.wikipedia.org/wiki/Geodesic_polyhedron
/// 12 vertices will always have 5 neighbors; the rest will have 6.
#[derive(Clone, GetSize)]
pub struct IcoTable<T: Clone> {
    /// Vertex information (position, data, neighbors)
    pub vertices: Vec<Vertex<T>>,
    /// All faces of the icosphere each consisting of three (sorted) vertex indices
    pub faces: Vec<Face>,
}

impl<T: Clone> IcoTable<T> {
    /// Generate table based on an existing subdivided icosaedron
    pub fn from_icosphere(icosphere: Subdivided<(), IcoSphereBase>, default_data: T) -> Self {
        let indices = icosphere.get_all_indices();
        let mut builder = AdjacencyBuilder::new(icosphere.raw_points().len());
        builder.add_indices(indices.as_slice());
        let neighbors = builder.finish().iter().map(|i| i.to_vec()).collect_vec();
        let vertex_positions: Vec<Vector3> = icosphere
            .raw_points()
            .iter()
            .map(|p| Vector3::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();

        let vertices = (0..vertex_positions.len())
            .map(|i| {
                Vertex::new(
                    vertex_positions[i],
                    default_data.clone(),
                    neighbors[i].clone(),
                )
            })
            .collect();

        let faces: Vec<Face> = indices
            .chunks(3)
            .map(|c| {
                let mut v = vec![c[0] as usize, c[1] as usize, c[2] as usize];
                v.sort();
                v.try_into().unwrap()
            })
            .collect_vec();

        Self { vertices, faces }
    }

    /// Generate table based on a minimum number of vertices on the subdivided icosaedron
    pub fn from_min_points(min_points: usize, default_data: T) -> Result<Self> {
        let icosphere = make_icosphere(min_points)?;
        Ok(Self::from_icosphere(icosphere, default_data))
    }

    /// Number of vertices in the table
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Check if the table is empty, i.e. has no vertices
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Set data associated with each vertex using a generator function
    pub fn set_vertex_data(&mut self, f: impl Fn(&Vector3) -> T) {
        self.vertices.iter_mut().for_each(|v| v.data = f(&v.pos));
    }

    /// Get data associated with each vertex
    pub fn vertex_data(&self) -> impl Iterator<Item = &T> {
        self.vertices.iter().map(|v| &v.data)
    }

    /// Transform vertex positions using a function
    pub fn transform_vertex_positions(&mut self, f: impl Fn(&Vector3) -> Vector3) {
        self.vertices.iter_mut().for_each(|v| v.pos = f(&v.pos));
    }

    /// Get projected barycentric coordinate for an arbitrary point
    ///
    /// See "Real-Time Collision Detection" by Christer Ericson (p141-142)
    pub fn barycentric(&self, p: &Vector3, face: &Face) -> Vector3 {
        let (a, b, c) = self.face_positions(face);
        // Check if P in vertex region outside A
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d1 = ab.dot(&ap);
        let d2 = ac.dot(&ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return Vector3::new(1.0, 0.0, 0.0);
        }
        // Check if P in vertex region outside B
        let bp = p - b;
        let d3 = ab.dot(&bp);
        let d4 = ac.dot(&bp);
        if d3 >= 0.0 && d4 <= d3 {
            return Vector3::new(0.0, 1.0, 0.0);
        }
        // Check if P in edge region of AB, if so return projection of P onto AB
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return Vector3::new(1.0 - v, v, 0.0);
        }
        // Check if P in vertex region outside C
        let cp = p - c;
        let d5 = ab.dot(&cp);
        let d6 = ac.dot(&cp);
        if d6 >= 0.0 && d5 <= d6 {
            return Vector3::new(0.0, 0.0, 1.0);
        }
        // Check if P in edge region of AC, if so return projection of P onto AC
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return Vector3::new(1.0 - w, 0.0, w);
        }
        // Check if P in edge region of BC, if so return projection of P onto BC
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return Vector3::new(0.0, 1.0 - w, w);
        }
        // P inside face region.
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        Vector3::new(1.0 - v - w, v, w)
    }

    /// Get barycentric coordinate for an arbitrary point on a face
    ///
    /// - Assume that `point` is on the plane defined by the face, i.e. no projection is done
    /// - https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    /// - http://realtimecollisiondetection.net/
    /// - https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    pub fn naive_barycentric(&self, p: &Vector3, face: &Face) -> Vector3 {
        let (a, b, c) = self.face_positions(face);
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d00 = ab.dot(&ab);
        let d01 = ab.dot(&ac);
        let d11 = ac.dot(&ac);
        let d20 = ap.dot(&ab);
        let d21 = ap.dot(&ac);
        let denom = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;
        Vector3::new(u, v, w)
    }

    /// Get the three vertices of a face
    pub fn face_positions(&self, face: &Face) -> (&Vector3, &Vector3, &Vector3) {
        let a = &self.vertices[face[0]].pos;
        let b = &self.vertices[face[1]].pos;
        let c = &self.vertices[face[2]].pos;
        (a, b, c)
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
            .map(|v| (v.pos - point).norm_squared())
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
    pub fn nearest_face(&self, point: &Vector3) -> Face {
        let point = point.normalize();
        let nearest = self.nearest_vertex(&point);
        let face: Face = self.vertices[nearest]
            .neighbors // neighbors to nearest
            .iter()
            .cloned()
            .map(|i| (i, (self.vertices[i].pos - point).norm_squared()))
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) // sort ascending
            .map(|(i, _)| i) // keep only indices
            .take(2) // take two next nearest distances
            .collect_tuple()
            .map(|(a, b)| [a, b, nearest]) // append nearest
            .expect("Face requires exactly three indices")
            .iter()
            .copied()
            .sorted_unstable() // we want sorted indices
            .collect_vec() // collect into array
            .try_into()
            .unwrap();

        assert_eq!(face.iter().unique().count(), 3);
        face
    }

    /// Save a VMD script to illustrate the icosphere
    pub fn save_vmd(&self, path: impl AsRef<Path>, scale: Option<f64>) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = scale.unwrap_or(1.0);
        writeln!(file, "draw delete all")?;
        for face in &self.faces {
            let a = &self.vertices[face[0]].pos.scale(s);
            let b = &self.vertices[face[1]].pos.scale(s);
            let c = &self.vertices[face[2]].pos.scale(s);
            let color = "red";
            vmd_draw_triangle(&mut file, a, b, c, color)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for IcoTable<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "# x y z θ φ data")?;
        for vertex in self.vertices.iter() {
            let (_r, theta, phi) = crate::to_spherical(&vertex.pos);
            writeln!(
                f,
                "{} {} {} {} {} {}",
                vertex.pos.x, vertex.pos.y, vertex.pos.z, theta, phi, vertex.data
            )?;
        }
        Ok(())
    }
}

impl IcoTable<f64> {
    /// Get data for a point on the surface using barycentric interpolation
    /// https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Interpolation_on_a_triangular_unstructured_grid
    pub fn interpolate(&self, point: &Vector3) -> f64 {
        let face = self.nearest_face(point);
        let bary = self.barycentric(point, &face);
        bary[0] * self.vertices[face[0]].data
            + bary[1] * self.vertices[face[1]].data
            + bary[2] * self.vertices[face[2]].data
    }
}

impl IcoTableOfSpheres {
    /// Interpolate data between two faces
    pub fn interpolate(
        &self,
        face_a: &Face,
        face_b: &Face,
        bary_a: &Vector3,
        bary_b: &Vector3,
    ) -> f64 {
        let data_ab =
            Matrix3::from_fn(|i, j| self.vertices[face_a[i]].data.vertices[face_b[j]].data);
        (bary_a.transpose() * data_ab * bary_b).to_scalar()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anglescan::make_icosphere;
    use approx::assert_relative_eq;

    #[test]
    fn test_icosphere_table() {
        let icosphere = make_icosphere(3).unwrap();
        let icotable = IcoTable::<f64>::from_icosphere(icosphere, 0.0);
        assert_eq!(icotable.vertices.len(), 12);
        assert_eq!(icotable.faces.len(), 20);

        let point = icotable.vertices[0].pos;

        assert_relative_eq!(point.x, 0.0);
        assert_relative_eq!(point.y, 1.0);
        assert_relative_eq!(point.z, 0.0);

        // find nearest vertex and face to vertex 0
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 2, 5]);
        assert_relative_eq!(bary[0], 1.0);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 0.0);

        // Nearest face to slightly displaced vertex 0
        let point = (icotable.vertices[0].pos + Vector3::new(1e-3, 0.0, 0.0)).normalize();
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 1, 5]);
        assert_relative_eq!(bary[0], 0.9991907334103153);
        assert_relative_eq!(bary[1], 0.000809266589684687);
        assert_relative_eq!(bary[2], 0.0);

        // find nearest vertex and face to vertex 2
        let point = icotable.vertices[2].pos;
        let face = icotable.nearest_face(&point);
        let bary = icotable.barycentric(&point, &face);
        assert_eq!(face, [0, 1, 2]);
        assert_relative_eq!(bary[0], 0.0);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 1.0);

        // Midpoint on edge between vertices 0 and 2
        let point = point + (icotable.vertices[0].pos - point) * 0.5;
        let bary = icotable.barycentric(&point, &face);
        assert_relative_eq!(bary[0], 0.5);
        assert_relative_eq!(bary[1], 0.0);
        assert_relative_eq!(bary[2], 0.5);
    }

    #[test]
    fn test_table_of_spheres() {
        let icotable = IcoTable::<f64>::from_min_points(42, 0.0).unwrap();
        let icotable_of_spheres = IcoTableOfSpheres::from_min_points(42, icotable).unwrap();
        assert_eq!(icotable_of_spheres.vertices.len(), 42);
        assert_eq!(icotable_of_spheres.vertices[0].data.vertices.len(), 42);
    }
}
