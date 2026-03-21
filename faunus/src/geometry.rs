//! Geometry: mass center, dipole moment, angles, dihedrals, gyration tensor, and molecular overlay.

use crate::{cell::SimulationCell, Point};
use nalgebra::{Matrix3, Rotation3, SymmetricEigen, Vector3};
use rand::Rng;

/// Compute the electric dipole moment of a charge distribution relative to a reference point.
///
/// Uses minimum-image distances to handle periodic boundary conditions:
/// **μ** = Σ qᵢ · (**rᵢ** − **r_ref**).
pub(crate) fn dipole_moment(
    charges_positions: impl IntoIterator<Item = (f64, Point)>,
    reference: &Point,
    cell: &impl SimulationCell,
) -> Point {
    charges_positions
        .into_iter()
        .fold(Point::zeros(), |mu, (q, pos)| {
            mu + q * cell.distance(&pos, reference)
        })
}

/// Mass-weighted gyration tensor with eigendecomposition.
///
/// Eigenvalues are sorted ascending (λ₁ ≤ λ₂ ≤ λ₃) and the eigenvector
/// columns are reordered to match. The rotation matrix built from the
/// eigenvectors maps body-frame → lab-frame.
#[derive(Debug, Clone)]
pub(crate) struct GyrationTensor {
    /// Eigenvalues sorted ascending.
    pub eigenvalues: [f64; 3],
    /// The 3×3 symmetric tensor.
    pub tensor: Matrix3<f64>,
    /// Rg² = trace = λ₁ + λ₂ + λ₃.
    pub rg_squared: f64,
    /// Rotation from principal axes to lab frame (columns = sorted eigenvectors).
    pub rotation: Rotation3<f64>,
}

impl GyrationTensor {
    /// Eigendecompose a symmetric 3×3 tensor and sort by ascending eigenvalue.
    pub fn from_tensor(tensor: Matrix3<f64>) -> Self {
        let eigen = SymmetricEigen::new(tensor);
        let mut order: [usize; 3] = [0, 1, 2];
        order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));

        let eigenvalues = order.map(|i| eigen.eigenvalues[i]);
        let rg_squared = eigenvalues.iter().sum();

        let mat = Matrix3::from_columns(&order.map(|i| eigen.eigenvectors.column(i).into_owned()));
        // Ensure right-handedness
        let rotation = if mat.determinant() < 0.0 {
            Rotation3::from_matrix_unchecked(-mat)
        } else {
            Rotation3::from_matrix_unchecked(mat)
        };

        Self {
            eigenvalues,
            tensor,
            rg_squared,
            rotation,
        }
    }

    /// Build from positions and masses relative to a precomputed center of mass,
    /// using periodic boundary conditions.
    pub fn from_positions_masses_com(
        positions: impl IntoIterator<Item = (Point, f64)>,
        com: &Point,
        cell: &impl SimulationCell,
    ) -> Option<Self> {
        let mut tensor = Matrix3::<f64>::zeros();
        let mut total_mass = 0.0;
        let mut count = 0usize;

        for (pos, mass) in positions {
            let r = cell.distance(&pos, com);
            total_mass += mass;
            tensor += r * r.transpose() * mass;
            count += 1;
        }

        if count < 2 || total_mass <= 0.0 {
            return None;
        }
        tensor /= total_mass;
        Some(Self::from_tensor(tensor))
    }

    /// Build from equal-mass positions (no PBC).
    pub fn from_equal_mass_positions(positions: &[Point]) -> Option<Self> {
        if positions.len() < 2 {
            return None;
        }
        let n = positions.len() as f64;
        let com: Point = positions.iter().sum::<Point>() / n;
        let mut tensor = Matrix3::<f64>::zeros();
        for p in positions {
            let r = p - com;
            tensor += r * r.transpose();
        }
        tensor /= n;
        Some(Self::from_tensor(tensor))
    }
}

/// Tolerance for detecting degenerate eigenvalues.
const EIGENVALUE_DEGENERACY_TOL: f64 = 1e-6;

/// Randomly flip the sign of eigenvectors corresponding to degenerate eigenvalues.
///
/// For degenerate eigenvalue pairs/triples, the principal axes are ambiguous;
/// random sign flips ensure unbiased MC sampling during molecular overlays.
fn randomize_degenerate_axes(
    rotation: &Rotation3<f64>,
    evals: &[f64; 3],
    rng: &mut (impl Rng + ?Sized),
) -> Rotation3<f64> {
    let mut mat = *rotation.matrix();

    let scale = evals.iter().sum::<f64>().max(1.0);
    let d01 = (evals[0] - evals[1]).abs() < EIGENVALUE_DEGENERACY_TOL * scale;
    let d12 = (evals[1] - evals[2]).abs() < EIGENVALUE_DEGENERACY_TOL * scale;

    // Non-degenerate axis used for determinant correction (avoids undoing a flip)
    let (flip_range, det_fix_col) = if d01 && d12 {
        (0..3, 2) // Triple degeneracy; any column works
    } else if d01 {
        (0..2, 2) // Axes 0,1 degenerate; fix via non-degenerate axis 2
    } else if d12 {
        (1..3, 0) // Axes 1,2 degenerate; fix via non-degenerate axis 0
    } else {
        return Rotation3::from_matrix_unchecked(mat);
    };

    for col in flip_range {
        if rng.gen::<bool>() {
            mat.column_mut(col).neg_mut();
        }
    }

    if mat.determinant() < 0.0 {
        mat.column_mut(det_fix_col).neg_mut();
    }
    Rotation3::from_matrix_unchecked(mat)
}

/// Overlay template positions onto a target molecule using gyration tensor alignment.
///
/// Aligns the principal axes of `template_positions` (equal-mass, no PBC) to
/// match the principal-axis frame of the target group defined by
/// `target_positions_masses` (mass-weighted, with PBC). Degenerate eigenvalues
/// are resolved with random sign flips to avoid MC bias.
///
/// Returns new positions in lab frame, centered on the target COM.
pub(crate) fn overlay_positions(
    template_positions: &[Point],
    target_positions_masses: impl IntoIterator<Item = (Point, f64)>,
    target_com: &Point,
    cell: &impl SimulationCell,
    rng: &mut (impl Rng + ?Sized),
) -> Option<Vec<Point>> {
    let target_gt =
        GyrationTensor::from_positions_masses_com(target_positions_masses, target_com, cell)?;
    let template_gt = GyrationTensor::from_equal_mass_positions(template_positions)?;

    // Randomize degenerate axes for both frames
    let target_rot = randomize_degenerate_axes(&target_gt.rotation, &target_gt.eigenvalues, rng);
    let template_rot =
        randomize_degenerate_axes(&template_gt.rotation, &template_gt.eigenvalues, rng);

    // R = R_target · R_template⁻¹ maps template body frame → target lab frame
    let align = target_rot * template_rot.inverse();

    // Template COM
    let n = template_positions.len() as f64;
    let template_com: Point = template_positions.iter().sum::<Point>() / n;

    let positions = template_positions
        .iter()
        .map(|p| {
            let mut pos = target_com + align * (p - template_com);
            cell.boundary(&mut pos);
            pos
        })
        .collect();

    Some(positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{BoundaryConditions, Shape};
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    #[test]
    fn dipole_moment_simple() {
        let cell = crate::cell::Endless;
        let origin = Point::zeros();
        // Two opposite charges along x: μ = q·d x̂
        let charges_positions = vec![
            (1.0, Point::new(1.0, 0.0, 0.0)),
            (-1.0, Point::new(-1.0, 0.0, 0.0)),
        ];
        let mu = super::dipole_moment(charges_positions, &origin, &cell);
        assert_relative_eq!(mu.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(mu.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(mu.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn dipole_moment_neutral_symmetric() {
        let cell = crate::cell::Endless;
        let origin = Point::zeros();
        // Symmetric arrangement: dipole moment should be zero
        let charges_positions = vec![
            (1.0, Point::new(1.0, 0.0, 0.0)),
            (1.0, Point::new(-1.0, 0.0, 0.0)),
            (-1.0, Point::new(0.0, 1.0, 0.0)),
            (-1.0, Point::new(0.0, -1.0, 0.0)),
        ];
        let mu = super::dipole_moment(charges_positions, &origin, &cell);
        assert_relative_eq!(mu.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn collinear_rod_eigenvalues() {
        let positions = [
            Point::new(-1.0, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
        ];
        let gt = GyrationTensor::from_equal_mass_positions(&positions).unwrap();
        assert_relative_eq!(gt.eigenvalues[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(gt.eigenvalues[1], 0.0, epsilon = 1e-10);
        assert!(gt.eigenvalues[2] > 0.0);
        assert_relative_eq!(gt.rg_squared, gt.eigenvalues.iter().sum::<f64>());
    }

    #[test]
    fn tetrahedron_isotropic() {
        let positions = [
            Point::new(1.0, 1.0, 1.0),
            Point::new(1.0, -1.0, -1.0),
            Point::new(-1.0, 1.0, -1.0),
            Point::new(-1.0, -1.0, 1.0),
        ];
        let gt = GyrationTensor::from_equal_mass_positions(&positions).unwrap();
        assert_relative_eq!(gt.eigenvalues[0], gt.eigenvalues[1], epsilon = 1e-10);
        assert_relative_eq!(gt.eigenvalues[1], gt.eigenvalues[2], epsilon = 1e-10);
    }

    #[test]
    fn rotation_is_right_handed() {
        let positions = [
            Point::new(3.0, 0.0, 0.0),
            Point::new(0.0, 1.0, 0.0),
            Point::new(0.0, 0.0, 0.5),
            Point::new(-1.0, 0.5, 0.2),
        ];
        let gt = GyrationTensor::from_equal_mass_positions(&positions).unwrap();
        assert_relative_eq!(gt.rotation.matrix().determinant(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn too_few_particles_returns_none() {
        let positions = [Point::new(1.0, 2.0, 3.0)];
        assert!(GyrationTensor::from_equal_mass_positions(&positions).is_none());
        assert!(GyrationTensor::from_equal_mass_positions(&[]).is_none());
    }

    #[test]
    fn overlay_preserves_com_and_rg() {
        // Template: T-shape in xy-plane
        let template = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(2.0, 0.0, 0.0),
            Point::new(-2.0, 0.0, 0.0),
            Point::new(0.0, 3.0, 0.0),
        ];

        // Target: same shape rotated 45° about z, shifted to (10, 5, 0)
        let angle = std::f64::consts::FRAC_PI_4;
        let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), angle);
        let shift = Point::new(10.0, 5.0, 0.0);
        let target: Vec<Point> = template.iter().map(|p| rot * p + shift).collect();
        let target_com = target.iter().sum::<Point>() / target.len() as f64;
        let target_with_mass: Vec<(Point, f64)> = target.iter().map(|p| (*p, 1.0)).collect();

        let cell = crate::cell::Endless;
        let mut rng = rand::thread_rng();
        let result =
            overlay_positions(&template, target_with_mass, &target_com, &cell, &mut rng).unwrap();

        // COM should match target COM
        let result_com = result.iter().sum::<Point>() / result.len() as f64;
        assert_relative_eq!(result_com.x, target_com.x, epsilon = 1e-8);
        assert_relative_eq!(result_com.y, target_com.y, epsilon = 1e-8);
        assert_relative_eq!(result_com.z, target_com.z, epsilon = 1e-8);

        // Rg² should be preserved (same shape, just rotated+translated)
        let template_gt = GyrationTensor::from_equal_mass_positions(&template).unwrap();
        let result_gt = GyrationTensor::from_equal_mass_positions(&result).unwrap();
        assert_relative_eq!(result_gt.rg_squared, template_gt.rg_squared, epsilon = 1e-8);
    }

    #[test]
    fn overlay_with_pbc() {
        let cell = crate::cell::Cuboid::new(20.0, 20.0, 20.0);
        let template = [
            Point::new(-1.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
            Point::new(0.0, 1.0, 0.0),
        ];
        // Target near box edge
        let target_com = Point::new(9.5, 0.0, 0.0);
        let target: Vec<(Point, f64)> = template
            .iter()
            .map(|p| {
                let mut pos = p + target_com;
                cell.boundary(&mut pos);
                (pos, 1.0)
            })
            .collect();

        let mut rng = rand::thread_rng();
        let result = overlay_positions(&template, target, &target_com, &cell, &mut rng).unwrap();

        // All result positions should be inside the cell
        for pos in &result {
            assert!(cell.is_inside(pos), "Position {pos:?} outside cell");
        }
    }
}

/// Calculate center of mass of a collection of points with masses.
/// Does not consider periodic boundary conditions.
pub(crate) fn mass_center<'a>(
    positions: impl IntoIterator<Item = &'a Point>,
    masses: &[f64],
) -> Point {
    let total_mass: f64 = masses.iter().sum();
    positions
        .into_iter()
        .zip(masses)
        .map(|(r, &m)| r * m)
        .sum::<Point>()
        / total_mass
}

/// Calculate center of mass of a collection of points with masses using PBC.
///
/// Uses the first atom as reference and unwraps all others via minimum image
/// convention to guarantee consistent geometry regardless of box wrapping.
#[cfg(test)]
pub(crate) fn mass_center_pbc<'a>(
    positions: impl IntoIterator<Item = &'a Point>,
    masses: &[f64],
    cell: &impl SimulationCell,
    _shift: Option<Point>,
) -> Point {
    let total_mass: f64 = masses.iter().sum();
    let mut iter = positions.into_iter().zip(masses.iter());
    let (&ref_pos, &ref_mass) = iter.next().expect("at least one position required");
    let mut com = ref_pos * ref_mass;
    for (&pos, &m) in iter {
        // Unwrap relative to reference atom using MIC
        let unwrapped = ref_pos + cell.distance(&pos, &ref_pos);
        com += unwrapped * m;
    }
    com /= total_mass;
    cell.boundary(&mut com);
    com
}

/// Calculate angle between two vectors in degrees.
#[inline(always)]
pub(crate) fn angle_vectors(v1: &Vector3<f64>, v2: &Vector3<f64>) -> f64 {
    let cos = v1.dot(v2) / (v1.norm() * v2.norm());
    cos.acos().to_degrees()
}

/// Calculate angle between three points with `b` being the vertex, in degrees.
#[inline(always)]
pub(crate) fn angle_points(a: &Point, b: &Point, c: &Point, pbc: &impl SimulationCell) -> f64 {
    angle_vectors(&pbc.distance(a, b), &pbc.distance(c, b))
}

/// Calculate dihedral angle between two planes defined by four points.
/// The first plane is given by points `a`, `b`, `c`.
/// The second plane is given by points `b`, `c`, `d`.
/// The angle is returned in degrees and adopts values between −180° and +180°.
pub(crate) fn dihedral_points(
    a: &Point,
    b: &Point,
    c: &Point,
    d: &Point,
    pbc: &impl SimulationCell,
) -> f64 {
    let ab = pbc.distance(b, a);
    let bc = pbc.distance(c, b);
    let cd = pbc.distance(d, c);

    // normalized vectors normal to the planes
    let abc = ab.cross(&bc).normalize();
    let bcd = bc.cross(&cd).normalize();

    let cos_angle = abc.dot(&bcd);
    let sin_angle = bc.normalize().dot(&abc.cross(&bcd));

    sin_angle.atan2(cos_angle).to_degrees()
}

#[test]
fn test_center_of_mass() {
    use float_cmp::assert_approx_eq;

    let positions = [
        Point::new(10.4, 11.3, 12.8),
        Point::new(7.3, 9.3, 2.6),
        Point::new(9.3, 10.1, 17.2),
    ];
    let masses = [1.46, 2.23, 10.73];

    let com = mass_center(&positions, &masses);

    assert_approx_eq!(f64, com.x, 9.10208044382802);
    assert_approx_eq!(f64, com.y, 10.09778085991678);
    assert_approx_eq!(f64, com.z, 14.49667128987517);

    let positions = [
        Point::new(10.4, 11.3, 12.8),
        Point::new(7.3, 9.3, 2.6),
        Point::new(9.3, 10.1, 17.2),
        Point::new(3.1, 2.4, 1.8),
    ];

    let masses = [1.46, 2.23, 10.73, 0.0];

    let com = mass_center(&positions, &masses);

    assert_approx_eq!(f64, com.x, 9.10208044382802);
    assert_approx_eq!(f64, com.y, 10.09778085991678);
    assert_approx_eq!(f64, com.z, 14.49667128987517);
}

#[test]
fn test_angle_vectors() {
    use float_cmp::assert_approx_eq;

    let v1 = Vector3::new(2.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 2.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 90.0);

    let v1 = Vector3::new(2.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, -2.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 90.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 0.0, 7.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 90.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(3.0, 0.0, 3.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 45.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(4.0, 0.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 0.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(-4.0, 0.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 180.0);

    let v1 = Vector3::new(1.0, -1.0, 3.5);
    let v2 = Vector3::new(1.2, 2.4, -0.7);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 110.40636490060925);
}

#[test]
fn test_angle_points() {
    use float_cmp::assert_approx_eq;

    let endless_cell = crate::cell::Endless;

    let p1 = Point::new(3.2, 3.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 5.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 90.0);

    let p1 = Point::new(3.2, 3.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 1.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 90.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(3.2, 3.3, 9.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 90.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(6.2, 3.3, 5.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 45.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(7.2, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 0.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(-1.2, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 180.0);

    let p1 = Point::new(4.2, 2.3, 6.0);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(4.4, 5.7, 1.8);
    assert_approx_eq!(
        f64,
        angle_points(&p1, &p2, &p3, &endless_cell),
        110.40636490060925
    );
}

#[test]
fn test_angle_points_pbc() {
    use float_cmp::assert_approx_eq;

    let cell = crate::cell::Cuboid::new(5.0, 10.0, 15.0);

    let p1 = Point::new(2.2, 3.3, 2.5);
    let p2 = Point::new(-2.0, 3.3, 2.5);
    let p3 = Point::new(-2.2, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &cell), 0.0);

    let p1 = Point::new(1.4, 3.3, 2.5);
    let p2 = Point::new(2.2, 3.3, 2.5);
    let p3 = Point::new(-2.3, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &cell), 180.0);

    let p1 = Point::new(1.5, -4.7, 1.2);
    let p2 = Point::new(1.5, 4.3, 1.2);
    let p3 = Point::new(1.5, -2.7, 4.2);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &cell), 45.0);
}

#[test]
fn test_dihedral_points() {
    use float_cmp::assert_approx_eq;

    let endless_cell = crate::cell::Endless;

    // cis conformation
    let p1 = Point::new(1.2, 5.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(3.2, 3.3, 2.5);
    let p4 = Point::new(3.2, 4.3, 2.5);
    assert_approx_eq!(f64, dihedral_points(&p1, &p2, &p3, &p4, &endless_cell), 0.0);

    // cis conformation
    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(1.2, -1.3, 3.2);
    assert_approx_eq!(f64, dihedral_points(&p1, &p2, &p3, &p4, &endless_cell), 0.0);

    // trans conformation
    let p1 = Point::new(1.2, -5.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(3.2, 3.3, 2.5);
    let p4 = Point::new(3.2, 4.3, 2.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        180.0
    );

    // trans conformation
    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(1.2, -1.3, 2.2);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        180.0
    );

    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(-13.2, -1.3, 2.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        90.0
    );

    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(2.2, -1.3, 2.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        -90.0
    );

    let p1 = Point::new(3.2, -5.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 3.3, 2.5);
    let p4 = Point::new(1.2, 4.3, 3.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        135.0
    );

    let p1 = Point::new(3.2, 5.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 3.3, 2.5);
    let p4 = Point::new(1.2, 4.3, 3.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        -45.0
    );

    let p1 = Point::new(3.2, 5.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 3.3, 2.5);
    let p4 = Point::new(1.2, 4.3, 1.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        45.0
    );

    // realistic data
    let p0 = Point::new(24.969, 13.428, 30.692);
    let p1 = Point::new(24.044, 12.661, 29.808);
    let p2 = Point::new(22.785, 13.482, 29.543);
    let p3 = Point::new(21.951, 13.670, 30.431);
    let p4 = Point::new(23.672, 11.328, 30.466);
    let p5 = Point::new(22.881, 10.326, 29.620);
    let p6 = Point::new(23.691, 9.935, 28.389);
    let p7 = Point::new(22.557, 9.096, 30.459);
    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p2, &p3, &endless_cell),
        -71.215151146714
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p4, &p5, &endless_cell),
        -171.9431994795364
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p6, &endless_cell),
        60.82226735264639
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p7, &endless_cell),
        -177.6364115152126
    );
}

#[test]
fn test_dihedral_points_pbc() {
    use crate::cell::BoundaryConditions;
    use float_cmp::assert_approx_eq;

    let cuboid = crate::cell::Cuboid::new(20.0, 10.0, 28.0);

    let mut p0 = Point::new(24.969, 13.428, 30.692);
    let mut p1 = Point::new(24.044, 12.661, 29.808);
    let mut p2 = Point::new(22.785, 13.482, 29.543);
    let mut p3 = Point::new(21.951, 13.670, 30.431);
    let mut p4 = Point::new(23.672, 11.328, 30.466);
    let mut p5 = Point::new(22.881, 10.326, 29.620);
    let mut p6 = Point::new(23.691, 9.935, 28.389);
    let mut p7 = Point::new(22.557, 9.096, 30.459);

    cuboid.boundary(&mut p0);
    cuboid.boundary(&mut p1);
    cuboid.boundary(&mut p2);
    cuboid.boundary(&mut p3);
    cuboid.boundary(&mut p4);
    cuboid.boundary(&mut p5);
    cuboid.boundary(&mut p6);
    cuboid.boundary(&mut p7);

    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p2, &p3, &cuboid),
        -71.215151146714
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p4, &p5, &cuboid),
        -171.9431994795364
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p6, &cuboid),
        60.82226735264639
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p7, &cuboid),
        -177.6364115152126
    );
}
