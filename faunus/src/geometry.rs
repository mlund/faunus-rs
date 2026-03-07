//! Gyration tensor decomposition and molecular overlay via principal-axis alignment.

use crate::{cell::SimulationCell, Point};
use nalgebra::{Matrix3, Rotation3, SymmetricEigen};
use rand::Rng;

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
