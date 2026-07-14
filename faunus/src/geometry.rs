//! Geometry: mass center, dipole moment, angles, dihedrals, gyration tensor, and molecular overlay.

use crate::{cell::SimulationCell, Point, UnitQuaternion};
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

/// Compute the traceless quadrupole tensor of a charge distribution relative to a reference point.
///
/// **Θ**_αβ = ½ Σ qᵢ (3 dᵢ_α dᵢ_β − dᵢ² δ_αβ),  dᵢ = rᵢ − r_ref  (Buckingham convention),
/// using minimum-image distances. The traceless form is reported because the isotropic
/// part Σ qᵢ dᵢ² does not contribute to the far-field potential and would otherwise
/// register a spurious moment for isotropic charge distributions.
pub(crate) fn quadrupole_moment(
    charges_positions: impl IntoIterator<Item = (f64, Point)>,
    reference: &Point,
    cell: &impl SimulationCell,
) -> Matrix3<f64> {
    let primitive = charges_positions
        .into_iter()
        .fold(Matrix3::zeros(), |q, (charge, pos)| {
            let d = cell.distance(&pos, reference);
            q + charge * d * d.transpose()
        });
    0.5 * (3.0 * primitive - primitive.trace() * Matrix3::identity())
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

/// Gather a molecule into a single periodic image, minimum-imaged about its first atom.
///
/// Stored coordinates are wrapped into the cell, so a molecule straddling a boundary has
/// atoms at opposite ends of an axis. Any arithmetic that treats those raw values as a rigid
/// body — a centroid, a gyration tensor, a displacement from the centre — then describes a
/// shell the size of the box rather than a molecule. Gather first.
///
/// Meaningful only while the molecule spans less than half the shortest box side; beyond
/// that, minimum image picks the wrong periodic copy and no gathering can recover it.
pub(crate) fn gather_molecule(positions: &[Point], cell: &impl SimulationCell) -> Vec<Point> {
    let Some(&first) = positions.first() else {
        return Vec::new();
    };
    positions
        .iter()
        .map(|p| first + cell.distance(p, &first))
        .collect()
}

/// Rotation that best maps a molecule's `reference` conformation onto its `current`
/// coordinates (Kabsch superposition).
///
/// This is what a group's stored orientation *means*, so it can be recovered from the
/// coordinates rather than carried alongside them and forgotten. `current` must already be
/// gathered into one periodic image; both sets are centred here.
///
/// `None` when there is nothing to fit — fewer than two atoms, or a mismatched conformation.
/// The fit is exact for a rigid body and a least-squares best fit otherwise. It is ambiguous,
/// though always consistent with the coordinates, for a molecule with rotational symmetry.
pub(crate) fn best_fit_rotation(reference: &[Point], current: &[Point]) -> Option<UnitQuaternion> {
    if reference.len() != current.len() || reference.len() < 2 {
        return None;
    }
    let n = reference.len() as f64;
    let ref_com: Point = reference.iter().sum::<Point>() / n;
    let cur_com: Point = current.iter().sum::<Point>() / n;

    // Covariance of the two centred point sets; its SVD gives the optimal rotation.
    let covariance = reference
        .iter()
        .zip(current)
        .fold(Matrix3::zeros(), |acc, (r, c)| {
            acc + (c - cur_com) * (r - ref_com).transpose()
        });

    let svd = covariance.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    // A negative determinant would be a reflection, not a rotation: flip the least significant
    // singular vector, which is the smallest change that restores a proper rotation.
    let sign = (u * v_t).determinant().signum();
    let matrix = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, sign)) * v_t;

    Some(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(matrix),
    ))
}

/// How far a molecule's coordinates depart from the ones `orientation` claims it has (RMSD, Å).
///
/// The honest test of a stored orientation, and the only one that works for every molecule: it
/// asks whether the quaternion *reproduces the coordinates*, not whether it equals some fitted
/// value. A symmetric or linear molecule has many orientations consistent with the same
/// coordinates — comparing against a best fit would call those disagreements, when the coordinates
/// cannot tell them apart at all.
///
/// `None` when there is nothing to compare — a mismatched or absent reference conformation.
#[cfg(test)]
pub(crate) fn orientation_residual(
    reference: &[Point],
    current: &[Point],
    orientation: &UnitQuaternion,
) -> Option<f64> {
    if reference.len() != current.len() || reference.is_empty() {
        return None;
    }
    let n = reference.len() as f64;
    let ref_com: Point = reference.iter().sum::<Point>() / n;
    let cur_com: Point = current.iter().sum::<Point>() / n;
    Some(
        (reference
            .iter()
            .zip(current)
            .map(|(r, c)| (orientation * (r - ref_com) - (c - cur_com)).norm_squared())
            .sum::<f64>()
            / n)
            .sqrt(),
    )
}

/// Relative spread below which a molecule counts as linear and its axial rotation as free.
const COLLINEARITY_TOL: f64 = 1e-9;

/// The direction a molecule lies along, if it is linear.
///
/// A superposition pins a rigid body's rotation only when the molecule spans at least a plane.
/// A diatomic — or any linear molecule — leaves the rotation *about its own axis* completely
/// undetermined: every axial angle reproduces the coordinates exactly, so the fit is free to
/// return any of them.
fn collinear_axis(points: &[Point]) -> Option<Point> {
    let n = points.len() as f64;
    let com: Point = points.iter().sum::<Point>() / n;
    let covariance = points.iter().fold(Matrix3::zeros(), |acc, p| {
        let d = p - com;
        acc + d * d.transpose()
    });
    let eigen = SymmetricEigen::new(covariance);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));
    let (largest, second) = (eigen.eigenvalues[order[0]], eigen.eigenvalues[order[1]]);
    (second <= COLLINEARITY_TOL * largest).then(|| eigen.eigenvectors.column(order[0]).normalize())
}

/// The component of `rotation` about `axis` (the twist of a swing-twist decomposition).
fn twist_about(rotation: &UnitQuaternion, axis: &Point) -> UnitQuaternion {
    let projected = axis * rotation.vector().dot(axis);
    let twist = nalgebra::Quaternion::new(rotation.w, projected.x, projected.y, projected.z);
    if twist.norm() < f64::EPSILON {
        UnitQuaternion::identity()
    } else {
        UnitQuaternion::new_normalize(twist)
    }
}

/// Rotation carrying `reference` onto `current`, but only when `current` really is a rigid
/// image of it — the residual of the fit is below `tolerance`.
///
/// This is the test for "is the orientation recoverable from the coordinates at all". It is,
/// for a molecule that has only been rotated and translated; it is not for one whose shape has
/// actually changed, where a best fit is just the closest lie and the stored orientation is the
/// only record of the rotations that were applied. Returning `None` there keeps a fit from
/// silently overwriting it.
///
/// A *linear* molecule is recoverable only up to its axial spin, which no coordinate can
/// witness. The free component is resolved against `prior` — the orientation the group already
/// holds — rather than left to whatever the superposition happens to produce, which would
/// otherwise re-spin a diatomic by an arbitrary angle every time its coordinates were revisited.
/// Express a lab-frame displacement in the body frame a group's orientation defines.
///
/// The inverse of the rotation [`rigid_body_rotation`] fits, and the single home of that
/// convention: a body-frame coordinate `b` sits at `com + R(q)·b` in the lab, so `b = q⁻¹·(r − com)`.
/// Anything that stores coordinates relative to a rotating molecule — an orientation-resolved
/// density, the reference conformation the GPU integrator rebuilds a rigid body from — goes through
/// here, so a flip of the convention cannot reach one of them and miss the other.
pub(crate) fn to_body_frame(displacement: &Point, orientation: &UnitQuaternion) -> Point {
    orientation.inverse_transform_vector(displacement)
}

pub(crate) fn rigid_body_rotation(
    reference: &[Point],
    current: &[Point],
    tolerance: f64,
    prior: &UnitQuaternion,
) -> Option<UnitQuaternion> {
    let rotation = best_fit_rotation(reference, current)?;
    let n = reference.len() as f64;
    let ref_com: Point = reference.iter().sum::<Point>() / n;
    let cur_com: Point = current.iter().sum::<Point>() / n;
    let residual = (reference
        .iter()
        .zip(current)
        .map(|(r, c)| (rotation * (r - ref_com) - (c - cur_com)).norm_squared())
        .sum::<f64>()
        / n)
        .sqrt();
    if residual > tolerance {
        return None;
    }

    let Some(axis) = collinear_axis(reference) else {
        return Some(rotation); // the fit is unique
    };
    // Keep the axial spin the group already had: it is unobservable, so re-deriving it would
    // change the stored orientation without anything in the coordinates having moved.
    let lab_axis = rotation * axis;
    let delta = prior * rotation.inverse();
    Some(twist_about(&delta, &lab_axis) * rotation)
}

/// Place a molecule's template at `com`, keeping its conformation and orientation.
///
/// The unoriented counterpart of [`overlay_positions`], for the cases where no principal-axis
/// frame exists to align against — a molecule of fewer than two atoms has no such frame.
pub(crate) fn place_at(template: &[Point], com: &Point, cell: &impl SimulationCell) -> Vec<Point> {
    let gathered = gather_molecule(template, cell);
    if gathered.is_empty() {
        return Vec::new();
    }
    let centroid: Point = gathered.iter().sum::<Point>() / gathered.len() as f64;
    gathered
        .iter()
        .map(|p| {
            let mut pos = com + (p - centroid);
            cell.boundary(&mut pos);
            pos
        })
        .collect()
}

/// Overlay template positions onto a target molecule using gyration tensor alignment.
///
/// Aligns the principal axes of `template_positions` to match the principal-axis frame of
/// the target group defined by `target_positions_masses` (mass-weighted, with PBC).
/// Degenerate eigenvalues are resolved with random sign flips to avoid MC bias.
///
/// The template arrives as raw stored coordinates and so is gathered first; the target is
/// minimum-imaged about its own mass center by [`GyrationTensor::from_positions_masses_com`].
///
/// Returns new positions in lab frame, centered on the target COM.
pub(crate) fn overlay_positions(
    template_positions: &[Point],
    target_positions_masses: impl IntoIterator<Item = (Point, f64)>,
    target_com: &Point,
    cell: &impl SimulationCell,
    rng: &mut (impl Rng + ?Sized),
) -> Option<Vec<Point>> {
    let template = gather_molecule(template_positions, cell);
    let template_com: Point = template.iter().sum::<Point>() / template.len() as f64;

    let target_gt =
        GyrationTensor::from_positions_masses_com(target_positions_masses, target_com, cell)?;
    // Already gathered, so there is no minimum image left to apply — hence `Endless`.
    let template_gt = GyrationTensor::from_positions_masses_com(
        template.iter().map(|&p| (p, 1.0)),
        &template_com,
        &crate::cell::Endless,
    )?;

    // Randomize degenerate axes for both frames
    let target_rot = randomize_degenerate_axes(&target_gt.rotation, &target_gt.eigenvalues, rng);
    let template_rot =
        randomize_degenerate_axes(&template_gt.rotation, &template_gt.eigenvalues, rng);

    // R = R_target · R_template⁻¹ maps template body frame → target lab frame
    let align = target_rot * template_rot.inverse();

    let positions = template
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
    use crate::cell::{BoundaryConditions, Endless, Shape};
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    /// Gyration tensor of equal-mass points that are already in one image.
    fn equal_mass_gyration(positions: &[Point]) -> Option<GyrationTensor> {
        let com: Point = positions.iter().sum::<Point>() / positions.len().max(1) as f64;
        GyrationTensor::from_positions_masses_com(
            positions.iter().map(|&p| (p, 1.0)),
            &com,
            &Endless,
        )
    }

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
        let gt = equal_mass_gyration(&positions).unwrap();
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
        let gt = equal_mass_gyration(&positions).unwrap();
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
        let gt = equal_mass_gyration(&positions).unwrap();
        assert_relative_eq!(gt.rotation.matrix().determinant(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn too_few_particles_returns_none() {
        let positions = [Point::new(1.0, 2.0, 3.0)];
        assert!(equal_mass_gyration(&positions).is_none());
        assert!(equal_mass_gyration(&[]).is_none());
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
        let template_gt = equal_mass_gyration(&template).unwrap();
        let result_gt = equal_mass_gyration(&result).unwrap();
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

    /// The rotation a molecule was placed with is exactly what the fit recovers.
    #[test]
    fn best_fit_rotation_recovers_an_applied_rotation() {
        let reference = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(4.5, 0.0, 1.5),
            Point::new(2.25, 3.9, -0.5),
            Point::new(-2.25, -3.9, -0.5),
        ];
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let applied = crate::transform::random_rotation(&mut rng);
            let shift = Point::new(11.0, -3.0, 7.0);
            let current: Vec<Point> = reference.iter().map(|p| applied * p + shift).collect();

            let fitted = best_fit_rotation(&reference, &current).unwrap();
            assert!(
                fitted.angle_to(&applied) < 1e-9,
                "recovered {fitted:?}, applied {applied:?}"
            );
        }
    }

    /// A reflection is not a rotation: a mirrored molecule must not be fitted with one.
    #[test]
    fn best_fit_rotation_never_returns_a_reflection() {
        let reference = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
            Point::new(0.0, 1.0, 0.0),
            Point::new(0.0, 0.0, 1.0),
        ];
        // Mirror through the xy-plane — no rotation can reproduce it.
        let current: Vec<Point> = reference
            .iter()
            .map(|p| Point::new(p.x, p.y, -p.z))
            .collect();

        let fitted = best_fit_rotation(&reference, &current).unwrap();
        let matrix = fitted.to_rotation_matrix();
        assert_relative_eq!(matrix.matrix().determinant(), 1.0, epsilon = 1e-9);
    }

    /// A linear molecule's spin about its own axis is unobservable, so it must not be invented.
    ///
    /// The superposition of a diatomic is rank-deficient: every axial angle reproduces the
    /// coordinates exactly, and the raw fit returns an arbitrary one — measured at up to 170°
    /// from the truth. Left unchecked, restoring a perfectly consistent checkpoint would re-spin
    /// every dimer in it and report the file as corrupt.
    #[test]
    fn a_linear_molecule_keeps_the_axial_spin_it_already_had() {
        let reference = [Point::new(-1.0, 0.0, 0.0), Point::new(1.0, 0.0, 0.0)];
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let truth = crate::transform::random_rotation(&mut rng);
            let current: Vec<Point> = reference.iter().map(|p| truth * p).collect();

            // Told what the group already believes, the fit must return exactly that.
            let recovered = rigid_body_rotation(&reference, &current, 1e-3, &truth)
                .expect("a rigidly rotated dimer is a rigid image");
            assert!(
                recovered.angle_to(&truth) < 1e-9,
                "re-spun a dimer by {:.1}° against coordinates that never moved",
                recovered.angle_to(&truth).to_degrees()
            );

            // Whatever it returns must still reproduce the coordinates.
            for (r, c) in reference.iter().zip(&current) {
                assert_relative_eq!((recovered * r - c).norm(), 0.0, epsilon = 1e-9);
            }
        }
    }

    /// A molecule spanning a plane pins its rotation outright; the prior is irrelevant.
    #[test]
    fn a_non_linear_molecule_ignores_the_prior() {
        let reference = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(2.0, 0.0, 0.0),
            Point::new(0.0, 1.5, 0.0),
        ];
        let mut rng = rand::thread_rng();
        let truth = crate::transform::random_rotation(&mut rng);
        let current: Vec<Point> = reference.iter().map(|p| truth * p).collect();

        let nonsense = crate::transform::random_rotation(&mut rng);
        let recovered = rigid_body_rotation(&reference, &current, 1e-3, &nonsense).unwrap();
        assert!(recovered.angle_to(&truth) < 1e-9);
    }

    /// A conformation that actually changed has no rigid-body rotation to recover.
    #[test]
    fn a_reshaped_molecule_has_no_rigid_body_rotation() {
        let reference = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(2.0, 0.0, 0.0),
            Point::new(0.0, 1.5, 0.0),
        ];
        let mut deformed = reference.to_vec();
        deformed[2].y += 0.9; // a pivot-scale change, far beyond any rounding
        assert!(
            rigid_body_rotation(&reference, &deformed, 1e-3, &UnitQuaternion::identity()).is_none()
        );
    }

    #[test]
    fn best_fit_rotation_needs_two_matched_atoms() {
        let one = [Point::new(1.0, 2.0, 3.0)];
        assert!(best_fit_rotation(&one, &one).is_none());
        let two = [Point::zeros(), Point::new(1.0, 0.0, 0.0)];
        assert!(best_fit_rotation(&two, &one).is_none());
    }

    /// All pairwise minimum-image separations, ascending — the molecule's shape,
    /// independent of where and how it is oriented.
    fn pair_distances(positions: &[Point], cell: &impl BoundaryConditions) -> Vec<f64> {
        let mut d = Vec::new();
        for (n, p) in positions.iter().enumerate() {
            for q in &positions[n + 1..] {
                d.push(cell.distance(p, q).norm());
            }
        }
        d.sort_by(f64::total_cmp);
        d
    }

    /// A template stored across a periodic boundary still overlays as one intact molecule.
    ///
    /// The template arrives as raw wrapped coordinates — a molecule sitting on the box edge
    /// has atoms at both ends of the axis. Treating those as a rigid body without gathering
    /// them first puts its centre mid-box and flings the atoms apart on overlay.
    #[test]
    fn overlay_of_a_wrapped_template_keeps_the_molecule_intact() {
        let cell = crate::cell::Cuboid::new(20.0, 20.0, 20.0);

        // Centre the molecule on the +x face so that it straddles the boundary.
        let shape = [
            Point::new(-2.0, 0.0, 0.0),
            Point::new(2.0, 0.0, 0.0),
            Point::new(0.0, 1.5, 0.0),
        ];
        let template: Vec<Point> = shape
            .iter()
            .map(|p| {
                let mut pos = p + Point::new(10.0, 0.0, 0.0);
                cell.boundary(&mut pos);
                pos
            })
            .collect();
        assert!(
            template.iter().any(|p| p.x < 0.0) && template.iter().any(|p| p.x > 0.0),
            "template must straddle the boundary for this test to mean anything"
        );

        // Target: the same molecule, intact, in the middle of the box.
        let target_com = Point::new(1.0, 2.0, 3.0);
        let target: Vec<(Point, f64)> = shape.iter().map(|p| (p + target_com, 1.0)).collect();

        let mut rng = rand::thread_rng();
        let result = overlay_positions(&template, target, &target_com, &cell, &mut rng).unwrap();

        // The overlay may reorient the molecule, but it must not deform it.
        let expected = pair_distances(&shape, &Endless);
        let got = pair_distances(&result, &cell);
        assert_eq!(got.len(), expected.len());
        for (got, want) in got.iter().zip(&expected) {
            assert_relative_eq!(got, want, epsilon = 1e-8);
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

#[cfg(test)]
mod body_frame_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The body frame is what the lab frame looks like with the group's own rotation taken out.
    #[test]
    fn inverse_quaternion_maps_lab_to_body_frame() {
        let orientation = UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::z_axis(),
            std::f64::consts::FRAC_PI_2,
        );
        let body = to_body_frame(&Point::new(0.0, 1.0, 0.0), &orientation);
        assert_relative_eq!(body.x, 1.0, epsilon = 1e-12);
        assert_relative_eq!(body.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(body.z, 0.0, epsilon = 1e-12);
    }

    /// Rotating a body-frame vector back by the same orientation returns the lab vector: the two
    /// directions are exact inverses, which is what `reconstruct_positions` on the GPU relies on.
    #[test]
    fn body_and_lab_frames_are_exact_inverses() {
        let orientation = UnitQuaternion::from_euler_angles(0.3, -0.7, 1.1);
        let lab = Point::new(1.5, -2.0, 0.75);
        let round_trip = orientation.transform_vector(&to_body_frame(&lab, &orientation));
        assert_relative_eq!((round_trip - lab).norm(), 0.0, epsilon = 1e-12);
    }
}
