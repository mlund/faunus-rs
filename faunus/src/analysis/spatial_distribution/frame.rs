use crate::{Point, UnitQuaternion};

/// Transform a lab-frame displacement into the rigid reference body frame.
pub(super) fn to_body_frame(displacement: &Point, orientation: &UnitQuaternion) -> Point {
    orientation.inverse_transform_vector(displacement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    #[test]
    fn inverse_quaternion_maps_lab_to_body_frame() {
        let axis = Vector3::z_axis();
        let orientation = UnitQuaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);
        let lab = Point::new(0.0, 1.0, 0.0);
        let body = to_body_frame(&lab, &orientation);
        assert_relative_eq!(body.x, 1.0, epsilon = 1e-12);
        assert_relative_eq!(body.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(body.z, 0.0, epsilon = 1e-12);
    }
}
