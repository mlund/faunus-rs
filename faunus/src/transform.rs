// Copyright 2023 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Transformations of particles and groups

use crate::{cell::VolumeScalePolicy, group::ParticleSelection, Point};
use anyhow::Ok;
use nalgebra::Quaternion;
use rand::prelude::*;

/// Generate a random unit vector by sphere picking
///
/// See also: <https://docs.rs/rand_distr/0.4.0/rand_distr/struct.UnitSphere.html>
pub fn random_unit_vector(rng: &mut impl Rng) -> Point {
    const RADIUS_SQUARED: f64 = 0.5 * 0.5;
    loop {
        let p = Point::new(
            rng.gen::<f64>() - 0.5,
            rng.gen::<f64>() - 0.5,
            rng.gen::<f64>() - 0.5,
        );
        let norm_squared = p.norm_squared();
        if norm_squared <= RADIUS_SQUARED {
            return p / norm_squared.sqrt();
        }
    }
}

/// This describes a transformation on a set of particles or a group.
/// For example, a translation by a vector, a rotation by an angle and axis,
/// or a contraction by `n` particles. It is mainly used to describe Monte Carlo moves.
#[derive(Clone, Debug)]
pub enum Transform {
    /// Translate all active particles by a vector
    Translate(Point),
    /// Translate a partial set of particles by a vector
    PartialTranslate(Point, ParticleSelection),
    /// Use a quaternion to rotatate around a given point
    Rotate(Point, Quaternion<f64>),
    /// Use a quaternion to rotatate a set of particles around a given point
    PartialRotate(Point, Quaternion<f64>, ParticleSelection),
    /// Scale coordinates from an old volume to a new, `(old_volume, new_volume)`
    VolumeScale(VolumeScalePolicy, (f64, f64)),
    /// Expand by `n` particles
    Expand(usize),
    /// Contract by `n` particles
    Contract(usize),
    /// Deactivate
    Deactivate,
    /// Activate
    Activate,
    /// Apply periodic boundary conditions to all particles
    Boundary,
    /// No operation
    None,
}

impl Transform {
    /// Transform a set of particles using a transformation
    ///
    /// The transformation is applied to the particles in a single group,
    /// given by `group_index`, in the `context`.
    pub fn on_group(
        &self,
        group_index: usize,
        context: &mut impl crate::Context,
    ) -> Result<(), anyhow::Error> {
        match self {
            Transform::Translate(displacement) => {
                Self::PartialTranslate(*displacement, ParticleSelection::Active)
                    .on_group(group_index, context)?;
            }
            Transform::PartialTranslate(displacement, selection) => {
                let indices = context.groups()[group_index]
                    .select(selection, context)
                    .unwrap();

                context.translate_particles(&indices, displacement);
            }
            _ => {
                todo!("Implement other transforms")
            }
        }
        Ok(())
    }
}

/// Rotate a collection of points by a random angle in random direction.
///
/// The optional `center` of rotation is subtracted before rotation,
/// and added again after.
pub(crate) fn rotate_random<'a>(
    positions: impl IntoIterator<Item = &'a mut Point>,
    center: &Point,
    rng: &mut ThreadRng,
) {
    let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
    let axis = crate::transform::random_unit_vector(rng);
    let matrix = nalgebra::Rotation3::new(axis * angle);
    let rotate = |pos: &mut Point| *pos = matrix * (*pos - center) + center;
    positions.into_iter().for_each(rotate);
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    // test random unit vector generation
    #[test]
    fn test_random_unit_vector() {
        let n = 5000;
        let mut rng = rand::thread_rng();
        let mut x_mean = 0.0;
        let mut y_mean = 0.0;
        let mut z_mean = 0.0;
        let mut rngsum = 0.0;
        for _ in 0..n {
            let v = random_unit_vector(&mut rng);
            assert_approx_eq!(f64, v.norm(), 1.0);
            x_mean += v.x;
            y_mean += v.y;
            z_mean += v.z;
            rngsum += rng.gen::<f64>();
        }
        assert_approx_eq!(f64, x_mean / n as f64, 0.0, epsilon = 0.025);
        assert_approx_eq!(f64, y_mean / n as f64, 0.0, epsilon = 0.025);
        assert_approx_eq!(f64, z_mean / n as f64, 0.0, epsilon = 0.025);
        assert_approx_eq!(f64, rngsum / n as f64, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_rotate_random() {
        let positions = [
            Point::new(10.4, 11.3, 12.8),
            Point::new(7.3, 9.3, 2.6),
            Point::new(9.3, 10.1, 17.2),
        ];
        let masses = [1.46, 2.23, 10.73];
        let com = crate::aux::center_of_mass(&positions, &masses);

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let mut cloned = positions;

            rotate_random(&mut cloned, &com, &mut rng);

            for (original, new) in positions.iter().zip(cloned.iter()) {
                assert_ne!(original, new);
            }

            let com_rotated = crate::aux::center_of_mass(&cloned, &masses);
            assert_approx_eq!(f64, com.x, com_rotated.x);
            assert_approx_eq!(f64, com.y, com_rotated.y);
            assert_approx_eq!(f64, com.z, com_rotated.z);
        }
    }
}
