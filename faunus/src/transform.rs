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

use crate::UnitQuaternion;
use crate::{
    cell::VolumeScalePolicy,
    group::{AtomKindId, ParticleSelection},
    Point,
};
use rand::prelude::*;

/// Generate a random unit vector by sphere picking
///
/// See also: <https://docs.rs/rand_distr/0.4.0/rand_distr/struct.UnitSphere.html>
pub fn random_unit_vector(rng: &mut (impl Rng + ?Sized)) -> Point {
    const RADIUS_SQUARED: f64 = 0.5 * 0.5;
    loop {
        let p = Point::new(
            rng.r#gen::<f64>() - 0.5,
            rng.r#gen::<f64>() - 0.5,
            rng.r#gen::<f64>() - 0.5,
        );
        let norm_squared = p.norm_squared();
        if norm_squared <= RADIUS_SQUARED {
            return p / norm_squared.sqrt();
        }
    }
}

/// Random displacement uniformly sampled in `[-max, max]`.
pub fn random_displacement(rng: &mut (impl Rng + ?Sized), max: f64) -> f64 {
    max * 2.0 * (rng.r#gen::<f64>() - 0.5)
}

/// Random quaternion for rotation about a random axis with angle in `[-max_angle, max_angle]`.
///
/// Returns `(quaternion, angle)`.
pub fn random_quaternion(rng: &mut (impl Rng + ?Sized), max_angle: f64) -> (UnitQuaternion, f64) {
    let axis = nalgebra::UnitVector3::new_normalize(random_unit_vector(rng));
    let angle = random_displacement(rng, max_angle);
    (UnitQuaternion::from_axis_angle(&axis, angle), angle)
}

/// Uniformly random rotation quaternion (Haar measure on SO(3)).
///
/// Uses Marsaglia's rejection method to sample a point uniformly on the
/// 4D unit sphere, which is equivalent to uniform rotation sampling.
/// See K. Shoemake, "Uniform random rotations", Graphics Gems III (1992).
pub fn random_rotation(rng: &mut (impl Rng + ?Sized)) -> UnitQuaternion {
    // Two pairs of uniform deviates, each rejected to lie inside the unit disk
    let (s1, x1, y1) = loop {
        let x = 2.0 * rng.r#gen::<f64>() - 1.0;
        let y = 2.0 * rng.r#gen::<f64>() - 1.0;
        let s = x * x + y * y;
        if s < 1.0 {
            break (s, x, y);
        }
    };
    let (s2, x2, y2) = loop {
        let x = 2.0 * rng.r#gen::<f64>() - 1.0;
        let y = 2.0 * rng.r#gen::<f64>() - 1.0;
        let s = x * x + y * y;
        if s < 1.0 {
            break (s, x, y);
        }
    };
    let factor = ((1.0 - s1) / s2).sqrt();
    UnitQuaternion::new_normalize(nalgebra::Quaternion::new(x1, y1, x2 * factor, y2 * factor))
}

/// A single group-level action for speciation (reaction ensemble) moves.
#[derive(Clone, Debug)]
pub enum SpeciationAction {
    /// Activate an empty group and set particle positions
    ActivateGroup {
        group_index: usize,
        positions: Vec<Point>,
    },
    /// Deactivate a full group
    DeactivateGroup(usize),
    /// Swap atom kind of a particle
    SwapAtomKind {
        group_index: usize,
        abs_index: usize,
        new_atom_id: AtomKindId,
    },
    /// Activate a single atom in an atomic mega-group
    ActivateAtom { group_index: usize, position: Point },
    /// Deactivate a single atom in an atomic mega-group by swapping it to end of active range
    DeactivateAtom {
        group_index: usize,
        abs_index: usize,
    },
}

impl SpeciationAction {
    /// The group this action touches.
    pub const fn group_index(&self) -> usize {
        match self {
            Self::ActivateGroup { group_index, .. }
            | Self::SwapAtomKind { group_index, .. }
            | Self::ActivateAtom { group_index, .. }
            | Self::DeactivateAtom { group_index, .. } => *group_index,
            Self::DeactivateGroup(group_index) => *group_index,
        }
    }
}

/// This describes a transformation on a set of particles or a group.
///
/// For example, a translation by a vector, a rotation by an angle and axis,
/// or a contraction by `n` particles. It is mainly used to describe Monte Carlo moves.
#[derive(Clone, Debug)]
pub enum Transform {
    /// Translate all active particles by a vector
    Translate(Point),
    /// Translate a partial set of particles by a vector
    PartialTranslate(Point, ParticleSelection),
    /// Rotate all active particles around their mass center
    Rotate(UnitQuaternion),
    /// Move selected particles to given positions (parallel to the selection).
    ///
    /// Used by the internal-coordinate moves, which rotate a sub-tree of a molecule about one of
    /// its own bonds or atoms. They pass positions rather than a rotation because the sub-tree
    /// must first be unwrapped by *following bonds*: taking the minimum image of each atom
    /// independently, as a rotation about a centre would, folds the part of a chain lying more
    /// than half a box length from that centre into the wrong periodic image and tears it apart.
    SetPositions(Vec<Point>, ParticleSelection),
    /// Scale coordinates to a new volume using the given policy
    VolumeScale(VolumeScalePolicy, f64),
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
    /// Sequence of group-level actions for reaction ensemble moves
    Speciation(Vec<SpeciationAction>),
    /// No operation
    None,
    /// Move a set of groups as one rigid cluster (roto-translation).
    ///
    /// `new_mass_centers` gives each group's target mass center (parallel to `groups`, already
    /// wrapped). The move computes them in *unwrapped* coordinates — recruiting neighbours by
    /// minimum image during cluster growth — so a cluster spanning the periodic boundary rotates
    /// correctly, unlike a single-pivot rotation on raw coordinates. `rotation` is the common
    /// orientation change (`None` for translation only); it is applied about each group's own mass
    /// center (a molecule spans less than half the box, so that step is PBC-safe), and the
    /// cluster-level rotation about the pivot is realised by the mass-center targets.
    ClusterTransform {
        groups: Vec<usize>,
        new_mass_centers: Vec<Point>,
        rotation: Option<UnitQuaternion>,
    },
}

impl Transform {
    /// Apply the transformation to a single group in the context.
    pub(crate) fn on_group(
        &self,
        group_index: usize,
        context: &mut impl crate::Context,
    ) -> anyhow::Result<()> {
        use crate::group::GroupSize;
        let needs_mass_center_update = match self {
            Self::Translate(displacement) => {
                let indices = context.groups()[group_index]
                    .select(&ParticleSelection::Active, context.topology_ref())?;
                context.translate_particles(&indices, displacement);
                true
            }
            Self::PartialTranslate(displacement, selection) => {
                let indices =
                    context.groups()[group_index].select(selection, context.topology_ref())?;
                context.translate_particles(&indices, displacement);
                true
            }
            Self::Rotate(quaternion) => {
                let indices = context.groups()[group_index]
                    .select(&ParticleSelection::Active, context.topology_ref())?;
                let center = context.mass_center(&indices);
                context.rotate_particles(&indices, quaternion, Some(-center));
                context.groups_mut()[group_index].rotate_by(quaternion);
                true
            }
            Self::SetPositions(positions, selection) => {
                let indices =
                    context.groups()[group_index].select(selection, context.topology_ref())?;
                anyhow::ensure!(
                    indices.len() == positions.len(),
                    "SetPositions: {} positions for {} selected particles",
                    positions.len(),
                    indices.len()
                );
                context.set_particle_positions(&indices, positions);
                true
            }
            Self::Activate => {
                context.resize_group(group_index, GroupSize::Full)?;
                true
            }
            Self::Expand(n) => {
                context.resize_group(group_index, GroupSize::Expand(*n))?;
                true
            }
            Self::Deactivate => {
                context.resize_group(group_index, GroupSize::Empty)?;
                false
            }
            Self::Contract(n) => {
                context.resize_group(group_index, GroupSize::Shrink(*n))?;
                false
            }
            _ => {
                todo!("Implement other transforms")
            }
        };
        if needs_mass_center_update {
            context.update_mass_center(group_index);
        }
        Ok(())
    }

    /// Apply the transformation to a group, saving affected particles as backup first.
    pub(crate) fn on_group_with_backup(
        &self,
        group_index: usize,
        context: &mut impl crate::Context,
    ) -> anyhow::Result<()> {
        let indices = match self {
            Self::Translate(_) | Self::Rotate(_) => context.groups()[group_index]
                .select(&ParticleSelection::Active, context.topology_ref())?,
            Self::PartialTranslate(_, selection) | Self::SetPositions(_, selection) => {
                context.groups()[group_index].select(selection, context.topology_ref())?
            }
            _ => vec![],
        };
        context.save_particle_backup(group_index, &indices);
        self.on_group(group_index, context)
    }

    /// Apply a system-wide transformation with backup (saves all particles, mass centers, cell).
    pub(crate) fn on_system_with_backup(
        &self,
        context: &mut impl crate::Context,
    ) -> anyhow::Result<()> {
        context.save_system_backup();
        self.on_system(context)
    }

    /// Apply a system-wide transformation to the context.
    pub(crate) fn on_system(&self, context: &mut impl crate::Context) -> anyhow::Result<()> {
        match self {
            Self::VolumeScale(policy, new_volume) => {
                context.scale_volume_and_positions(*new_volume, *policy)?;
            }
            Self::Speciation(actions) => {
                for action in actions {
                    match action {
                        SpeciationAction::ActivateGroup {
                            group_index,
                            positions,
                        } => {
                            let start = context.groups()[*group_index].start();
                            let indices = start..start + positions.len();
                            context.set_positions(indices, positions.iter());
                            Self::Activate.on_group(*group_index, context)?;
                        }
                        SpeciationAction::DeactivateGroup(group_index) => {
                            Self::Deactivate.on_group(*group_index, context)?;
                        }
                        SpeciationAction::SwapAtomKind {
                            group_index: _,
                            abs_index,
                            new_atom_id,
                        } => {
                            context.set_atom_kind(*abs_index, *new_atom_id);
                        }
                        SpeciationAction::ActivateAtom {
                            group_index,
                            position,
                        } => {
                            // Place atom at the first inactive slot and expand by one
                            let group = &context.groups()[*group_index];
                            let slot = group.start() + group.len();
                            context.set_positions(slot..slot + 1, [position]);
                            Self::Expand(1).on_group(*group_index, context)?;
                        }
                        SpeciationAction::DeactivateAtom {
                            group_index,
                            abs_index,
                        } => {
                            // Swap with last active to keep active atoms contiguous, then shrink
                            let group = &context.groups()[*group_index];
                            let last_active = group.start() + group.len() - 1;
                            if *abs_index != last_active {
                                context.swap_particles(*abs_index, last_active);
                            }
                            Self::Contract(1).on_group(*group_index, context)?;
                        }
                    }
                }
            }
            Self::ClusterTransform {
                groups,
                new_mass_centers,
                rotation,
            } => {
                for (&gi, &new_com) in groups.iter().zip(new_mass_centers) {
                    let old_com = context.groups()[gi]
                        .mass_center()
                        .copied()
                        .expect("cluster groups are molecular and have a mass center");
                    let indices = context.groups()[gi]
                        .select(&ParticleSelection::Active, context.topology_ref())?;
                    if let Some(q) = rotation {
                        // Rotate the molecule about its own mass center — PBC-safe because a molecule
                        // spans less than half the box. Rotation about the center leaves it invariant,
                        // so the subsequent shift alone places the mass center at its cluster target.
                        context.rotate_particles(&indices, q, Some(-old_com));
                        context.groups_mut()[gi].rotate_by(q);
                    }
                    context.translate_particles(&indices, &(new_com - old_com));
                    context.update_mass_center(gi);
                }
            }
            _ => {
                todo!("Implement other system-wide transforms")
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    fn rotate_random<'a>(
        positions: impl IntoIterator<Item = &'a mut Point>,
        center: &Point,
        rng: &mut impl Rng,
    ) {
        let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let axis = random_unit_vector(rng);
        let matrix = nalgebra::Rotation3::new(axis * angle);
        let rotate = |pos: &mut Point| *pos = matrix * (*pos - center) + center;
        positions.into_iter().for_each(rotate);
    }

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
            rngsum += rng.r#gen::<f64>();
        }
        assert_approx_eq!(f64, x_mean / n as f64, 0.0, epsilon = 0.025);
        assert_approx_eq!(f64, y_mean / n as f64, 0.0, epsilon = 0.025);
        assert_approx_eq!(f64, z_mean / n as f64, 0.0, epsilon = 0.025);
        assert_approx_eq!(f64, rngsum / n as f64, 0.5, epsilon = 0.01);
    }

    #[test]
    fn rotate_updates_group_quaternion() {
        use crate::backend::Backend;
        use crate::group::GroupCollection;
        let mut rng = rand::thread_rng();
        let mut context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(std::path::Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        // Pick a molecular group (index 1 has multiple atoms)
        let group_index = 1;
        assert!(context.groups()[group_index].len() > 1);

        let axis = nalgebra::UnitVector3::new_normalize(Point::new(0.0, 0.0, 1.0));
        let q1 = UnitQuaternion::from_axis_angle(&axis, 0.5);
        let transform = Transform::Rotate(q1);
        transform.on_group(group_index, &mut context).unwrap();
        assert!(context.groups()[group_index].quaternion().angle_to(&q1) < 1e-12);

        // Second rotation composes
        let q2 = UnitQuaternion::from_axis_angle(&axis, 0.3);
        let transform2 = Transform::Rotate(q2);
        transform2.on_group(group_index, &mut context).unwrap();
        let expected = q2 * q1;
        assert!(
            context.groups()[group_index]
                .quaternion()
                .angle_to(&expected)
                < 1e-12
        );
    }

    /// A change of internal geometry leaves the group's rigid-body orientation alone.
    #[test]
    fn set_positions_does_not_update_quaternion() {
        use crate::backend::Backend;
        use crate::group::{GroupCollection, ParticleSelection};
        let mut rng = rand::thread_rng();
        let mut context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(std::path::Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        use crate::context::WithTopology;
        let group_index = 1;
        let selection = ParticleSelection::Active;
        let indices = context.groups()[group_index]
            .select(&selection, context.topology_ref())
            .unwrap();
        let positions = vec![Point::new(1.0, 2.0, 3.0); indices.len()];
        let transform = Transform::SetPositions(positions, selection);
        transform.on_group(group_index, &mut context).unwrap();
        assert_eq!(
            *context.groups()[group_index].quaternion(),
            UnitQuaternion::identity()
        );
    }

    #[test]
    fn test_rotate_random() {
        let positions = [
            Point::new(10.4, 11.3, 12.8),
            Point::new(7.3, 9.3, 2.6),
            Point::new(9.3, 10.1, 17.2),
        ];
        let masses = [1.46, 2.23, 10.73];
        let com = crate::geometry::mass_center(&positions, &masses);

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let mut cloned = positions;

            rotate_random(&mut cloned, &com, &mut rng);

            for (original, new) in positions.iter().zip(cloned.iter()) {
                assert_ne!(original, new);
            }

            let com_rotated = crate::geometry::mass_center(&cloned, &masses);
            assert_approx_eq!(f64, com.x, com_rotated.x);
            assert_approx_eq!(f64, com.y, com_rotated.y);
            assert_approx_eq!(f64, com.z, com_rotated.z);
        }
    }
}
