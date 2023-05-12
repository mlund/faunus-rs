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

use crate::{cell::SimulationCell, Point, PointParticle};
use anyhow::Ok;
use nalgebra::Quaternion;
use serde::{Deserialize, Serialize};

/// Policies for how to scale a volume
///
/// This is used to scale a volume to a new volume. Each variant
/// takes a tuple of (old_volume, new_volume) and scales a given point.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum VolumeScalePolicy {
    /// Isotropic scaling (equal scaling in all directions)
    Isotropic,
    /// Isochoric scaling along z (constant volume)
    IsochoricZ,
    /// Scale along z-axis only
    ScaleZ,
    /// Scale along x and y
    ScaleXY,
}

/// Trait for scaling a point according to a policy.
/// Typically implemented for a unit cell, othorhombic, triclinic, etc.
pub trait VolumeScale {
    /// Scale a point according to the policy
    fn volume_scale(
        &self,
        policy: VolumeScalePolicy,
        old_volume: f64,
        new_volume: f64,
        point: &mut Point,
    ) -> Result<(), anyhow::Error>;
}

impl VolumeScale for crate::cell::UnitCell {
    fn volume_scale(
        &self,
        policy: VolumeScalePolicy,
        old_volume: f64,
        new_volume: f64,
        point: &mut Point,
    ) -> Result<(), anyhow::Error> {
        if self.shape() != crate::cell::CellShape::Orthorhombic {
            return Err(anyhow::Error::msg(
                "Currently only orthorhombic cells are supported for volume scaling",
            ));
        }
        match policy {
            VolumeScalePolicy::Isotropic => {
                point.scale_mut((new_volume / old_volume).cbrt());
            }
            VolumeScalePolicy::IsochoricZ => {
                let factor = (new_volume / old_volume).cbrt();
                point.x = factor;
                point.y = factor;
                point.z = factor.powi(2).recip();
            }
            VolumeScalePolicy::ScaleZ => {
                point.z *= new_volume / old_volume;
            }
            VolumeScalePolicy::ScaleXY => {
                let factor = (new_volume / old_volume).sqrt();
                point.x *= factor;
                point.y *= factor;
            }
        }
        Ok(())
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
    PartialTranslate(Point, Vec<usize>),
    /// Use a quaternion to rotatate around a given point
    Rotate(Point, Quaternion<f64>),
    /// Use a quaternion to rotatate a set of particles around a given point
    PartialRotate(Point, Quaternion<f64>, Vec<usize>),
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

/// Transform a set of particles using a transformation
///
/// The transformation is applied to the particles in a single group,
/// given by `group_index`, in the `context`.
pub fn transform(
    context: &mut impl crate::Context,
    group_index: usize,
    transformation: &Transform,
) -> Result<(), anyhow::Error> {
    match transformation {
        Transform::Translate(displacement) => {
            let group_len = context.groups()[group_index].len();
            transform(
                context,
                group_index,
                &Transform::PartialTranslate(*displacement, (0..group_len).collect()),
            )?
        }
        Transform::PartialTranslate(displacement, indices) => {
            let mut particles =
                context.group_particles_partial(group_index, indices.iter().copied());
            let positions = particles.iter_mut().map(|p| p.pos_mut());
            partial_transform(context, positions, displacement);
            context.set_particles_partial(group_index, particles.iter(), indices.iter().copied())?
        }
        _ => {
            todo!("Implement other transforms")
        }
    }
    Ok(())
}

/// Translates a set of particles by a vector and applies periodic boundary conditions
fn partial_transform<'a>(
    cell: &impl SimulationCell,
    positions: impl Iterator<Item = &'a mut Point>,
    displacement: &Point,
) {
    positions.for_each(|pos| {
        *pos += displacement;
        cell.boundary(pos);
    });
}
