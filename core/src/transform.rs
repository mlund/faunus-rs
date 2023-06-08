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

use crate::{
    cell::BoundaryConditions, cell::VolumeScalePolicy, group::ParticleSelection, Point,
    PointParticle,
};
use anyhow::Ok;
use nalgebra::Quaternion;
use rand::prelude::*;

/// Generate a random unit vector by sphere picking
pub fn random_unit_vector(rng: &mut ThreadRng) -> Point {
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

impl Transform {
    /// Transform a set of particles using a transformation
    ///
    /// The transformation is applied to the particles in a single group,
    /// given by `group_index`, in the `context`.
    pub fn on_group(
        &self,
        context: &mut impl crate::Context,
        group_index: usize,
    ) -> Result<(), anyhow::Error> {
        match self {
            Transform::Translate(displacement) => {
                let group_len = context.groups()[group_index].len();
                Self::PartialTranslate(*displacement, (0..group_len).collect())
                    .on_group(context, group_index)?;
            }
            Transform::PartialTranslate(displacement, indices) => {
                let indices = context.groups()[group_index]
                    .select(&ParticleSelection::RelIndex(indices.clone()))
                    .unwrap();
                let mut particles = context.get_particles(indices.clone());
                let positions = particles.iter_mut().map(|p| p.pos_mut());
                translate(context.cell(), positions, displacement);
                context.set_particles(indices, particles.iter())?
            }
            _ => {
                todo!("Implement other transforms")
            }
        }
        Ok(())
    }
}

/// Translates a set of particles by a vector and applies periodic boundary conditions
fn translate<'a>(
    pbc: &impl BoundaryConditions,
    positions: impl Iterator<Item = &'a mut Point>,
    displacement: &Point,
) {
    positions.for_each(|pos| {
        *pos += displacement;
        pbc.boundary(pos);
    });
}
