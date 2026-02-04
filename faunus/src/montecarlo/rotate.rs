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

use super::MoveStatistics;
use crate::montecarlo;
use crate::propagate::Displacement;
use crate::transform::{random_unit_vector, Transform};
use crate::{Change, Context, GroupChange};
use nalgebra::UnitVector3;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Move for translating a random molecule.
///
/// This will pick a random molecule of type `molecule_id` and translate it by a random displacement.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotateMolecule {
    /// Name of the molecule type to translate.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Id of the molecule type to translate.
    #[serde(skip)]
    molecule_id: usize,
    /// Maximum angular displacement (radians).
    #[serde(alias = "dp")]
    max_displacement: f64,
    /// Move selection weight.
    weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    repeat: usize,
    /// Move statisticcs.
    #[serde(skip_deserializing)]
    statistics: MoveStatistics,
}

impl RotateMolecule {
    /// Create a new `TranslateMolecule` move.
    pub fn new(
        molecule_name: &str,
        molecule_id: usize,
        max_displacement: f64,
        weight: f64,
        repeat: usize,
    ) -> Self {
        Self {
            molecule_name: molecule_name.to_owned(),
            molecule_id,
            max_displacement,
            weight,
            repeat,
            statistics: MoveStatistics::default(),
        }
    }

    /// Propose a translation of a molecule.
    ///
    /// Translates molecule in given `context` and return a change object
    /// describing the change as well as the magnitude of the displacement.
    pub(crate) fn propose_move(
        &mut self,
        context: &mut impl Context,
        rng: &mut impl Rng,
    ) -> Option<(Change, Displacement)> {
        match montecarlo::random_group(context, rng, self.molecule_id) {
            Some(group_index) => {
                let axis = random_unit_vector(rng);
                let uaxis = UnitVector3::new_normalize(axis);
                let angle = self.max_displacement * 2.0 * (rng.gen::<f64>() - 0.5);
                let quaternion = crate::UnitQuaternion::from_axis_angle(&uaxis, angle);
                Transform::Rotate(axis, quaternion)
                    .on_group(group_index, context)
                    .unwrap();
                Some((
                    Change::SingleGroup(group_index, GroupChange::RigidBody),
                    Displacement::Angle(angle),
                ))
            }
            None => None,
        }
    }

    /// Get immutable reference to the statistics of the move.
    pub(crate) const fn get_statistics(&self) -> &MoveStatistics {
        &self.statistics
    }

    /// Get mutable reference to the statistics of the move.
    pub(crate) const fn get_statistics_mut(&mut self) -> &mut MoveStatistics {
        &mut self.statistics
    }

    /// Get weight of the move.
    pub(crate) const fn weight(&self) -> f64 {
        self.weight
    }

    /// Number of times the move should be repeated if selected.
    pub(crate) const fn repeat(&self) -> usize {
        self.repeat
    }

    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.molecule_id = context
            .topology()
            .moleculekinds()
            .iter()
            .position(|x| x.name() == &self.molecule_name)
            .ok_or(anyhow::Error::msg(
                "Molecule kind in the definition of 'RotateMolecule' move does not exist.",
            ))?;
        Ok(())
    }
}

impl crate::Info for RotateMolecule {
    fn short_name(&self) -> Option<&'static str> {
        Some("rotate_molecule")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Rigid body rotation of random molecule")
    }
}
