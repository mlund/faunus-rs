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

//! Torsion angles

use derive_getters::Getters;
use interatomic::threebody::{
    cosine::CosineTorsion, harmonic::HarmonicTorsion, ThreebodyAngleEnergy,
};
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{group::Group, Context};

use super::Indexed;

/// Force field definition for torsion, e.g. harmonic, cosine, etc.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum TorsionKind {
    /// Harmonic torsion.
    Harmonic(HarmonicTorsion),
    /// Cosine angle as used in e.g. GROMOS-96.
    Cosine(CosineTorsion),
    /// Unspecified torsion type.
    #[default]
    Unspecified,
}

/// Definition of torsion between three indexed atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Getters, Validate)]
#[serde(deny_unknown_fields)]
pub struct Torsion {
    /// Indices of the three atoms in the angle.
    /// The atoms are bonded as 1-2-3.
    #[validate(custom(function = "super::validate_unique_indices"))]
    index: [usize; 3],
    /// Kind of torsion, e.g. harmonic, cosine, etc.
    #[serde(default)]
    kind: TorsionKind,
}

impl Torsion {
    /// Create new torsion
    pub const fn new(index: [usize; 3], kind: TorsionKind) -> Self {
        Self { index, kind }
    }

    /// Check if torsion contains atom with index
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }

    /// Calculate energy of a torsion in a specific group.
    /// Returns 0.0 if any of the interacting particles is inactive.
    pub fn energy(&self, context: &impl Context, group: &Group) -> f64 {
        let indices = match self.index.map(|rel| group.to_absolute_index(rel)) {
            [Ok(i), Ok(j), Ok(k)] => [i, j, k],
            _ => return 0.0,
        };

        let angle = context.get_angle(&indices);
        self.threebody_angle_energy(angle)
    }

    /// Calculate energy of an intermolecular torsion.
    /// Returns 0.0 if any of the interacting particles is inactive.
    pub fn energy_intermolecular(
        &self,
        context: &impl Context,
        term: &crate::energy::IntermolecularBonded,
    ) -> f64 {
        // any of the particles is inactive
        if self.index.iter().any(|&i| !term.is_active(i)) {
            return 0.0;
        }

        let angle = context.get_angle(&self.index);
        self.threebody_angle_energy(angle)
    }
}

impl Indexed for Torsion {
    fn index(&self) -> &[usize] {
        &self.index
    }
}

impl ThreebodyAngleEnergy for Torsion {
    fn threebody_angle_energy(&self, angle: f64) -> f64 {
        match &self.kind {
            TorsionKind::Harmonic(x) => x.threebody_angle_energy(angle),
            TorsionKind::Cosine(x) => x.threebody_angle_energy(angle),
            TorsionKind::Unspecified => 0.0,
        }
    }
}
