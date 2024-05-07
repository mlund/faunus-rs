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

//! Dihedral angles

use derive_getters::Getters;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::Indexed;

/// Force field definition for dihedral angle potentials between four atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum DihedralKind {
    /// Harmonic dihedral type (force constant, equilibrium angle).
    Harmonic { k: f64, aeq: f64 },
    /// Proper periodic dihedral type (force constant, periodicity, phase).
    ProperPeriodic { k: f64, n: f64, phi: f64 },
    /// Improper harmonic dihedral type (force constant, equilibrium angle).
    ImproperHarmonic { k: f64, aeq: f64 },
    /// Amber-style improper torsion, where atom3 is the central atom bonded to atoms 1, 2, and 4.
    /// Atoms 1, 2, and 4 are only bonded to atom 3 in this instance.
    /// (force constant, periodicity, phase).
    ImproperAmber { k: f64, n: f64, phi: f64 },
    /// CHARMM-style improper torsion between 4 atoms. The first atom must be the central atom. (force constant, periodicity, phase).
    ImproperCHARMM { k: f64, n: f64, phi: f64 },
    /// Unspecified dihedral type.
    #[default]
    Unspecified,
}

impl DihedralKind {
    /// Check if dihedral is improper
    pub fn is_improper(&self) -> bool {
        matches!(
            self,
            DihedralKind::ImproperHarmonic { .. }
                | DihedralKind::ImproperAmber { .. }
                | DihedralKind::ImproperCHARMM { .. }
        )
    }
}

/// Valence dihedral between four atoms separated by three covalent bonds.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Getters, Validate)]
#[serde(deny_unknown_fields)]
pub struct Dihedral {
    /// Indices of the four atoms in the dihedral.
    /// The indices are bonded as 1-2-3-4.
    #[validate(custom(function = "super::validate_unique_indices"))]
    index: [usize; 4],
    /// Kind of dihedral, e.g. harmonic, proper periodic, improper harmonic, etc.
    #[serde(default)]
    kind: DihedralKind,
    /// Optional 1-4 electrostatic scaling factor
    electrostatic_scaling: Option<f64>,
    /// Optional 1-4 Lennard-Jones scaling factor
    lj_scaling: Option<f64>,
}

impl Dihedral {
    /// Create new dihedral
    pub fn new(
        index: [usize; 4],
        kind: DihedralKind,
        electrostatic_scaling: Option<f64>,
        lj_scaling: Option<f64>,
    ) -> Self {
        Self {
            index,
            kind,
            electrostatic_scaling,
            lj_scaling,
        }
    }

    /// Determines if the dihedral contains the given atom index
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }

    /// Determines if the dihedral is improper
    pub fn is_improper(&self) -> bool {
        self.kind.is_improper()
    }

    /// Shift all indices by a given offset
    pub fn shift(&mut self, offset: isize) {
        for i in &mut self.index {
            *i = i.checked_add_signed(offset).unwrap();
        }
    }
}

impl Indexed for Dihedral {
    fn index(&self) -> &[usize] {
        &self.index
    }
}
