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

use serde::{Deserialize, Serialize};

/// Force field definition for dihedral angle potentials between four atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum DihedralKind {
    /// Harmonic dihedral type (force constant, equilibrium angle)
    Harmonic(f64, f64),
    /// Proper periodic dihedral type (force constant, periodicity, phase)
    ProperPeriodic(f64, f64, f64),
    /// Improper harmonic dihedral type (force constant, equilibrium angle)
    ImproperHarmonic(f64, f64),
    /// Amber-style improper torsion, where atom3 is the central atom bonded to atoms 1, 2, and 4.
    /// Atoms 1, 2, and 4 are only bonded to atom 3 in this instance.
    /// (force constant, periodicity, phase)
    ImproperAmber(f64, f64, f64),
    /// CHARMM-style improper torsion between 4 atoms. The first atom must be the central atom. (force constant, periodicity, phase)
    ImproperCHARMM(f64, f64, f64),
    /// Unspecified dihedral type
    #[default]
    None,
}

impl DihedralKind {
    /// Check if dihedral is improper
    pub fn is_improper(&self) -> bool {
        matches!(
            self,
            DihedralKind::ImproperHarmonic(..)
                | DihedralKind::ImproperAmber(..)
                | DihedralKind::ImproperCHARMM(..)
        )
    }
}

/// Valence dihedral between four atoms separated by three covalent bonds.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Dihedral {
    /// Indices of the four atoms in the dihedral.
    /// The indices are bonded as 1-2-3-4.
    pub index: [usize; 4],
    /// Kind of dihedral, e.g. harmonic, proper periodic, improper harmonic, etc.
    pub kind: DihedralKind,
    /// Optional 1-4 electrostatic scaling factor
    pub electrostatic_scaling: Option<f64>,
    /// Optional 1-4 Lennard-Jones scaling factor
    pub lj_scaling: Option<f64>,
}

impl Dihedral {
    /// Create new dihedral
    pub fn new(index: [usize; 4], kind: DihedralKind) -> Self {
        Self {
            index,
            kind,
            electrostatic_scaling: None,
            lj_scaling: None,
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
}
