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

use serde::{Deserialize, Serialize};

/// Force field definition for torsion, e.g. harmonic, cosine, etc.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum TorsionKind {
    /// Harmonic torsion (force constant, equilibrium angle)
    Harmonic { k: f64, aeq: f64 },
    /// Cosine angle as used in e.g. GROMOS-96 (force constant, equilibrium angle)
    Cosine { k: f64, aeq: f64 },
    /// Unspecified torsion type
    #[default]
    Unspecified,
}

/// Definition of torsion between three indexed atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Torsion {
    /// Indices of the three atoms in the angle
    pub index: [usize; 3],
    /// Kind of torsion, e.g. harmonic, cosine, etc.
    pub kind: TorsionKind,
}

impl Torsion {
    /// Create new torsion
    pub fn new(index: [usize; 3], kind: TorsionKind) -> Self {
        Self { index, kind }
    }

    /// Check if torsion contains atom with index
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }

    /// Shift all indices by a given offset
    pub fn shift(&mut self, offset: isize) {
        for i in &mut self.index {
            *i = i.checked_add_signed(offset).unwrap();
        }
    }
}
