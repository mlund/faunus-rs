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

//! Bonds between atoms

use float_cmp::approx_eq;
use serde::{Deserialize, Serialize};

/// Force field definition for bonds, e.g. harmonic, FENE, Morse, etc.
///
/// Each varient stores the parameters for the bond type, like force constant, equilibrium distance, etc.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum BondKind {
    /// Harmonic bond type (force constant, equilibrium distance).
    /// See <https://en.wikipedia.org/wiki/Harmonic_oscillator>.
    Harmonic(f64, f64),
    /// Finite extensible nonlinear elastic bond type (force constant, equilibrium distance, maximum distance)
    /// See <https://en.wikipedia.org/wiki/Finitely_extensible_nonlinear_elastic_potential>.
    FENE(f64, f64, f64),
    /// Morse bond type (force constant, equilibrium distance, depth of potential well).
    /// See <https://en.wikipedia.org/wiki/Morse_potential>.
    Morse(f64, f64, f64),
    /// Harmonic Urey-Bradley bond type (force constant, equilibrium distance)
    /// See <https://manual.gromacs.org/documentation/current/reference-manual/functions/bonded-interactions.html#urey-bradley-potential>
    /// for more information.
    UreyBradley(f64, f64),
    /// Undefined bond type
    #[default]
    Unspecified,
}

/// Bond order describing the multiplicity of a bond between two atoms.
///
/// See <https://en.wikipedia.org/wiki/Bond_order> for more information.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub enum BondOrder {
    #[default]
    /// Undefined bond order
    Unspecified,
    /// Single bond, e.g. diatomic hydrogen, H–H
    Single,
    /// Double bond, e.g. diatomic oxygen, O=O
    Double,
    /// Triple bond, e.g. diatomic nitrogen, N≡N
    Triple,
    Quadruple,
    Quintuple,
    Sextuple,
    Amide,
    Aromatic,
    Custom(f64),
}

impl From<BondOrder> for f64 {
    fn from(value: BondOrder) -> Self {
        match value {
            BondOrder::Unspecified => 0.0,
            BondOrder::Single => 1.0,
            BondOrder::Double => 2.0,
            BondOrder::Triple => 3.0,
            BondOrder::Quadruple => 4.0,
            BondOrder::Quintuple => 5.0,
            BondOrder::Sextuple => 6.0,
            BondOrder::Amide => 1.25,
            BondOrder::Aromatic => 1.5,
            BondOrder::Custom(value) => value,
        }
    }
}

impl From<f64> for BondOrder {
    fn from(value: f64) -> Self {
        match value {
            x if approx_eq!(f64, x, 0.0) => BondOrder::Unspecified,
            x if approx_eq!(f64, x, 1.0) => BondOrder::Single,
            x if approx_eq!(f64, x, 2.0) => BondOrder::Double,
            x if approx_eq!(f64, x, 3.0) => BondOrder::Triple,
            x if approx_eq!(f64, x, 4.0) => BondOrder::Quadruple,
            x if approx_eq!(f64, x, 5.0) => BondOrder::Quintuple,
            x if approx_eq!(f64, x, 6.0) => BondOrder::Sextuple,
            x if approx_eq!(f64, x, 1.25) => BondOrder::Amide,
            x if approx_eq!(f64, x, 1.5) => BondOrder::Aromatic,
            _ => BondOrder::Custom(value),
        }
    }
}

/// Describes a bond between two atoms
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Bond {
    /// Indices of the two atoms in the bond
    pub index: [usize; 2],
    /// Kind of bond, e.g. harmonic, FENE, Morse, etc.
    pub kind: BondKind,
    /// Bond order
    pub order: BondOrder,
}

impl Bond {
    /// Create new bond
    pub fn new(index: [usize; 2], kind: BondKind, order: Option<BondOrder>) -> Self {
        Self {
            index,
            kind,
            order: order.unwrap_or_default(),
        }
    }

    /// Create new bond where indices are offset by `shift`. Panics if overflow.
    pub fn shift(&self, shift: isize) -> Self {
        Self {
            index: [
                self.index[0].checked_add_signed(shift).unwrap(),
                self.index[1].checked_add_signed(shift).unwrap(),
            ],
            kind: self.kind.clone(),
            order: self.order.clone(),
        }
    }

    /// Check if bond contains atom with index
    pub fn contains(&self, index: usize) -> bool {
        self.index.contains(&index)
    }
}
