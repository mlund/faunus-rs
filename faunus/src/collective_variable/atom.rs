// Copyright 2025 Mikael Lund
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

//! Single-atom collective variables: AtomPosition.

use super::{impl_single_atom_with_dim_builder, CvKind, EvalContext};
use crate::axes::Axes;
use crate::group::GroupCollection;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// AtomPosition
// ---------------------------------------------------------------------------

/// Position of a single atom, optionally projected onto an axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomPosition {
    #[serde(alias = "dimension")]
    projection: Axes,
    index: usize,
}

#[typetag::serde(name = "atom_position")]
impl CvKind for AtomPosition {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        self.projection
            .project(GroupCollection::position(context, self.index))
            .norm()
    }

    fn name(&self) -> &'static str {
        "AtomPosition"
    }
}

impl_single_atom_with_dim_builder!(AtomPosition, "atom_position", |projection, index| {
    AtomPosition { projection, index }
});
