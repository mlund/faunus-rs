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

use super::{CvKind, CvKindBuilder, EvalContext};
use crate::axes::Axes;
use crate::group::GroupCollection;
use crate::selection::Selection;
use serde::{Deserialize, Serialize};

/// Position of a single atom, optionally projected onto an axis.
///
/// The selection is resolved live each evaluation so that speciation and
/// GCMC moves are handled correctly. Returns NaN if the selection does
/// not match exactly one active atom.
///
/// For single-axis projections (x, y, z) returns the signed component;
/// for multi-axis (xy, xz, yz, xyz) returns the Euclidean norm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomPosition {
    #[serde(alias = "dimension")]
    projection: Axes,
    selection: Selection,
}

#[typetag::serde(name = "atom_position")]
impl CvKind for AtomPosition {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let indices =
            self.selection
                .resolve_atoms_live(context.topology_ref(), context.groups(), &|i| {
                    context.atom_kind(i)
                });
        if indices.len() != 1 {
            return f64::NAN;
        }
        let pos = GroupCollection::position(context, indices[0]);
        match self.projection {
            Axes::X => pos.x,
            Axes::Y => pos.y,
            Axes::Z => pos.z,
            other => other.project(pos).norm(),
        }
    }

    fn name(&self) -> &'static str {
        "AtomPosition"
    }
}

/// Builder for AtomPosition CV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomPositionBuilder {
    pub selection: Selection,
    #[serde(default, alias = "dimension")]
    pub projection: Axes,
}

#[typetag::serde(name = "atom_position")]
impl CvKindBuilder for AtomPositionBuilder {
    fn build(&self, context: &dyn EvalContext) -> anyhow::Result<Box<dyn CvKind>> {
        let indices =
            self.selection
                .resolve_atoms_live(context.topology_ref(), context.groups(), &|i| {
                    context.atom_kind(i)
                });
        // Warn rather than error: speciation/GCMC can change the active count later
        if indices.len() != 1 {
            log::warn!(
                "atom_position: selection '{}' matched {} atoms (expected 1); \
                 will return NaN until exactly one matches",
                self.selection,
                indices.len()
            );
        }
        Ok(Box::new(AtomPosition {
            projection: self.projection,
            selection: self.selection.clone(),
        }))
    }

    fn description(&self) -> Option<String> {
        Some(format!(
            "selection: {}, projection: {:?}",
            self.selection, self.projection
        ))
    }
}
