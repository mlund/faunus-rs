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

//! Dynamic selection collective variables: Count, Charge.
//!
//! These CVs re-resolve selections at each evaluation, allowing them to track
//! changing particle counts (e.g., grand canonical ensemble).
//! They are self-building since no build-time resolution is needed.

use super::{impl_self_building_cv, CvKind, EvalContext};
use crate::selection::Selection;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Count (self-building)
// ---------------------------------------------------------------------------

/// Number of active atoms matching a selection (re-resolves each evaluation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Count {
    pub selection: Selection,
}

#[typetag::serde(name = "count")]
impl CvKind for Count {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        self.selection
            .resolve_atoms_live(context.topology_ref(), context.groups(), &|i| {
                context.get_atomkind(i)
            })
            .len() as f64
    }

    fn name(&self) -> &'static str {
        "Count"
    }
}

impl_self_building_cv!(Count, "count");

// ---------------------------------------------------------------------------
// Charge (self-building)
// ---------------------------------------------------------------------------

/// Sum of charges of active atoms matching a selection (re-resolves each evaluation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Charge {
    pub selection: Selection,
}

#[typetag::serde(name = "charge")]
impl CvKind for Charge {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let indices =
            self.selection
                .resolve_atoms_live(context.topology_ref(), context.groups(), &|i| {
                    context.get_atomkind(i)
                });
        let atomkinds = context.topology_ref().atomkinds();
        indices
            .iter()
            .map(|&i| atomkinds[context.get_atomkind(i)].charge())
            .sum()
    }

    fn name(&self) -> &'static str {
        "Charge"
    }
}

impl_self_building_cv!(Charge, "charge");
