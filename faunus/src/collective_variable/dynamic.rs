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

//! Dynamic selection collective variables: Count, Concentration, Charge.
//!
//! These CVs re-resolve selections at each evaluation, allowing them to track
//! changing particle counts (e.g., grand canonical ensemble).
//! They are self-building since no build-time resolution is needed.

use super::{impl_self_building_cv, CvKind, EvalContext};
use crate::cell::Shape;
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
                context.atom_kind(i)
            })
            .len() as f64
    }

    fn name(&self) -> &'static str {
        "Count"
    }
}

impl_self_building_cv!(Count, "count", |s| Some(format!(
    "selection: {}",
    s.selection
)));

// ---------------------------------------------------------------------------
// Concentration (self-building)
// ---------------------------------------------------------------------------

/// Molar concentration (mol/L) of active atoms matching a selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molarity {
    pub selection: Selection,
}

#[typetag::serde(name = "molarity")]
impl CvKind for Molarity {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let n = self
            .selection
            .resolve_atoms_live(context.topology_ref(), context.groups(), &|i| {
                context.atom_kind(i)
            })
            .len() as f64;
        let volume = context.cell().volume().unwrap_or(f64::INFINITY);
        n / (volume * crate::MOLAR_TO_INV_ANGSTROM3)
    }

    fn name(&self) -> &'static str {
        "Molarity"
    }
}

impl_self_building_cv!(Molarity, "molarity", |s| Some(format!(
    "selection: {}",
    s.selection
)));

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
                    context.atom_kind(i)
                });
        let atomkinds = context.topology_ref().atomkinds();
        indices
            .iter()
            .map(|&i| atomkinds[context.atom_kind(i)].charge())
            .sum()
    }

    fn name(&self) -> &'static str {
        "Charge"
    }
}

impl_self_building_cv!(Charge, "charge", |s| Some(format!(
    "selection: {}",
    s.selection
)));
