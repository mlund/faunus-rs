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
use crate::topology::GroupKind;
use serde::{Deserialize, Serialize};

/// Count matching entities using the same convention as speciation and `count_active`:
/// one COM per Molecular group, atom count for Atomic/Reservoir groups.
fn count_by_group_kind(selection: &Selection, context: &dyn EvalContext) -> f64 {
    let topology = context.topology_ref();
    let groups = context.groups();
    selection
        .resolve_groups_live(topology, groups, &|i| context.atom_kind(i))
        .iter()
        .map(
            |&gi| match topology.moleculekinds()[groups[gi].molecule()].group_kind() {
                GroupKind::Molecular => 1.0,
                GroupKind::Atomic | GroupKind::Reservoir => groups[gi].len() as f64,
            },
        )
        .sum()
}

// ---------------------------------------------------------------------------
// Count (self-building)
// ---------------------------------------------------------------------------

/// Number of active entities matching a selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Count {
    pub selection: Selection,
}

#[typetag::serde(name = "count")]
impl CvKind for Count {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        count_by_group_kind(&self.selection, context)
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

/// Molar concentration (mol/L) of active entities matching a selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molarity {
    pub selection: Selection,
}

#[typetag::serde(name = "molarity")]
impl CvKind for Molarity {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let volume = context.cell().volume().unwrap_or(f64::INFINITY);
        count_by_group_kind(&self.selection, context) / (volume * crate::MOLAR_TO_INV_ANGSTROM3)
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
        let topology = context.topology_ref();
        self.selection
            .resolve_atoms_live(topology, context.groups(), &|i| context.atom_kind(i))
            .iter()
            .map(|&i| topology.atomkinds()[context.atom_kind(i)].charge())
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
