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

//! Group-based collective variables: Size, EndToEnd, MassCenterPosition, MassCenterSeparation.
//!
//! These CVs resolve group indices at build time and evaluate against those fixed indices.

use super::{
    impl_single_group_builder, impl_single_group_with_dim_builder, CvKind, CvKindBuilder,
    EvalContext,
};
use crate::cell::BoundaryConditions;
use crate::dimension::Dimension;
use crate::selection::Selection;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Size
// ---------------------------------------------------------------------------

/// Number of active atoms in a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Size {
    group: usize,
}

#[typetag::serde(name = "size")]
impl CvKind for Size {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        context.groups()[self.group].len() as f64
    }

    fn name(&self) -> &'static str {
        "Size"
    }
}

impl_single_group_builder!(Size, "size", |group| Size { group });

// ---------------------------------------------------------------------------
// EndToEnd
// ---------------------------------------------------------------------------

/// Distance between first and last active atoms of a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndToEnd {
    dimension: Dimension,
    group: usize,
}

#[typetag::serde(name = "end_to_end")]
impl CvKind for EndToEnd {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let g = &context.groups()[self.group];
        let active = g.iter_active();
        let first = active.start;
        let last = active.end.saturating_sub(1);
        if first >= active.end {
            return 0.0;
        }
        self.dimension
            .filter(context.get_distance(first, last))
            .norm()
    }

    fn name(&self) -> &'static str {
        "EndToEnd"
    }
}

impl_single_group_with_dim_builder!(EndToEnd, "end_to_end", |dimension, group| EndToEnd {
    dimension,
    group
});

// ---------------------------------------------------------------------------
// MassCenterPosition
// ---------------------------------------------------------------------------

/// Position of the mass center of a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassCenterPosition {
    dimension: Dimension,
    group: usize,
}

#[typetag::serde(name = "mass_center_position")]
impl CvKind for MassCenterPosition {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        context.groups()[self.group]
            .mass_center()
            .map(|com| self.dimension.filter(*com).norm())
            .unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "MassCenterPosition"
    }
}

impl_single_group_with_dim_builder!(
    MassCenterPosition,
    "mass_center_position",
    |dimension, group| MassCenterPosition { dimension, group }
);

// ---------------------------------------------------------------------------
// MassCenterSeparation (two groups - manual impl)
// ---------------------------------------------------------------------------

/// Distance between mass centers of two groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassCenterSeparation {
    dimension: Dimension,
    group1: usize,
    group2: usize,
}

#[typetag::serde(name = "mass_center_separation")]
impl CvKind for MassCenterSeparation {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let groups = context.groups();
        match (
            groups[self.group1].mass_center(),
            groups[self.group2].mass_center(),
        ) {
            (Some(a), Some(b)) => self.dimension.filter(context.cell().distance(a, b)).norm(),
            _ => 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "MassCenterSeparation"
    }
}

/// Builder for MassCenterSeparation CV (two selections).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassCenterSeparationBuilder {
    pub selection: Selection,
    pub selection2: Selection,
    #[serde(default)]
    pub dimension: Dimension,
}

#[typetag::serde(name = "mass_center_separation")]
impl CvKindBuilder for MassCenterSeparationBuilder {
    fn build(&self, context: &dyn EvalContext) -> Result<Box<dyn CvKind>> {
        let indices1 = self
            .selection
            .resolve_groups(context.topology_ref(), context.groups());
        if indices1.len() != 1 {
            bail!(
                "MassCenterSeparation: selection '{}' must match exactly one group, found {}",
                self.selection,
                indices1.len()
            );
        }

        let indices2 = self
            .selection2
            .resolve_groups(context.topology_ref(), context.groups());
        if indices2.len() != 1 {
            bail!(
                "MassCenterSeparation: selection2 '{}' must match exactly one group, found {}",
                self.selection2,
                indices2.len()
            );
        }

        Ok(Box::new(MassCenterSeparation {
            dimension: self.dimension,
            group1: indices1[0],
            group2: indices2[0],
        }))
    }
}
