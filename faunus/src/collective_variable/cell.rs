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

//! Cell geometry collective variables: Volume, BoxLength.

use super::{impl_self_building_cv, CvKind, CvKindBuilder, EvalContext};
use crate::cell::Shape;
use crate::dimension::Dimension;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Volume (self-building: no resolution needed)
// ---------------------------------------------------------------------------

/// Volume of the simulation cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Volume;

#[typetag::serde(name = "volume")]
impl CvKind for Volume {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        context.cell().volume().unwrap_or(f64::INFINITY)
    }

    fn name(&self) -> &'static str {
        "Volume"
    }
}

impl_self_building_cv!(Volume, "volume");

// ---------------------------------------------------------------------------
// BoxLength (needs dimension → component resolution)
// ---------------------------------------------------------------------------

/// Length of a simulation box edge along a single axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxLength {
    component: usize,
}

#[typetag::serde(name = "box_length")]
impl CvKind for BoxLength {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        context
            .cell()
            .bounding_box()
            .map(|bb| 2.0 * bb[self.component])
            .unwrap_or(f64::INFINITY)
    }

    fn name(&self) -> &'static str {
        "BoxLength"
    }
}

/// Builder for BoxLength CV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxLengthBuilder {
    #[serde(default)]
    pub dimension: Dimension,
}

#[typetag::serde(name = "box_length")]
impl CvKindBuilder for BoxLengthBuilder {
    fn build(&self, _context: &dyn EvalContext) -> Result<Box<dyn CvKind>> {
        let component = match self.dimension {
            Dimension::X => 0,
            Dimension::Y => 1,
            Dimension::Z => 2,
            _ => bail!("BoxLength requires dimension x, y, or z"),
        };
        Ok(Box::new(BoxLength { component }))
    }
}
