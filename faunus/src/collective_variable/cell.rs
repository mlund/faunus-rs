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

//! Cell geometry collective variable: Volume.

use super::{CvKind, CvKindBuilder, EvalContext};
use crate::cell::Shape;
use crate::dimension::Dimension;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Volume
// ---------------------------------------------------------------------------

/// Cell volume, area, or length depending on `dimension`.
///
/// - `xyz` (default): true cell volume via `Shape::volume()`
/// - Sub-dimensions (e.g. `z`, `xy`): derived as `volume / orthogonal_lengths`.
///   For a cylinder with `xy`, this gives πr² (not the bounding rectangle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Volume {
    dimension: Dimension,
}

/// Compute the effective measure (volume, area, or length) for the given dimension.
///
/// - 3D (`XYZ`): true cell volume
/// - 2D (e.g. `XY`): volume divided by the orthogonal bounding-box length,
///   giving geometry-correct areas (e.g. πr² for a cylinder cross-section)
/// - 1D (e.g. `Z`): bounding-box length along that axis
fn effective_measure(cell: &crate::cell::Cell, dimension: Dimension) -> Option<f64> {
    match dimension.ndim() {
        3 => cell.volume(),
        2 => {
            let volume = cell.volume()?;
            let bb = cell.bounding_box()?;
            let orthogonal_length = dimension.complement().effective_volume(bb);
            (orthogonal_length > 0.0).then(|| volume / orthogonal_length)
        }
        1 => cell.bounding_box().map(|bb| dimension.effective_volume(bb)),
        _ => None,
    }
}

#[typetag::serde(name = "volume")]
impl CvKind for Volume {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        effective_measure(context.cell(), self.dimension).unwrap_or(f64::INFINITY)
    }

    fn name(&self) -> &'static str {
        "Volume"
    }
}

/// Builder for Volume CV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeBuilder {
    #[serde(default)]
    pub dimension: Dimension,
}

#[typetag::serde(name = "volume")]
impl CvKindBuilder for VolumeBuilder {
    fn build(&self, _context: &dyn EvalContext) -> Result<Box<dyn CvKind>> {
        Ok(Box::new(Volume {
            dimension: self.dimension,
        }))
    }
}
