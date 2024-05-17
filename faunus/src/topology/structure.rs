// Copyright 2023-2024 Mikael Lund
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

//! Loading molecular structures.

use std::path::Path;

use crate::cell::SimulationCell;
use crate::Point;

/// Obtain positions of particles from the provided structure file using the `chemfiles` crate.
#[cfg(feature = "chemfiles")]
pub(crate) fn positions_from_structure_file(
    filename: &impl AsRef<Path>,
    cell: Option<&impl SimulationCell>,
) -> anyhow::Result<Vec<Point>> {
    Ok(super::chemfiles_interface::positions_from_frame(
        &super::chemfiles_interface::frame_from_file(filename)?,
        cell,
    ))
}

#[cfg(not(feature = "chemfiles"))]
pub(crate) fn positions_from_structure_file(
    filename: &impl AsRef<Path>,
    cell: Option<&impl SimulationCell>,
) -> anyhow::Result<Vec<Point>> {
    todo!("Not implemented. Use the `chemfiles` feature.")
}
