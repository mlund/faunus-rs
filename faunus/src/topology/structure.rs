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

use super::io;
use crate::cell::SimulationCell;
use crate::Point;
use nalgebra::Vector3;

/// Obtain positions of particles from the provided structure file.
///
/// If a simulation cell is given, coordinates are shifted by `-0.5 * bounding_box`
/// to convert from file convention (corner origin) to Faunus convention (center origin).
pub(crate) fn positions_from_structure_file(
    filename: &impl AsRef<Path>,
    cell: Option<&impl SimulationCell>,
) -> anyhow::Result<Vec<Point>> {
    let data = io::read_structure(filename)?;

    let shift = cell.map_or_else(Vector3::default, |cell| {
        cell.bounding_box()
            .map_or_else(Vector3::default, |b| -0.5 * b)
    });

    let positions = data.positions.into_iter().map(|pos| pos + shift).collect();
    Ok(positions)
}
