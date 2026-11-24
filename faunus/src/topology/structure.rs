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

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::io;
use crate::cell::SimulationCell;
use crate::Point;

/// Where a structure file has its coordinate origin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    /// At a box corner, as most structure files and the trajectory output have it.
    #[default]
    Corner,
    /// At the box centre, as Faunus cells are.
    Center,
}

/// Read particle positions from a structure file, in cell coordinates.
///
/// Positions beyond a wall are an error: left in place they give an infinite energy and a
/// run that rejects every move without saying why. A periodic overhang is kept, since
/// molecules made whole across a boundary are normal and minimum-image distances handle them.
pub fn positions_from_structure_file(
    filename: &impl AsRef<Path>,
    cell: &impl SimulationCell,
    origin: Origin,
) -> anyhow::Result<Vec<Point>> {
    let data = io::read_structure(filename)?;
    let shift = match origin {
        Origin::Corner => cell.bounding_box().map_or_else(Point::zeros, |b| -0.5 * b),
        Origin::Center => Point::zeros(),
    };
    let positions: Vec<Point> = data.positions.into_iter().map(|pos| pos + shift).collect();

    let outside = positions
        .iter()
        .filter(|pos| {
            let mut wrapped = **pos;
            cell.boundary(&mut wrapped);
            !cell.is_inside(&wrapped)
        })
        .count();
    if outside > 0 {
        let hint = match origin {
            Origin::Corner => {
                "coordinates are read with the origin at a box corner; \
                 for a box-centred file use `!FromFile {file: ..., origin: center}`"
            }
            Origin::Center => "coordinates are read as box-centred (`origin: center`)",
        };
        anyhow::bail!(
            "{outside} of {} positions in '{}' lie outside the cell: {hint}",
            positions.len(),
            filename.as_ref().display()
        );
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{Cuboid, Slit};

    fn xyz_file(dir: &tempfile::TempDir, positions: &[[f64; 3]]) -> std::path::PathBuf {
        let path = dir.path().join("structure.xyz");
        let atoms: String = positions
            .iter()
            .map(|[x, y, z]| format!("A {x} {y} {z}\n"))
            .collect();
        std::fs::write(&path, format!("{}\n\n{atoms}", positions.len())).unwrap();
        path
    }

    #[test]
    fn origin_selects_whether_coordinates_are_shifted() {
        let dir = tempfile::tempdir().unwrap();
        let file = xyz_file(&dir, &[[1.0, 2.0, 3.0]]);
        let cell = Cuboid::new(10.0, 10.0, 10.0);
        let corner = positions_from_structure_file(&file, &cell, Origin::Corner).unwrap();
        let center = positions_from_structure_file(&file, &cell, Origin::Center).unwrap();
        assert_eq!(corner, vec![Point::new(-4.0, -3.0, -2.0)]);
        assert_eq!(center, vec![Point::new(1.0, 2.0, 3.0)]);
    }

    /// A box-centred file read with the corner default lands beyond the slit walls; that
    /// used to give an infinite energy and zero acceptance with no error.
    #[test]
    fn position_beyond_a_wall_is_an_error_that_names_the_origin() {
        let dir = tempfile::tempdir().unwrap();
        let file = xyz_file(&dir, &[[0.0, 0.0, 0.0], [0.0, 0.0, -4.0]]);
        let slit = Slit::new(10.0, 10.0, 10.0);
        let err = positions_from_structure_file(&file, &slit, Origin::Corner).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("1 of 2"), "{message}");
        assert!(message.contains("origin: center"), "{message}");
        assert!(positions_from_structure_file(&file, &slit, Origin::Center).is_ok());
    }

    /// Molecules kept whole across a periodic boundary overhang the box; minimum-image
    /// distances handle that, so it is neither an error nor wrapped.
    #[test]
    fn periodic_overhang_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let file = xyz_file(&dir, &[[11.0, 5.0, 5.0]]);
        let cell = Cuboid::new(10.0, 10.0, 10.0);
        let positions = positions_from_structure_file(&file, &cell, Origin::Corner).unwrap();
        assert_eq!(positions, vec![Point::new(6.0, 0.0, 0.0)]);
    }
}
