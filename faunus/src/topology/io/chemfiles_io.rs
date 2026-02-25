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

//! Chemfiles-based reader/writer for non-XYZ formats.

use super::{StructureData, StructureIO};
use crate::Point;
use std::path::Path;

#[derive(Debug)]
pub(crate) struct ChemfilesFormat;

impl StructureIO for ChemfilesFormat {
    fn read(&self, path: &Path) -> anyhow::Result<StructureData> {
        let mut trajectory = chemfiles::Trajectory::open(path, 'r')?;
        let mut frame = chemfiles::Frame::new();
        trajectory.read(&mut frame)?;

        let names: Vec<String> = frame.iter_atoms().map(|a| a.name()).collect();

        let positions: Vec<Point> = frame
            .positions()
            .iter()
            .map(|pos| Point::new(pos[0], pos[1], pos[2]))
            .collect();

        Ok(StructureData {
            names,
            positions,
            comment: None,
            ..Default::default()
        })
    }

    fn write(&self, path: &Path, data: &StructureData, append: bool) -> anyhow::Result<()> {
        let mode = if append { 'a' } else { 'w' };
        let mut trajectory = chemfiles::Trajectory::open(path, mode)?;

        let mut frame = chemfiles::Frame::new();

        for (name, pos) in data.names.iter().zip(data.positions.iter()) {
            let pos_arr: [f64; 3] = (*pos).into();
            frame.add_atom(&chemfiles::Atom::new(name.as_str()), pos_arr, None);
        }

        trajectory.write(&frame)?;
        Ok(())
    }
}
