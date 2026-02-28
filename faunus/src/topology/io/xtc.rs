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

//! Gromacs XTC trajectory writer using the `molly` crate.

use super::{StructureData, StructureIO};
use std::fs::OpenOptions;
use std::path::Path;

#[derive(Debug)]
pub struct XtcFormat;

impl StructureIO for XtcFormat {
    fn read(&self, path: &Path) -> anyhow::Result<StructureData> {
        anyhow::bail!(
            "XTC format does not contain atom names; use XYZ or PDB for reading structures: {}",
            path.display()
        )
    }

    fn write(&self, path: &Path, data: &StructureData, append: bool) -> anyhow::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .map_err(|e| anyhow::anyhow!("Cannot open '{}' for writing: {}", path.display(), e))?;

        let mut writer = molly::XTCWriter::new(file);

        // Convert Å → nm (Gromacs convention)
        const ANGSTROM_TO_NM: f32 = 0.1;

        let positions: Vec<f32> = data
            .positions
            .iter()
            .flat_map(|p| {
                [
                    p.x as f32 * ANGSTROM_TO_NM,
                    p.y as f32 * ANGSTROM_TO_NM,
                    p.z as f32 * ANGSTROM_TO_NM,
                ]
            })
            .collect();

        // Only diagonal elements needed for orthorhombic boxes
        #[allow(clippy::option_if_let_else)]
        let boxvec = if let Some(b) = &data.box_lengths {
            molly::BoxVec::from_cols_array(&[
                b.x as f32 * ANGSTROM_TO_NM,
                0.0,
                0.0,
                0.0,
                b.y as f32 * ANGSTROM_TO_NM,
                0.0,
                0.0,
                0.0,
                b.z as f32 * ANGSTROM_TO_NM,
            ])
        } else {
            molly::BoxVec::ZERO
        };

        let frame = molly::Frame {
            step: data.step.unwrap_or(0),
            time: data.time.unwrap_or(0.0),
            boxvec,
            precision: 1000.0, // standard Gromacs precision (0.001 nm resolution)
            positions,
        };

        writer.write_frame(&frame)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;

    #[test]
    fn read_xtc_returns_error() {
        let dir = std::env::temp_dir().join("faunus_test_read.xtc");
        std::fs::write(&dir, b"").unwrap();
        assert!(XtcFormat.read(&dir).is_err());
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn write_and_read_back_xtc() {
        let dir = std::env::temp_dir().join("faunus_test_roundtrip.xtc");
        let data = StructureData {
            names: vec!["H".into(), "O".into()],
            positions: vec![Point::new(1.0, 2.0, 3.0), Point::new(4.0, 5.0, 6.0)],
            comment: None,
            step: Some(42),
            time: Some(10.5),
            box_lengths: Some(Point::new(10.0, 10.0, 10.0)),
        };

        XtcFormat.write(&dir, &data, false).unwrap();

        // Read back with molly and verify
        let mut reader = molly::XTCReader::open(&dir).unwrap();
        let mut frame = molly::Frame::default();
        reader.read_frame(&mut frame).unwrap();

        assert_eq!(frame.step, 42);
        assert_eq!(frame.natoms(), 2);
        assert!((frame.time - 10.5).abs() < 1e-3);

        // Positions should be in nm (input was Å)
        assert!((frame.positions[0] - 0.1).abs() < 0.001);
        assert!((frame.positions[1] - 0.2).abs() < 0.001);
        assert!((frame.positions[2] - 0.3).abs() < 0.001);
        assert!((frame.positions[3] - 0.4).abs() < 0.001);
        assert!((frame.positions[4] - 0.5).abs() < 0.001);
        assert!((frame.positions[5] - 0.6).abs() < 0.001);

        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn write_append_xtc() {
        let dir = std::env::temp_dir().join("faunus_test_append.xtc");
        let frame1 = StructureData {
            names: vec!["H".into()],
            positions: vec![Point::new(1.0, 2.0, 3.0)],
            comment: None,
            step: Some(0),
            time: Some(0.0),
            box_lengths: Some(Point::new(10.0, 10.0, 10.0)),
        };
        let frame2 = StructureData {
            names: vec!["O".into()],
            positions: vec![Point::new(4.0, 5.0, 6.0)],
            comment: None,
            step: Some(1),
            time: Some(1.0),
            box_lengths: Some(Point::new(10.0, 10.0, 10.0)),
        };

        XtcFormat.write(&dir, &frame1, false).unwrap();
        XtcFormat.write(&dir, &frame2, true).unwrap();

        // Read back both frames
        let mut reader = molly::XTCReader::open(&dir).unwrap();
        let frames = reader.read_all_frames().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].step, 0);
        assert_eq!(frames[1].step, 1);

        std::fs::remove_file(&dir).ok();
    }
}
