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

//! Native XYZ format reader and writer.

use super::{StructureData, StructureIO};
use crate::Point;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug)]
pub(crate) struct XyzFormat;

impl StructureIO for XyzFormat {
    fn read(&self, path: &Path) -> anyhow::Result<StructureData> {
        let file = File::open(path)
            .map_err(|e| anyhow::anyhow!("Cannot open '{}': {}", path.display(), e))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Line 1: atom count
        let count_line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty XYZ file: {}", path.display()))??;
        let num_atoms: usize = count_line.trim().parse().map_err(|_| {
            anyhow::anyhow!(
                "Invalid atom count '{}' in {}",
                count_line.trim(),
                path.display()
            )
        })?;

        // Line 2: comment
        let comment_line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing comment line in {}", path.display()))??;
        let comment = Some(comment_line.trim().to_string()).filter(|s| !s.is_empty());

        // Atom lines
        let mut names = Vec::with_capacity(num_atoms);
        let mut positions = Vec::with_capacity(num_atoms);

        for (i, line_result) in lines.enumerate() {
            if i >= num_atoms {
                break;
            }
            let line = line_result?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                anyhow::bail!(
                    "Malformed atom line {} in {}: '{}'",
                    i + 3,
                    path.display(),
                    line
                );
            }
            names.push(parts[0].to_string());
            let x: f64 = parts[1]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid x coordinate on line {}", i + 3))?;
            let y: f64 = parts[2]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid y coordinate on line {}", i + 3))?;
            let z: f64 = parts[3]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid z coordinate on line {}", i + 3))?;
            positions.push(Point::new(x, y, z));
        }

        if names.len() != num_atoms {
            anyhow::bail!(
                "Expected {} atoms but found {} in {}",
                num_atoms,
                names.len(),
                path.display()
            );
        }

        Ok(StructureData {
            names,
            positions,
            comment,
            ..Default::default()
        })
    }

    fn write(&self, path: &Path, data: &StructureData, append: bool) -> anyhow::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .map_err(|e| anyhow::anyhow!("Cannot open '{}' for writing: {}", path.display(), e))?;

        let mut writer = BufWriter::new(file);

        // Line 1: atom count
        writeln!(writer, "{}", data.names.len())?;

        // Line 2: comment
        writeln!(writer, "{}", data.comment.as_deref().unwrap_or(""))?;

        // Atom lines
        for (name, pos) in data.names.iter().zip(data.positions.iter()) {
            writeln!(writer, "{} {} {} {}", name, pos.x, pos.y, pos.z)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn read_mol2_xyz() {
        let data = XyzFormat.read(Path::new("tests/files/mol2.xyz")).unwrap();
        assert_eq!(data.names.len(), 3);
        assert_eq!(data.positions.len(), 3);
        assert_eq!(data.names, vec!["OW", "OW", "X"]);
        assert_approx_eq!(f64, data.positions[0].x, 0.1);
        assert_approx_eq!(f64, data.positions[0].y, 0.6);
        assert_approx_eq!(f64, data.positions[0].z, 0.2);
        assert_approx_eq!(f64, data.positions[2].x, 0.7);
        assert_approx_eq!(f64, data.positions[2].y, 0.1);
        assert_approx_eq!(f64, data.positions[2].z, 0.3);
        assert_eq!(data.comment.as_deref(), Some("System"));
    }

    #[test]
    fn read_mol2_absolute_xyz() {
        let data = XyzFormat
            .read(Path::new("tests/files/mol2_absolute.xyz"))
            .unwrap();
        assert_eq!(data.names.len(), 15);
        assert_eq!(data.positions.len(), 15);
    }

    #[test]
    fn read_structure_xyz() {
        let data = XyzFormat
            .read(Path::new("tests/files/structure.xyz"))
            .unwrap();
        assert_eq!(data.names.len(), 21);
        assert_eq!(data.positions.len(), 21);
        assert_eq!(data.names[0], "OW");
        assert_eq!(data.names[1], "HW");
    }

    #[test]
    fn read_empty_file() {
        let dir = std::env::temp_dir().join("faunus_test_empty.xyz");
        std::fs::write(&dir, "").unwrap();
        assert!(XyzFormat.read(&dir).is_err());
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn read_bad_atom_count() {
        let dir = std::env::temp_dir().join("faunus_test_bad_count.xyz");
        std::fs::write(&dir, "abc\ncomment\n").unwrap();
        assert!(XyzFormat.read(&dir).is_err());
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn read_malformed_line() {
        let dir = std::env::temp_dir().join("faunus_test_malformed.xyz");
        std::fs::write(&dir, "1\ncomment\nOW 1.0 2.0\n").unwrap();
        let err = XyzFormat.read(&dir).unwrap_err();
        assert!(err.to_string().contains("Malformed"));
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn read_too_few_atoms() {
        let dir = std::env::temp_dir().join("faunus_test_few.xyz");
        std::fs::write(&dir, "3\ncomment\nOW 1.0 2.0 3.0\n").unwrap();
        let err = XyzFormat.read(&dir).unwrap_err();
        assert!(err.to_string().contains("Expected 3"));
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = std::env::temp_dir().join("faunus_test_roundtrip.xyz");
        let data = StructureData {
            names: vec!["H".into(), "O".into()],
            positions: vec![Point::new(1.0, 2.0, 3.0), Point::new(4.0, 5.0, 6.0)],
            comment: Some("test frame".into()),
            ..Default::default()
        };
        XyzFormat.write(&dir, &data, false).unwrap();

        let read_back = XyzFormat.read(&dir).unwrap();
        assert_eq!(read_back.names, data.names);
        assert_eq!(read_back.positions.len(), 2);
        assert_approx_eq!(f64, read_back.positions[0].x, 1.0);
        assert_approx_eq!(f64, read_back.positions[1].z, 6.0);
        assert_eq!(read_back.comment.as_deref(), Some("test frame"));
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn write_append_mode() {
        let dir = std::env::temp_dir().join("faunus_test_append.xyz");
        let frame1 = StructureData {
            names: vec!["H".into()],
            positions: vec![Point::new(1.0, 2.0, 3.0)],
            comment: Some("frame1".into()),
            ..Default::default()
        };
        let frame2 = StructureData {
            names: vec!["O".into(), "N".into()],
            positions: vec![Point::new(4.0, 5.0, 6.0), Point::new(7.0, 8.0, 9.0)],
            comment: Some("frame2".into()),
            ..Default::default()
        };

        XyzFormat.write(&dir, &frame1, false).unwrap();
        XyzFormat.write(&dir, &frame2, true).unwrap();

        // Verify the file contains both frames
        let content = std::fs::read_to_string(&dir).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines[0], "1"); // frame1 count
        assert_eq!(lines[1], "frame1");
        assert_eq!(lines[3], "2"); // frame2 count
        assert_eq!(lines[4], "frame2");
        assert_eq!(lines.len(), 7);
        std::fs::remove_file(&dir).ok();
    }
}
