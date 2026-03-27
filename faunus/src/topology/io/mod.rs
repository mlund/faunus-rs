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

//! Format-agnostic structure file I/O.
//!
//! XYZ format is always available natively. XTC (Gromacs) is handled by the `molly` crate.

pub mod frame_state;
pub(crate) mod psf;
mod xtc;
mod xyz;

use crate::Point;
use std::path::Path;

/// Conversion factor from nanometers to ångströms (Gromacs XTC convention).
#[cfg(feature = "cli")]
pub const NM_TO_ANGSTROM: f64 = 10.0;

/// Conversion factor from ångströms to nanometers (Gromacs XTC convention).
pub const ANGSTROM_TO_NM: f64 = 0.1;

/// Format-agnostic in-memory representation of a structure frame.
#[derive(Debug, Default)]
pub struct StructureData {
    pub names: Vec<String>,
    pub positions: Vec<Point>,
    pub comment: Option<String>,
    /// Simulation step number (used by trajectory formats like XTC).
    pub step: Option<u32>,
    /// Time in picoseconds (used by trajectory formats like XTC).
    pub time: Option<f32>,
    /// Bounding box lengths (used by trajectory formats like XTC).
    pub box_lengths: Option<Point>,
}

/// Trait for reading and writing molecular structure files.
pub trait StructureIO: std::fmt::Debug {
    /// Read a structure from a file path.
    fn read(&self, path: &Path) -> anyhow::Result<StructureData>;
    /// Write a structure frame to a file path. If `append` is true, append to existing file.
    fn write(&self, path: &Path, data: &StructureData, append: bool) -> anyhow::Result<()>;
}

/// Return a reader/writer for the given file path based on its extension.
pub fn format_for_path(path: &Path) -> anyhow::Result<Box<dyn StructureIO>> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("xyz") => Ok(Box::new(xyz::XyzFormat)),
        Some("xtc") => Ok(Box::new(xtc::XtcFormat)),
        Some(ext) => anyhow::bail!("Unsupported format '.{ext}'"),
        None => anyhow::bail!("Cannot determine format: no file extension"),
    }
}

/// Convenience: read a structure file, auto-detecting format.
pub fn read_structure(path: &impl AsRef<Path>) -> anyhow::Result<StructureData> {
    format_for_path(path.as_ref())?.read(path.as_ref())
}

/// Convenience: write a structure frame, auto-detecting format.
pub fn write_structure_frame(
    path: &impl AsRef<Path>,
    data: &StructureData,
    append: bool,
) -> anyhow::Result<()> {
    format_for_path(path.as_ref())?.write(path.as_ref(), data, append)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_for_xyz() {
        assert!(format_for_path(Path::new("test.xyz")).is_ok());
    }

    #[test]
    fn format_for_xtc() {
        assert!(format_for_path(Path::new("test.xtc")).is_ok());
    }

    #[test]
    fn format_for_no_extension() {
        assert!(format_for_path(Path::new("noext")).is_err());
    }

    #[test]
    fn format_for_unsupported_extension() {
        assert!(format_for_path(Path::new("test.pdb")).is_err());
    }
}
