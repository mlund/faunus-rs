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
//! XYZ format is always available natively. Other formats require the `chemfiles` feature.

#[cfg(feature = "chemfiles")]
mod chemfiles_io;
mod xyz;

use crate::Point;
use std::path::Path;

/// Format-agnostic in-memory representation of a structure frame.
#[derive(Debug)]
pub(crate) struct StructureData {
    pub names: Vec<String>,
    pub positions: Vec<Point>,
    pub comment: Option<String>,
}

/// Trait for reading and writing molecular structure files.
pub(crate) trait StructureIO: std::fmt::Debug {
    /// Read a structure from a file path.
    fn read(&self, path: &Path) -> anyhow::Result<StructureData>;
    /// Write a structure frame to a file path. If `append` is true, append to existing file.
    fn write(&self, path: &Path, data: &StructureData, append: bool) -> anyhow::Result<()>;
}

/// Return a reader/writer for the given file path based on its extension.
/// XYZ is always available. Other formats require the `chemfiles` feature.
pub(crate) fn format_for_path(path: &Path) -> anyhow::Result<Box<dyn StructureIO>> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("xyz") => Ok(Box::new(xyz::XyzFormat)),
        #[cfg(feature = "chemfiles")]
        Some(_) => Ok(Box::new(chemfiles_io::ChemfilesFormat)),
        #[cfg(not(feature = "chemfiles"))]
        Some(ext) => anyhow::bail!("Format '.{ext}' requires the `chemfiles` feature"),
        None => anyhow::bail!("Cannot determine format: no file extension"),
    }
}

/// Convenience: read a structure file, auto-detecting format.
pub(crate) fn read_structure(path: &impl AsRef<Path>) -> anyhow::Result<StructureData> {
    format_for_path(path.as_ref())?.read(path.as_ref())
}

/// Convenience: write a structure frame, auto-detecting format.
pub(crate) fn write_structure_frame(
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
    fn format_for_no_extension() {
        assert!(format_for_path(Path::new("noext")).is_err());
    }

    #[test]
    #[cfg(not(feature = "chemfiles"))]
    fn format_for_pdb_without_chemfiles() {
        let err = format_for_path(Path::new("test.pdb")).unwrap_err();
        assert!(err.to_string().contains("chemfiles"));
    }

    #[test]
    #[cfg(feature = "chemfiles")]
    fn format_for_pdb_with_chemfiles() {
        assert!(format_for_path(Path::new("test.pdb")).is_ok());
    }
}
