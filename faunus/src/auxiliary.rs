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

//! Implementation of auxiliary functions for computing various properties.

use crate::{cell::SimulationCell, Point};
use flate2::write::GzEncoder;
use flate2::Compression;
use nalgebra::Vector3;
use std::fmt::Display;
use std::io::Write;
use std::path::Path;

/// If the output file has a `.gz` extension, return a `GzEncoder` wrapped around the file.
fn open_compressed(path: &Path) -> anyhow::Result<Box<dyn Write + Send>> {
    let file = std::fs::File::create(path)
        .map_err(|err| anyhow::anyhow!("Error creating file {path:?}: {err}"))?;
    if path.extension().unwrap_or_default() == "gz" {
        Ok(Box::new(GzEncoder::new(file, Compression::default())))
    } else {
        Ok(Box::new(file))
    }
}

/// Column-data file format, inferred from file extension.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum ColumnFormat {
    /// Space-separated with `# ` header prefix (`.dat`).
    #[default]
    Whitespace,
    /// Comma-separated, plain header row (`.csv`).
    Csv,
    /// Tab-separated, plain header row (`.tsv`).
    Tsv,
}

impl ColumnFormat {
    /// Infer format from path: `.csv` (or `.csv.gz`) → Csv, else Whitespace.
    pub fn from_path(path: &Path) -> Self {
        let stem = if path.extension().unwrap_or_default() == "gz" {
            path.file_stem().map(Path::new)
        } else {
            Some(path)
        };
        match stem.and_then(|p| p.extension()).and_then(|e| e.to_str()) {
            Some("csv") => Self::Csv,
            Some("tsv") => Self::Tsv,
            _ => Self::Whitespace,
        }
    }

    const fn separator(self) -> &'static str {
        match self {
            Self::Whitespace => " ",
            Self::Csv => ",",
            Self::Tsv => "\t",
        }
    }

    const fn comment_prefix(self) -> &'static str {
        match self {
            Self::Whitespace => "# ",
            Self::Csv | Self::Tsv => "",
        }
    }
}

/// Format-aware writer for column data (.dat, .csv, optionally gzip-compressed).
///
/// Writes a header row on construction and provides [`write_row`](Self::write_row)
/// for appending data rows with the correct separator.
pub struct ColumnWriter {
    inner: Box<dyn Write + Send>,
    sep: &'static str,
}

impl std::fmt::Debug for ColumnWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnWriter").finish_non_exhaustive()
    }
}

impl ColumnWriter {
    /// Open a file, infer the format from its extension, and write the header.
    pub fn open(path: &Path, columns: &[&str]) -> anyhow::Result<Self> {
        let inner = open_compressed(path)?;
        let format = ColumnFormat::from_path(path);
        Self::new(inner, format, columns)
    }

    /// Wrap an existing writer, write the header row.
    pub fn new(
        mut inner: Box<dyn Write + Send>,
        format: ColumnFormat,
        columns: &[&str],
    ) -> anyhow::Result<Self> {
        let sep = format.separator();
        write!(inner, "{}", format.comment_prefix())?;
        for (i, col) in columns.iter().enumerate() {
            if i > 0 {
                write!(inner, "{sep}")?;
            }
            write!(inner, "{col}")?;
        }
        writeln!(inner)?;
        Ok(Self { inner, sep })
    }

    /// Write a row of values using the format's separator.
    pub fn write_row(&mut self, values: &[&dyn Display]) -> std::io::Result<()> {
        for (i, val) in values.iter().enumerate() {
            if i > 0 {
                write!(self.inner, "{}", self.sep)?;
            }
            write!(self.inner, "{val}")?;
        }
        writeln!(self.inner)
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod column_writer_tests {
    use super::*;

    #[test]
    fn format_from_extension() {
        assert_eq!(
            ColumnFormat::from_path(Path::new("out.dat")),
            ColumnFormat::Whitespace
        );
        assert_eq!(
            ColumnFormat::from_path(Path::new("out.dat.gz")),
            ColumnFormat::Whitespace
        );
        assert_eq!(
            ColumnFormat::from_path(Path::new("out.csv")),
            ColumnFormat::Csv
        );
        assert_eq!(
            ColumnFormat::from_path(Path::new("out.csv.gz")),
            ColumnFormat::Csv
        );
        assert_eq!(
            ColumnFormat::from_path(Path::new("out.txt")),
            ColumnFormat::Whitespace
        );
    }

    fn collect_output(
        format: ColumnFormat,
        columns: &[&str],
    ) -> (ColumnWriter, std::sync::Arc<std::sync::Mutex<Vec<u8>>>) {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let shared = buf.clone();

        /// Wrapper to make `Arc<Mutex<Vec<u8>>>` implement `Write`.
        struct SharedBuf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl Write for SharedBuf {
            fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().write(data)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let writer = ColumnWriter::new(Box::new(SharedBuf(buf)), format, columns).unwrap();
        (writer, shared)
    }

    #[test]
    fn whitespace_output() {
        let (mut w, buf) = collect_output(ColumnFormat::Whitespace, &["x", "y"]);
        w.write_row(&[&1, &format_args!("{:.2}", 3.14)]).unwrap();
        let bytes = buf.lock().unwrap();
        assert_eq!(String::from_utf8_lossy(&bytes), "# x y\n1 3.14\n");
    }

    #[test]
    fn csv_output() {
        let (mut w, buf) = collect_output(ColumnFormat::Csv, &["x", "y"]);
        w.write_row(&[&1, &format_args!("{:.2}", 3.14)]).unwrap();
        let bytes = buf.lock().unwrap();
        assert_eq!(String::from_utf8_lossy(&bytes), "x,y\n1,3.14\n");
    }
}

/// Calculate center of mass of a collection of points with masses.
/// Does not consider periodic boundary conditions.
pub(crate) fn mass_center<'a>(
    positions: impl IntoIterator<Item = &'a Point>,
    masses: &[f64],
) -> Point {
    let total_mass: f64 = masses.iter().sum();
    positions
        .into_iter()
        .zip(masses)
        .map(|(r, &m)| r * m)
        .sum::<Point>()
        / total_mass
}

/// Calculate center of mass of a collection of points with masses using PBC.
///
/// Uses the first atom as reference and unwraps all others via minimum image
/// convention to guarantee consistent geometry regardless of box wrapping.
#[cfg(test)]
pub(crate) fn mass_center_pbc<'a>(
    positions: impl IntoIterator<Item = &'a Point>,
    masses: &[f64],
    cell: &impl SimulationCell,
    _shift: Option<Point>,
) -> Point {
    let total_mass: f64 = masses.iter().sum();
    let mut iter = positions.into_iter().zip(masses.iter());
    let (&ref_pos, &ref_mass) = iter.next().expect("at least one position required");
    let mut com = ref_pos * ref_mass;
    for (&pos, &m) in iter {
        // Unwrap relative to reference atom using MIC
        let unwrapped = ref_pos + cell.distance(&pos, &ref_pos);
        com += unwrapped * m;
    }
    com /= total_mass;
    cell.boundary(&mut com);
    com
}

#[test]
fn test_center_of_mass() {
    use float_cmp::assert_approx_eq;

    let positions = [
        Point::new(10.4, 11.3, 12.8),
        Point::new(7.3, 9.3, 2.6),
        Point::new(9.3, 10.1, 17.2),
    ];
    let masses = [1.46, 2.23, 10.73];

    let com = mass_center(&positions, &masses);

    assert_approx_eq!(f64, com.x, 9.10208044382802);
    assert_approx_eq!(f64, com.y, 10.09778085991678);
    assert_approx_eq!(f64, com.z, 14.49667128987517);

    let positions = [
        Point::new(10.4, 11.3, 12.8),
        Point::new(7.3, 9.3, 2.6),
        Point::new(9.3, 10.1, 17.2),
        Point::new(3.1, 2.4, 1.8),
    ];

    let masses = [1.46, 2.23, 10.73, 0.0];

    let com = mass_center(&positions, &masses);

    assert_approx_eq!(f64, com.x, 9.10208044382802);
    assert_approx_eq!(f64, com.y, 10.09778085991678);
    assert_approx_eq!(f64, com.z, 14.49667128987517);
}

/// Calculate angle between two vectors.
/// The angle is returned in degrees.
#[inline(always)]
pub(crate) fn angle_vectors(v1: &Vector3<f64>, v2: &Vector3<f64>) -> f64 {
    let cos = v1.dot(v2) / (v1.norm() * v2.norm());
    cos.acos().to_degrees()
}

#[test]
fn test_angle_vectors() {
    use float_cmp::assert_approx_eq;

    let v1 = Vector3::new(2.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 2.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 90.0);

    let v1 = Vector3::new(2.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, -2.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 90.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 0.0, 7.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 90.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(3.0, 0.0, 3.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 45.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(4.0, 0.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 0.0);

    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(-4.0, 0.0, 0.0);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 180.0);

    let v1 = Vector3::new(1.0, -1.0, 3.5);
    let v2 = Vector3::new(1.2, 2.4, -0.7);
    assert_approx_eq!(f64, angle_vectors(&v1, &v2), 110.40636490060925);
}

/// Calculate angle between three points with `b` being the vertext of the angle.
/// The angle is returned in degrees.
#[inline(always)]
pub(crate) fn angle_points(a: &Point, b: &Point, c: &Point, pbc: &impl SimulationCell) -> f64 {
    // b->a
    let ba = pbc.distance(a, b);
    // b->c
    let bc = pbc.distance(c, b);
    angle_vectors(&ba, &bc)
}

#[test]
fn test_angle_points() {
    use float_cmp::assert_approx_eq;

    let endless_cell = crate::cell::Endless;

    let p1 = Point::new(3.2, 3.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 5.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 90.0);

    let p1 = Point::new(3.2, 3.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 1.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 90.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(3.2, 3.3, 9.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 90.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(6.2, 3.3, 5.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 45.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(7.2, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 0.0);

    let p1 = Point::new(4.2, 3.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(-1.2, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &endless_cell), 180.0);

    let p1 = Point::new(4.2, 2.3, 6.0);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(4.4, 5.7, 1.8);
    assert_approx_eq!(
        f64,
        angle_points(&p1, &p2, &p3, &endless_cell),
        110.40636490060925
    );
}

#[test]
fn test_angle_points_pbc() {
    use float_cmp::assert_approx_eq;

    let cell = crate::cell::Cuboid::new(5.0, 10.0, 15.0);

    let p1 = Point::new(2.2, 3.3, 2.5);
    let p2 = Point::new(-2.0, 3.3, 2.5);
    let p3 = Point::new(-2.2, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &cell), 0.0);

    let p1 = Point::new(1.4, 3.3, 2.5);
    let p2 = Point::new(2.2, 3.3, 2.5);
    let p3 = Point::new(-2.3, 3.3, 2.5);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &cell), 180.0);

    let p1 = Point::new(1.5, -4.7, 1.2);
    let p2 = Point::new(1.5, 4.3, 1.2);
    let p3 = Point::new(1.5, -2.7, 4.2);
    assert_approx_eq!(f64, angle_points(&p1, &p2, &p3, &cell), 45.0);
}

/// Calculate dihedral angle between two planes defined by four points.
/// The first plane is given by points `a`, `b`, `c`.
/// The second plane is given by points `b`, `c`, `d`.
/// The angle is returned in degrees and adopts values between −180° and +180°.
pub(crate) fn dihedral_points(
    a: &Point,
    b: &Point,
    c: &Point,
    d: &Point,
    pbc: &impl SimulationCell,
) -> f64 {
    let ab = pbc.distance(b, a);
    let bc = pbc.distance(c, b);
    let cd = pbc.distance(d, c);

    // normalized vectors normal to the planes
    let abc = ab.cross(&bc).normalize();
    let bcd = bc.cross(&cd).normalize();

    let cos_angle = abc.dot(&bcd);
    let sin_angle = bc.normalize().dot(&abc.cross(&bcd));

    sin_angle.atan2(cos_angle).to_degrees()
}

#[test]
fn test_dihedral_points() {
    use float_cmp::assert_approx_eq;

    let endless_cell = crate::cell::Endless;

    // cis conformation
    let p1 = Point::new(1.2, 5.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(3.2, 3.3, 2.5);
    let p4 = Point::new(3.2, 4.3, 2.5);
    assert_approx_eq!(f64, dihedral_points(&p1, &p2, &p3, &p4, &endless_cell), 0.0);

    // cis conformation
    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(1.2, -1.3, 3.2);
    assert_approx_eq!(f64, dihedral_points(&p1, &p2, &p3, &p4, &endless_cell), 0.0);

    // trans conformation
    let p1 = Point::new(1.2, -5.3, 2.5);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(3.2, 3.3, 2.5);
    let p4 = Point::new(3.2, 4.3, 2.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        180.0
    );

    // trans conformation
    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(1.2, -1.3, 2.2);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        180.0
    );

    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(-13.2, -1.3, 2.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        90.0
    );

    let p1 = Point::new(1.2, 3.3, 5.2);
    let p2 = Point::new(1.2, 3.3, 2.5);
    let p3 = Point::new(1.2, -1.3, 2.5);
    let p4 = Point::new(2.2, -1.3, 2.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        -90.0
    );

    let p1 = Point::new(3.2, -5.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 3.3, 2.5);
    let p4 = Point::new(1.2, 4.3, 3.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        135.0
    );

    let p1 = Point::new(3.2, 5.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 3.3, 2.5);
    let p4 = Point::new(1.2, 4.3, 3.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        -45.0
    );

    let p1 = Point::new(3.2, 5.3, 2.5);
    let p2 = Point::new(3.2, 3.3, 2.5);
    let p3 = Point::new(1.2, 3.3, 2.5);
    let p4 = Point::new(1.2, 4.3, 1.5);
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p2, &p3, &p4, &endless_cell),
        45.0
    );

    // realistic data
    let p0 = Point::new(24.969, 13.428, 30.692);
    let p1 = Point::new(24.044, 12.661, 29.808);
    let p2 = Point::new(22.785, 13.482, 29.543);
    let p3 = Point::new(21.951, 13.670, 30.431);
    let p4 = Point::new(23.672, 11.328, 30.466);
    let p5 = Point::new(22.881, 10.326, 29.620);
    let p6 = Point::new(23.691, 9.935, 28.389);
    let p7 = Point::new(22.557, 9.096, 30.459);
    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p2, &p3, &endless_cell),
        -71.215151146714
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p4, &p5, &endless_cell),
        -171.9431994795364
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p6, &endless_cell),
        60.82226735264639
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p7, &endless_cell),
        -177.6364115152126
    );

    // TODO: test periodic boundary conditions
}

#[test]
fn test_dihedral_points_pbc() {
    use crate::cell::BoundaryConditions;
    use float_cmp::assert_approx_eq;

    let cuboid = crate::cell::Cuboid::new(20.0, 10.0, 28.0);

    let mut p0 = Point::new(24.969, 13.428, 30.692);
    let mut p1 = Point::new(24.044, 12.661, 29.808);
    let mut p2 = Point::new(22.785, 13.482, 29.543);
    let mut p3 = Point::new(21.951, 13.670, 30.431);
    let mut p4 = Point::new(23.672, 11.328, 30.466);
    let mut p5 = Point::new(22.881, 10.326, 29.620);
    let mut p6 = Point::new(23.691, 9.935, 28.389);
    let mut p7 = Point::new(22.557, 9.096, 30.459);

    cuboid.boundary(&mut p0);
    cuboid.boundary(&mut p1);
    cuboid.boundary(&mut p2);
    cuboid.boundary(&mut p3);
    cuboid.boundary(&mut p4);
    cuboid.boundary(&mut p5);
    cuboid.boundary(&mut p6);
    cuboid.boundary(&mut p7);

    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p2, &p3, &cuboid),
        -71.215151146714
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p0, &p1, &p4, &p5, &cuboid),
        -171.9431994795364
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p6, &cuboid),
        60.82226735264639
    );
    assert_approx_eq!(
        f64,
        dihedral_points(&p1, &p4, &p5, &p7, &cuboid),
        -177.6364115152126
    );
}
