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

//! Auxiliary functions for I/O and numerical integration.

use flate2::write::GzEncoder;
use flate2::Compression;
use std::fmt::Display;
use std::io::Write;
use std::path::Path;

/// Parse a named section from a YAML input file into a typed config struct.
pub(crate) fn parse_yaml_section<T: serde::de::DeserializeOwned>(
    input: &Path,
    key: &str,
) -> anyhow::Result<T> {
    let yaml = std::fs::read_to_string(input)?;
    let value: serde_yml::Value = serde_yml::from_str(&yaml)?;
    let section = value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Missing `{key}:` section in input file"))?;
    Ok(serde_yml::from_value(section.clone())?)
}

/// Resolve max thread count: 0 means use all available cores.
pub(crate) fn resolve_thread_count(max_threads: usize) -> usize {
    if max_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        max_threads
    }
}

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
pub(crate) struct ColumnWriter {
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
    pub(crate) fn open(path: &Path, columns: &[&str]) -> anyhow::Result<Self> {
        let inner = open_compressed(path)?;
        let format = ColumnFormat::from_path(path);
        Self::new(inner, format, columns)
    }

    /// Wrap an existing writer, write the header row.
    pub(crate) fn new(
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
    pub(crate) fn write_row(&mut self, values: &[&dyn Display]) -> std::io::Result<()> {
        for (i, val) in values.iter().enumerate() {
            if i > 0 {
                write!(self.inner, "{}", self.sep)?;
            }
            write!(self.inner, "{val}")?;
        }
        writeln!(self.inner)
    }

    pub(crate) fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Composite Simpson's rule over `n` equally spaced points on [0, 1].
///
/// Uses Simpson's 1/3 for odd `n`; for even `n`, applies Simpson's 3/8 on the
/// last 3 intervals. Returns 0 for fewer than 2 points.
pub(crate) fn simpson_integrate(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let h = 1.0 / (n - 1) as f64;
    if n == 2 {
        return h * (values[0] + values[1]) / 2.0;
    }
    if n == 3 {
        return h / 3.0 * (values[0] + 4.0 * values[1] + values[2]);
    }

    if n % 2 == 1 {
        let mut sum = values[0] + values[n - 1];
        for i in (1..n - 1).step_by(2) {
            sum += 4.0 * values[i];
        }
        for i in (2..n - 1).step_by(2) {
            sum += 2.0 * values[i];
        }
        sum * h / 3.0
    } else {
        // Simpson's 1/3 requires odd point count; split even n into
        // an odd-length 1/3 block plus a 4-point 3/8 tail to avoid
        // the double-counting that a trapezoidal correction would cause.
        let m = n - 3;
        let mut sum = values[0] + values[m - 1];
        for i in (1..m - 1).step_by(2) {
            sum += 4.0 * values[i];
        }
        for i in (2..m - 1).step_by(2) {
            sum += 2.0 * values[i];
        }
        let result_13 = sum * h / 3.0;
        let result_38 = 3.0 * h / 8.0
            * (values[n - 4] + 3.0 * values[n - 3] + 3.0 * values[n - 2] + values[n - 1]);
        result_13 + result_38
    }
}

/// Incremental weighted mean using West's algorithm.
///
/// When all weights are 1.0, reduces to Welford's unweighted mean.
/// See [West (1979)](https://doi.org/10.1145/359146.359153).
#[derive(Clone, Debug, Default)]
pub(crate) struct WeightedMean {
    sum_w: f64,
    mean: f64,
    count: u64,
}

impl WeightedMean {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a value with the given weight. Zero-weight samples are ignored.
    pub fn add(&mut self, value: f64, weight: f64) {
        if weight == 0.0 {
            return;
        }
        self.sum_w += weight;
        self.mean += weight * (value - self.mean) / self.sum_w;
        self.count += 1;
    }

    /// Current weighted mean, or NaN if no samples have been added.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    /// Whether no values have been added.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Number of values added.
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Sum of all weights.
    #[allow(dead_code)]
    pub fn sum_weights(&self) -> f64 {
        self.sum_w
    }
}

/// Running block average with mean and standard error of the mean.
///
/// Wraps [`average::Variance`] with convenience methods for reporting.
/// Each call to [`add`](Self::add) represents one block measurement.
#[derive(Clone, Debug, Default)]
pub(crate) struct BlockAverage(average::Variance);

use average::Estimate as _;

impl BlockAverage {
    pub fn new() -> Self {
        Self(average::Variance::new())
    }

    /// Record a per-block value.
    pub fn add(&mut self, value: f64) {
        self.0.add(value);
    }

    /// Mean over all blocks.
    pub fn mean(&self) -> f64 {
        self.0.mean()
    }

    /// Standard error of the mean (SEM = σ / √n).
    pub fn error(&self) -> f64 {
        self.0.error()
    }

    /// Serialize as YAML mapping `{ mean: ..., error: ... }`.
    pub fn to_yaml(&self) -> Option<serde_yml::Value> {
        let mut m = serde_yml::Mapping::new();
        m.try_insert("mean", self.mean())?;
        m.try_insert("error", self.error())?;
        Some(serde_yml::Value::Mapping(m))
    }
}

/// Extension trait to reduce YAML mapping construction boilerplate.
pub(crate) trait MappingExt {
    /// Insert a serializable value, returning `None` if serialization fails.
    fn try_insert(&mut self, key: &str, value: impl serde::Serialize) -> Option<()>;
}

impl MappingExt for serde_yml::Mapping {
    fn try_insert(&mut self, key: &str, value: impl serde::Serialize) -> Option<()> {
        self.insert(key.into(), serde_yml::to_value(value).ok()?);
        Some(())
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
        w.write_row(&[&1, &format_args!("{:.2}", 3.15)]).unwrap();
        let bytes = buf.lock().unwrap();
        assert_eq!(String::from_utf8_lossy(&bytes), "# x y\n1 3.15\n");
    }

    #[test]
    fn csv_output() {
        let (mut w, buf) = collect_output(ColumnFormat::Csv, &["x", "y"]);
        w.write_row(&[&1, &format_args!("{:.2}", 3.15)]).unwrap();
        let bytes = buf.lock().unwrap();
        assert_eq!(String::from_utf8_lossy(&bytes), "x,y\n1,3.15\n");
    }
}

#[cfg(test)]
mod weighted_mean_tests {
    use super::WeightedMean;
    use approx::assert_relative_eq;

    #[test]
    fn uniform_weights_match_simple_mean() {
        let mut wm = WeightedMean::new();
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        for &v in &values {
            wm.add(v, 1.0);
        }
        assert_relative_eq!(wm.mean(), 3.0);
        assert_eq!(wm.len(), 5);
    }

    #[test]
    fn weighted_mean() {
        let mut wm = WeightedMean::new();
        // weight 3 on value 2, weight 1 on value 6 → mean = (6+6)/4 = 3.0
        wm.add(2.0, 3.0);
        wm.add(6.0, 1.0);
        assert_relative_eq!(wm.mean(), 3.0);
        assert_eq!(wm.len(), 2);
        assert_relative_eq!(wm.sum_weights(), 4.0);
    }

    #[test]
    fn single_value() {
        let mut wm = WeightedMean::new();
        wm.add(42.0, 0.5);
        assert_relative_eq!(wm.mean(), 42.0);
    }
}

#[cfg(test)]
mod simpson_tests {
    use super::simpson_integrate;

    #[test]
    fn linear_odd() {
        let values: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
        assert!((simpson_integrate(&values) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn quadratic_odd() {
        let n = 11;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / (n - 1) as f64;
                x * x
            })
            .collect();
        assert!((simpson_integrate(&values) - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn two_points() {
        assert!((simpson_integrate(&[0.0, 1.0]) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn single_point() {
        assert_eq!(simpson_integrate(&[1.0]), 0.0);
    }

    #[test]
    fn linear_even() {
        let n = 10;
        let values: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        assert!((simpson_integrate(&values) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn quadratic_even() {
        let n = 10;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / (n - 1) as f64;
                x * x
            })
            .collect();
        assert!((simpson_integrate(&values) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn four_points() {
        let n = 4;
        let values: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        assert!((simpson_integrate(&values) - 0.5).abs() < 1e-14);
    }
}
