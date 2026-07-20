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

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::io::{self, BufRead, Write};
use std::ops::Mul;
use std::path::{Path, PathBuf};

/// Read a YAML file, applying Jinja2 template rendering when the file contains a Jinja
/// opener outside of comments (see [`looks_like_template`]).
///
/// Top-level keys prefixed with `_` are silently removed, allowing
/// sections to be temporarily disabled (e.g. `_umbrella:` instead of `umbrella:`).
///
/// Plain YAML files (no template tags) pass through unchanged.
/// See [minijinja](https://docs.rs/minijinja) for the template language.
pub fn read_yaml(path: impl AsRef<Path>) -> anyhow::Result<String> {
    use anyhow::Context;
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read '{}'", path.display()))?;
    // Only invoke the template engine when template tags are present
    let yaml = if looks_like_template(&raw) {
        let mut env = minijinja::Environment::new();
        // Strict mode fails fast on undefined variables with the variable name
        // and line, instead of silently producing `undefined` values that
        // explode later in arithmetic with confusing messages like
        // "tried to use * operator on unsupported types number and undefined".
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
        env.render_str(&raw, minijinja::context! {})
            .map_err(|err| {
                let mut msg = format!("Template error in '{}': {err:#}", path.display());
                // Strict mode catches plain undefined accesses with `UndefinedError`;
                // arithmetic/coercion on undefined surfaces as `InvalidOperation`
                // with the literal type "undefined" in the message. Match either.
                let touches_undefined = err.kind() == minijinja::ErrorKind::UndefinedError
                    || err.to_string().contains("undefined");
                if touches_undefined {
                    msg.push_str(
                        "\nhint: faunus renders templates with no external context. \
                     Set defaults inline, e.g. `{% set var = var | default(value) %}`.",
                    );
                }
                anyhow::anyhow!(msg)
            })?
    } else {
        raw
    };
    // Name the file here: a serde error alone reports only a line and column, and every
    // input path funnels through this function.
    strip_underscore_keys(&yaml).with_context(|| format!("Cannot parse '{}'", path.display()))
}

/// Decide whether `raw` should be rendered as a Jinja template before YAML parsing.
///
/// Templating is triggered by a Jinja opener (`{%` or `{#`), but the trigger must not fire on
/// one that appears only inside a YAML comment — otherwise a previously-valid input aborts at
/// load time (`# note: use {% ... %}` should stay a plain comment). We therefore look for the
/// opener in each line with its trailing YAML comment removed. Detecting the opener (rather
/// than a matched pair) keeps multi-line tags and block comments working, e.g. a
/// `{# … #}` section-disabling comment whose delimiters span several lines. Residual
/// limitation: an opener inside a quoted string value still triggers — matching the original
/// behaviour, and rare in practice.
fn looks_like_template(raw: &str) -> bool {
    raw.lines().any(|line| {
        let code = strip_line_comment(line);
        code.contains("{%") || code.contains("{#")
    })
}

/// Strip a trailing YAML end-of-line comment, leaving Jinja `#}` intact.
///
/// A YAML comment starts at a `#` that begins the line or follows whitespace; a `#`
/// immediately followed by `}` is a Jinja `#}` close and is not treated as a comment.
fn strip_line_comment(line: &str) -> &str {
    let b = line.as_bytes();
    for i in 0..b.len() {
        let is_comment_start = b[i] == b'#'
            && (i == 0 || b[i - 1].is_ascii_whitespace())
            && !(i + 1 < b.len() && b[i + 1] == b'}');
        if is_comment_start {
            return &line[..i];
        }
    }
    line
}

/// Remove top-level YAML keys that start with `_`.
fn strip_underscore_keys(yaml: &str) -> anyhow::Result<String> {
    let mut value: yaml_serde::Value = yaml_serde::from_str(yaml)?;
    if let yaml_serde::Value::Mapping(ref mut map) = value {
        let disabled: Vec<_> = map
            .keys()
            .filter(|k| k.as_str().is_some_and(|s| s.starts_with('_')))
            .cloned()
            .collect();
        if disabled.is_empty() {
            return Ok(yaml.to_string());
        }
        for key in &disabled {
            log::info!("Ignoring disabled section `{}`", key.as_str().unwrap());
            map.remove(key);
        }
    }
    Ok(yaml_serde::to_string(&value)?)
}

/// Parse a named section from a YAML input file into a typed config struct.
// Only the `cli`-gated umbrella / Wang-Landau drivers use this.
#[cfg(feature = "cli")]
pub fn parse_yaml_section<T: serde::de::DeserializeOwned>(
    input: &Path,
    key: &str,
) -> anyhow::Result<T> {
    let yaml = read_yaml(input)?;
    let value: yaml_serde::Value = yaml_serde::from_str(&yaml)?;
    let section = value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Missing `{key}:` section in input file"))?;
    from_section_value(key, section)
}

/// Deserialize a section's already-extracted [`Value`](yaml_serde::Value),
/// labeling any error with the section name so the user knows where to look.
///
/// The location is gone once the document is a parsed `Value`, so the section
/// name is the best anchor we can attach without re-parsing from the source.
/// `section` is the dotted path to the section (e.g. `system/medium`), not a
/// single literal key, so the message reads as a location rather than a key.
pub fn from_section_value<T: serde::de::DeserializeOwned>(
    section: &str,
    value: &yaml_serde::Value,
) -> anyhow::Result<T> {
    yaml_serde::from_value(value.clone())
        .map_err(|e| anyhow::anyhow!("in `{section}` section: {e}"))
}

/// Deserialize a YAML sequence of (typically tagged) entries one at a time so an
/// error names the offending entry by 1-based index and YAML tag, e.g.
/// `in `analysis` entry 1 (!RadialDistribution): unknown field ...`.
///
/// A null section (the key present with no value) yields an empty list, matching
/// how `from_value::<Vec<_>>` treats `Null`.
pub fn from_tagged_list<T: serde::de::DeserializeOwned>(
    section: &str,
    value: &yaml_serde::Value,
) -> anyhow::Result<Vec<T>> {
    if value.is_null() {
        return Ok(Vec::new());
    }
    let seq = value
        .as_sequence()
        .ok_or_else(|| anyhow::anyhow!("`{section}` must be a list"))?;
    seq.iter()
        .enumerate()
        .map(|(index, entry)| {
            yaml_serde::from_value::<T>(entry.clone()).map_err(|e| {
                // `Tag`'s Display renders as `!Name`; untagged entries get no suffix.
                let tag = match entry {
                    yaml_serde::Value::Tagged(tagged) => format!(" ({})", tagged.tag),
                    _ => String::new(),
                };
                anyhow::anyhow!("in `{section}` entry {}{}: {e}", index + 1, tag)
            })
        })
        .collect()
}

/// Valid top-level sections of an input document.
///
/// `version` and `comment` are free-form annotations that faunus itself writes
/// into output files and that shipped atom/energy libraries carry; they are read
/// by nothing but must not be rejected when such a file is used as input.
const TOP_LEVEL_KEYS: &[&str] = &[
    "atoms",
    "molecules",
    "system",
    "energy",
    "analysis",
    "propagate",
    "include",
    "umbrella",
    "wang_landau",
    "version",
    "comment",
];

/// Valid keys directly under `system:`.
const SYSTEM_KEYS: &[&str] = &["cell", "medium", "energy", "blocks", "intermolecular"];

/// Reject unknown keys at the document root and directly under `system:`.
///
/// The document is parsed piecemeal: each section reader extracts only the keys
/// it recognizes (`value.get("propagate")`, …) and ignores the rest. A
/// misspelled section name (`analysiss:`) or a stray key would therefore vanish
/// silently instead of erroring. These two levels have no owning struct that
/// could carry `deny_unknown_fields`, so we validate them explicitly.
///
/// `_`-prefixed keys are intentionally-disabled sections and are always allowed.
pub fn validate_section_keys(root: &yaml_serde::Value) -> anyhow::Result<()> {
    check_allowed_keys("the document root", root, TOP_LEVEL_KEYS)?;
    if let Some(system) = root.get("system") {
        check_allowed_keys("`system`", system, SYSTEM_KEYS)?;
    }
    Ok(())
}

fn check_allowed_keys(
    location: &str,
    value: &yaml_serde::Value,
    allowed: &[&str],
) -> anyhow::Result<()> {
    let yaml_serde::Value::Mapping(map) = value else {
        return Ok(());
    };
    for key in map.keys() {
        // A non-string key can never name a valid section, so reject it rather
        // than skipping it (which would let a mis-typed scalar key slip through).
        let Some(key) = key.as_str() else {
            anyhow::bail!("non-string key {key:?} in {location}");
        };
        if key.starts_with('_') || allowed.contains(&key) {
            continue;
        }
        let hint = did_you_mean(key, allowed)
            .map(|s| format!(" (did you mean `{s}`?)"))
            .unwrap_or_default();
        anyhow::bail!(
            "unknown key `{key}` in {location}{hint}; allowed keys: {}",
            allowed.join(", "),
        );
    }
    Ok(())
}

/// Closest allowed key within a small edit distance, for a typo hint.
fn did_you_mean<'a>(key: &str, allowed: &[&'a str]) -> Option<&'a str> {
    allowed
        .iter()
        .map(|&candidate| (candidate, levenshtein(key, candidate)))
        // Only suggest genuinely-close matches, scaled to the word length.
        .filter(|&(candidate, dist)| dist <= candidate.len().div_ceil(2))
        .min_by_key(|&(_, dist)| dist)
        .map(|(candidate, _)| candidate)
}

/// Levenshtein edit distance between two ASCII-ish strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0; b_chars.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b_chars.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_chars.len()]
}

/// Serialize `value` to `path` as a YAML file, naming the file on failure.
///
/// For checkpoints and other whole-struct dumps; multi-section outputs that
/// share one file handle keep writing sections directly.
pub fn write_yaml_file<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    use anyhow::Context;
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create '{}'", path.display()))?;
    yaml_serde::to_writer(file, value)
        .with_context(|| format!("Cannot write '{}'", path.display()))?;
    Ok(())
}

/// Deserialize a whole YAML file at `path` into `T`, naming the file on failure.
pub fn read_yaml_file<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    use anyhow::Context;
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open '{}'", path.display()))?;
    yaml_serde::from_reader(file).with_context(|| format!("Cannot parse '{}'", path.display()))
}

/// Resolve max thread count: 0 means use all available cores.
// Only the `cli`-gated umbrella / Wang-Landau drivers use this.
#[cfg(feature = "cli")]
pub fn resolve_thread_count(max_threads: usize) -> usize {
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

/// Open a file for reading, transparently decompressing `.gz`.
fn open_read_compressed(path: &Path) -> anyhow::Result<Box<dyn BufRead + Send>> {
    let file = std::fs::File::open(path)
        .map_err(|err| anyhow::anyhow!("Error opening file {path:?}: {err}"))?;
    if path.extension().unwrap_or_default() == "gz" {
        Ok(Box::new(std::io::BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(std::io::BufReader::new(file)))
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
/// The file and its header row are created lazily, on the first
/// [`write_row`](Self::write_row) — constructing a writer for a path touches no disk. A run
/// that is only validated (`--check`) or that fails during setup therefore never truncates an
/// existing output file: you cannot destroy a file you have not written a row to.
pub(crate) struct ColumnWriter {
    format: ColumnFormat,
    state: WriterState,
}

enum WriterState {
    /// A path-backed writer not yet opened; the file and header are created on the first row.
    Pending {
        path: PathBuf,
        columns: Vec<String>,
    },
    Open(Box<dyn Write + Send>),
}

impl std::fmt::Debug for ColumnWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnWriter").finish_non_exhaustive()
    }
}

impl ColumnWriter {
    /// Prepare a file-backed writer, inferring the format from the extension. The file is
    /// created and the header written on the first [`write_row`](Self::write_row), not here.
    pub(crate) fn open(path: &Path, columns: &[&str]) -> anyhow::Result<Self> {
        Ok(Self {
            format: ColumnFormat::from_path(path),
            state: WriterState::Pending {
                path: path.to_path_buf(),
                columns: columns.iter().map(|c| c.to_string()).collect(),
            },
        })
    }

    /// Wrap an already-open writer and write the header now. Used only by tests, where the sink
    /// is an in-memory buffer rather than a path, so there is nothing to defer.
    #[cfg(test)]
    pub(crate) fn new(
        mut inner: Box<dyn Write + Send>,
        format: ColumnFormat,
        columns: &[&str],
    ) -> anyhow::Result<Self> {
        write_header(&mut inner, format, columns)?;
        Ok(Self {
            format,
            state: WriterState::Open(inner),
        })
    }

    /// Open the file and write the header on first use, returning the underlying writer. A
    /// path that cannot be created surfaces here, on the first row, rather than at construction.
    fn writer(&mut self) -> std::io::Result<&mut (dyn Write + Send)> {
        if let WriterState::Pending { path, columns } = &self.state {
            let mut inner =
                open_compressed(path).map_err(|e| io::Error::other(format!("{e:#}")))?;
            write_header(&mut inner, self.format, columns)?;
            self.state = WriterState::Open(inner);
        }
        let WriterState::Open(inner) = &mut self.state else {
            unreachable!("just transitioned to Open");
        };
        Ok(inner.as_mut())
    }

    /// Write a row of values using the format's separator.
    pub(crate) fn write_row(&mut self, values: &[&dyn Display]) -> std::io::Result<()> {
        let sep = self.format.separator();
        let inner = self.writer()?;
        for (i, val) in values.iter().enumerate() {
            if i > 0 {
                write!(inner, "{sep}")?;
            }
            write!(inner, "{val}")?;
        }
        writeln!(inner)
    }

    pub(crate) fn flush(&mut self) -> std::io::Result<()> {
        match &mut self.state {
            WriterState::Open(inner) => inner.flush(),
            // Nothing was written, so there is no open file to flush.
            WriterState::Pending { .. } => Ok(()),
        }
    }
}

/// Write the `# col1 col2 …` header row in `format` to an already-open writer.
fn write_header(
    inner: &mut (impl Write + ?Sized),
    format: ColumnFormat,
    columns: &[impl AsRef<str>],
) -> std::io::Result<()> {
    let sep = format.separator();
    write!(inner, "{}", format.comment_prefix())?;
    for (i, col) in columns.iter().enumerate() {
        if i > 0 {
            write!(inner, "{sep}")?;
        }
        write!(inner, "{}", col.as_ref())?;
    }
    writeln!(inner)
}

/// Reader for single-column numeric data files (.csv, .csv.gz, .dat, .dat.gz).
///
/// Skips header lines (starting with `#` or non-numeric) and parses one `f64` per row.
#[allow(dead_code)]
pub(crate) struct ColumnReader {
    inner: Box<dyn BufRead + Send>,
    format: ColumnFormat,
}

#[allow(dead_code)]
impl ColumnReader {
    pub(crate) fn open(path: &Path) -> anyhow::Result<Self> {
        let inner = open_read_compressed(path)?;
        let format = ColumnFormat::from_path(path);
        Ok(Self { inner, format })
    }
}

impl TryFrom<ColumnReader> for Vec<f64> {
    type Error = anyhow::Error;

    fn try_from(reader: ColumnReader) -> anyhow::Result<Self> {
        let sep = reader.format.separator();
        let mut values = Vec::new();
        for line in reader.inner.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // Take first column only
            let field = trimmed.split(sep).next().unwrap_or(trimmed);
            if let Ok(v) = field.parse::<f64>() {
                values.push(v);
            }
        }
        Ok(values)
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

/// Fit isotropic rotational diffusion coefficient from the trace of Q̃(τ).
///
/// Minimizes the sum of squared residuals between the observed trace values
/// and the Favro isotropic model `Tr(Q̃(τ)) = ¾(1 - exp(-2Dτ))`.
/// Uses Newton–Raphson iteration on the single parameter D.
///
/// The factor 2D (not 6D) follows from Favro Eq. 9 with D_x = D_y = D_z = D:
/// each diagonal element is `Q_ii = ¼(1 - exp(-2Dτ))`, so `Tr = ¾(1 - exp(-2Dτ))`.
///
/// Returns `None` if the input is empty or the fit fails to converge.
///
/// See [Favro (1960)](https://doi.org/10.1103/PhysRev.119.53), Eq. 9.
pub(crate) fn fit_isotropic_d_rot(lags: &[f64], trace: &[f64]) -> Option<f64> {
    if lags.is_empty() || lags.len() != trace.len() || lags.iter().any(|&t| t <= 0.0) {
        return None;
    }

    // Initial guess from the last data point: Tr = ¾(1 - exp(-2Dτ)) → D = -ln(1 - 4/3 Tr) / (2τ)
    let last = lags.len() - 1;
    let arg = 1.0 - 4.0 / 3.0 * trace[last];
    let mut d = if arg > 0.0 {
        -arg.ln() / (2.0 * lags[last])
    } else {
        0.01 // trace ≥ ¾ means fully decorrelated; use a safe default for Newton iteration
    };

    // Newton–Raphson: minimize Σ (model(τ) - data(τ))²
    // model(τ) = ¾(1 - exp(-2Dτ))
    // ∂model/∂D = ¾ · 2τ · exp(-2Dτ) = 1.5τ · exp(-2Dτ)
    // ∂²model/∂D² = -¾ · 4τ² · exp(-2Dτ) = -3τ² · exp(-2Dτ)
    for _ in 0..50 {
        let mut gradient = 0.0;
        let mut hessian = 0.0;
        for (&tau, &tr) in lags.iter().zip(trace.iter()) {
            let e = (-2.0 * d * tau).exp();
            let model = 0.75 * (1.0 - e);
            let residual = model - tr;
            let dm = 1.5 * tau * e;
            let d2m = -3.0 * tau * tau * e;
            gradient += 2.0 * residual * dm;
            hessian += 2.0 * (dm * dm + residual * d2m);
        }
        if hessian.abs() < 1e-30 {
            break;
        }
        let step = gradient / hessian;
        d -= step;
        if d < 0.0 {
            d = 1e-10;
        }
        if step.abs() < 1e-12 * d.abs() {
            return Some(d);
        }
    }
    None // did not converge within iteration limit
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
    #[allow(dead_code)]
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

/// Mean ± SEM snapshot of a block-averaged scalar; canonical YAML shape.
///
/// Produced by `&BlockAverage * scale` (mean is scaled signed, error
/// scaled by `|scale|` since `Var(cX) = c² Var(X)`). Derives `Serialize`
/// and `Deserialize`, so the `{ mean, error }` mappings in `output.yaml`
/// round-trip back into a `BlockSummary` for free.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub(crate) struct BlockSummary {
    pub mean: f64,
    pub error: f64,
}

impl BlockSummary {
    /// Snapshot whose `error` is a sample standard deviation (fluctuation
    /// width) rather than a SEM. Used where the spread itself is the reported
    /// quantity (e.g. charge/dipole fluctuations, equipartition temperatures),
    /// keeping the same `{ mean, error }` YAML shape as block averages.
    pub fn from_fluctuation(mean: f64, std: f64) -> Self {
        Self { mean, error: std }
    }
}

/// Running average with automatic hierarchical blocking for error estimation.
#[derive(Clone, Debug, Default)]
pub(crate) struct BlockAverage(HierarchicalBlockAverage);

impl BlockAverage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one observation.
    pub fn add(&mut self, value: f64) {
        self.0.add(value, 1.0);
    }

    /// Mean over all observations, or `None` if nothing was sampled.
    ///
    /// There is no honest `f64` for the unsampled case: `0.0` reads as a measured zero,
    /// and a NaN sentinel silently defeats the callers' own guards, since every
    /// comparison against NaN is false. `Option` makes each caller say what it means.
    pub fn checked_mean(&self) -> Option<f64> {
        self.0.mean()
    }

    /// Standard error of the mean, corrected for serial correlation by blocking.
    /// NaN below two observations, where the spread is unknown rather than zero.
    pub fn error(&self) -> f64 {
        self.0.error()
    }

    /// Snapshot scaled to a derived quantity: the mean signs with `scale`, the error
    /// takes its magnitude (variance scales by c²). `None` when unsampled.
    pub fn scaled(&self, scale: f64) -> Option<BlockSummary> {
        self.summary().map(|summary| BlockSummary {
            mean: summary.mean * scale,
            error: summary.error * scale.abs(),
        })
    }

    /// Sample standard deviation across observations (σ). Only the unit tests need this.
    #[cfg(test)]
    pub fn stddev(&self) -> f64 {
        self.0.sample_stddev()
    }

    /// Number of observations recorded. Only the unit tests need this.
    #[cfg(test)]
    pub fn n(&self) -> u64 {
        self.0.n()
    }

    /// Snapshot of mean ± SEM in the same `{mean, error}` shape used throughout YAML
    /// output, or `None` when unsampled. Scaled views go through [`scaled`](Self::scaled).
    pub fn summary(&self) -> Option<BlockSummary> {
        self.checked_mean().map(|mean| BlockSummary {
            mean,
            error: self.error(),
        })
    }

    /// Serialize as YAML mapping `{ mean, error }`, or null when unsampled. Serializing
    /// the `Option` keeps the null in place of the mapping, so a caller's `?` reports the
    /// unsampled quantity rather than dropping its siblings from the output.
    pub fn to_yaml(&self) -> Option<yaml_serde::Value> {
        yaml_serde::to_value(self.summary()).ok()
    }
}

/// Fewer effective block means make their sample variance too noisy to be a useful
/// error estimate.
const MIN_EFFECTIVE_BLOCKS: f64 = 16.0;

/// Weighted running moments for one blocking level.
#[derive(Clone, Debug, Default)]
struct WeightedMoments {
    sum_w: f64,
    sum_w2: f64,
    mean: f64,
    /// Weighted sum of squared deviations, `S = Σ wᵢ (xᵢ − x̄)²`.
    sum_sq: f64,
    count: u64,
}

impl WeightedMoments {
    fn add(&mut self, value: f64, weight: f64) {
        self.sum_w += weight;
        self.sum_w2 += weight * weight;
        let delta = value - self.mean;
        self.mean += weight / self.sum_w * delta;
        self.sum_sq += weight * delta * (value - self.mean);
        self.count += 1;
    }

    fn effective_n(&self) -> f64 {
        if self.sum_w2 == 0.0 {
            0.0
        } else {
            self.sum_w * self.sum_w / self.sum_w2
        }
    }

    /// Standard error of the mean. NaN below two effective observations: one sample
    /// fixes the mean but says nothing about its spread, and 0.0 would claim the
    /// opposite — that the mean is known exactly.
    fn error(&self) -> f64 {
        let neff = self.effective_n();
        if neff <= 1.0 || self.sum_w <= 0.0 {
            f64::NAN
        } else {
            (self.sum_sq / (self.sum_w * (neff - 1.0))).sqrt()
        }
    }

    #[cfg(test)]
    fn sample_stddev(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            (self.sum_sq / (self.count - 1) as f64).sqrt()
        }
    }

    /// Uncertainty of the error estimate itself, σ(σ) ≈ σ/√(2(M−1)) for M effective
    /// blocks (Flyvbjerg & Petersen 1989). Sets the scale below which a rise in the
    /// blocking curve is indistinguishable from noise.
    ///
    /// Infinite below two effective blocks, where σ(σ) is undefined. Levels that thin are
    /// filtered out by [`has_reliable_error`](Self::has_reliable_error) long before the
    /// plateau scan; the guard keeps the undefined case from silently reading as "flat".
    fn error_uncertainty(&self) -> f64 {
        let neff = self.effective_n();
        if neff <= 1.0 {
            f64::INFINITY
        } else {
            self.error() / (2.0 * (neff - 1.0)).sqrt()
        }
    }

    /// Whether this level's block means are numerous enough to estimate a variance.
    ///
    /// Gated on the *effective* sample size, not the raw block count, to stay consistent
    /// with `error()`, whose degrees of freedom are `neff − 1`: under skewed weights a
    /// level can hold hundreds of blocks yet carry barely one independent observation.
    fn has_reliable_error(&self) -> bool {
        self.effective_n() >= MIN_EFFECTIVE_BLOCKS
    }
}

/// One point on the blocking curve: an error estimate and the noise on that estimate.
/// Paired in a struct so a level's error can never be compared against another's σ(σ).
#[derive(Clone, Copy, Debug)]
struct BlockingPoint {
    error: f64,
    noise: f64,
}

#[derive(Clone, Copy, Debug)]
struct WeightedSample {
    value: f64,
    weight: f64,
}

impl WeightedSample {
    fn merge(self, other: Self) -> Self {
        let weight = self.weight + other.weight;
        Self {
            value: self.value + other.weight / weight * (other.value - self.value),
            weight,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BlockingLevel {
    moments: WeightedMoments,
    pending: Option<WeightedSample>,
}

/// Shared hierarchy for weighted and unweighted running averages.
#[derive(Clone, Debug, Default)]
struct HierarchicalBlockAverage {
    levels: Vec<BlockingLevel>,
}

impl HierarchicalBlockAverage {
    fn add(&mut self, value: f64, weight: f64) {
        let mut sample = WeightedSample { value, weight };
        let mut index = 0;
        loop {
            if index == self.levels.len() {
                self.levels.push(BlockingLevel::default());
            }
            let level = &mut self.levels[index];
            level.moments.add(sample.value, sample.weight);
            if let Some(previous) = level.pending.take() {
                sample = previous.merge(sample);
                index += 1;
            } else {
                level.pending = Some(sample);
                break;
            }
        }
    }

    /// Level 0 — every observation, unblocked. The mean and the uncorrelated spread
    /// are read from here; only the error consults the coarser levels.
    fn base(&self) -> Option<&WeightedMoments> {
        self.levels.first().map(|level| &level.moments)
    }

    fn mean(&self) -> Option<f64> {
        self.base().map(|moments| moments.mean)
    }

    #[cfg(test)]
    fn effective_n(&self) -> f64 {
        self.base().map_or(0.0, WeightedMoments::effective_n)
    }

    /// Standard error of the mean, corrected for serial correlation by blocking.
    ///
    /// Correlation can only inflate the variance of a mean, so the naive SEM is a floor:
    /// a blocked estimate below it means the blocking curve dipped through noise, or the
    /// data are anti-correlated — a regime this does not claim to resolve, and where the
    /// uncorrelated estimate is the conservative reading.
    fn error(&self) -> f64 {
        let naive = self.base().map_or(f64::NAN, WeightedMoments::error);
        let curve: Vec<BlockingPoint> = self
            .levels
            .iter()
            .map(|level| &level.moments)
            .filter(|moments| moments.has_reliable_error())
            .map(|moments| BlockingPoint {
                error: moments.error(),
                noise: moments.error_uncertainty(),
            })
            .collect();
        Self::plateau_error(&curve).map_or(naive, |error| error.max(naive))
    }

    /// The Flyvbjerg-Petersen plateau, or `None` for a run too short to block.
    ///
    /// Coarsening blocks decorrelates them, so the SEM estimate climbs while adjacent blocks
    /// still share information and levels off once they no longer do. The plateau is the
    /// first level that no *later* level rises above by more than the noise on the two
    /// estimates combined — the curve must stay flat, not merely pause. Testing only the
    /// next level stops at the first slow step of a curve that is still climbing, which is
    /// how a strongly correlated observable reads at fine resolution: successive levels
    /// there differ by less than σ(σ) even though the curve has far to go.
    ///
    /// The coarsest level is never itself a plateau: nothing follows it to show it has
    /// levelled off. A curve still rising at that point belongs to a run too short to
    /// resolve its correlation time, so report the largest estimate rather than a
    /// confident-looking small one.
    fn plateau_error(curve: &[BlockingPoint]) -> Option<f64> {
        let levelled_off = |(index, point): (usize, &BlockingPoint)| {
            curve[index + 1..]
                .iter()
                .all(|later| later.error - point.error <= point.noise.hypot(later.noise))
                .then_some(point.error)
        };
        curve
            .split_last()
            .and_then(|(_, rising)| rising.iter().enumerate().find_map(levelled_off))
            .or_else(|| curve.iter().map(|point| point.error).reduce(f64::max))
    }

    #[cfg(test)]
    fn sample_stddev(&self) -> f64 {
        self.base().map_or(0.0, WeightedMoments::sample_stddev)
    }

    fn n(&self) -> u64 {
        self.base().map_or(0, |moments| moments.count)
    }
}

/// Running reliability-weighted average with automatic hierarchical blocking.
///
/// Each [`add`](Self::add) records one trajectory sample and its reweighting
/// factor (`1.0` for an unbiased run). Adjacent samples are merged recursively
/// into blocks of 2, 4, 8, … samples. The reported error is read off the blocking
/// plateau, while the mean always uses every sample.
#[derive(Clone, Debug, Default)]
pub(crate) struct WeightedBlockAverage(HierarchicalBlockAverage);

impl WeightedBlockAverage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a trajectory value with a reweighting factor. Zero-weight samples
    /// carry no information and do not advance the blocking hierarchy.
    pub fn add(&mut self, value: f64, weight: f64) {
        if weight == 0.0 {
            return;
        }
        self.0.add(value, weight);
    }

    /// Weight-normalised mean, or NaN if no samples have been added.
    pub fn mean(&self) -> f64 {
        self.0.mean().unwrap_or(f64::NAN)
    }

    /// Weight-normalised mean, or `None` if empty — so callers reporting a value
    /// cannot leak the NaN sentinel into the output.
    pub fn checked_mean(&self) -> Option<f64> {
        self.0.mean()
    }

    /// Kish effective sample size of the original weighted observations.
    #[cfg(test)]
    pub fn effective_n(&self) -> f64 {
        self.0.effective_n()
    }

    /// Standard error of the mean, corrected for serial correlation by blocking.
    pub fn error(&self) -> f64 {
        self.0.error()
    }

    /// Number of original trajectory samples recorded.
    pub fn n(&self) -> u64 {
        self.0.n()
    }

    /// Snapshot of mean ± SEM in the canonical `{ mean, error }` YAML shape, or
    /// `None` when empty so an unsampled accumulator serializes as null rather than
    /// a `{nan, 0}` mapping.
    pub fn summary(&self) -> Option<BlockSummary> {
        self.checked_mean().map(|mean| BlockSummary {
            mean,
            error: self.error(),
        })
    }
}

impl Mul<f64> for &WeightedBlockAverage {
    type Output = BlockSummary;

    fn mul(self, scale: f64) -> BlockSummary {
        BlockSummary {
            mean: self.mean() * scale,
            error: self.error() * scale.abs(),
        }
    }
}

/// Extension trait to reduce YAML mapping construction boilerplate.
pub(crate) trait MappingExt {
    /// Insert a serializable value, returning `None` if serialization fails.
    fn try_insert(&mut self, key: &str, value: impl serde::Serialize) -> Option<()>;
}

impl MappingExt for yaml_serde::Mapping {
    fn try_insert(&mut self, key: &str, value: impl serde::Serialize) -> Option<()> {
        self.insert(key.into(), yaml_serde::to_value(value).ok()?);
        Some(())
    }
}

#[cfg(test)]
mod section_parse_tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Widget {
        #[allow(dead_code)]
        name: String,
    }

    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    enum Item {
        Widget(Widget),
    }

    #[test]
    fn tagged_list_error_names_entry_and_tag() {
        let value: yaml_serde::Value =
            yaml_serde::from_str("- !Widget {name: a}\n- !Widget {name: b, oops: 1}\n").unwrap();
        let err = from_tagged_list::<Item>("things", &value)
            .unwrap_err()
            .to_string();
        assert!(err.contains("entry 2"), "{err}");
        assert!(err.contains("!Widget"), "{err}");
        assert!(err.contains("oops"), "{err}");
    }

    #[test]
    fn tagged_list_requires_sequence() {
        let value: yaml_serde::Value = yaml_serde::from_str("name: a").unwrap();
        let err = from_tagged_list::<Item>("things", &value)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be a list"), "{err}");
    }

    #[test]
    fn tagged_list_null_is_empty() {
        // A present-but-null section must behave like an empty list, not an error.
        let value: yaml_serde::Value = yaml_serde::from_str("~").unwrap();
        assert!(from_tagged_list::<Item>("things", &value)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn section_value_error_names_section() {
        let value: yaml_serde::Value = yaml_serde::from_str("name: a\noops: 1").unwrap();
        let err = from_section_value::<Widget>("system/widget", &value)
            .unwrap_err()
            .to_string();
        assert!(err.contains("`system/widget` section"), "{err}");
        assert!(err.contains("oops"), "{err}");
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

    #[test]
    fn column_reader_roundtrip_csv_gz() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.csv.gz");
        let values = vec![1.5, 2.7, 3.15];

        let mut w = ColumnWriter::open(&path, &["cv"]).unwrap();
        for &v in &values {
            w.write_row(&[&v]).unwrap();
        }
        drop(w);

        let reader = ColumnReader::open(&path).unwrap();
        let loaded: Vec<f64> = reader.try_into().unwrap();
        assert_eq!(loaded, values);
    }

    /// The writer must not touch disk until the first row: this is what keeps a validated
    /// (`--check`) or setup-failed run from truncating an existing output file. Opening an
    /// existing file and then never writing must leave its contents intact.
    #[test]
    fn column_writer_opens_lazily_on_first_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.dat");

        std::fs::write(&path, "PRIOR\n").unwrap();
        let mut w = ColumnWriter::open(&path, &["step", "value"]).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "PRIOR\n",
            "constructing the writer must not truncate the existing file"
        );

        w.write_row(&[&1, &2]).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# step value\n1 2\n",
            "the first row creates the file with a header, replacing prior contents"
        );

        // A writer that is only opened (never written) leaves a missing file missing.
        let untouched = dir.path().join("never.dat");
        let _ = ColumnWriter::open(&untouched, &["x"]).unwrap();
        assert!(!untouched.exists(), "opening alone must create no file");
    }

    #[test]
    fn column_reader_skips_comments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.dat");
        std::fs::write(&path, "# header\n1.0\n2.0\n").unwrap();

        let reader = ColumnReader::open(&path).unwrap();
        let loaded: Vec<f64> = reader.try_into().unwrap();
        assert_eq!(loaded, vec![1.0, 2.0]);
    }
}

#[cfg(test)]
mod fit_d_rot_tests {
    use super::fit_isotropic_d_rot;

    #[test]
    fn recovers_known_d() {
        let d_true = 0.05;
        let lags: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let trace: Vec<f64> = lags
            .iter()
            .map(|&tau| 0.75 * (1.0 - (-2.0 * d_true * tau).exp()))
            .collect();
        let d_fit = fit_isotropic_d_rot(&lags, &trace).unwrap();
        assert!(
            (d_fit - d_true).abs() < 1e-10,
            "d_fit={d_fit}, expected={d_true}"
        );
    }

    #[test]
    fn recovers_small_d() {
        let d_true = 0.001;
        let lags: Vec<f64> = (1..=500).map(|i| i as f64).collect();
        let trace: Vec<f64> = lags
            .iter()
            .map(|&tau| 0.75 * (1.0 - (-2.0 * d_true * tau).exp()))
            .collect();
        let d_fit = fit_isotropic_d_rot(&lags, &trace).unwrap();
        assert!(
            (d_fit - d_true).abs() / d_true < 1e-8,
            "d_fit={d_fit}, expected={d_true}"
        );
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(fit_isotropic_d_rot(&[], &[]).is_none());
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
mod block_tests {
    use super::BlockAverage;
    use approx::assert_relative_eq;

    #[test]
    fn scaling_signs_the_mean_and_absolutes_the_error() {
        let mut b = BlockAverage::new();
        b.add(1.0);
        b.add(3.0);
        let base = b.scaled(1.0).unwrap();
        let scaled = b.scaled(-2.0).unwrap();
        // mean: signed scaling
        assert_relative_eq!(scaled.mean, base.mean * -2.0);
        // error: absolute scaling (variance scales by c²)
        assert_relative_eq!(scaled.error, base.error * 2.0);
        // The unscaled-by-1 form matches direct accessors
        assert_relative_eq!(base.mean, b.checked_mean().unwrap());
        assert_relative_eq!(base.error, b.error());
    }

    /// An unsampled accumulator has no mean to report. Reporting 0.0 — what
    /// `average::Variance` returns and what this wrapper used to pass through — is
    /// indistinguishable from a genuine measured zero, and no caller could tell them
    /// apart because the sample count is not on the public surface.
    #[test]
    fn an_unsampled_average_has_no_summary_and_serializes_as_null() {
        let empty = BlockAverage::new();
        assert_eq!(empty.checked_mean(), None);
        assert!(empty.summary().is_none());
        assert!(empty.scaled(2.0).is_none());
        assert_eq!(empty.to_yaml(), Some(yaml_serde::Value::Null));
    }

    /// One observation fixes the mean but says nothing about its spread. `0.0` claims
    /// the opposite — that the mean is known exactly.
    #[test]
    fn a_single_observation_reports_an_unknown_error() {
        let mut average = BlockAverage::new();
        average.add(5.0);

        assert_eq!(average.checked_mean(), Some(5.0));
        assert!(average.error().is_nan());
        let summary = average.summary().expect("a sampled average has a summary");
        assert_relative_eq!(summary.mean, 5.0);
        assert!(summary.error.is_nan());
    }

    #[test]
    fn short_run_uses_the_ordinary_standard_error() {
        let mut average = BlockAverage::new();
        for value in [1.0, 2.0, 4.0, 8.0, 16.0] {
            average.add(value);
        }

        let sample_variance = 37.2_f64;
        assert_relative_eq!(average.checked_mean().unwrap(), 6.2, epsilon = 1e-12);
        assert_relative_eq!(average.stddev(), sample_variance.sqrt(), epsilon = 1e-12);
        assert_relative_eq!(
            average.error(),
            (sample_variance / 5.0).sqrt(),
            epsilon = 1e-12
        );
        assert_eq!(average.n(), 5);
    }
}

#[cfg(test)]
mod weighted_block_tests {
    use super::{BlockAverage, WeightedBlockAverage};
    use approx::assert_relative_eq;

    /// Both wrappers delegate to the same engine, so comparing them to each other
    /// proves nothing. Pin them to a hand-computed SEM instead: five values whose
    /// sample variance is 37.2, giving √(37.2/5) over a run too short to block.
    #[test]
    fn unit_weights_match_the_analytical_standard_error() {
        let mut plain = BlockAverage::new();
        let mut weighted = WeightedBlockAverage::new();
        for value in [1.0, 2.0, 4.0, 8.0, 16.0] {
            plain.add(value);
            weighted.add(value, 1.0);
        }
        let expected_error = (37.2_f64 / 5.0).sqrt();
        for (mean, error) in [
            (plain.checked_mean().unwrap(), plain.error()),
            (weighted.mean(), weighted.error()),
        ] {
            assert_relative_eq!(mean, 6.2, epsilon = 1e-12);
            assert_relative_eq!(error, expected_error, epsilon = 1e-12);
        }
        assert_eq!(weighted.n(), 5);
        assert_relative_eq!(weighted.effective_n(), 5.0, epsilon = 1e-12);
    }

    /// A rescaled weight set leaves the mean and the effective sample size
    /// unchanged: only relative weights matter.
    #[test]
    fn weighted_mean_and_effective_n() {
        let mut wba = WeightedBlockAverage::new();
        // weight 3 on 2.0, weight 1 on 6.0 → mean = (6 + 6)/4 = 3.0
        wba.add(2.0, 3.0);
        wba.add(6.0, 1.0);
        assert_relative_eq!(wba.mean(), 3.0, epsilon = 1e-12);
        // N_eff = (3+1)² / (9+1) = 16/10 = 1.6
        assert_relative_eq!(wba.effective_n(), 1.6, epsilon = 1e-12);
    }

    /// A single effective sample fixes the mean and says nothing about its spread, so the
    /// error is unknown rather than zero — reporting 0.0 would claim the mean is exact.
    #[test]
    fn single_effective_sample_has_an_unknown_error() {
        let mut wba = WeightedBlockAverage::new();
        wba.add(5.0, 1.0);
        assert_eq!(wba.checked_mean(), Some(5.0));
        assert!(wba.error().is_nan());

        let empty = WeightedBlockAverage::new();
        assert_eq!(empty.checked_mean(), None);
        assert!(empty.mean().is_nan());
        assert!(empty.error().is_nan());
        assert!(empty.summary().is_none());
    }

    #[test]
    fn scaling_signs_the_mean_and_absolutes_the_error() {
        let mut wba = WeightedBlockAverage::new();
        wba.add(1.0, 2.0);
        wba.add(3.0, 2.0);
        let scaled = &wba * -2.0;
        assert_relative_eq!(scaled.mean, wba.mean() * -2.0, epsilon = 1e-12);
        assert_relative_eq!(scaled.error, wba.error() * 2.0, epsilon = 1e-12);
    }

    #[test]
    fn hierarchical_blocking_preserves_weighted_mean_with_incomplete_block() {
        let mut average = WeightedBlockAverage::new();
        let mut weighted_sum = 0.0;
        let mut sum_weights = 0.0;

        for index in 0..513 {
            let value = (index % 7) as f64;
            let weight = (index % 5 + 1) as f64;
            average.add(value, weight);
            weighted_sum += weight * value;
            sum_weights += weight;
        }

        assert_relative_eq!(average.mean(), weighted_sum / sum_weights, epsilon = 1e-12);
        assert_eq!(average.n(), 513);
    }
}

/// Analytical tests of the blocking engine itself. Both public wrappers delegate here,
/// so the statistics are pinned once, against closed-form answers rather than against
/// each other.
#[cfg(test)]
mod blocking_engine_tests {
    use super::{HierarchicalBlockAverage, WeightedMoments};
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};

    /// Seeded so the blocking curves below are the same on every run and platform.
    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0x4d59_5df4_d0f3_3173)
    }

    fn engine(values: &[f64]) -> HierarchicalBlockAverage {
        let mut average = HierarchicalBlockAverage::default();
        for value in values {
            average.add(*value, 1.0);
        }
        average
    }

    /// The correlation-blind estimate the blocking curve starts from.
    fn naive_error(average: &HierarchicalBlockAverage) -> f64 {
        average.base().map_or(0.0, WeightedMoments::error)
    }

    /// σ/√N over an independent sample — the answer blocking must reproduce.
    fn analytical_sem(values: &[f64]) -> f64 {
        values.iter().collect::<average::Variance>().error()
    }

    /// Independent samples have no plateau to find: every level estimates the same
    /// variance, so blocking must land on the uncorrelated SEM, √(1/12N) for U(0,1).
    #[test]
    fn independent_samples_reproduce_the_uncorrelated_sem() {
        const N: usize = 4096;
        let mut rng = test_rng();
        let values: Vec<f64> = (0..N).map(|_| rng.gen::<f64>()).collect();
        let average = engine(&values);

        assert_relative_eq!(
            average.error(),
            (1.0 / (12.0 * N as f64)).sqrt(),
            max_relative = 0.35
        );
        assert_relative_eq!(
            average.error(),
            analytical_sem(&values),
            max_relative = 0.35
        );
    }

    /// Each independent draw repeated eight times: the correlation time is exactly 8
    /// samples, so the true SEM is that of the 64 underlying draws and the naive
    /// estimate understates it by √8. Blocking must recover the former.
    #[test]
    fn known_correlation_time_recovers_the_independent_sem() {
        const BLOCKS: usize = 64;
        const REPEATS: usize = 8;
        let mut rng = test_rng();
        let independent: Vec<f64> = (0..BLOCKS).map(|_| rng.gen::<f64>()).collect();
        let correlated: Vec<f64> = independent
            .iter()
            .flat_map(|value| std::iter::repeat_n(*value, REPEATS))
            .collect();
        let average = engine(&correlated);
        let truth = analytical_sem(&independent);

        // Level 3 blocks exactly one repeat run, so its block means *are* the draws.
        assert_relative_eq!(average.levels[3].moments.error(), truth, epsilon = 1e-12);
        assert_relative_eq!(average.error(), truth, max_relative = 0.15);
        assert!(
            average.error() > 2.5 * naive_error(&average),
            "blocking must expose the √8 the naive SEM misses"
        );
    }

    /// Anti-correlated data: every block of two averages to exactly 0.5, so all coarse
    /// levels report zero spread. Correlation only ever inflates the variance of a mean,
    /// so the naive SEM is a floor — reporting the coarse levels verbatim would claim a
    /// visibly fluctuating observable is known exactly.
    #[test]
    fn anti_correlated_samples_never_report_below_the_naive_sem() {
        let values: Vec<f64> = (0..512).map(|index| (index % 2) as f64).collect();
        let average = engine(&values);

        assert!(average.levels[1..]
            .iter()
            .filter(|level| level.moments.has_reliable_error())
            .all(|level| level.moments.error() == 0.0));
        assert_relative_eq!(average.error(), naive_error(&average), epsilon = 1e-12);
        assert!(average.error() > 0.0);
    }

    /// A single dominant weight leaves ~1 effective observation at every level however
    /// many blocks each holds. Gating on the raw count would call such a level reliable
    /// and prefer it; gating on `effective_n` disqualifies all of them.
    #[test]
    fn skewed_weights_disqualify_every_blocking_level() {
        let mut average = HierarchicalBlockAverage::default();
        average.add(1.0, 1e4);
        for index in 0..511 {
            average.add((index % 7) as f64, 1.0);
        }

        assert!(average.levels[0].moments.count >= 512);
        assert!(average.levels[0].moments.effective_n() < 2.0);
        assert!(!average
            .levels
            .iter()
            .any(|level| level.moments.has_reliable_error()));
        assert_relative_eq!(average.error(), naive_error(&average), epsilon = 1e-12);
    }

    /// A slow mode buried under fast noise: the curve creeps at fine resolution — the first
    /// step rises by less than σ(σ) — then climbs steeply once blocks span the slow mode.
    /// Judging the plateau from the next level alone reads that creep as a plateau and
    /// reports the naive SEM for data correlated by a factor of ~2.5. Real observables look
    /// like this (`cluster_lj`'s cluster size does), and a random walk does not: it rises
    /// too steeply at every level to expose the mistake.
    #[test]
    fn a_slowly_rising_curve_is_not_mistaken_for_a_plateau() {
        const RUN: usize = 256;
        let mut rng = test_rng();
        let mut slow = 0.0;
        let values: Vec<f64> = (0..4096)
            .map(|index| {
                if index % RUN == 0 {
                    // A few percent of the noise variance: gentle enough that the first
                    // blocking step is lost in σ(σ), yet it dominates once blocks span a run.
                    slow = (rng.gen::<f64>() - 0.5) * 0.18;
                }
                slow + rng.gen::<f64>() - 0.5
            })
            .collect();
        let average = engine(&values);

        let first_step =
            average.levels[1].moments.error() / average.levels[0].moments.error() - 1.0;
        assert!(
            first_step
                < average.levels[0].moments.error_uncertainty() / average.levels[0].moments.error(),
            "test is only meaningful while the first step hides inside σ(σ)"
        );
        assert!(
            average.error() > 1.8 * naive_error(&average),
            "a creeping curve is still a rising curve, not a plateau"
        );
    }

    /// A random walk stays correlated at every resolved scale, so the blocking curve
    /// never levels off. With no plateau the run is too short to resolve the correlation
    /// time, and the largest eligible estimate is the conservative reading.
    #[test]
    fn unresolved_correlation_falls_back_to_the_largest_estimate() {
        let mut rng = test_rng();
        let mut position = 0.0;
        let values: Vec<f64> = (0..512)
            .map(|_| {
                position += rng.gen::<f64>() - 0.5;
                position
            })
            .collect();
        let average = engine(&values);

        let largest = average
            .levels
            .iter()
            .filter(|level| level.moments.has_reliable_error())
            .map(|level| level.moments.error())
            .fold(f64::NEG_INFINITY, f64::max);
        assert_relative_eq!(average.error(), largest, epsilon = 1e-12);
        assert!(average.error() > 3.0 * naive_error(&average));
    }
}

#[cfg(test)]
mod template_detect_tests {
    use super::looks_like_template;

    #[test]
    fn real_statement_line_is_a_template() {
        assert!(looks_like_template(
            "{% set lz = 30.0 %}\ncell: !Cuboid [{{ lz }}, 1, 1]\n"
        ));
    }

    #[test]
    fn jinja_comment_block_is_a_template() {
        assert!(looks_like_template("foo: 1  {# inline jinja comment #}\n"));
    }

    #[test]
    fn tag_inside_yaml_comment_is_not_a_template() {
        // Regression for the false positive: `{%`/`{#` in a plain YAML comment must not
        // route the whole file through the strict template engine.
        assert!(!looks_like_template(
            "spacing: 5.0  # try {% production %} later\n"
        ));
        assert!(!looks_like_template(
            "# see docs on {# templating #}\nspacing: 5.0\n"
        ));
    }

    #[test]
    fn multiline_block_comment_is_a_template() {
        // docs/index.md documents a `{# … #}` block comment (delimiters on separate lines)
        // to disable whole YAML sections; it must still be detected and rendered.
        assert!(looks_like_template(
            "{# Disabled section:\numbrella:\n  coordinate: ...\n#}\natoms: []\n"
        ));
    }

    #[test]
    fn multiline_statement_is_a_template() {
        assert!(looks_like_template(
            "{% set xs = [\n  1, 2, 3,\n] %}\nn: {{ xs | length }}\n"
        ));
    }

    #[test]
    fn hash_inside_jinja_comment_still_detects() {
        // A `#` inside `{# … #}` must not defeat detection of the opener.
        assert!(looks_like_template("foo: 1  {# see item #7 #}\n"));
    }

    #[test]
    fn plain_yaml_is_not_a_template() {
        assert!(!looks_like_template("atoms:\n  - {name: A, mass: 1.0}\n"));
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

    // --- read_yaml template tests ---

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("faunus_test_{name}.yaml"));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn read_yaml_plain_passthrough() {
        let path = write_temp("plain", "key: value\nlist: [1, 2, 3]");
        let result = super::read_yaml(&path).unwrap();
        assert_eq!(result, "key: value\nlist: [1, 2, 3]");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn read_yaml_template_variables() {
        let path = write_temp("vars", "{% set x = 42 %}\nvalue: {{ x }}");
        let result = super::read_yaml(&path).unwrap();
        assert!(result.contains("value: 42"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn read_yaml_template_comment_only() {
        let path = write_temp("comment", "{# This is a comment #}\nplain: yaml");
        let result = super::read_yaml(&path).unwrap();
        assert!(result.contains("plain: yaml"));
        assert!(!result.contains("comment"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn read_yaml_syntax_error_reports_file() {
        let path = write_temp("bad", "{% for %}");
        let err = super::read_yaml(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Template error"), "missing prefix: {msg}");
        assert!(msg.contains("faunus_test_bad"), "missing filename: {msg}");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn read_yaml_missing_file() {
        let err = super::read_yaml("/nonexistent/file.yaml").unwrap_err();
        assert!(format!("{err}").contains("Cannot read"));
    }

    #[test]
    fn read_yaml_yaml_tags_preserved() {
        let path = write_temp(
            "tags",
            "{% set v = 1.0 %}\natom: {σ: {{ v }}, hydrophobicity: !Lambda 0.0}",
        );
        let result = super::read_yaml(&path).unwrap();
        assert!(result.contains("!Lambda 0.0"), "YAML tag lost: {result}");
        assert!(result.contains("σ: 1.0"), "variable not rendered: {result}");
        std::fs::remove_file(path).ok();
    }
}
