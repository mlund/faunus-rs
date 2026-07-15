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
use std::io::{BufRead, Write};
use std::ops::Mul;
use std::path::Path;

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
    let mut value: serde_yml::Value = serde_yml::from_str(yaml)?;
    if let serde_yml::Value::Mapping(ref mut map) = value {
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
    Ok(serde_yml::to_string(&value)?)
}

/// Parse a named section from a YAML input file into a typed config struct.
// Only the `cli`-gated umbrella / Wang-Landau drivers use this.
#[cfg(feature = "cli")]
pub fn parse_yaml_section<T: serde::de::DeserializeOwned>(
    input: &Path,
    key: &str,
) -> anyhow::Result<T> {
    let yaml = read_yaml(input)?;
    let value: serde_yml::Value = serde_yml::from_str(&yaml)?;
    let section = value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Missing `{key}:` section in input file"))?;
    from_section_value(key, section)
}

/// Deserialize a section's already-extracted [`Value`](serde_yml::Value),
/// labeling any error with the section name so the user knows where to look.
///
/// The location is gone once the document is a parsed `Value`, so the section
/// name is the best anchor we can attach without re-parsing from the source.
/// `section` is the dotted path to the section (e.g. `system/medium`), not a
/// single literal key, so the message reads as a location rather than a key.
pub fn from_section_value<T: serde::de::DeserializeOwned>(
    section: &str,
    value: &serde_yml::Value,
) -> anyhow::Result<T> {
    serde_yml::from_value(value.clone()).map_err(|e| anyhow::anyhow!("in `{section}` section: {e}"))
}

/// Deserialize a YAML sequence of (typically tagged) entries one at a time so an
/// error names the offending entry by 1-based index and YAML tag, e.g.
/// `in `analysis` entry 1 (!RadialDistribution): unknown field ...`.
///
/// A null section (the key present with no value) yields an empty list, matching
/// how `from_value::<Vec<_>>` treats `Null`.
pub fn from_tagged_list<T: serde::de::DeserializeOwned>(
    section: &str,
    value: &serde_yml::Value,
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
            serde_yml::from_value::<T>(entry.clone()).map_err(|e| {
                // `Tag`'s Display renders as `!Name`; untagged entries get no suffix.
                let tag = match entry {
                    serde_yml::Value::Tagged(tagged) => format!(" ({})", tagged.tag),
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
pub fn validate_section_keys(root: &serde_yml::Value) -> anyhow::Result<()> {
    check_allowed_keys("the document root", root, TOP_LEVEL_KEYS)?;
    if let Some(system) = root.get("system") {
        check_allowed_keys("`system`", system, SYSTEM_KEYS)?;
    }
    Ok(())
}

fn check_allowed_keys(
    location: &str,
    value: &serde_yml::Value,
    allowed: &[&str],
) -> anyhow::Result<()> {
    let serde_yml::Value::Mapping(map) = value else {
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
    serde_yml::to_writer(file, value)
        .with_context(|| format!("Cannot write '{}'", path.display()))?;
    Ok(())
}

/// Deserialize a whole YAML file at `path` into `T`, naming the file on failure.
pub fn read_yaml_file<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    use anyhow::Context;
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open '{}'", path.display()))?;
    serde_yml::from_reader(file).with_context(|| format!("Cannot parse '{}'", path.display()))
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

/// Running block average with mean and standard error of the mean.
///
/// Wraps [`average::Variance`] with convenience methods for reporting.
/// Each call to [`add`](Self::add) represents one block measurement.
#[derive(Clone, Debug, Default)]
pub(crate) struct BlockAverage(average::Variance);

use average::Estimate as _;

impl Mul<f64> for &BlockAverage {
    type Output = BlockSummary;

    fn mul(self, scale: f64) -> BlockSummary {
        BlockSummary {
            mean: self.mean() * scale,
            error: self.error() * scale.abs(),
        }
    }
}

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

    /// Sample standard deviation across blocks (σ). Only the unit tests need this.
    #[cfg(test)]
    pub fn stddev(&self) -> f64 {
        self.0.sample_variance().sqrt()
    }

    /// Number of blocks recorded. Only the unit tests need this.
    #[cfg(test)]
    pub fn n(&self) -> u64 {
        self.0.len()
    }

    /// Snapshot of mean ± SEM in the same `{mean, error}` shape used
    /// throughout YAML output. Scaled views go through `&self * scale`.
    pub fn summary(&self) -> BlockSummary {
        BlockSummary {
            mean: self.mean(),
            error: self.error(),
        }
    }

    /// Serialize as YAML mapping `{ mean, error }` via [`BlockSummary`].
    pub fn to_yaml(&self) -> Option<serde_yml::Value> {
        serde_yml::to_value(self.summary()).ok()
    }
}

/// Running block average with a reliability-weighted mean and standard error.
///
/// Each [`add`](Self::add) records one block measurement together with a
/// reweighting factor (`1.0` for an unbiased run, the trajectory weight for a
/// penalty/Wang–Landau rerun). The mean is weight-normalised; the error is a
/// standard error built on the Kish effective sample size
/// `N_eff = (Σwᵢ)² / Σwᵢ²`, so `SEM² = S / (Σwᵢ · (N_eff − 1))` with
/// `S = Σ wᵢ (xᵢ − x̄)²` accumulated by West's algorithm. With every weight
/// `1.0` this reduces exactly to the unweighted [`BlockAverage`] SEM.
#[derive(Clone, Debug, Default)]
pub(crate) struct WeightedBlockAverage {
    sum_w: f64,
    sum_w2: f64,
    mean: f64,
    /// Weighted sum of squared deviations, `S = Σ wᵢ (xᵢ − x̄)²`.
    sum_sq: f64,
    count: u64,
}

impl WeightedBlockAverage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a per-block value with a reweighting factor. Zero-weight samples
    /// are ignored (they carry no information and would only perturb `N_eff`).
    pub fn add(&mut self, value: f64, weight: f64) {
        if weight == 0.0 {
            return;
        }
        self.sum_w += weight;
        self.sum_w2 += weight * weight;
        let delta = value - self.mean;
        self.mean += weight / self.sum_w * delta;
        self.sum_sq += weight * delta * (value - self.mean);
        self.count += 1;
    }

    /// Weight-normalised mean, or NaN if no samples have been added.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.mean
        }
    }

    /// Weight-normalised mean, or `None` if empty — so callers reporting a value
    /// cannot leak the NaN sentinel into the output.
    pub fn checked_mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Kish effective sample size `(Σwᵢ)² / Σwᵢ²`; the number of equally-weighted
    /// samples that would carry the same information.
    pub fn effective_n(&self) -> f64 {
        if self.sum_w2 == 0.0 {
            0.0
        } else {
            self.sum_w * self.sum_w / self.sum_w2
        }
    }

    /// Standard error of the weighted mean. Zero for fewer than two effective
    /// samples, where the spread is undefined.
    pub fn error(&self) -> f64 {
        let neff = self.effective_n();
        if neff <= 1.0 || self.sum_w <= 0.0 {
            0.0
        } else {
            (self.sum_sq / (self.sum_w * (neff - 1.0))).sqrt()
        }
    }

    /// Number of blocks recorded.
    pub fn n(&self) -> u64 {
        self.count
    }

    /// Snapshot of mean ± SEM in the canonical `{ mean, error }` YAML shape, or
    /// `None` when empty so an unsampled accumulator serializes as null rather than
    /// a `{nan, 0}` mapping.
    pub fn summary(&self) -> Option<BlockSummary> {
        (self.count > 0).then(|| BlockSummary {
            mean: self.mean,
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

impl MappingExt for serde_yml::Mapping {
    fn try_insert(&mut self, key: &str, value: impl serde::Serialize) -> Option<()> {
        self.insert(key.into(), serde_yml::to_value(value).ok()?);
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
        let value: serde_yml::Value =
            serde_yml::from_str("- !Widget {name: a}\n- !Widget {name: b, oops: 1}\n").unwrap();
        let err = from_tagged_list::<Item>("things", &value)
            .unwrap_err()
            .to_string();
        assert!(err.contains("entry 2"), "{err}");
        assert!(err.contains("!Widget"), "{err}");
        assert!(err.contains("oops"), "{err}");
    }

    #[test]
    fn tagged_list_requires_sequence() {
        let value: serde_yml::Value = serde_yml::from_str("name: a").unwrap();
        let err = from_tagged_list::<Item>("things", &value)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be a list"), "{err}");
    }

    #[test]
    fn tagged_list_null_is_empty() {
        // A present-but-null section must behave like an empty list, not an error.
        let value: serde_yml::Value = serde_yml::from_str("~").unwrap();
        assert!(from_tagged_list::<Item>("things", &value)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn section_value_error_names_section() {
        let value: serde_yml::Value = serde_yml::from_str("name: a\noops: 1").unwrap();
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
    use super::{BlockAverage, BlockSummary};
    use approx::assert_relative_eq;

    #[test]
    fn block_summary_serde_roundtrip() {
        let original = BlockSummary {
            mean: -0.0615,
            error: 0.0013,
        };
        let yaml = serde_yml::to_string(&original).unwrap();
        let parsed: BlockSummary = serde_yml::from_str(&yaml).unwrap();
        assert_relative_eq!(parsed.mean, original.mean);
        assert_relative_eq!(parsed.error, original.error);
    }

    #[test]
    fn block_average_mul_scales_mean_signed_error_absolute() {
        let mut b = BlockAverage::new();
        b.add(1.0);
        b.add(3.0);
        let base = &b * 1.0;
        let scaled = &b * -2.0;
        // mean: signed scaling
        assert_relative_eq!(scaled.mean, base.mean * -2.0);
        // error: absolute scaling (variance scales by c²)
        assert_relative_eq!(scaled.error, base.error * 2.0);
        // The unscaled-by-1 form matches direct accessors
        assert_relative_eq!(base.mean, b.mean());
        assert_relative_eq!(base.error, b.error());
    }
}

#[cfg(test)]
mod weighted_block_tests {
    use super::{BlockAverage, WeightedBlockAverage};
    use approx::assert_relative_eq;

    /// With every weight 1.0 the weighted accumulator must reproduce the
    /// unweighted block mean and SEM exactly.
    #[test]
    fn unit_weights_match_block_average() {
        let values = [1.0, 2.0, 4.0, 8.0, 16.0];
        let mut plain = BlockAverage::new();
        let mut weighted = WeightedBlockAverage::new();
        for &v in &values {
            plain.add(v);
            weighted.add(v, 1.0);
        }
        assert_relative_eq!(weighted.mean(), plain.mean(), epsilon = 1e-12);
        assert_relative_eq!(weighted.error(), plain.error(), epsilon = 1e-12);
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

    /// A dominant weight collapses the effective sample size toward one, and the
    /// error toward zero — a single frame carries no spread.
    #[test]
    fn single_effective_sample_has_zero_error() {
        let mut wba = WeightedBlockAverage::new();
        wba.add(5.0, 1.0);
        assert_eq!(wba.error(), 0.0);
        let empty = WeightedBlockAverage::new();
        assert!(empty.mean().is_nan());
        assert_eq!(empty.error(), 0.0);
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
            "{# Disabled section:\numbrella:\n  cv: ...\n#}\natoms: []\n"
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
    fn read_yaml_template_for_loop() {
        let path = write_temp(
            "loop",
            "items:\n{% for i in range(3) %}\n  - {{ i }}\n{% endfor %}",
        );
        let result = super::read_yaml(&path).unwrap();
        assert!(result.contains("- 0"));
        assert!(result.contains("- 1"));
        assert!(result.contains("- 2"));
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
    fn read_yaml_template_math() {
        let path = write_temp(
            "math",
            "{% set a = 3.8 %}{% set b = 5.0 %}\nσ: {{ (a + b) / 2 }}",
        );
        let result = super::read_yaml(&path).unwrap();
        assert!(result.contains("σ: 4.4"));
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
