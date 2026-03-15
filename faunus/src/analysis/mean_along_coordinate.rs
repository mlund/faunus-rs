// Copyright 2025 Mikael Lund
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

//! Average of one collective variable binned along another.
//!
//! Each sample evaluates two CVs: a *property* (CV1) and a *coordinate* (CV2).
//! CV2 is discretised into uniform bins of width `resolution`, and the
//! property value is accumulated into a per-bin running mean.

use super::{Analyze, Frequency};
use crate::collective_variable::{CollectiveVariable, CollectiveVariableBuilder};
use crate::Context;
use anyhow::Result;
use average::{Estimate, Mean};
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

/// YAML builder for [`MeanAlongCoordinate`].
///
/// CV1 (property to average) fields are flattened at the top level.
/// CV2 (binning coordinate) lives under the `coordinate:` key and
/// must include a `resolution` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanAlongCoordinateBuilder {
    #[serde(flatten)]
    pub cv: CollectiveVariableBuilder,
    pub coordinate: CollectiveVariableBuilder,
    pub file: PathBuf,
    pub frequency: Frequency,
}

impl MeanAlongCoordinateBuilder {
    pub fn build(&self, context: &impl Context) -> Result<MeanAlongCoordinate> {
        let cv = self.cv.build(context)?;
        let coordinate = self.coordinate.build(context)?;
        let resolution = coordinate
            .axis()
            .resolution
            .filter(|&r| r > 0.0)
            .ok_or_else(|| anyhow::anyhow!("coordinate requires a positive 'resolution' field"))?;

        Ok(MeanAlongCoordinate {
            cv,
            coordinate,
            resolution,
            bins: BTreeMap::new(),
            cv_mean: Mean::new(),
            coord_mean: Mean::new(),
            output_file: self.file.clone(),
            frequency: self.frequency,
        })
    }
}

/// Tracks the mean of one CV binned along another CV.
///
/// Uses a [`BTreeMap`] for automatic ordering and no range requirement.
#[derive(Debug)]
pub struct MeanAlongCoordinate {
    cv: CollectiveVariable,
    coordinate: CollectiveVariable,
    resolution: f64,
    bins: BTreeMap<i64, Mean>,
    cv_mean: Mean,
    coord_mean: Mean,
    #[debug(skip)]
    output_file: PathBuf,
    frequency: Frequency,
}

impl MeanAlongCoordinate {
    /// Map a coordinate value to a bin index.
    fn bin_index(&self, value: f64) -> i64 {
        (value / self.resolution).floor() as i64
    }

    /// Center of a bin given its index.
    fn bin_center(&self, index: i64) -> f64 {
        (index as f64 + 0.5) * self.resolution
    }

    /// Write current averages to file, recreating it each time.
    fn write_output(&self) -> Result<()> {
        if self.bins.is_empty() {
            return Ok(());
        }
        let mut stream = crate::auxiliary::open_compressed(&self.output_file)?;
        let cv_name = &self.cv.axis().name;
        let coord_name = &self.coordinate.axis().name;
        writeln!(stream, "# {coord_name} mean({cv_name}) count")?;
        for (&idx, mean) in &self.bins {
            let center = self.bin_center(idx);
            let avg = mean.mean();
            let count = mean.len();
            writeln!(stream, "{center:.6} {avg:.6} {count}")?;
        }
        stream.flush()?;
        Ok(())
    }
}

impl crate::Info for MeanAlongCoordinate {
    fn short_name(&self) -> Option<&'static str> {
        Some("meanalongcoordinate")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Mean of collective variable along coordinate")
    }
}

impl<T: Context> Analyze<T> for MeanAlongCoordinate {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }
        let coord_value = self.coordinate.evaluate(context);
        let cv_value = self.cv.evaluate(context);
        self.coord_mean.add(coord_value);
        self.cv_mean.add(cv_value);
        let idx = self.bin_index(coord_value);
        self.bins.entry(idx).or_default().add(cv_value);
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.cv_mean.len() as usize
    }

    fn write_to_disk(&mut self) -> Result<()> {
        self.write_output()
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        if self.bins.is_empty() {
            return None;
        }
        let mut map = serde_yaml::Mapping::new();
        map.insert(
            "property".into(),
            serde_yaml::Value::String(self.cv.axis().name.clone()),
        );
        map.insert(
            "mean_property".into(),
            serde_yaml::to_value(self.cv_mean.mean()).ok()?,
        );
        map.insert(
            "coordinate".into(),
            serde_yaml::Value::String(self.coordinate.axis().name.clone()),
        );
        map.insert(
            "mean_coordinate".into(),
            serde_yaml::to_value(self.coord_mean.mean()).ok()?,
        );
        map.insert(
            "num_samples".into(),
            serde_yaml::Value::Number((self.cv_mean.len() as usize).into()),
        );
        map.insert(
            "num_bins".into(),
            serde_yaml::Value::Number(self.bins.len().into()),
        );
        Some(serde_yaml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
property: volume
coordinate:
  property: volume
  resolution: 0.5
file: test.dat
frequency: !Every 100
"#;
        let builder: MeanAlongCoordinateBuilder = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(builder.coordinate.resolution, Some(0.5));
        assert_eq!(builder.file.to_str().unwrap(), "test.dat");
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !MeanAlongCoordinate
  property: volume
  coordinate:
    property: volume
    resolution: 0.5
  file: test.dat
  frequency: !Every 100
"#;
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(
            builders[0],
            AnalysisBuilder::MeanAlongCoordinate(_)
        ));
    }

    #[test]
    fn bin_index_and_center() {
        let resolution = 1.0;
        let bin_index = |v: f64| (v / resolution).floor() as i64;
        let bin_center = |i: i64| (i as f64 + 0.5) * resolution;

        assert_eq!(bin_index(0.0), 0);
        assert_eq!(bin_index(0.9), 0);
        assert_eq!(bin_index(1.0), 1);
        assert_eq!(bin_index(-0.1), -1);
        assert_eq!(bin_index(-1.0), -1);

        assert!((bin_center(0) - 0.5).abs() < 1e-10);
        assert!((bin_center(-1) - (-0.5)).abs() < 1e-10);
        assert!((bin_center(1) - 1.5).abs() < 1e-10);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::analysis::Analyze;
    use crate::backend::Backend;
    use std::path::Path;

    fn make_context() -> Backend {
        let mut rng = rand::thread_rng();
        Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap()
    }

    #[test]
    fn build_and_sample() {
        let ctx = make_context();
        let yaml = r#"
property: volume
coordinate:
  property: volume
  resolution: 100.0
file: /tmp/faunus_test_mean_along.dat
frequency: !Every 1
"#;
        let builder: MeanAlongCoordinateBuilder = serde_yaml::from_str(yaml).unwrap();
        let mut analysis = builder.build(&ctx).unwrap();

        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 0);

        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 1);
        assert_eq!(analysis.bins.len(), 1);

        analysis.sample(&ctx, 2).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 2);
        // Same volume each time, so still 1 bin
        assert_eq!(analysis.bins.len(), 1);
    }

    #[test]
    fn missing_resolution_fails() {
        let ctx = make_context();
        let yaml = r#"
property: volume
coordinate:
  property: volume
file: /tmp/faunus_test_mean_along_fail.dat
frequency: !Every 1
"#;
        let builder: MeanAlongCoordinateBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(builder.build(&ctx).is_err());
    }
}
