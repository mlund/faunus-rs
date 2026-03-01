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

//! Collective variable time series analysis.
//!
//! Evaluates a collective variable at each sample step, tracks a running
//! average, and optionally streams `{step, value, average}` to a file.

use super::{Analyze, Frequency};
use crate::collective_variable::{CollectiveVariableBuilder, ConcreteCollectiveVariable};
use crate::Context;
use anyhow::Result;
use average::{Estimate, Mean};
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

/// YAML builder for [`CollectiveVariableAnalysis`].
///
/// Uses `#[serde(flatten)]` on [`CollectiveVariableBuilder`] so users write
/// CV fields (`property`, `range`, …) at the same level as `file` and
/// `frequency`, avoiding a nested `cv:` block. Same approach as
/// [`ConstrainBuilder`](crate::energy::constrain::ConstrainBuilder).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveVariableAnalysisBuilder {
    #[serde(flatten)]
    pub cv: CollectiveVariableBuilder,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    pub frequency: Frequency,
}

impl CollectiveVariableAnalysisBuilder {
    /// Resolve selections against live context and open the output file, if any.
    pub fn build(&self, context: &impl Context) -> Result<CollectiveVariableAnalysis> {
        let cv = self.cv.build_concrete(context)?;

        let stream = if let Some(path) = &self.file {
            let mut stream = crate::auxiliary::open_compressed(path)?;
            writeln!(stream, "# step value average")?;
            Some(stream)
        } else {
            None
        };

        Ok(CollectiveVariableAnalysis {
            cv,
            stream,
            frequency: self.frequency,
            mean: Mean::new(),
            mean_squared: Mean::new(),
            num_samples: 0,
        })
    }
}

/// Monitors a single collective variable over the course of a simulation.
///
/// Each sampled step writes `{step, value, running_average}` to an optional
/// output file, mirroring the C++ Faunus `FileReactionCoordinate` analysis.
#[derive(Debug)]
pub struct CollectiveVariableAnalysis {
    cv: ConcreteCollectiveVariable,
    #[debug(skip)]
    stream: Option<Box<dyn Write>>,
    frequency: Frequency,
    mean: Mean,
    mean_squared: Mean,
    num_samples: usize,
}

impl CollectiveVariableAnalysis {
    /// Running mean of all sampled CV values (Welford's algorithm via `average` crate).
    pub fn mean(&self) -> f64 {
        self.mean.mean()
    }
}

impl crate::Info for CollectiveVariableAnalysis {
    fn short_name(&self) -> Option<&'static str> {
        Some("collectivevariable")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Collective variable time series")
    }
}

impl<T: Context> Analyze<T> for CollectiveVariableAnalysis {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }
        let value = self.cv.evaluate(context);
        self.mean.add(value);
        self.mean_squared.add(value * value);
        self.num_samples += 1;

        if let Some(ref mut stream) = self.stream {
            writeln!(stream, "{} {:.6} {:.6}", step, value, self.mean.mean())?;
        }
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn flush(&mut self) {
        if let Some(ref mut stream) = self.stream {
            let _ = stream.flush();
        }
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        let mut map = serde_yaml::Mapping::new();
        map.insert(
            "property".into(),
            serde_yaml::Value::String(self.cv.axis().name.clone()),
        );
        map.insert(
            "num_samples".into(),
            serde_yaml::Value::Number(self.num_samples.into()),
        );
        map.insert("mean".into(), serde_yaml::to_value(self.mean.mean()).ok()?);
        map.insert(
            "rms".into(),
            serde_yaml::to_value(self.mean_squared.mean().sqrt()).ok()?,
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
range: [1000.0, 5000.0]
frequency: !Every 100
"#;
        let builder: CollectiveVariableAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(builder.cv.range, (1000.0, 5000.0));
        assert!(builder.file.is_none());
        assert!(matches!(builder.frequency, Frequency::Every(100)));
    }

    #[test]
    fn deserialize_builder_with_file() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
file: rc.dat
frequency: !Every 50
"#;
        let builder: CollectiveVariableAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(builder.file.as_ref().unwrap().to_str().unwrap(), "rc.dat");
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
frequency: !Every 100
"#;
        let builder: CollectiveVariableAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
        let serialized = serde_yaml::to_string(&builder).unwrap();
        let roundtrip: CollectiveVariableAnalysisBuilder =
            serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(roundtrip.cv.range, (1000.0, 5000.0));
        assert!(matches!(roundtrip.frequency, Frequency::Every(100)));
    }

    #[test]
    fn deserialize_missing_frequency() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
"#;
        let result = serde_yaml::from_str::<CollectiveVariableAnalysisBuilder>(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !CollectiveVariable
  property: volume
  range: [1000.0, 5000.0]
  frequency: !Every 100
"#;
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(
            builders[0],
            AnalysisBuilder::CollectiveVariable(_)
        ));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::analysis::Analyze;
    use crate::cell::Shape;
    use crate::context::WithCell;
    use crate::platform::reference::ReferencePlatform;
    use std::path::Path;

    fn make_context() -> ReferencePlatform {
        let mut rng = rand::thread_rng();
        ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap()
    }

    #[test]
    fn build_and_sample_volume() {
        let ctx = make_context();
        let yaml = r#"
property: volume
range: [0.0, 1e10]
frequency: !Every 1
"#;
        let builder: CollectiveVariableAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
        let mut analysis = builder.build(&ctx).unwrap();

        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 0);

        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 1);

        analysis.sample(&ctx, 2).unwrap();
        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 2);

        let expected_volume = ctx.cell().volume().unwrap();
        assert!((analysis.mean() - expected_volume).abs() < 1e-10);
    }

    #[test]
    fn frequency_filtering() {
        let ctx = make_context();
        let yaml = r#"
property: volume
range: [0.0, 1e10]
frequency: !Every 10
"#;
        let builder: CollectiveVariableAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
        let mut analysis = builder.build(&ctx).unwrap();

        // Step 1 should not sample (not multiple of 10)
        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 0);

        // Step 10 should sample
        analysis.sample(&ctx, 10).unwrap();
        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 1);

        // Step 20 should sample
        analysis.sample(&ctx, 20).unwrap();
        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 2);
    }

    #[test]
    fn build_via_analysis_builder() {
        let ctx = make_context();
        let yaml = r#"
- !CollectiveVariable
  property: volume
  range: [0.0, 1e10]
  frequency: !Every 1
"#;
        let builders: Vec<crate::analysis::AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
        let analysis = builders[0].build(&ctx).unwrap();
        assert_eq!(analysis.short_name(), Some("collectivevariable"));
    }
}
