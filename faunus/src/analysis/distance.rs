use super::{Analyze, Frequency};
use crate::selection::Selection;
use crate::Context;
use anyhow::Result;
use derive_builder::Builder;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

/// Measures the mass-center distance between two groups of molecules.
///
/// At each sample step, resolves both selections to groups and writes
/// all pairwise COM distances. When both selections match the same groups,
/// pairs are deduplicated to avoid double counting.
#[derive(Debug, Builder)]
#[builder(build_fn(skip), derive(Deserialize, Serialize))]
pub struct MassCenterDistance {
    /// Pair of selection expressions for the two molecule groups.
    selections: (Selection, Selection),
    /// Stream distances to this file at each sample.
    #[builder_field_attr(serde(rename = "file"))]
    #[allow(dead_code)]
    output_file: PathBuf,
    /// Stream object
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    #[debug(skip)]
    stream: Box<dyn Write>,
    /// Sample frequency.
    frequency: Frequency,
    /// Counter for the number of samples taken.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip_deserializing))]
    num_samples: usize,
}

impl MassCenterDistanceBuilder {
    fn validate(&self) -> Result<()> {
        if self.selections.is_none() || self.output_file.is_none() || self.frequency.is_none() {
            anyhow::bail!("Missing required fields for MassCenterDistance analysis.");
        }
        Ok(())
    }

    pub fn build(&self) -> Result<MassCenterDistance> {
        self.validate()?;
        let stream = crate::auxiliary::open_compressed(self.output_file.as_ref().unwrap())?;
        Ok(MassCenterDistance {
            selections: self.selections.clone().unwrap(),
            output_file: self.output_file.clone().unwrap(),
            stream,
            frequency: self.frequency.unwrap(),
            num_samples: 0,
        })
    }
}

impl crate::Info for MassCenterDistance {
    fn short_name(&self) -> Option<&'static str> {
        Some("com distance")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Mass center distance between two molecule selections")
    }
}

impl<T: Context> Analyze<T> for MassCenterDistance {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }
        let topology = context.topology_ref();
        let groups = context.groups();
        let indices1 = self.selections.0.resolve_groups(topology, groups);
        let indices2 = self.selections.1.resolve_groups(topology, groups);
        let same_selection = self.selections.0.source() == self.selections.1.source();

        for &i in &indices1 {
            for &j in &indices2 {
                if same_selection && i >= j {
                    continue; // avoid double-counting and self-pairing
                }
                let com1 = groups[i].mass_center();
                let com2 = groups[j].mass_center();
                if let Some((a, b)) = com1.zip(com2) {
                    writeln!(self.stream.as_mut(), "{:.3}", (a - b).norm())?;
                    self.num_samples += 1;
                } else {
                    log::error!("Skipping COM distance calculation due to missing COM.");
                }
            }
        }
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        let mut map = serde_yaml::Mapping::new();
        map.insert("num_samples".into(), serde_yaml::Value::Number(self.num_samples.into()));
        Some(serde_yaml::Value::Mapping(map))
    }
}

impl<T: Context> From<MassCenterDistance> for Box<dyn Analyze<T>> {
    fn from(analysis: MassCenterDistance) -> Self {
        Box::new(analysis)
    }
}
