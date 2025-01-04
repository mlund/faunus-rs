use super::{Analyze, Frequency};
use crate::Context;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

/// Writes structure of the system in the specified format during the simulation.
#[cfg(feature = "chemfiles")]
#[derive(Debug, Builder)]
#[builder(derive(Deserialize, Serialize))]
pub struct StructureWriter {
    /// Output file name (xyz, pdb, etc.)
    #[builder_field_attr(serde(rename = "file"))]
    output_file: String,
    /// Trajectory object
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    trajectory: Option<chemfiles::Trajectory>,
    /// Sample frequency.
    frequency: Frequency,
    /// Counter for the number of samples taken.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip_deserializing))]
    num_samples: usize,
}

impl<T: Context> From<StructureWriter> for Box<dyn Analyze<T>> {
    fn from(analysis: StructureWriter) -> Box<dyn Analyze<T>> {
        Box::new(analysis)
    }
}

#[cfg(feature = "chemfiles")]
impl StructureWriter {
    pub fn new(output_file: &str, frequency: Frequency) -> StructureWriter {
        StructureWriter {
            output_file: output_file.to_owned(),
            frequency,
            trajectory: None,
            num_samples: 0,
        }
    }
}

#[cfg(feature = "chemfiles")]
impl crate::Info for StructureWriter {
    fn short_name(&self) -> Option<&'static str> {
        Some("structure printer")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Writes structure of the system at specified frequency into an output trajectory.")
    }
}

#[cfg(feature = "chemfiles")]
impl<T: Context> Analyze<T> for StructureWriter {
    fn sample(&mut self, context: &T, step: usize) -> anyhow::Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }

        let frame = context.to_frame();

        if self.trajectory.is_none() {
            self.trajectory = Some(chemfiles::Trajectory::open(&self.output_file, 'w')?);
        }

        self.trajectory.as_mut().unwrap().write(&frame)?;
        self.num_samples += 1;
        Ok(())
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }
}
