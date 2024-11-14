use super::{Analyze, Frequency};
use crate::Context;

/// Writes structure of the system in the specified format during the simulation.
#[cfg(feature = "chemfiles")]
#[derive(Debug)]
pub struct StructureWriter {
    output_file: String,
    trajectory: Option<chemfiles::Trajectory>,
    frequency: Frequency,
    num_samples: usize,
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
