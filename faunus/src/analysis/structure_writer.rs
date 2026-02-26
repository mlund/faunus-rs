use super::{Analyze, Frequency};
use crate::cell::Shape;
use crate::topology::io::{self, StructureData};
use crate::Context;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

/// Writes structure of the system in the specified format during the simulation.
#[derive(Debug, Builder)]
#[builder(derive(Deserialize, Serialize))]
pub struct StructureWriter {
    /// Output file name (xyz, pdb, etc.)
    #[builder_field_attr(serde(rename = "file"))]
    output_file: String,
    /// Sample frequency.
    frequency: Frequency,
    /// Counter for the number of samples taken.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip_deserializing))]
    num_samples: usize,
}

impl<T: Context> From<StructureWriter> for Box<dyn Analyze<T>> {
    fn from(analysis: StructureWriter) -> Self {
        Box::new(analysis)
    }
}

impl StructureWriter {
    pub fn new(output_file: &str, frequency: Frequency) -> Self {
        Self {
            output_file: output_file.to_owned(),
            frequency,
            num_samples: 0,
        }
    }
}

impl crate::Info for StructureWriter {
    fn short_name(&self) -> Option<&'static str> {
        Some("structure printer")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Writes structure of the system at specified frequency into an output trajectory.")
    }
}

impl<T: Context> Analyze<T> for StructureWriter {
    fn sample(&mut self, context: &T, step: usize) -> anyhow::Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }

        let topology = context.topology();
        let particles = context.get_all_particles();

        // Shift from Faunus convention (center at origin) to file convention (corner at origin)
        let shift = context
            .cell()
            .bounding_box()
            .map(|b| 0.5 * b)
            .unwrap_or_default();

        let mut names = Vec::with_capacity(particles.len());
        let mut positions = Vec::with_capacity(particles.len());

        for group in context.groups().iter() {
            let molecule = &topology.moleculekinds()[group.molecule()];
            for (i, &atom_idx) in molecule.atom_indices().iter().enumerate() {
                let atom_name = molecule.atom_names()[i]
                    .as_deref()
                    .unwrap_or(topology.atomkinds()[atom_idx].name());
                names.push(atom_name.to_string());
                positions.push(particles[i + group.start()].pos + shift);
            }
        }

        let box_lengths = context.cell().bounding_box();

        let data = StructureData {
            names,
            positions,
            step: Some(step as u32),
            box_lengths,
            ..Default::default()
        };

        let append = self.num_samples > 0;
        io::write_structure_frame(&self.output_file, &data, append)?;
        self.num_samples += 1;
        Ok(())
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        let mut map = serde_yaml::Mapping::new();
        map.insert(
            "file".into(),
            serde_yaml::Value::String(self.output_file.clone()),
        );
        map.insert(
            "num_samples".into(),
            serde_yaml::Value::Number(self.num_samples.into()),
        );
        Some(serde_yaml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;

    #[test]
    fn deserialize_trajectory_builders() {
        let yaml = std::fs::read_to_string("tests/files/trajectory_xyz.yaml").unwrap();
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(builders.len(), 2);

        // Verify first entry: xyz trajectory
        let AnalysisBuilder::StructureWriter(ref b) = builders[0] else {
            panic!("expected StructureWriter variant");
        };
        let writer = b.build().unwrap();
        assert_eq!(writer.output_file, "traj.xyz");
        assert!(matches!(writer.frequency, Frequency::Every(100)));

        // Verify second entry: xtc trajectory
        let AnalysisBuilder::StructureWriter(ref b) = builders[1] else {
            panic!("expected StructureWriter variant");
        };
        let writer = b.build().unwrap();
        assert_eq!(writer.output_file, "traj.xtc");
        assert!(matches!(writer.frequency, Frequency::Every(50)));
    }
}
