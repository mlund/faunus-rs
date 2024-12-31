use super::{Analyze, Frequency};
use crate::topology::Topology;
use crate::Context;
use anyhow::Result;
use derive_builder::Builder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

/// Writes structure of the system in the specified format during the simulation.
#[derive(Debug, Builder)]
#[builder(build_fn(skip), derive(Deserialize, Serialize))]
pub struct MassCenterDistance {
    /// Pair of molecule names to calculate the distance between. May be identical.
    #[allow(dead_code)]
    molecules: (String, String),
    /// Pair of molecule id's to calculate the distance between. May be identical.
    #[builder(setter(skip))]
    molids: (usize, usize),
    /// Stream distances to this file at each sample.
    #[builder_field_attr(serde(rename = "file"))]
    #[allow(dead_code)]
    output_file: PathBuf,
    /// Stream object
    #[builder(setter(skip))]
    encoder: Option<GzEncoder<std::fs::File>>,
    /// Sample frequency.
    frequency: Frequency,
    /// Counter for the number of samples taken.
    #[builder(setter(skip))]
    num_samples: usize,
}

impl MassCenterDistanceBuilder {
    fn validate(&self) -> Result<()> {
        if self.molecules.is_none() || self.output_file.is_none() || self.frequency.is_none() {
            anyhow::bail!("Missing required fields for Mass Center Distance Analysis.");
        }
        Ok(())
    }
    pub fn build(&self, topology: &Topology) -> Result<MassCenterDistance> {
        self.validate()?;
        let molecules = self.molecules.clone().unwrap();
        MassCenterDistance::new(
            (&molecules.0, &molecules.1),
            self.output_file.clone().unwrap(),
            self.frequency.unwrap(),
            topology,
        )
    }
}

impl crate::Info for MassCenterDistance {
    fn short_name(&self) -> Option<&'static str> {
        Some("com distance")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Writes structure of the system at specified frequency into an output trajectory.")
    }
}

impl MassCenterDistance {
    /// Create a new `MassCenterDistance` object.
    pub fn new(
        molecules: (&str, &str),
        output_file: PathBuf,
        frequency: Frequency,
        topology: &Topology,
    ) -> Result<Self> {
        let get_id = |name| {
            topology
                .find_molecule(name)
                .map(|m| m.id())
                .expect("Molecule now found.")
        };
        let molids = (get_id(molecules.0), get_id(molecules.1));
        let stream = std::fs::File::create(&output_file)?;
        Ok(Self {
            molecules: (molecules.0.to_owned(), molecules.1.to_owned()),
            molids,
            output_file,
            encoder: Some(GzEncoder::new(stream, Compression::default())),
            frequency,
            num_samples: 0,
        })
    }
}

impl<T: Context> Analyze<T> for MassCenterDistance {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, _step: usize) -> Result<()> {
        let sel1 = crate::group::GroupSelection::ByMoleculeId(self.molids.0);
        let sel2 = crate::group::GroupSelection::ByMoleculeId(self.molids.1);
        let indices1 = context.select(&sel1);
        let indices2 = context.select(&sel2);

        for i in &indices1 {
            for j in &indices2 {
                // Avoid double counting when the two molids are identical
                if self.molids.0 == self.molids.1 && i >= j {
                    continue;
                }
                let com1 = context.groups()[*i].mass_center();
                let com2 = context.groups()[*j].mass_center();
                if let Some((a, b)) = com1.zip(com2) {
                    let distance = (a - b).norm();
                    writeln!(self.encoder.as_mut().unwrap(), "{:.3}", distance)?;
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
}

impl<T: Context> From<MassCenterDistance> for Box<dyn Analyze<T>> {
    fn from(analysis: MassCenterDistance) -> Box<dyn Analyze<T>> {
        Box::new(analysis)
    }
}
