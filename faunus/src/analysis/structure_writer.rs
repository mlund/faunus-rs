use super::{Analyze, Frequency};
use crate::auxiliary::MappingExt;
use crate::cell::Shape;
use crate::selection::{Selection, SelectionCache};
use crate::topology::io::{self, frame_state::FrameStateWriter, psf, StructureData};
use crate::Context;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::Path;

/// Writes structure of the system in the specified format during the simulation.
#[derive(Debug, Builder)]
#[builder(derive(Deserialize, Serialize), build_fn(validate = "Self::validate"))]
pub struct StructureWriter {
    /// Output file name (xyz, pdb, etc.)
    #[builder_field_attr(serde(rename = "file"))]
    output_file: String,
    /// Sample frequency.
    frequency: Frequency,
    /// Write a `.aux` frame state file alongside the trajectory.
    #[builder_field_attr(serde(default))]
    #[builder(default)]
    save_frame_state: bool,
    /// Optional molecule selection filter (VMD-like expression).
    #[builder_field_attr(serde(default))]
    #[builder(setter(strip_option), default)]
    // strip_option: avoid double-Option in builder serde
    selection: Option<Selection>,
    /// Counter for the number of samples taken.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip_deserializing))]
    num_samples: usize,
    /// Lazy-opened so the header can capture group topology from the first frame.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    frame_state_writer: Option<FrameStateWriter>,
    /// Cached resolved group indices for selection.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    group_cache: SelectionCache,
}

impl StructureWriterBuilder {
    fn validate(&self) -> Result<(), String> {
        // Frame state (.aux) encodes full-system group topology; a filtered
        // selection would produce a mismatch during rerun deserialization.
        if self.save_frame_state == Some(true) && self.selection.is_some() {
            return Err("save_frame_state cannot be combined with selection".into());
        }
        Ok(())
    }
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
            save_frame_state: false,
            selection: None,
            num_samples: 0,
            frame_state_writer: None,
            group_cache: SelectionCache::default(),
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

impl StructureWriter {
    /// Resolve selected group indices, using cache to avoid re-resolution.
    fn selected_group_indices<T: Context>(&mut self, context: &T) -> Cow<'_, [usize]> {
        match &self.selection {
            Some(sel) => {
                let gen = context.group_lists_generation();
                Cow::Borrowed(
                    self.group_cache
                        .get_or_resolve(gen, || context.resolve_groups_live(sel)),
                )
            }
            None => Cow::Owned((0..context.groups().len()).collect()),
        }
    }

    fn write_frame<T: Context>(&mut self, context: &T, step: usize) -> anyhow::Result<()> {
        let topology = context.topology();
        let all_groups = context.groups();
        let group_indices = self.selected_group_indices(context);

        let num_particles: usize = group_indices
            .iter()
            .map(|&i| all_groups[i].capacity())
            .sum();
        let mut names = Vec::with_capacity(num_particles);
        let mut positions = Vec::with_capacity(num_particles);

        for &gi in group_indices.iter() {
            let group = &all_groups[gi];
            let molecule = &topology.moleculekinds()[group.molecule()];
            // capacity() not len(): XTC requires fixed particle count per frame
            for i in 0..group.capacity() {
                let topo_i = molecule.topology_index(i);
                names.push(
                    molecule
                        .resolved_atom_name(topo_i, topology.atomkinds())
                        .to_string(),
                );
                positions.push(context.position(i + group.start()));
            }
        }

        let (box_lengths, shift) = match context.cell().orthorhombic_expansion() {
            Some(expansion) => {
                if self.num_samples == 0 {
                    log::info!(
                        "Expanding {} → {} particles for orthorhombic output",
                        names.len(),
                        names.len() * (1 + expansion.translations.len())
                    );
                }
                let n = names.len();
                let extra = n * expansion.translations.len();
                names.reserve(extra);
                positions.reserve(extra);
                for translation in &expansion.translations {
                    names.extend_from_within(..n);
                    for i in 0..n {
                        positions.push(positions[i] + translation);
                    }
                }
                (Some(expansion.box_lengths), 0.5 * expansion.box_lengths)
            }
            None => {
                let bb = context.cell().bounding_box();
                (bb, bb.map(|b| 0.5 * b).unwrap_or_default())
            }
        };

        // Shift from Faunus convention (center at origin) to file convention (corner at origin)
        for pos in &mut positions {
            *pos += shift;
        }

        let data = StructureData {
            names,
            positions,
            step: Some(step as u32),
            box_lengths,
            ..Default::default()
        };

        let append = self.num_samples > 0;
        io::write_structure_frame(&self.output_file, &data, append)?;

        // Write frame state alongside the trajectory frame
        if self.save_frame_state {
            if self.frame_state_writer.is_none() {
                let aux_path = io::frame_state::aux_path_from_traj(Path::new(&self.output_file));
                let groups: Vec<(u32, u32)> = context
                    .groups()
                    .iter()
                    .map(|g| (g.molecule() as u32, g.capacity() as u32))
                    .collect();
                let n_particles = context.num_particles() as u32;
                let w = FrameStateWriter::create(&aux_path, &groups, n_particles)?;
                log::info!("Writing frame state to {}", aux_path.display());
                self.frame_state_writer = Some(w);
            }
            let writer = self.frame_state_writer.as_mut().unwrap();
            let groups = context.groups();
            let quaternions: Vec<_> = groups.iter().map(|g| *g.quaternion()).collect();
            let mass_centers: Vec<_> = groups
                .iter()
                .map(|g| g.mass_center().copied().unwrap_or_default())
                .collect();
            let group_sizes: Vec<u32> = groups.iter().map(|g| g.len() as u32).collect();
            let atom_ids: Vec<u32> = (0..context.num_particles())
                .map(|i| context.atom_kind(i) as u32)
                .collect();
            writer.write_frame(&quaternions, &mass_centers, &group_sizes, &atom_ids)?;
        }

        self.num_samples += 1;
        Ok(())
    }
}

impl<T: Context> Analyze<T> for StructureWriter {
    fn perform_sample(&mut self, context: &T, step: usize, _weight: f64) -> anyhow::Result<()> {
        self.write_frame(context, step)
    }

    fn finalize(&mut self, context: &T) -> anyhow::Result<()> {
        if self.frequency.should_perform_at_end() {
            self.write_frame(context, self.num_samples)?;
        }
        if self.num_samples > 0 {
            // into_owned() releases the borrow on self.group_cache so self.output_file is accessible
            let group_indices = self.selected_group_indices(context).into_owned();
            let base = Path::new(&self.output_file);
            let topology = context.topology();
            let all_groups = context.groups();
            let filtered: Vec<_> = group_indices
                .iter()
                .map(|&i| all_groups[i].clone())
                .collect();
            let psf_path = base.with_extension("psf");
            psf::write_psf(&psf_path, &topology, &filtered)?;
            let tcl_path = base.with_extension("tcl");
            let psf_name = psf_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("traj.psf");
            let traj_name = base
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&self.output_file);
            psf::write_vmd_script(&tcl_path, &topology, psf_name, traj_name)?;
            log::info!("VMD visualization: vmd -e {}", tcl_path.display());
        }
        Ok(())
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        let mut map = serde_yml::Mapping::new();
        map.try_insert("file", &self.output_file)?;
        map.try_insert("num_samples", self.num_samples)?;
        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;

    #[test]
    fn deserialize_trajectory_builders() {
        let yaml = std::fs::read_to_string("tests/files/trajectory_xyz.yaml").unwrap();
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(&yaml).unwrap();
        assert_eq!(builders.len(), 3);

        // Verify first entry: xyz trajectory
        let AnalysisBuilder::StructureWriter(ref b) = builders[0] else {
            panic!("expected StructureWriter variant");
        };
        let writer = b.build().unwrap();
        assert_eq!(writer.output_file, "traj.xyz");
        assert!(matches!(writer.frequency, Frequency::Every(100)));
        assert!(writer.selection.is_none());

        // Verify second entry: xtc trajectory
        let AnalysisBuilder::StructureWriter(ref b) = builders[1] else {
            panic!("expected StructureWriter variant");
        };
        let writer = b.build().unwrap();
        assert_eq!(writer.output_file, "traj.xtc");
        assert!(matches!(writer.frequency, Frequency::Every(50)));
        assert!(writer.selection.is_none());

        // Verify third entry: xyz with selection filter
        let AnalysisBuilder::StructureWriter(ref b) = builders[2] else {
            panic!("expected StructureWriter variant");
        };
        let writer = b.build().unwrap();
        assert_eq!(writer.output_file, "selected.xyz");
        assert!(matches!(writer.frequency, Frequency::Every(10)));
        assert!(writer.selection.is_some());
        assert_eq!(writer.selection.unwrap().source(), "molecule water");
    }
}
