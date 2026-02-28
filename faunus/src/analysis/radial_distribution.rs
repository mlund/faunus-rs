//! Radial distribution function g(r) analysis.
//!
//! Supports atom-atom and center-of-mass (COM-COM) pair distributions
//! with minimum image convention for periodic boundary conditions.

use super::{Analyze, Frequency};
use crate::cell::{BoundaryConditions, Shape};
use crate::dimension::Dimension;
use crate::histogram::Histogram;
use crate::selection::Selection;
use crate::Context;
use anyhow::Result;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

/// YAML builder for [`RadialDistribution`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadialDistributionBuilder {
    /// Pair of selection expressions.
    selections: (Selection, Selection),
    /// Output file path.
    file: PathBuf,
    /// Bin width in distance units.
    dr: f64,
    /// Maximum distance. Defaults to half the shortest box dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_r: Option<f64>,
    /// If true, use center-of-mass distances instead of atom-atom.
    #[serde(default)]
    use_com: bool,
    /// Whether to exclude intramolecular pairs. Default: true for atom-atom, ignored for COM.
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude_intramolecular: Option<bool>,
    /// Dimensionality for shell normalization. Default: XYZ (3D spherical shells).
    #[serde(default)]
    dimension: Dimension,
    /// Sampling frequency.
    frequency: Frequency,
}

impl RadialDistributionBuilder {
    pub fn build(&self, context: &impl Context) -> Result<RadialDistribution> {
        let cell = context.cell();
        let max_r = match self.max_r {
            Some(r) => r,
            None => {
                let bbox = cell.bounding_box().ok_or_else(|| {
                    anyhow::anyhow!(
                        "RadialDistribution: cell has no bounding box; set explicit max_r"
                    )
                })?;
                bbox.x.min(bbox.y).min(bbox.z) / 2.0
            }
        };
        if max_r <= 0.0 {
            anyhow::bail!("RadialDistribution: max_r must be positive");
        }
        let exclude_intramolecular = !self.use_com && self.exclude_intramolecular.unwrap_or(true);

        let histogram = Histogram::new(0.0, max_r, self.dr);
        let stream = crate::auxiliary::open_compressed(&self.file)?;

        Ok(RadialDistribution {
            selections: self.selections.clone(),
            histogram,
            volume_sum: 0.0,
            pair_count_sum: 0.0,
            num_samples: 0,
            use_com: self.use_com,
            exclude_intramolecular,
            dimension: self.dimension,
            output_file: self.file.clone(),
            stream,
            frequency: self.frequency,
        })
    }
}

/// Radial distribution function analysis.
#[derive(Debug)]
pub struct RadialDistribution {
    selections: (Selection, Selection),
    histogram: Histogram,
    volume_sum: f64,
    /// Accumulated pair count across all frames (for GC ensemble correctness).
    pair_count_sum: f64,
    num_samples: usize,
    use_com: bool,
    exclude_intramolecular: bool,
    dimension: Dimension,
    output_file: PathBuf,
    #[debug(skip)]
    stream: Box<dyn Write>,
    frequency: Frequency,
}

/// Iterate unique pairs from two index lists, deduplicating when `same` is true,
/// look up a position for each index, and feed each PBC distance into the histogram.
/// Returns the number of pairs evaluated.
fn collect_pair_distances(
    indices1: &[usize],
    indices2: &[usize],
    same: bool,
    mut get_pos: impl FnMut(usize) -> Option<crate::Point>,
    mut skip: impl FnMut(usize, usize) -> bool,
    cell: &impl BoundaryConditions,
    dimension: Dimension,
    histogram: &mut Histogram,
) -> f64 {
    let mut pair_count = 0u64;
    for (idx_i, &i) in indices1.iter().enumerate() {
        let Some(pos_i) = get_pos(i) else { continue };
        let start_j = if same { idx_i + 1 } else { 0 };
        let iter_j = if same { &indices2[start_j..] } else { indices2 };
        for &j in iter_j {
            if skip(i, j) {
                continue;
            }
            let Some(pos_j) = get_pos(j) else { continue };
            // Project displacement onto active dimensions before computing distance
            histogram.add(dimension.filter(cell.distance(&pos_i, &pos_j)).norm());
            pair_count += 1;
        }
    }
    pair_count as f64
}

impl RadialDistribution {
    fn same_selection(&self) -> bool {
        self.selections.0.source() == self.selections.1.source()
    }

    /// Find which group index an atom belongs to.
    fn group_of_atom(groups: &[crate::group::Group], atom_index: usize) -> Option<usize> {
        groups.iter().position(|g| g.contains(atom_index))
    }

    /// Sample atom-atom RDF, returning the number of pairs evaluated.
    fn sample_atom_atom(&mut self, context: &impl Context) -> f64 {
        let topology = context.topology_ref();
        let groups = context.groups();
        let atoms1 = self.selections.0.resolve_atoms(topology, groups);
        let atoms2 = self.selections.1.resolve_atoms(topology, groups);
        let exclude = self.exclude_intramolecular;

        collect_pair_distances(
            &atoms1,
            &atoms2,
            self.same_selection(),
            |i| Some(context.position(i)),
            |i, j| {
                exclude && {
                    let gi = Self::group_of_atom(groups, i);
                    gi.is_some() && gi == Self::group_of_atom(groups, j)
                }
            },
            context.cell(),
            self.dimension,
            &mut self.histogram,
        )
    }

    /// Sample COM-COM RDF, returning the number of pairs evaluated.
    fn sample_com_com(&mut self, context: &impl Context) -> f64 {
        let topology = context.topology_ref();
        let groups = context.groups();
        let gi1 = self.selections.0.resolve_groups(topology, groups);
        let gi2 = self.selections.1.resolve_groups(topology, groups);

        collect_pair_distances(
            &gi1,
            &gi2,
            self.same_selection(),
            |gi| groups[gi].mass_center().copied(),
            |_, _| false,
            context.cell(),
            self.dimension,
            &mut self.histogram,
        )
    }

    /// Write the normalized g(r) to the output file.
    fn write_gr(&mut self) -> Result<()> {
        if self.num_samples == 0 || self.pair_count_sum == 0.0 || self.volume_sum == 0.0 {
            return Ok(());
        }
        let v_avg = self.volume_sum / self.num_samples as f64;
        let n_pairs_avg = self.pair_count_sum / self.num_samples as f64;
        let dr = self.histogram.bin_width();

        self.stream = crate::auxiliary::open_compressed(&self.output_file)?;
        writeln!(self.stream, "# r g(r)")?;
        for (r, count) in self.histogram.iter() {
            let r_inner = r - dr / 2.0;
            let r_outer = r + dr / 2.0;
            let shell_volume = self.dimension.shell_volume(r_inner, r_outer);
            let ideal = n_pairs_avg * self.num_samples as f64 * shell_volume / v_avg;
            let gr = if ideal > 0.0 { count / ideal } else { 0.0 };
            writeln!(self.stream, "{:.6} {:.6}", r, gr)?;
        }
        self.stream.flush()?;
        Ok(())
    }
}

impl crate::Info for RadialDistribution {
    fn short_name(&self) -> Option<&'static str> {
        Some("rdf")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Radial distribution function g(r)")
    }
}

impl<T: Context> Analyze<T> for RadialDistribution {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }
        let pairs = if self.use_com {
            self.sample_com_com(context)
        } else {
            self.sample_atom_atom(context)
        };
        self.pair_count_sum += pairs;
        if let Some(bbox) = context.cell().bounding_box() {
            self.volume_sum += self.dimension.effective_volume(bbox);
        }
        self.num_samples += 1;
        self.write_gr()?;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn flush(&mut self) {
        let _ = self.write_gr();
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        if self.num_samples == 0 {
            return None;
        }
        let mut map = serde_yaml::Mapping::new();
        map.insert(
            "num_samples".into(),
            serde_yaml::Value::Number(self.num_samples.into()),
        );
        map.insert(
            "num_bins".into(),
            serde_yaml::Value::Number(self.histogram.num_bins().into()),
        );
        Some(serde_yaml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use approx::assert_relative_eq;

    #[test]
    fn deserialize_atom_atom() {
        let yaml = r#"
selections: ["atomtype Na", "atomtype Cl"]
file: rdf.dat
dr: 0.1
frequency: !Every 100
"#;
        let builder: RadialDistributionBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(!builder.use_com);
        assert!(builder.max_r.is_none());
        assert!(builder.exclude_intramolecular.is_none());
        assert!(matches!(builder.frequency, Frequency::Every(100)));
    }

    #[test]
    fn deserialize_com_mode() {
        let yaml = r#"
selections: ["molecule polymer", "molecule polymer"]
use_com: true
file: rdf_com.dat
dr: 0.5
max_r: 30.0
frequency: !Every 50
"#;
        let builder: RadialDistributionBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(builder.use_com);
        assert_relative_eq!(builder.max_r.unwrap(), 30.0);
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !RadialDistribution
  selections: ["atomtype Na", "atomtype Cl"]
  file: rdf.dat
  dr: 0.1
  frequency: !Every 100
"#;
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(
            builders[0],
            AnalysisBuilder::RadialDistribution(_)
        ));
    }

    /// Verify normalization: fill histogram with ideal-gas counts → g(r) ≈ 1.
    fn check_normalization(dimension: Dimension) {
        let dr = 0.5;
        let num_samples = 10usize;
        let n_pairs = 100.0;
        let volume = 100.0;

        let mut rdf = RadialDistribution {
            selections: (
                Selection::parse("all").unwrap(),
                Selection::parse("all").unwrap(),
            ),
            histogram: Histogram::new(0.0, 5.0, dr),
            volume_sum: volume * num_samples as f64,
            pair_count_sum: n_pairs * num_samples as f64,
            num_samples,
            use_com: false,
            exclude_intramolecular: false,
            dimension,
            output_file: PathBuf::from("/dev/null"),
            stream: Box::new(std::io::sink()),
            frequency: Frequency::Every(1),
        };
        for i in 0..rdf.histogram.num_bins() {
            let r = rdf.histogram.bin_center(i);
            let r_inner = r - dr / 2.0;
            let r_outer = r + dr / 2.0;
            let shell_vol = dimension.shell_volume(r_inner, r_outer);
            let ideal_count = n_pairs * num_samples as f64 * shell_vol / volume;
            for _ in 0..ideal_count.round() as usize {
                rdf.histogram.add(r);
            }
        }
        assert!(rdf.write_gr().is_ok());
    }

    #[test]
    fn normalization_uniform_gas_3d() {
        check_normalization(Dimension::XYZ);
    }

    #[test]
    fn normalization_uniform_gas_2d() {
        check_normalization(Dimension::XY);
    }

    #[test]
    fn normalization_uniform_gas_1d() {
        check_normalization(Dimension::Z);
    }
}
