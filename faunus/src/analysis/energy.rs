//! Energy analysis — streams per-step energy values to disk.
//!
//! Two modes:
//! - **Total**: writes every energy term plus the total (one row per sample).
//! - **Partial**: writes the nonbonded energy between two VMD-like selections.

use super::{Analyze, Frequency, Sampling};
use crate::auxiliary::{ColumnWriter, MappingExt, WeightedMean};
use crate::selection::{Atoms, CachedSelection, Selection};
use crate::ObserveContext;
use crate::WithHamiltonian;
use anyhow::Result;
use derive_more::Debug;
use serde::Deserialize;
use std::fmt::Display;
use std::path::PathBuf;

/// YAML builder for [`EnergyAnalysis`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyAnalysisBuilder {
    file: PathBuf,
    frequency: Frequency,
    /// Optional pair of VMD-like selections for partial nonbonded energy.
    #[serde(default)]
    selections: Option<(Selection, Selection)>,
}

/// What kind of energy to report.
enum EnergyMode {
    /// Per-term breakdown + total.
    Total,
    /// Nonbonded energy between two selections.
    Partial(Box<(CachedSelection<Atoms>, CachedSelection<Atoms>)>),
}

/// Streams energy values to an output file.
#[derive(Debug)]
pub struct EnergyAnalysis {
    mode: EnergyMode,
    #[debug(skip)]
    stream: ColumnWriter,
    /// Frequency and frame count, owned by the framework.
    sampling: Sampling,
    mean: WeightedMean,
}

impl EnergyAnalysisBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        crate::analysis::prefix_in_place(&mut self.file, dir)
    }

    pub fn build(
        &self,
        context: &(impl ObserveContext + WithHamiltonian),
    ) -> Result<EnergyAnalysis> {
        let (stream, mode) = if let Some((sel1, sel2)) = &self.selections {
            let topology = context.topology_ref();
            let groups = context.groups();
            for sel in [sel1, sel2] {
                if sel
                    .resolve_atoms(topology, groups, &|i| context.atom_kind(i))
                    .is_empty()
                {
                    anyhow::bail!("Energy: selection '{}' resolves to no atoms", sel.source());
                }
            }
            let stream = ColumnWriter::open(&self.file, &["step", "energy", "average"])?;
            (
                stream,
                EnergyMode::Partial(Box::new((
                    CachedSelection::atoms(sel1.clone()),
                    CachedSelection::atoms(sel2.clone()),
                ))),
            )
        } else {
            let hamiltonian = context.hamiltonian();
            let names: Vec<&str> = hamiltonian
                .energy_terms()
                .iter()
                .filter_map(crate::Info::short_name)
                .collect();
            let mut cols: Vec<&str> = vec!["step"];
            cols.extend(&names);
            cols.push("total");
            let stream = ColumnWriter::open(&self.file, &cols)?;
            (stream, EnergyMode::Total)
        };

        Ok(EnergyAnalysis {
            mode,
            stream,
            sampling: Sampling::new(self.frequency),
            mean: WeightedMean::new(),
        })
    }
}

impl crate::Info for EnergyAnalysis {
    fn short_name(&self) -> Option<&'static str> {
        Some("energy")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Energy analysis")
    }
}

impl<T: ObserveContext + WithHamiltonian> Analyze<T> for EnergyAnalysis {
    fn sampling(&self) -> &Sampling {
        &self.sampling
    }
    fn sampling_mut(&mut self) -> &mut Sampling {
        &mut self.sampling
    }

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        // Resolving needs `&mut` on the caches, so do it before borrowing the rest of `self`.
        let partial_atoms = match &mut self.mode {
            EnergyMode::Partial(pair) => Some((
                pair.0.resolve(context).to_vec(),
                pair.1.resolve(context).to_vec(),
            )),
            EnergyMode::Total => None,
        };
        match &self.mode {
            EnergyMode::Total => {
                let hamiltonian = context.hamiltonian();
                let terms = hamiltonian.per_term_energies(context, &crate::Change::Everything);
                let total: f64 = terms.iter().map(|(_, e)| *e).sum();
                self.mean.add(total, weight);
                let formatted: Vec<_> = terms.iter().map(|(_, e)| format!("{e:.6}")).collect();
                let total_str = format!("{total:.6}");
                let mut row: Vec<&dyn Display> = vec![&step];
                row.extend(formatted.iter().map(|s| s as &dyn Display));
                row.push(&total_str);
                self.stream.write_row(&row)?;
            }
            EnergyMode::Partial(..) => {
                let (a1, a2) = partial_atoms.expect("resolved above for the Partial mode");
                let hamiltonian = context.hamiltonian();
                let energy: f64 = hamiltonian
                    .energy_terms()
                    .iter()
                    .filter_map(|term| term.nonbonded_energy_between_atoms(context, &a1, &a2))
                    .sum();
                self.mean.add(energy, weight);
                let mean = self.mean.mean();
                self.stream.write_row(&[
                    &step,
                    &format_args!("{energy:.6}"),
                    &format_args!("{mean:.6}"),
                ])?;
            }
        }
        Ok(())
    }

    fn results(&self) -> Option<serde_yml::Value> {
        if self.sampling.num_samples() == 0 {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("mean", self.mean.mean())?;
        if let EnergyMode::Partial(pair) = &self.mode {
            map.try_insert(
                "selections",
                [pair.0.selection().source(), pair.1.selection().source()],
            )?;
        }
        Some(serde_yml::Value::Mapping(map))
    }
}

impl std::fmt::Debug for EnergyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Total => write!(f, "Total"),
            Self::Partial(pair) => write!(
                f,
                "Partial({:?}, {:?})",
                pair.0.selection().source(),
                pair.1.selection().source()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;

    #[test]
    fn deserialize_total() {
        let yaml = r#"
file: energy.dat
frequency: !Every 100
"#;
        let builder: EnergyAnalysisBuilder = serde_yml::from_str(yaml).unwrap();
        assert!(builder.selections.is_none());
        assert!(matches!(builder.frequency, Frequency::Every(100)));
    }

    #[test]
    fn deserialize_partial() {
        let yaml = r#"
file: partial.dat
frequency: !Every 50
selections: ["molecule water", "atomtype Na"]
"#;
        let builder: EnergyAnalysisBuilder = serde_yml::from_str(yaml).unwrap();
        assert!(builder.selections.is_some());
        let (sel1, sel2) = builder.selections.unwrap();
        assert_eq!(sel1.source(), "molecule water");
        assert_eq!(sel2.source(), "atomtype Na");
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !Energy
  file: energy.dat
  frequency: !Every 100
"#;
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        assert!(matches!(builders[0], AnalysisBuilder::Energy(_)));
    }

    #[test]
    fn roundtrip_via_analysis_builder_partial() {
        let yaml = r#"
- !Energy
  file: partial.dat
  frequency: !Every 50
  selections: ["all", "all"]
"#;
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        assert!(matches!(builders[0], AnalysisBuilder::Energy(_)));
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
    #[ignore = "topology_pass.yaml uses unimplemented Morse potential"]
    fn build_and_sample_total() {
        let ctx = make_context();
        let tmp = tempfile::tempdir().unwrap();
        let sink = tmp.path().join("energy.dat");
        let yaml = format!(
            "file: {}\nfrequency: !Every 1\n",
            sink.to_string_lossy().replace('\\', "/")
        );
        let builder: EnergyAnalysisBuilder = serde_yml::from_str(&yaml).unwrap();
        let mut analysis = builder.build(&ctx).unwrap();

        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 0);
        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 1);

        let yaml_out = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        let mean = yaml_out.get("mean").unwrap().as_f64().unwrap();
        assert!(mean.is_finite());
    }

    #[test]
    #[ignore = "topology_pass.yaml uses unimplemented Morse potential"]
    fn build_and_sample_partial() {
        let ctx = make_context();
        let tmp = tempfile::tempdir().unwrap();
        let sink = tmp.path().join("energy.dat");
        let yaml = format!(
            "file: {}\nfrequency: !Every 1\nselections: [\"all\", \"all\"]\n",
            sink.to_string_lossy().replace('\\', "/")
        );
        let builder: EnergyAnalysisBuilder = serde_yml::from_str(&yaml).unwrap();
        let mut analysis = builder.build(&ctx).unwrap();

        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 1);

        let yaml_out = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        let mean = yaml_out.get("mean").unwrap().as_f64().unwrap();
        assert!(mean.is_finite());
        assert!(yaml_out.get("selections").is_some());
    }
}
