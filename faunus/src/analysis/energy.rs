//! Energy analysis — streams per-step energy values to disk.
//!
//! Two modes:
//! - **Total**: writes every energy term plus the total (one row per sample).
//! - **Partial**: writes the nonbonded energy between two VMD-like selections.

use super::{Analyze, Frequency};
use crate::selection::Selection;
use crate::Context;
use anyhow::Result;
use average::{Estimate, Mean};
use derive_more::Debug;
use serde::Deserialize;
use std::io::Write;
use std::path::PathBuf;

/// YAML builder for [`EnergyAnalysis`].
#[derive(Debug, Clone, Deserialize)]
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
    Partial(Selection, Selection),
}

/// Streams energy values to an output file.
#[derive(Debug)]
pub struct EnergyAnalysis {
    mode: EnergyMode,
    #[debug(skip)]
    stream: Box<dyn Write>,
    frequency: Frequency,
    mean: Mean,
    num_samples: usize,
}

impl EnergyAnalysisBuilder {
    pub fn build(&self, context: &impl Context) -> Result<EnergyAnalysis> {
        let mut stream = crate::auxiliary::open_compressed(&self.file)?;

        let mode = if let Some((sel1, sel2)) = &self.selections {
            let topology = context.topology_ref();
            let groups = context.groups();
            for sel in [sel1, sel2] {
                if sel.resolve_atoms(topology, groups).is_empty() {
                    anyhow::bail!("Energy: selection '{}' resolves to no atoms", sel.source());
                }
            }
            writeln!(stream, "# step energy average")?;
            EnergyMode::Partial(sel1.clone(), sel2.clone())
        } else {
            // Write header from current hamiltonian terms.
            let hamiltonian = context.hamiltonian();
            let names: Vec<&str> = hamiltonian
                .energy_terms()
                .iter()
                .filter_map(crate::Info::short_name)
                .collect();
            write!(stream, "# step")?;
            for name in &names {
                write!(stream, " {name}")?;
            }
            writeln!(stream, " total")?;
            EnergyMode::Total
        };

        Ok(EnergyAnalysis {
            mode,
            stream,
            frequency: self.frequency,
            mean: Mean::new(),
            num_samples: 0,
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

impl<T: Context> Analyze<T> for EnergyAnalysis {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }
        match &self.mode {
            EnergyMode::Total => {
                let hamiltonian = context.hamiltonian();
                let terms = hamiltonian.per_term_energies(context, &crate::Change::Everything);
                let total: f64 = terms.iter().map(|(_, e)| *e).sum();
                self.mean.add(total);
                write!(self.stream, "{step}")?;
                for (.., energy) in &terms {
                    write!(self.stream, " {energy:.6}")?;
                }
                writeln!(self.stream, " {total:.6}")?;
            }
            EnergyMode::Partial(sel1, sel2) => {
                // Re-resolve each sample since group membership can change (GC ensemble)
                let topology = context.topology_ref();
                let groups = context.groups();
                let a1 = sel1.resolve_atoms(topology, groups);
                let a2 = sel2.resolve_atoms(topology, groups);
                let hamiltonian = context.hamiltonian();
                let energy: f64 = hamiltonian
                    .energy_terms()
                    .iter()
                    .filter_map(|term| term.nonbonded_energy_between_atoms(context, &a1, &a2))
                    .sum();
                self.mean.add(energy);
                writeln!(self.stream, "{step} {energy:.6} {:.6}", self.mean.mean())?;
            }
        }
        self.num_samples += 1;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn flush(&mut self) {
        let _ = self.stream.flush();
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
        map.insert("mean".into(), serde_yaml::to_value(self.mean.mean()).ok()?);
        if let EnergyMode::Partial(sel1, sel2) = &self.mode {
            map.insert(
                "selections".into(),
                serde_yaml::to_value([sel1.source(), sel2.source()]).ok()?,
            );
        }
        Some(serde_yaml::Value::Mapping(map))
    }
}

impl std::fmt::Debug for EnergyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Total => write!(f, "Total"),
            Self::Partial(s1, s2) => write!(f, "Partial({:?}, {:?})", s1.source(), s2.source()),
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
        let builder: EnergyAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: EnergyAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
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
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builders[0], AnalysisBuilder::Energy(_)));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::analysis::Analyze;
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
    #[ignore = "topology_pass.yaml uses unimplemented Morse potential"]
    fn build_and_sample_total() {
        let ctx = make_context();
        let yaml = r#"
file: /dev/null
frequency: !Every 1
"#;
        let builder: EnergyAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
        let mut analysis = builder.build(&ctx).unwrap();

        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 0);
        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 1);

        let yaml_out = Analyze::<ReferencePlatform>::to_yaml(&analysis).unwrap();
        let mean = yaml_out.get("mean").unwrap().as_f64().unwrap();
        assert!(mean.is_finite());
    }

    #[test]
    #[ignore = "topology_pass.yaml uses unimplemented Morse potential"]
    fn build_and_sample_partial() {
        let ctx = make_context();
        let yaml = r#"
file: /dev/null
frequency: !Every 1
selections: ["all", "all"]
"#;
        let builder: EnergyAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
        let mut analysis = builder.build(&ctx).unwrap();

        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<ReferencePlatform>::num_samples(&analysis), 1);

        let yaml_out = Analyze::<ReferencePlatform>::to_yaml(&analysis).unwrap();
        let mean = yaml_out.get("mean").unwrap().as_f64().unwrap();
        assert!(mean.is_finite());
        assert!(yaml_out.get("selections").is_some());
    }
}
