//! Multipole analysis: per-group charge statistics and dipole moment.
//!
//! Computes mean charge ⟨Z⟩, charge capacitance C = ⟨Z²⟩ − ⟨Z⟩²,
//! and mean dipole moment magnitude ⟨|μ|⟩ averaged over all groups
//! matching a selection. Handles atom-type swaps (titration) and
//! GCMC (only active groups contribute).

use super::{Analyze, Frequency};
use crate::auxiliary::{MappingExt, WeightedMean};
use crate::selection::Selection;
use crate::topology::GroupKind;
use crate::Context;
use anyhow::Result;
use derive_more::Debug;
use serde::{Deserialize, Serialize};

/// Per-atom charge accumulator with name for YAML output.
#[derive(Debug, Clone)]
struct PerAtomCharge {
    name: String,
    q: WeightedMean,
    q_squared: WeightedMean,
}

/// YAML builder for [`MultipoleAnalysis`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipoleAnalysisBuilder {
    pub selection: Selection,
    pub frequency: Frequency,
}

impl MultipoleAnalysisBuilder {
    pub fn build(&self, context: &impl Context) -> Result<MultipoleAnalysis> {
        let topology = context.topology_ref();
        let groups = context.resolve_groups_live(&self.selection);
        if groups.is_empty() {
            anyhow::bail!(
                "Multipole: selection '{}' matched no groups",
                self.selection.source()
            );
        }
        let molecule_kinds: std::collections::BTreeSet<_> = groups
            .iter()
            .map(|&gi| context.groups()[gi].molecule())
            .collect();
        if molecule_kinds.len() > 1 {
            anyhow::bail!(
                "Multipole: selection '{}' matched multiple molecule kinds; per-atom analysis requires a single molecule kind",
                self.selection.source()
            );
        }
        let molecule_kind = *molecule_kinds.iter().next().unwrap();
        log::info!(
            "Multipole: selection '{}' matched {} groups",
            self.selection.source(),
            groups.len()
        );
        Ok(MultipoleAnalysis {
            selection: self.selection.clone(),
            frequency: self.frequency,
            num_samples: 0,
            charge: WeightedMean::new(),
            charge_squared: WeightedMean::new(),
            dipole_scalar: WeightedMean::new(),
            dipole_squared: WeightedMean::new(),
            per_atom: Vec::new(),
            molecule_kind: Some(molecule_kind),
            track_per_atom: topology.moleculekinds()[molecule_kind].group_kind()
                == GroupKind::Molecular,
        })
    }
}

/// Per-group charge and dipole moment analysis.
#[derive(Debug)]
pub struct MultipoleAnalysis {
    selection: Selection,
    frequency: Frequency,
    num_samples: usize,
    charge: WeightedMean,
    charge_squared: WeightedMean,
    dipole_scalar: WeightedMean,
    dipole_squared: WeightedMean,
    /// Per-atom charge stats, lazy-initialized on first sample.
    /// Stored because `to_yaml()` has no access to topology.
    per_atom: Vec<PerAtomCharge>,
    /// Molecule kind validated at build time.
    molecule_kind: Option<usize>,
    /// True only for single-kind molecular selections.
    track_per_atom: bool,
}

impl crate::Info for MultipoleAnalysis {
    fn short_name(&self) -> Option<&'static str> {
        Some("multipole")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Per-group charge and dipole moment analysis")
    }
}

impl<T: Context> Analyze<T> for MultipoleAnalysis {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, _step: usize, weight: f64) -> Result<()> {
        let topology = context.topology_ref();
        let atomkinds = topology.atomkinds();
        let moleculekinds = topology.moleculekinds();

        for &gi in &context.resolve_groups_live(&self.selection) {
            let group = &context.groups()[gi];
            let mol = group.molecule();

            // Lazy-init per-atom accumulators from the validated molecular kind.
            if self.track_per_atom && self.per_atom.is_empty() {
                let molkind = &moleculekinds[mol];
                self.per_atom = (0..molkind.atoms().len())
                    .map(|i| PerAtomCharge {
                        name: molkind.resolved_atom_name(i, atomkinds).to_owned(),
                        q: WeightedMean::new(),
                        q_squared: WeightedMean::new(),
                    })
                    .collect();
            }

            let track_per_atom = self.track_per_atom
                && self.molecule_kind.is_some_and(|m| m == mol)
                && group.capacity() == self.per_atom.len();
            let mut z = 0.0;
            for i in group.iter_active() {
                let q = atomkinds[context.atom_kind(i)].charge();
                z += q;
                if track_per_atom {
                    let rel = i - group.start();
                    if let Some(per_atom) = self.per_atom.get_mut(rel) {
                        per_atom.q.add(q, weight);
                        per_atom.q_squared.add(q * q, weight);
                    }
                }
            }
            self.charge.add(z, weight);
            self.charge_squared.add(z * z, weight);

            if let Some(mu) = crate::collective_variable::group::group_dipole_moment(gi, context) {
                let mu_norm = mu.norm();
                self.dipole_scalar.add(mu_norm, weight);
                self.dipole_squared.add(mu_norm * mu_norm, weight);
            }
        }

        self.num_samples += 1;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        if self.num_samples == 0 {
            return None;
        }
        let mut map = serde_yml::Mapping::new();

        let z_mean = self.charge.mean();
        let capacitance = (self.charge_squared.mean() - z_mean * z_mean).max(0.0);
        let z_std = capacitance.sqrt();

        let mu_mean = self.dipole_scalar.mean();
        let mu_var = (self.dipole_squared.mean() - mu_mean * mu_mean).max(0.0);
        let mu_std = mu_var.sqrt();

        map.try_insert("selection", self.selection.source())?;
        map.try_insert("num_samples", self.num_samples)?;
        map.try_insert("charge", format!("{z_mean:.4} ± {z_std:.4}"))?;
        map.try_insert("capacitance", capacitance)?;
        map.try_insert("dipole_moment", format!("{mu_mean:.4} ± {mu_std:.4}"))?;

        if !self.per_atom.is_empty() {
            let atoms: Vec<serde_yml::Value> = self
                .per_atom
                .iter()
                .enumerate()
                .filter_map(|(idx, atom)| {
                    let q_mean = atom.q.mean();
                    let variance = (atom.q_squared.mean() - q_mean * q_mean).max(0.0);
                    if variance < f64::EPSILON {
                        return None;
                    }
                    let mut entry = serde_yml::Mapping::new();
                    entry.try_insert("index", idx)?;
                    entry.try_insert("name", atom.name.as_str())?;
                    entry.try_insert("⟨q⟩", q_mean)?;
                    entry.try_insert("⟨q²⟩-⟨q⟩²", variance)?;
                    Some(serde_yml::Value::Mapping(entry))
                })
                .collect();
            if !atoms.is_empty() {
                map.try_insert("atoms", atoms)?;
            }
        }

        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Analyze;
    use crate::backend::Backend;
    use crate::group::GroupCollection;
    use tempfile::NamedTempFile;

    fn backend_from_str(yaml: &str) -> Backend {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        let mut rng = rand::thread_rng();
        Backend::new(tmp.path(), None, &mut rng).unwrap()
    }

    #[test]
    fn per_atom_stats_are_reported_for_single_molecular_kind() {
        let mut ctx = backend_from_str(
            r#"
atoms:
  - {name: A0, mass: 1.0, charge: 0.0, sigma: 1.0}
  - {name: A1, mass: 1.0, charge: 1.0, sigma: 1.0}
molecules:
  - name: MOL
    atoms: [A0]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: MOL
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
        );
        let builder = MultipoleAnalysisBuilder {
            selection: Selection::parse("all").unwrap(),
            frequency: Frequency::Every(1),
        };
        let mut analysis = builder.build(&ctx).unwrap();

        analysis.sample(&ctx, 0).unwrap();
        ctx.set_atom_kind(0, 1);
        analysis.sample(&ctx, 1).unwrap();

        let yaml = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        let atoms = yaml
            .get("atoms")
            .and_then(serde_yml::Value::as_sequence)
            .unwrap();
        assert_eq!(atoms.len(), 1);
        assert_eq!(
            atoms[0].get("index").and_then(serde_yml::Value::as_u64),
            Some(0)
        );
        assert_eq!(
            atoms[0].get("name").and_then(serde_yml::Value::as_str),
            Some("A0")
        );
    }

    #[test]
    fn build_fails_for_mixed_molecule_kinds() {
        let ctx = backend_from_str(
            r#"
atoms:
  - {name: A, mass: 1.0, sigma: 1.0}
  - {name: B, mass: 1.0, sigma: 1.0}
molecules:
  - name: MOL
    atoms: [A]
  - name: ATOM
    atoms: [B]
    atomic: true
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: MOL
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: ATOM
      N: 2
      insert: !RandomAtomPos {}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
        );
        let builder = MultipoleAnalysisBuilder {
            selection: Selection::parse("all").unwrap(),
            frequency: Frequency::Every(1),
        };

        let err = builder.build(&ctx).unwrap_err().to_string();
        assert!(
            err.contains("matched multiple molecule kinds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn atomic_group_does_not_emit_per_atom_stats() {
        let ctx = backend_from_str(
            r#"
atoms:
  - {name: X, mass: 1.0, charge: 1.0, sigma: 1.0}
molecules:
  - name: particle
    atoms: [X]
    atomic: true
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: particle
      N: 20
      active: 8
      insert: !RandomAtomPos {}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
        );
        let builder = MultipoleAnalysisBuilder {
            selection: Selection::parse("all").unwrap(),
            frequency: Frequency::Every(1),
        };
        let mut analysis = builder.build(&ctx).unwrap();

        analysis.sample(&ctx, 0).unwrap();

        let yaml = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        assert!(yaml.get("atoms").is_none());
    }
}
