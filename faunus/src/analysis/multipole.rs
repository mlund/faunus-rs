//! Multipole analysis: per-group charge, dipole, and quadrupole statistics.
//!
//! Computes mean charge ⟨Z⟩, charge capacitance C = ⟨Z²⟩ − ⟨Z⟩²,
//! mean dipole moment magnitude ⟨|μ|⟩, and the traceless quadrupole tensor
//! ⟨Θ_αβ⟩ (PBC-aware, relative to COM) averaged over all groups matching a
//! selection. Handles atom-type swaps (titration) and
//! GCMC (only active groups contribute).

use super::{Analyze, Frequency, Sampling};
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
#[serde(deny_unknown_fields)]
pub struct MultipoleAnalysisBuilder {
    pub selection: Selection,
    pub frequency: Frequency,
}

impl MultipoleAnalysisBuilder {
    /// No-op: this analysis writes its result via YAML, not a `file:` field.
    pub fn apply_output_dir(&mut self, _dir: &std::path::Path) -> Result<()> {
        Ok(())
    }

    pub fn build(&self, context: &impl Context) -> Result<MultipoleAnalysis> {
        let topology = context.topology_ref();
        let groups = context.resolve_groups(&self.selection);
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
            sampling: Sampling::new(self.frequency),
            charge: WeightedMean::new(),
            charge_squared: WeightedMean::new(),
            dipole_scalar: WeightedMean::new(),
            dipole_squared: WeightedMean::new(),
            quadrupole: Default::default(),
            quadrupole_squared: Default::default(),
            quadrupole_norm: WeightedMean::new(),
            quadrupole_norm_squared: WeightedMean::new(),
            per_atom: Vec::new(),
            molecule_kind: Some(molecule_kind),
            track_per_atom: topology.moleculekinds()[molecule_kind].group_kind()
                == GroupKind::Molecular,
        })
    }
}

/// Per-group charge, dipole, and quadrupole analysis.
#[derive(Debug)]
pub struct MultipoleAnalysis {
    selection: Selection,
    /// Frequency and frame count, owned by the framework.
    sampling: Sampling,
    charge: WeightedMean,
    charge_squared: WeightedMean,
    dipole_scalar: WeightedMean,
    dipole_squared: WeightedMean,
    /// Traceless quadrupole tensor components [xx, xy, xz, yy, yz, zz].
    quadrupole: [WeightedMean; 6],
    quadrupole_squared: [WeightedMean; 6],
    /// Frobenius norm |Θ|_F = sqrt(Σ Θ_αβ²).
    quadrupole_norm: WeightedMean,
    quadrupole_norm_squared: WeightedMean,
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
        Some("Per-group charge, dipole, and quadrupole analysis")
    }
}

impl<T: Context> Analyze<T> for MultipoleAnalysis {
    fn sampling(&self) -> &Sampling {
        &self.sampling
    }
    fn sampling_mut(&mut self) -> &mut Sampling {
        &mut self.sampling
    }

    fn perform_sample(&mut self, context: &T, _step: usize, weight: f64) -> Result<()> {
        let topology = context.topology_ref();
        let atomkinds = topology.atomkinds();
        let moleculekinds = topology.moleculekinds();

        for &gi in &context.resolve_groups(&self.selection) {
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

            if let Some(qt) =
                crate::collective_variable::group::group_quadrupole_moment(gi, context)
            {
                let comps = [
                    qt[(0, 0)],
                    qt[(0, 1)],
                    qt[(0, 2)],
                    qt[(1, 1)],
                    qt[(1, 2)],
                    qt[(2, 2)],
                ];
                for (acc, &v) in self.quadrupole.iter_mut().zip(&comps) {
                    acc.add(v, weight);
                }
                for (acc, &v) in self.quadrupole_squared.iter_mut().zip(&comps) {
                    acc.add(v * v, weight);
                }
                let norm = qt.norm();
                self.quadrupole_norm.add(norm, weight);
                self.quadrupole_norm_squared.add(norm * norm, weight);
            }
        }

        Ok(())
    }

    fn results(&self) -> Option<serde_yml::Value> {
        if self.sampling.num_samples() == 0 {
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
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("charge", format!("{z_mean:.4} ± {z_std:.4}"))?;
        map.try_insert("capacitance", capacitance)?;
        map.try_insert("dipole_moment", format!("{mu_mean:.4} ± {mu_std:.4}"))?;

        let qnorm_mean = self.quadrupole_norm.mean();
        let qnorm_std = (self.quadrupole_norm_squared.mean() - qnorm_mean * qnorm_mean)
            .max(0.0)
            .sqrt();
        map.try_insert(
            "quadrupole_moment",
            format!("{qnorm_mean:.4} ± {qnorm_std:.4}"),
        )?;

        // Two flat lists (order [xx, xy, xz, yy, yz, zz]) rather than per-component
        // strings, so the tensor loads straight into e.g. a NumPy array.
        let (values, errors): (Vec<f64>, Vec<f64>) = self
            .quadrupole
            .iter()
            .zip(&self.quadrupole_squared)
            .map(|(mean_acc, sq_acc)| {
                let mean = mean_acc.mean();
                let std = (sq_acc.mean() - mean * mean).max(0.0).sqrt();
                (mean, std)
            })
            .unzip();
        let mut qt_map = serde_yml::Mapping::new();
        qt_map.try_insert("order", ["xx", "xy", "xz", "yy", "yz", "zz"])?;
        qt_map.try_insert("values", values)?;
        qt_map.try_insert("errors", errors)?;
        map.try_insert("quadrupole_tensor", serde_yml::Value::Mapping(qt_map))?;

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
    fn quadrupole_tensor_components_match_analytical() {
        // Two +q charges at ±(d, d, 0) with d = 2, COM at the origin. The
        // primitive tensor is 2q·[[d², d², 0], [d², d², 0], [0, 0, 0]] with
        // trace 4qd², so the traceless Θ = ½(3Q − tr(Q)I) is (order
        // [xx, xy, xz, yy, yz, zz]): [4, 12, 0, 4, 0, −8]. A like-charge,
        // off-axis placement pins the diagonal, an off-diagonal, and the sign —
        // unlike an antisymmetric pair, whose tensor vanishes for any impl.
        let ctx = backend_from_str(
            r#"
atoms:
  - {name: P, mass: 1.0, charge: 1.0, sigma: 1.0}
molecules:
  - name: PAIR
    atoms: [P, P]
system:
  cell: !Cuboid [100.0, 100.0, 100.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: PAIR
      N: 1
      insert: !Manual [[2.0, 2.0, 0.0], [-2.0, -2.0, 0.0]]
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
        let qt = yaml.get("quadrupole_tensor").unwrap();
        let values: Vec<f64> = qt
            .get("values")
            .and_then(|v| v.as_sequence())
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected = [4.0, 12.0, 0.0, 4.0, 0.0, -8.0];
        for (got, want) in values.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-9, "got {got}, expected {want}");
        }
        // Θ must be traceless: xx + yy + zz = 0.
        assert!((values[0] + values[3] + values[5]).abs() < 1e-9);

        // Frobenius norm counts off-diagonals twice: √(4²+4²+8² + 2·12²) = √384.
        let norm_str = yaml.get("quadrupole_moment").unwrap().as_str().unwrap();
        let norm: f64 = norm_str.split('±').next().unwrap().trim().parse().unwrap();
        assert!((norm - 384.0_f64.sqrt()).abs() < 1e-3, "norm = {norm}");
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
