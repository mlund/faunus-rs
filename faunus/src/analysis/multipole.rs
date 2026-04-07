//! Multipole analysis: per-group charge statistics and dipole moment.
//!
//! Computes mean charge ⟨Z⟩, charge capacitance C = ⟨Z²⟩ − ⟨Z⟩²,
//! and mean dipole moment magnitude ⟨|μ|⟩ averaged over all groups
//! matching a selection. Handles atom-type swaps (titration) and
//! GCMC (only active groups contribute).

use super::{Analyze, Frequency};
use crate::auxiliary::{MappingExt, WeightedMean};
use crate::selection::Selection;
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
        let groups = context.resolve_groups_live(&self.selection);
        if groups.is_empty() {
            anyhow::bail!(
                "Multipole: selection '{}' matched no groups",
                self.selection.source()
            );
        }
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
            molecule_kind: None,
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
    /// Molecule kind for per-atom tracking; set on first sample.
    molecule_kind: Option<usize>,
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

            // Lazy-init per-atom accumulators from the first group's molecule kind
            if self.molecule_kind.is_none() {
                self.molecule_kind = Some(mol);
                let molkind = &moleculekinds[mol];
                self.per_atom = (0..molkind.atoms().len())
                    .map(|i| PerAtomCharge {
                        name: molkind.resolved_atom_name(i, atomkinds).to_owned(),
                        q: WeightedMean::new(),
                        q_squared: WeightedMean::new(),
                    })
                    .collect();
            }

            let track_per_atom = self.molecule_kind.is_some_and(|m| m == mol);
            let mut z = 0.0;
            for i in group.iter_active() {
                let q = atomkinds[context.atom_kind(i)].charge();
                z += q;
                if track_per_atom {
                    let rel = i - group.start();
                    self.per_atom[rel].q.add(q, weight);
                    self.per_atom[rel].q_squared.add(q * q, weight);
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
