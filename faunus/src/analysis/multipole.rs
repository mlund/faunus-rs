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

        // Resolve live to capture GCMC activation changes and atom-type swaps from titration
        for &gi in &context.resolve_groups_live(&self.selection) {
            let group = &context.groups()[gi];

            // Uses live atom_kind so swapped charges (titration) are reflected
            let z: f64 = group
                .iter_active()
                .map(|i| atomkinds[context.atom_kind(i)].charge())
                .sum();
            self.charge.add(z, weight);
            self.charge_squared.add(z * z, weight); // for capacitance C = ⟨Z²⟩ − ⟨Z⟩²

            // Returns None for groups without COM (e.g. empty groups)
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
        let capacitance = self.charge_squared.mean() - z_mean * z_mean;
        let z_std = capacitance.max(0.0).sqrt();

        let mu_mean = self.dipole_scalar.mean();
        let mu_var = self.dipole_squared.mean() - mu_mean * mu_mean;
        let mu_std = mu_var.max(0.0).sqrt();

        map.try_insert("selection", self.selection.source())?;
        map.try_insert("num_samples", self.num_samples)?;
        map.try_insert("charge", format!("{z_mean:.4} ± {z_std:.4}"))?;
        map.try_insert("capacitance", capacitance)?;
        map.try_insert("dipole_moment", format!("{mu_mean:.4} ± {mu_std:.4}"))?;

        Some(serde_yml::Value::Mapping(map))
    }
}
