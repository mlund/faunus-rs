// Copyright 2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Scaled Widom insertion for single-ion excess chemical potential.
//!
//! Implements the method of [Svensson & Woodward (1988)](https://doi.org/10.1080/00268978800100203)
//! where charge scaling maintains electroneutrality in the finite periodic box.

use super::{Analyze, Frequency};
use crate::cell::BoundaryConditions;
use crate::energy::builder::PairInteraction;
use crate::energy::pairpot::ShortRange;
use crate::topology::AtomKind;
use crate::Context;
use crate::auxiliary::BlockAverage;
use anyhow::Result;
use derive_builder::Builder;
use derive_more::Debug;
use interatomic::coulomb::Temperature;
use interatomic::twobody::IsotropicTwobodyEnergy;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Coulomb scheme wrapper for `ion_potential` calls without baked-in charge products.
#[derive(Clone, Debug)]
enum CoulombEvaluator {
    Plain(interatomic::coulomb::pairwise::Plain),
    ReactionField(interatomic::coulomb::pairwise::ReactionField),
    RealSpaceEwald(interatomic::coulomb::pairwise::RealSpaceEwald),
    Ewald(interatomic::coulomb::pairwise::EwaldTruncated),
    Fanourgakis(interatomic::coulomb::pairwise::Fanourgakis),
}

macro_rules! dispatch_coulomb_evaluator {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            CoulombEvaluator::Plain(s) => s.$method($($arg),*),
            CoulombEvaluator::ReactionField(s) => s.$method($($arg),*),
            CoulombEvaluator::RealSpaceEwald(s) => s.$method($($arg),*),
            CoulombEvaluator::Ewald(s) => s.$method($($arg),*),
            CoulombEvaluator::Fanourgakis(s) => s.$method($($arg),*),
        }
    };
}

impl CoulombEvaluator {
    /// Electric potential at distance `r` from a point charge `z`.
    /// Units: [charge / length] (dimensionless in Å/e convention).
    fn ion_potential(&self, charge: f64, distance: f64) -> f64 {
        use interatomic::coulomb::pairwise::MultipolePotential;
        dispatch_coulomb_evaluator!(self, ion_potential, charge, distance)
    }
}

/// Extract `CoulombEvaluator` from a `PairInteraction` Coulomb variant.
fn extract_coulomb_evaluator(interaction: &PairInteraction) -> Option<CoulombEvaluator> {
    match interaction {
        PairInteraction::CoulombPlain(s) => Some(CoulombEvaluator::Plain(s.clone())),
        PairInteraction::CoulombReactionField(s) => {
            Some(CoulombEvaluator::ReactionField(s.clone()))
        }
        PairInteraction::CoulombRealSpaceEwald(s) => {
            Some(CoulombEvaluator::RealSpaceEwald(s.clone()))
        }
        PairInteraction::CoulombEwald(s) => Some(CoulombEvaluator::Ewald(s.clone())),
        PairInteraction::CoulombFanourgakis(s) => Some(CoulombEvaluator::Fanourgakis(s.clone())),
        _ => None,
    }
}

fn default_insertions() -> Option<usize> {
    Some(10)
}

fn default_lambda_points() -> Option<usize> {
    Some(11)
}

/// Scaled Widom insertion analysis for single-ion excess chemical potential.
///
/// A ghost particle is inserted at random positions; the excess chemical
/// potential is decomposed into short-range and electrostatic contributions.
/// Charge scaling maintains electroneutrality in the finite periodic box.
///
/// Reference: [Svensson & Woodward, Mol. Phys. 64:2, 247-259 (1988)](https://doi.org/10.1080/00268978800100203)
#[derive(Debug, Builder)]
#[builder(build_fn(skip), derive(Deserialize, Serialize))]
pub struct ScaledWidomInsertion {
    /// Ghost atom type name
    atom: String,

    /// Number of ghost insertions per sample call
    #[builder_field_attr(serde(default = "default_insertions"))]
    insertions: usize,

    /// Number of lambda quadrature points on [0,1]
    #[builder_field_attr(serde(default = "default_lambda_points"))]
    lambda_points: usize,

    /// Pair interactions for the ghost (same syntax as nonbonded.default)
    #[allow(dead_code)] // consumed at build time; kept for struct completeness
    default: Vec<PairInteraction>,

    /// Sample frequency
    frequency: Frequency,

    // --- constructed at build time ---
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    ghost_charge: f64,

    /// Precomputed 1/(lambda_points - 1) for lambda quadrature
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    lambda_inv: f64,

    /// 1/kT in mol/kJ
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    beta: f64,

    /// SR potentials: ghost vs each atom type j
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    sr_potentials: Vec<ShortRange>,

    /// Per-atom-type charges, indexed by atom type id
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    atom_charges: Vec<f64>,

    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    coulomb_scheme: Option<CoulombEvaluator>,

    /// TO_CHEMISTRY_UNIT / permittivity
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    coulomb_prefactor: f64,

    // --- per-block accumulators (reset each sample call) ---

    /// Σ β·ΔU_el(λ_k)·exp(−β·ΔU(λ_k)) per λ-point (numerator of I(λ))
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    block_lambda_num: Vec<f64>,

    /// Σ exp(−β·ΔU(λ_k)) per λ-point (denominator of I(λ))
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    block_lambda_den: Vec<f64>,

    /// Σ exp(−β·ΔU_sr) over insertions in this block
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    block_sr_boltzmann: f64,

    /// Σ exp(−β·ΔU_total) over insertions (unscaled Widom, no charge scaling)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    block_unscaled_sum: f64,

    /// Scratch buffer for I(λ_k) = num[k]/den[k], reused each sample
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    integrand: Vec<f64>,

    // --- running statistics ---
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    mu_sr: BlockAverage,

    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    mu_el: BlockAverage,

    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    mu_total: BlockAverage,

    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    mu_unscaled: BlockAverage,

    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    num_samples: usize,

    /// Seeded RNG for deterministic ghost insertion positions
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    #[debug(skip)]
    rng: rand::rngs::StdRng,
}

impl ScaledWidomInsertionBuilder {
    /// Build analysis from the builder, resolving atom types against topology.
    pub fn build(
        &self,
        context: &impl Context,
        medium: Option<&interatomic::coulomb::Medium>,
    ) -> Result<ScaledWidomInsertion> {
        let atom_name = self
            .atom
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("ScaledWidomInsertion: missing 'atom' field"))?;
        let frequency = self
            .frequency
            .ok_or_else(|| anyhow::anyhow!("ScaledWidomInsertion: missing 'frequency' field"))?;
        let insertions = self.insertions.unwrap_or(10);
        let lambda_points = self.lambda_points.unwrap_or(11);
        let interactions = self
            .default
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ScaledWidomInsertion: missing 'default' field"))?;

        let topology = context.topology();
        let atomkinds = topology.atomkinds();
        let ghost_atom = topology
            .find_atom(atom_name)
            .ok_or_else(|| anyhow::anyhow!("atom type '{}' not found in topology", atom_name))?;
        let ghost_charge = ghost_atom.charge();

        let temperature = medium
            .map(|m| m.temperature())
            .ok_or_else(|| anyhow::anyhow!("ScaledWidomInsertion requires medium (temperature)"))?;
        let beta = 1.0 / (crate::R_IN_KJ_PER_MOL * temperature);

        // Build SR potentials and charge table, indexed by atom type
        let sr_interactions: Vec<_> = interactions.iter().filter(|i| !i.is_coulomb()).collect();
        let mut sr_potentials = Vec::with_capacity(atomkinds.len());
        let mut atom_charges = Vec::with_capacity(atomkinds.len());
        for atom_j in atomkinds {
            sr_potentials.push(build_sr_for_pair(ghost_atom, atom_j, &sr_interactions)?);
            atom_charges.push(atom_j.charge());
        }

        let coulomb_scheme = interactions.iter().find_map(extract_coulomb_evaluator);
        let permittivity = medium.map(|m| m.permittivity()).unwrap_or(1.0);
        let coulomb_prefactor = interatomic::coulomb::TO_CHEMISTRY_UNIT / permittivity;

        Ok(ScaledWidomInsertion {
            atom: atom_name.to_string(),
            insertions,
            lambda_points,
            default: interactions.clone(),
            frequency,
            ghost_charge,
            lambda_inv: 1.0 / (lambda_points - 1).max(1) as f64,
            beta,
            sr_potentials,
            atom_charges,
            coulomb_scheme,
            coulomb_prefactor,
            block_lambda_num: vec![0.0; lambda_points],
            block_lambda_den: vec![0.0; lambda_points],
            block_sr_boltzmann: 0.0,
            block_unscaled_sum: 0.0,
            integrand: vec![0.0; lambda_points],
            mu_sr: BlockAverage::new(),
            mu_el: BlockAverage::new(),
            mu_total: BlockAverage::new(),
            mu_unscaled: BlockAverage::new(),
            num_samples: 0,
            rng: rand::rngs::StdRng::seed_from_u64(0xB0BA_CAFE_F00D),
        })
    }
}

/// Build combined short-range potential for a ghost-atom pair.
fn build_sr_for_pair(
    ghost: &AtomKind,
    atom_j: &AtomKind,
    sr_interactions: &[&PairInteraction],
) -> Result<ShortRange> {
    match sr_interactions.len() {
        0 => Ok(ShortRange::None),
        1 => sr_interactions[0].to_short_range(ghost, atom_j),
        _ => {
            let potentials: Vec<Box<dyn IsotropicTwobodyEnergy>> = sr_interactions
                .iter()
                .map(|i| -> Result<Box<dyn IsotropicTwobodyEnergy>> {
                    let sr = i.to_short_range(ghost, atom_j)?;
                    Ok(Box::new(sr) as Box<dyn IsotropicTwobodyEnergy>)
                })
                .collect::<Result<_>>()?;
            let sum = potentials
                .into_iter()
                .reduce(|a, b| Box::new(interatomic::twobody::Combined::new(a, b)))
                .unwrap();
            Ok(ShortRange::Dynamic(interatomic::twobody::ArcPotential::new(
                sum,
            )))
        }
    }
}

use crate::auxiliary::simpson_integrate;

impl ScaledWidomInsertion {
    fn reset_block(&mut self) {
        self.block_lambda_num.fill(0.0);
        self.block_lambda_den.fill(0.0);
        self.block_sr_boltzmann = 0.0;
        self.block_unscaled_sum = 0.0;
    }
}

impl crate::Info for ScaledWidomInsertion {
    fn short_name(&self) -> Option<&'static str> {
        Some("scaled_widom_insertion")
    }

    fn long_name(&self) -> Option<&'static str> {
        Some("Scaled Widom insertion for single-ion chemical potential")
    }

    fn citation(&self) -> Option<&'static str> {
        Some("doi:10.1080/00268978800100203")
    }
}

impl<T: Context> Analyze<T> for ScaledWidomInsertion {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }

        // N_T enters the charge scaling factor 1 − λz_α/(z_β N_T)
        let n_total: usize = context.groups().iter().map(|g| g.len()).sum();
        let n_total_f = n_total as f64;
        let z_alpha = self.ghost_charge;
        let prefactor = self.coulomb_prefactor;
        let beta = self.beta;
        let n_lambda = self.lambda_points;
        let cell = context.cell();

        self.reset_block();
        let lambda_inv = self.lambda_inv;

        for _ in 0..self.insertions {
            let r_ghost = crate::cell::random_point_inside(cell, &mut self.rng);
            let mut du_sr = 0.0;
            let mut phi = 0.0; // Φ(r) = Σ_j scheme.ion_potential(z_j, r_j)
            let mut s = 0.0; // S(r) = Σ_j scheme.ion_potential(1, r_j)

            for group in context.groups() {
                for j in group.iter_active() {
                    let pos_j = context.position(j);
                    let dist_sq = cell.distance_squared(&r_ghost, &pos_j);
                    let atom_type = context.atom_kind(j) as usize;

                    du_sr += self.sr_potentials[atom_type].isotropic_twobody_energy(dist_sq);

                    if let Some(ref scheme) = self.coulomb_scheme {
                        let dist = dist_sq.sqrt();
                        phi += scheme.ion_potential(self.atom_charges[atom_type], dist);
                        s += scheme.ion_potential(1.0, dist);
                    }
                }
            }

            // Hard-core overlaps give exp(−β·∞) = 0, correctly contributing
            // nothing to the Widom averages without needing explicit rejection.
            let sr_boltz = (-beta * du_sr).exp();
            let du_unscaled = du_sr + z_alpha * phi * prefactor;
            self.block_sr_boltzmann += sr_boltz;
            self.block_unscaled_sum += (-beta * du_unscaled).exp();

            // Lambda quadrature for scaled electrostatic contribution (eq. 7).
            // ΔU_el(λ) = λ z_α [Φ(r) − λ z_α S(r)/N_T] × prefactor
            // The sr_boltz factor couples SR overlap to the electrostatic weight.
            for k in 0..n_lambda {
                let lambda_k = k as f64 * lambda_inv;
                let du_el =
                    lambda_k * z_alpha * (phi - lambda_k * z_alpha * s / n_total_f) * prefactor;
                let boltz = sr_boltz * (-beta * du_el).exp();
                self.block_lambda_den[k] += boltz;
                self.block_lambda_num[k] += beta * du_el * boltz;
            }
        }

        // Per-block excess chemical potentials (in units of kT)
        let n_ins = self.insertions as f64;
        let mu_sr_block = -(self.block_sr_boltzmann / n_ins).ln();

        // I(λ_k) = num[k] / den[k]; integrated with Simpson's rule → β Δμ_el
        for (dest, (&num, &den)) in self.integrand.iter_mut().zip(
            self.block_lambda_num
                .iter()
                .zip(self.block_lambda_den.iter()),
        ) {
            *dest = if den > 0.0 { num / den } else { 0.0 };
        }
        let mu_el_block = simpson_integrate(&self.integrand);

        let mu_total_block = mu_sr_block + mu_el_block;
        let mu_unscaled_block = -(self.block_unscaled_sum / n_ins).ln();

        // Feed per-block values into running statistics for mean ± SEM
        self.mu_sr.add(mu_sr_block);
        self.mu_el.add(mu_el_block);
        self.mu_total.add(mu_total_block);
        self.mu_unscaled.add(mu_unscaled_block);
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
        map.insert("atom".into(), serde_yml::Value::String(self.atom.clone()));
        map.insert(
            "num_samples".into(),
            serde_yml::Value::Number(self.num_samples.into()),
        );
        let mut excess = serde_yml::Mapping::new();
        excess.insert("short_range".into(), self.mu_sr.to_yaml()?);
        excess.insert("electrostatic".into(), self.mu_el.to_yaml()?);
        excess.insert("total".into(), self.mu_total.to_yaml()?);
        map.insert(
            "excess_chemical_potential (kT)".into(),
            serde_yml::Value::Mapping(excess),
        );

        map.insert("unscaled_widom (kT)".into(), self.mu_unscaled.to_yaml()?);

        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
atom: Na
insertions: 20
lambda_points: 11
frequency: !Every 10
default:
  - !Coulomb { cutoff: 1000.0 }
  - !WCA { mixing: arithmetic }
"#;
        let builder: ScaledWidomInsertionBuilder = serde_yml::from_str(yaml).unwrap();
        assert_eq!(builder.atom.as_deref().unwrap(), "Na");
        assert_eq!(builder.insertions.unwrap(), 20);
        assert_eq!(builder.lambda_points.unwrap(), 11);
        let interactions = builder.default.as_ref().unwrap();
        assert_eq!(interactions.len(), 2);
        assert!(interactions[0].is_coulomb());
        assert!(!interactions[1].is_coulomb());
    }

    #[test]
    fn info_trait() {
        let swi = ScaledWidomInsertion {
            atom: "Na".to_string(),
            insertions: 10,
            lambda_points: 11,
            default: vec![],
            frequency: Frequency::Every(10),
            ghost_charge: 1.0,
            lambda_inv: 0.1,
            beta: 1.0,
            sr_potentials: vec![],
            atom_charges: vec![],
            coulomb_scheme: None,
            coulomb_prefactor: 0.0,
            block_lambda_num: vec![],
            block_lambda_den: vec![],
            block_sr_boltzmann: 0.0,
            block_unscaled_sum: 0.0,
            integrand: vec![],
            mu_sr: BlockAverage::new(),
            mu_el: BlockAverage::new(),
            mu_total: BlockAverage::new(),
            mu_unscaled: BlockAverage::new(),
            num_samples: 0,
            rng: rand::rngs::StdRng::seed_from_u64(0),
        };
        use crate::Info;
        assert_eq!(swi.short_name(), Some("scaled_widom_insertion"));
        assert!(swi.citation().unwrap().starts_with("doi:"));
    }
}
