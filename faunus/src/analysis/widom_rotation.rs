// Copyright 2026 Mikael Lund
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

//! Widom rotational perturbation of a molecule about its center of mass.
//!
//! For each frozen snapshot a tagged rigid molecule is held at its center of
//! mass and rigidly reoriented to `M` trial orientations sampled uniformly on
//! SO(3). We evaluate only the molecule↔environment energy. A rotation about the
//! center of mass leaves every intramolecular distance unchanged, so the bonded
//! and intramolecular non-bonded energy are rotation-invariant and cancel.
//!
//! The run's own Hamiltonian supplies the orientational energy landscape `u(Ω)`,
//! which we reduce to:
//! - the orientationally-averaged one-body potential of mean force
//!   `W = -RT·ln⟨M⁻¹ Σ exp(-u/RT)⟩` (compare with an umbrella profile);
//! - the mean interaction energy `⟨u⟩`;
//! - the orientational entropy relative to a free rotor,
//!   `S_orient/R = -Σₖ wₖ ln(M wₖ)`, taken straight from the Boltzmann weights
//!   `wₖ` of the trial orientations. This is the exact SO(3) entropy, with no
//!   model for the shape of the well; the restriction-of-order/free-energy
//!   connection is due to [Akke et al. (1993)](https://doi.org/10.1021/ja00074a073),
//!   whose per-vector `S² → entropy` mapping we do *not* use. Together with `W`
//!   and `⟨u⟩` it forms an `F = U - TS` decomposition;
//! - the Lipari–Szabo generalized order parameter `S²` for chosen molecular
//!   vectors ([Lipari & Szabo (1982)](https://doi.org/10.1021/ja00381a009));
//! - optionally the mean torque and the librational stiffness of the cage.
//!
//! `W`, `⟨u⟩` and the stiffness are in kJ/mol; the entropy is dimensionless
//! (`≤ 0`, zero for a free rotor) and the torque is in kT per radian.
//!
//! With implicit solvent these energies are potentials of mean force (free
//! energies relative to pure solvent), not mechanical energies.
//!
//! The analysis emits one set of numbers per run. For spatial resolution, run
//! separate umbrella windows and combine them afterwards.

use super::widom::WidomAccumulator;
use super::{Analyze, Frequency, Sampling};
use crate::auxiliary::{BlockAverage, BlockSummary, ColumnWriter, MappingExt};
use crate::cell::BoundaryConditions;
use crate::context::{PerturbContext, Perturbation};
use crate::energy::EnergyChange;
use crate::geometry::GyrationTensor;
use crate::selection::{CachedSelection, Groups, Selection};
use crate::ObserveContext;
use crate::{Point, UnitQuaternion};
use anyhow::Result;
use derive_more::Debug;
use nalgebra::{Matrix3, Matrix4, Quaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// First super-Fibonacci constant, `√2`.
const PHI: f64 = std::f64::consts::SQRT_2;
/// Second super-Fibonacci constant (root of `ψ³ = ψ + 4`).
const PSI: f64 = 1.533_751_168_755_204_3;

/// Below this effective sample size the weighted orientation cloud is too sparse
/// to define a 3×3 covariance, so per-snapshot stiffness is skipped.
const MIN_NEFF_FOR_STIFFNESS: f64 = 3.0;

/// Guard against dividing by a vanishing orientational variance.
const VARIANCE_EPSILON: f64 = 1e-12;

/// Low-discrepancy set of `n` orientations sampled uniformly on SO(3).
///
/// Deterministic super-Fibonacci spiral of unit quaternions with equal weights,
/// so `M⁻¹ Σ f(Ωₖ)` approaches the Haar average of `f`. Being quasi-random rather
/// than random, it converges faster than plain Monte Carlo but carries a
/// discrepancy-bounded error rather than a statistical one.
/// See [Alexa (2022)](https://doi.org/10.1109/CVPR52688.2022.00811).
fn super_fibonacci(n: usize) -> Vec<UnitQuaternion> {
    use std::f64::consts::TAU;
    (0..n)
        .map(|i| {
            let s = i as f64 + 0.5;
            let t = s / n as f64;
            let (radial, axial) = (t.sqrt(), (1.0 - t).sqrt());
            let (alpha, beta) = (TAU * s / PHI, TAU * s / PSI);
            UnitQuaternion::new_normalize(Quaternion::new(
                axial * beta.cos(),
                radial * alpha.sin(),
                radial * alpha.cos(),
                axial * beta.sin(),
            ))
        })
        .collect()
}

/// A molecular vector whose orientational order parameter is measured.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorSpec {
    /// Vector between two atoms/beads, given by their indices within the molecule.
    Pair([usize; 2]),
    /// Gyration-tensor principal axis: 0 = smallest, 1 = middle, 2 = largest.
    Axis(usize),
    /// Explicit unit vector in the molecule's initial reference frame.
    Body([f64; 3]),
}

impl VectorSpec {
    /// Human-readable label for output columns.
    fn label(&self) -> String {
        match self {
            Self::Pair([i, j]) => format!("pair({i},{j})"),
            Self::Axis(a) => format!("axis{a}"),
            Self::Body([x, y, z]) => format!("body({x:.2},{y:.2},{z:.2})"),
        }
    }

    /// Lab-frame reference direction (unit) for group `gi` at its current
    /// orientation. Each trial orientation later rotates this rigidly.
    fn reference<T: ObserveContext>(
        &self,
        gi: usize,
        indices: &[usize],
        com: &Point,
        context: &T,
    ) -> Result<Point> {
        let dir = match self {
            Self::Pair([i, j]) => {
                let (a, b) = (indices[*i], indices[*j]);
                context
                    .cell()
                    .distance(&context.position(a), &context.position(b))
            }
            Self::Axis(axis) => {
                let positions_masses = indices
                    .iter()
                    .map(|&i| (context.position(i), context.atom_mass(i)));
                let tensor = GyrationTensor::from_positions_masses_com(
                    positions_masses,
                    com,
                    context.cell(),
                )
                .ok_or_else(|| anyhow::anyhow!("cannot form gyration tensor for axis vector"))?;
                tensor.rotation.matrix().column(*axis).into_owned()
            }
            Self::Body(v) => {
                let body = Vector3::new(v[0], v[1], v[2]);
                context.groups()[gi].quaternion().transform_vector(&body)
            }
        };
        let norm = dir.norm();
        if norm < VARIANCE_EPSILON {
            anyhow::bail!("degenerate reference vector for {}", self.label());
        }
        Ok(dir / norm)
    }
}

/// Default torque finite-difference step in radians.
const fn default_dtheta() -> f64 {
    0.01
}

/// YAML builder for [`WidomRotation`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WidomRotationBuilder {
    /// Selection of molecular (non-atomic) groups to perturb.
    pub selection: Selection,
    /// Number of trial orientations `M`.
    pub orientations: usize,
    /// Molecular vectors for the order parameter `S²`; omit to skip `S²`.
    #[serde(default)]
    pub vectors: Vec<VectorSpec>,
    /// Measure the mean torque by a small virtual rotation.
    #[serde(default)]
    pub torque: bool,
    /// Finite rotation step for the torque, in radians.
    #[serde(default = "default_dtheta")]
    pub dtheta: f64,
    /// Measure the librational stiffness of the cage.
    #[serde(default)]
    pub stiffness: bool,
    /// Optional per-sample output stream (gzip CSV when `.csv.gz`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    /// Sampling frequency.
    pub frequency: Frequency,
}

impl WidomRotationBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        crate::analysis::prefix_opt(&mut self.file, dir)
    }

    /// `thermal_energy` is `R·T` in kJ/mol.
    pub fn build(
        &self,
        context: &impl ObserveContext,
        thermal_energy: f64,
    ) -> Result<WidomRotation> {
        anyhow::ensure!(
            self.orientations > 0,
            "WidomRotation: 'orientations' must be > 0"
        );
        if self.torque {
            anyhow::ensure!(
                self.dtheta.abs() > VARIANCE_EPSILON,
                "WidomRotation: 'dtheta' must be non-zero when 'torque' is set"
            );
        }

        let groups = context.resolve_groups(&self.selection);
        anyhow::ensure!(
            !groups.is_empty(),
            "WidomRotation: selection '{}' matched no groups",
            self.selection.source()
        );

        // A single molecule kind keeps vector indices meaningful across matches.
        let topology = context.topology_ref();
        let mol_id = context.groups()[groups[0]].molecule();
        for &gi in &groups {
            anyhow::ensure!(
                context.groups()[gi].molecule() == mol_id,
                "WidomRotation: selection '{}' spans multiple molecule kinds",
                self.selection.source()
            );
        }
        let kind = topology.moleculekind(mol_id);
        anyhow::ensure!(
            !kind.atomic() && kind.len() >= 2,
            "WidomRotation: molecule '{}' has no rigid-body orientation",
            kind.name()
        );
        self.validate_vectors(kind.len())?;

        let stream = self
            .file
            .as_ref()
            .map(|path| ColumnWriter::open(path, &["step", "W/kJ/mol", "mean_S2", "N_eff"]))
            .transpose()?;

        Ok(WidomRotation {
            selection: CachedSelection::groups(self.selection.clone()),
            quaternions: super_fibonacci(self.orientations),
            vectors: self.vectors.clone(),
            thermal_energy,
            widom: WidomAccumulator::default(),
            order: self.vectors.iter().map(|_| BlockAverage::new()).collect(),
            mean_order: BlockAverage::new(),
            mean_interaction: BlockAverage::new(),
            entropy: BlockAverage::new(),
            neff: BlockAverage::new(),
            torque: self.torque.then(|| TorqueProbe {
                axes: std::array::from_fn(|_| WidomAccumulator::default()),
                dtheta: self.dtheta,
            }),
            stiffness: self
                .stiffness
                .then(|| std::array::from_fn(|_| BlockAverage::new())),
            sampling: Sampling::new(self.frequency),
            num_blocks: 0,
            stream,
        })
    }

    fn validate_vectors(&self, molecule_len: usize) -> Result<()> {
        for spec in &self.vectors {
            match spec {
                VectorSpec::Pair([i, j]) => anyhow::ensure!(
                    i != j && *i < molecule_len && *j < molecule_len,
                    "WidomRotation: pair {spec:?} out of range for molecule of {molecule_len} atoms"
                ),
                VectorSpec::Axis(a) => {
                    anyhow::ensure!(*a < 3, "WidomRotation: axis index {a} must be 0, 1 or 2")
                }
                VectorSpec::Body(v) => anyhow::ensure!(
                    Vector3::new(v[0], v[1], v[2]).norm() > VARIANCE_EPSILON,
                    "WidomRotation: body vector {v:?} must be non-zero"
                ),
            }
        }
        Ok(())
    }
}

/// Mean-torque probe: one Widom accumulator per lab axis.
#[derive(Clone, Debug, Default)]
struct TorqueProbe {
    axes: [WidomAccumulator; 3],
    dtheta: f64,
}

/// Widom rotational perturbation analysis. See the module documentation.
#[derive(Debug)]
pub struct WidomRotation {
    selection: CachedSelection<Groups>,
    quaternions: Vec<UnitQuaternion>,
    vectors: Vec<VectorSpec>,
    thermal_energy: f64,
    /// Orientational partition function → `W`.
    widom: WidomAccumulator,
    /// `S²` per vector, and averaged over vectors.
    order: Vec<BlockAverage>,
    mean_order: BlockAverage,
    mean_interaction: BlockAverage,
    /// Orientational configurational entropy in units of the gas constant.
    entropy: BlockAverage,
    neff: BlockAverage,
    torque: Option<TorqueProbe>,
    /// Principal librational stiffnesses (kJ/mol/rad²), ascending.
    stiffness: Option<[BlockAverage; 3]>,
    /// Frequency and frame count, owned by the framework.
    sampling: Sampling,
    /// Molecule scans, one per matching molecule per frame. Each is an independent block, so this
    /// is the count behind every reported error bar.
    num_blocks: usize,
    #[debug(skip)]
    stream: Option<ColumnWriter>,
}

/// Rigidly reorient group `gi` and return its molecule↔environment energy there, in kT.
///
/// The group, and every cache that describes it, is left exactly as it was found — see
/// [`PerturbContext::measure`].
///
/// A rigid-body change excludes the rotation-invariant intramolecular energy, so the energy read
/// is exactly the interaction with the rest of the system. It is `+∞` for a pose that overlaps a
/// neighbour, which [`WidomRotation::weights`] expects and handles.
fn energy_at_orientation<T: PerturbContext>(
    trial: &mut T,
    gi: usize,
    rotation: &UnitQuaternion,
    thermal_energy: f64,
) -> anyhow::Result<f64> {
    trial.measure(
        &Perturbation::Rotate {
            group: gi,
            rotation: *rotation,
        },
        |perturbed, change| perturbed.hamiltonian().energy(perturbed, change) / thermal_energy,
    )
}

impl WidomRotation {
    /// Scan all trial orientations of group `gi` on `trial`, returning the per-orientation
    /// energies in kT. `trial` is left unchanged.
    fn scan_energies<T: PerturbContext>(
        &self,
        trial: &mut T,
        gi: usize,
    ) -> anyhow::Result<Vec<f64>> {
        self.quaternions
            .iter()
            .map(|q| energy_at_orientation(trial, gi, q, self.thermal_energy))
            .collect()
    }

    /// Boltzmann weights from energies (kT), min-shifted for numerical safety.
    ///
    /// Returns `None` when every trial orientation is forbidden (all energies
    /// `+∞`, e.g. a large molecule that clashes in any pose): the conditional
    /// orientational distribution is then undefined, and only `W` (which the
    /// Widom accumulator records as `+∞`) remains meaningful.
    fn weights(energies: &[f64]) -> Option<Vec<f64>> {
        let min = energies.iter().copied().fold(f64::INFINITY, f64::min);
        if !min.is_finite() {
            return None;
        }
        let unnormalized: Vec<f64> = energies.iter().map(|u| (-(u - min)).exp()).collect();
        let sum: f64 = unnormalized.iter().sum();
        (sum > 0.0 && sum.is_finite()).then(|| unnormalized.iter().map(|w| w / sum).collect())
    }

    /// Accumulate all snapshot observables for one molecule.
    fn accumulate(&mut self, energies: &[f64], references: &[Point]) {
        // W: pool every orientation (an all-forbidden pose gives W→∞); one block
        // per molecule-snapshot for errors.
        for &u in energies {
            self.widom.collect(u, 1.0);
        }
        self.widom.end_block();

        // The remaining observables need the conditional orientational distribution.
        let Some(weights) = Self::weights(energies) else {
            return;
        };

        let mean_u: f64 = energies.iter().zip(&weights).map(|(u, w)| u * w).sum();
        self.mean_interaction.add(mean_u * self.thermal_energy);

        // Orientational configurational entropy relative to a free rotor.
        let count = energies.len() as f64;
        let entropy: f64 = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * (count * w).ln())
            .sum();
        self.entropy.add(entropy);

        let neff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        self.neff.add(neff);

        self.accumulate_order(&weights, references);
        if let Some(stiffness) = self.stiffness.as_mut() {
            Self::accumulate_stiffness(
                stiffness,
                &self.quaternions,
                &weights,
                neff,
                self.thermal_energy,
            );
        }
    }

    /// Per-vector and mean order parameter `S²`.
    fn accumulate_order(&mut self, weights: &[f64], references: &[Point]) {
        if references.is_empty() {
            return;
        }
        let mut sum = 0.0;
        for (accumulator, reference) in self.order.iter_mut().zip(references) {
            let mut tensor = Matrix3::zeros();
            for (q, &w) in self.quaternions.iter().zip(weights) {
                let v = q.transform_vector(reference);
                tensor += w * (v * v.transpose());
            }
            let s2 = 1.5 * tensor.iter().map(|x| x * x).sum::<f64>() - 0.5;
            accumulator.add(s2);
            sum += s2;
        }
        self.mean_order.add(sum / references.len() as f64);
    }

    /// Librational stiffness `K = RT·Cov⁻¹` from the weighted orientation cloud;
    /// its principal values are `RT / var` along each principal libration axis.
    fn accumulate_stiffness(
        stiffness: &mut [BlockAverage; 3],
        quaternions: &[UnitQuaternion],
        weights: &[f64],
        neff: f64,
        thermal_energy: f64,
    ) {
        if neff < MIN_NEFF_FOR_STIFFNESS {
            return; // cloud too sparse to define a covariance
        }
        // Weighted-mean orientation: dominant eigenvector of Σ w qqᵀ (Markley).
        let mut markley = Matrix4::zeros();
        for (q, &w) in quaternions.iter().zip(weights) {
            let c = q.into_inner().coords;
            markley += w * (c * c.transpose());
        }
        let eigen = markley.symmetric_eigen();
        let dominant = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        let mean = UnitQuaternion::from_quaternion(Quaternion::from(
            eigen.eigenvectors.column(dominant).into_owned(),
        ));

        // Covariance of the librational rotation vectors about the mean.
        let mut covariance = Matrix3::zeros();
        for (q, &w) in quaternions.iter().zip(weights) {
            let theta = (q * mean.inverse()).scaled_axis();
            covariance += w * (theta * theta.transpose());
        }
        // Sort softest first so a given accumulator always tracks the same
        // principal axis; roundoff can make a vanishing variance slightly negative.
        let mut variances: Vec<f64> = covariance
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .collect();
        variances.sort_by(|a, b| b.total_cmp(a));
        for (accumulator, &variance) in stiffness.iter_mut().zip(&variances) {
            // A vanishing variance means the cloud is degenerate along this libration
            // axis and the stiffness diverges. Skip that axis alone — dropping the
            // snapshot would discard the well-defined axes, and recording a non-finite
            // value would poison the block average for good.
            if variance.is_finite() && variance > VARIANCE_EPSILON {
                accumulator.add(thermal_energy / variance);
            }
        }
    }

    /// Mean torque about each lab axis via a small virtual rotation, analogous to the
    /// virtual-translate force.
    fn accumulate_torque<T: PerturbContext>(
        &mut self,
        trial: &mut T,
        gi: usize,
        reference_energy: f64,
    ) -> anyhow::Result<()> {
        let Some(probe) = self.torque.as_mut() else {
            return Ok(());
        };
        let axes = [Vector3::x_axis(), Vector3::y_axis(), Vector3::z_axis()];
        for (accumulator, axis) in probe.axes.iter_mut().zip(axes) {
            let rotation = UnitQuaternion::from_axis_angle(&axis, probe.dtheta);
            let energy = energy_at_orientation(trial, gi, &rotation, self.thermal_energy)?;
            accumulator.collect(energy - reference_energy, 1.0);
        }
        Ok(())
    }
}

impl_info!(
    WidomRotation,
    "widom_rotation",
    "Widom rotational perturbation about the center of mass",
    "doi:10.1063/1.1734110" // Widom insertion method
);

impl<T: PerturbContext> Analyze<T> for WidomRotation {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, step: usize, _weight: f64) -> Result<()> {
        let groups = self.selection.resolve(context).to_vec();
        if groups.is_empty() {
            return Ok(()); // e.g. all molecules removed under GCMC
        }

        let mut trial = context.clone();
        for gi in groups.iter().map(|gi| gi.get()) {
            let indices: Vec<usize> = context.groups()[gi].iter_active().collect();
            let com = context.mass_center(&indices);
            let references = self
                .vectors
                .iter()
                .map(|v| v.reference(gi, &indices, &com, context))
                .collect::<Result<Vec<_>>>()?;

            // The torque compares each rotated energy against this one, so it must be read against
            // the change a rotation of this group derives — not one named independently here. No
            // refresh is owed: nothing has been perturbed, so the caches still describe the
            // coordinates.
            let unrotated = Perturbation::Rotate {
                group: gi,
                rotation: UnitQuaternion::identity(),
            };
            let reference_energy =
                trial.hamiltonian().energy(&trial, &unrotated.change()) / self.thermal_energy;
            let energies = self.scan_energies(&mut trial, gi)?;
            self.accumulate(&energies, &references);
            self.accumulate_torque(&mut trial, gi, reference_energy)?;
            // Each molecule-scan is one independent block.
            self.num_blocks += 1;
        }

        if let Some(stream) = self.stream.as_mut() {
            let w = self.widom.mean_free_energy() * self.thermal_energy;
            stream.write_row(&[
                &step,
                &format_args!("{w:.6e}"),
                &format_args!("{:.6}", self.mean_order.mean()),
                &format_args!("{:.3}", self.neff.mean()),
            ])?;
        }
        Ok(())
    }

    fn results(&self) -> Option<serde_yml::Value> {
        // A frame in which no molecule matched leaves every block empty.
        if self.num_blocks == 0 {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("num_blocks", self.num_blocks)?;
        // Pooled log-sum-exp estimate (matches the module formula and the CSV),
        // with the block-to-block standard error.
        map.try_insert(
            "W/kJ/mol",
            BlockSummary {
                mean: self.widom.mean_free_energy() * self.thermal_energy,
                error: self.widom.free_energy().error() * self.thermal_energy,
            },
        )?;
        map.try_insert("mean_interaction/kJ/mol", self.mean_interaction.summary())?;
        map.try_insert("orientational_entropy/R", self.entropy.summary())?;
        map.try_insert("N_eff", self.neff.summary())?;

        if !self.vectors.is_empty() {
            let per_vector: Vec<serde_yml::Value> = self
                .vectors
                .iter()
                .zip(&self.order)
                .filter_map(|(spec, accumulator)| {
                    let mut entry = serde_yml::Mapping::new();
                    entry.try_insert("vector", spec.label())?;
                    entry.try_insert("S2", accumulator.summary())?;
                    Some(serde_yml::Value::Mapping(entry))
                })
                .collect();
            map.insert("S2".into(), serde_yml::Value::Sequence(per_vector));
            map.try_insert("mean_S2", self.mean_order.summary())?;
        }

        if let Some(probe) = &self.torque {
            let mut torque = serde_yml::Mapping::new();
            for (axis, accumulator) in ["x", "y", "z"].iter().zip(&probe.axes) {
                let value = -accumulator.mean_free_energy() / probe.dtheta;
                torque.try_insert(axis, value)?;
            }
            map.insert(
                "torque/kT_per_rad".into(),
                serde_yml::Value::Mapping(torque),
            );
        }

        if let Some(stiffness) = &self.stiffness {
            // An axis that was degenerate in every snapshot has no samples; report it
            // as null rather than letting an empty average read as zero stiffness,
            // which would claim free rotation about an axis that is in fact locked.
            let values: Vec<serde_yml::Value> = stiffness
                .iter()
                .map(|accumulator| match accumulator.n() {
                    0 => serde_yml::Value::Null,
                    _ => {
                        serde_yml::to_value(accumulator.summary()).unwrap_or(serde_yml::Value::Null)
                    }
                })
                .collect();
            map.insert(
                "stiffness/kJ_per_mol_per_rad2".into(),
                serde_yml::Value::Sequence(values),
            );
        }
        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn super_fibonacci_is_unit_norm_and_deterministic() {
        let first = super_fibonacci(256);
        let second = super_fibonacci(256);
        assert_eq!(first.len(), 256);
        for (a, b) in first.iter().zip(&second) {
            assert_approx_eq!(f64, a.into_inner().norm(), 1.0, epsilon = 1e-12);
            assert_approx_eq!(f64, (a.into_inner() - b.into_inner()).norm(), 0.0);
        }
    }

    #[test]
    fn super_fibonacci_covers_so3_isotropically() {
        // A uniform SO(3) set carries any fixed vector isotropically → S² ≈ 0.
        let quaternions = super_fibonacci(4096);
        let reference = Vector3::new(0.0, 0.0, 1.0);
        let weight = 1.0 / quaternions.len() as f64;
        let mut tensor = Matrix3::zeros();
        for q in &quaternions {
            let v = q.transform_vector(&reference);
            tensor += weight * (v * v.transpose());
        }
        let s2 = 1.5 * tensor.iter().map(|x| x * x).sum::<f64>() - 0.5;
        assert!(
            s2.abs() < 1e-2,
            "isotropic set should give S² ≈ 0, got {s2}"
        );
    }

    #[test]
    fn s2_equal_weights_is_isotropic_and_concentrated_is_locked() {
        let quaternions = super_fibonacci(2048);
        let reference = Vector3::new(1.0, 0.0, 0.0);

        // Equal weights → isotropic.
        let equal = vec![1.0 / quaternions.len() as f64; quaternions.len()];
        let iso = order_parameter(&quaternions, &equal, &reference);
        assert!(iso.abs() < 1e-2);

        // All weight on one orientation → fully locked (S² = 1).
        let mut single = vec![0.0; quaternions.len()];
        single[7] = 1.0;
        let locked = order_parameter(&quaternions, &single, &reference);
        assert_approx_eq!(f64, locked, 1.0, epsilon = 1e-10);
    }

    /// A fully collapsed cloud has zero librational variance about every axis, so
    /// every stiffness diverges and nothing may be recorded.
    #[test]
    fn stiffness_skips_fully_degenerate_orientation_cloud() {
        let mut stiffness = std::array::from_fn(|_| BlockAverage::new());
        let identical = vec![UnitQuaternion::identity(); 4];
        let weights = vec![0.25; 4];

        WidomRotation::accumulate_stiffness(&mut stiffness, &identical, &weights, 4.0, 2.5);

        for accumulator in &stiffness {
            assert_eq!(
                accumulator.n(),
                0,
                "diverging stiffness must not be recorded"
            );
        }
    }

    /// A cloud that librates only within a plane is degenerate about the plane
    /// normal alone. The two well-defined axes must still be recorded.
    #[test]
    fn stiffness_skips_only_the_degenerate_axis() {
        let mut stiffness = std::array::from_fn(|_| BlockAverage::new());
        let angle = 0.1;
        let planar: Vec<UnitQuaternion> = [
            Vector3::x_axis(),
            Vector3::x_axis(),
            Vector3::y_axis(),
            Vector3::y_axis(),
        ]
        .iter()
        .zip([angle, -angle, angle, -angle])
        .map(|(axis, a)| UnitQuaternion::from_axis_angle(axis, a))
        .chain(std::iter::once(UnitQuaternion::identity()))
        .collect();
        let weights = vec![0.2; 5];
        let thermal_energy = 2.5;

        WidomRotation::accumulate_stiffness(&mut stiffness, &planar, &weights, 5.0, thermal_energy);

        // Softest first: the two in-plane axes carry the samples, the collapsed
        // out-of-plane axis carries none.
        let variance = 0.4 * angle * angle;
        for accumulator in &stiffness[..2] {
            assert_eq!(accumulator.n(), 1);
            assert_approx_eq!(
                f64,
                accumulator.mean(),
                thermal_energy / variance,
                epsilon = 1e-6
            );
        }
        assert_eq!(stiffness[2].n(), 0, "degenerate axis must not be recorded");
    }

    #[test]
    fn weights_flat_landscape_is_uniform() {
        let energies = vec![5.0; 100];
        let weights = WidomRotation::weights(&energies).unwrap();
        for w in &weights {
            assert_approx_eq!(f64, *w, 0.01, epsilon = 1e-12);
        }
        let neff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        assert_approx_eq!(f64, neff, 100.0, epsilon = 1e-9);
    }

    #[test]
    fn weights_survive_extreme_energies() {
        let weights = WidomRotation::weights(&[-1000.0, 1000.0]).unwrap();
        assert!(weights.iter().all(|w| w.is_finite()));
        assert_approx_eq!(f64, weights.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        assert!(weights[0] > weights[1]); // lower energy dominates
    }

    #[test]
    fn weights_none_when_all_orientations_forbidden() {
        // Every pose clashes → no accessible orientation → weights undefined.
        assert!(WidomRotation::weights(&[f64::INFINITY; 8]).is_none());
        // A single accessible orientation is enough for well-defined weights.
        assert!(WidomRotation::weights(&[f64::INFINITY, 0.0, f64::INFINITY]).is_some());
    }

    /// Helper mirroring the internal `S²` computation for a single vector.
    fn order_parameter(
        quaternions: &[UnitQuaternion],
        weights: &[f64],
        reference: &Vector3<f64>,
    ) -> f64 {
        let mut tensor = Matrix3::zeros();
        for (q, &w) in quaternions.iter().zip(weights) {
            let v = q.transform_vector(reference);
            tensor += w * (v * v.transpose());
        }
        1.5 * tensor.iter().map(|x| x * x).sum::<f64>() - 0.5
    }

    #[test]
    fn deserialize_vector_specs() {
        let yaml = r#"
selection: "molecule MOL"
orientations: 100
vectors:
  - !pair [0, 5]
  - !axis 2
  - !body [0.0, 0.0, 1.0]
frequency: !Every 10
"#;
        let builder: WidomRotationBuilder = serde_yml::from_str(yaml).unwrap();
        assert_eq!(builder.orientations, 100);
        assert_eq!(builder.vectors.len(), 3);
        assert!(matches!(builder.vectors[0], VectorSpec::Pair([0, 5])));
        assert!(matches!(builder.vectors[1], VectorSpec::Axis(2)));
        assert!(matches!(builder.vectors[2], VectorSpec::Body(_)));
    }

    #[test]
    fn vector_validation_rejects_out_of_range() {
        let builder = WidomRotationBuilder {
            selection: Selection::parse("molecule MOL").unwrap(),
            orientations: 10,
            vectors: vec![VectorSpec::Pair([0, 9])],
            torque: false,
            dtheta: 0.01,
            stiffness: false,
            file: None,
            frequency: Frequency::Every(1),
        };
        assert!(builder.validate_vectors(3).is_err()); // index 9 ≥ 3
        assert!(builder.validate_vectors(9).is_err()); // index 9 == len is out of range
        assert!(builder.validate_vectors(10).is_ok()); // indices 0, 9 < 10
        assert!(WidomRotationBuilder {
            vectors: vec![VectorSpec::Axis(3)],
            ..builder.clone()
        }
        .validate_vectors(5)
        .is_err());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::backend::Backend;
    use crate::change::{Change, GroupChange};
    use crate::context::{Context, ObserveContext, WithHamiltonian};
    use crate::group::GroupCollection;
    use float_cmp::assert_approx_eq;

    const RT_300: f64 = crate::R_IN_KJ_PER_MOL * 300.0;

    fn backend_from_str(yaml: &str) -> Backend {
        Backend::from_yaml_str(yaml, None, &mut rand::thread_rng()).unwrap()
    }

    /// A charged dimer along x, optionally in a linear external field `q*z`.
    fn dimer_backend(external: &str) -> Backend {
        backend_from_str(&format!(
            r#"
atoms:
  - {{name: P, mass: 1.0, charge: 1.0, sigma: 2.0}}
  - {{name: N, mass: 1.0, charge: -1.0, sigma: 2.0}}
molecules:
  - name: DIMER
    atoms: [P, N]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {{permittivity: !Vacuum, temperature: 300.0}}
  energy: {{{external}}}
  blocks:
    - molecule: DIMER
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#
        ))
    }

    fn builder(torque: bool, stiffness: bool) -> WidomRotationBuilder {
        WidomRotationBuilder {
            selection: Selection::parse("molecule DIMER").unwrap(),
            orientations: 200,
            vectors: vec![VectorSpec::Pair([0, 1]), VectorSpec::Axis(2)],
            torque,
            dtheta: 0.01,
            stiffness,
            file: None,
            frequency: Frequency::Every(1),
        }
    }

    #[test]
    fn full_observables_are_reported_and_state_is_preserved() {
        let ctx = dimer_backend(r#"custom_external: [{selection: "all", function: "q * z"}]"#);
        let mut analysis = builder(true, true).build(&ctx, RT_300).unwrap();

        let gi = ctx.resolve_groups(analysis.selection.selection())[0];
        let indices: Vec<usize> = ctx.groups()[gi].iter_active().collect();
        let quaternion_before = *ctx.groups()[gi].quaternion();
        let com_before = ctx.mass_center(&indices);

        analysis.sample(&ctx, 1).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 1);

        // The perturbation must not touch the molecule's carried orientation or COM.
        assert_approx_eq!(
            f64,
            (ctx.groups()[gi].quaternion().into_inner() - quaternion_before.into_inner()).norm(),
            0.0
        );
        assert_approx_eq!(f64, (ctx.mass_center(&indices) - com_before).norm(), 0.0);

        let yaml = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        for key in ["W/kJ/mol", "mean_interaction/kJ/mol", "N_eff", "mean_S2"] {
            assert!(yaml.get(key).is_some(), "missing key {key}");
        }
        assert!(yaml.get("torque/kT_per_rad").is_some());
        assert!(yaml.get("stiffness/kJ_per_mol_per_rad2").is_some());

        let s2 = yaml
            .get("S2")
            .and_then(serde_yml::Value::as_sequence)
            .unwrap();
        assert_eq!(s2.len(), 2);
        for entry in s2 {
            let value = entry
                .get("S2")
                .unwrap()
                .get("mean")
                .unwrap()
                .as_f64()
                .unwrap();
            assert!((-0.5..=1.0001).contains(&value), "S² out of range: {value}");
        }
        let neff = yaml
            .get("N_eff")
            .unwrap()
            .get("mean")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((1.0..=200.0).contains(&neff), "N_eff out of range: {neff}");
    }

    #[test]
    fn flat_landscape_gives_zero_free_energy_and_isotropy() {
        // No interactions → every orientation has equal (zero) energy.
        let ctx = dimer_backend("");
        let mut analysis = builder(false, false).build(&ctx, RT_300).unwrap();
        for step in 0..3 {
            analysis.sample(&ctx, step).unwrap();
        }
        let yaml = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        let w = yaml
            .get("W/kJ/mol")
            .unwrap()
            .get("mean")
            .unwrap()
            .as_f64()
            .unwrap();
        let mean_s2 = yaml
            .get("mean_S2")
            .unwrap()
            .get("mean")
            .unwrap()
            .as_f64()
            .unwrap();
        let neff = yaml
            .get("N_eff")
            .unwrap()
            .get("mean")
            .unwrap()
            .as_f64()
            .unwrap();
        assert_approx_eq!(f64, w, 0.0, epsilon = 1e-9);
        assert!(
            mean_s2.abs() < 1e-2,
            "expected isotropy, got S² = {mean_s2}"
        );
        assert_approx_eq!(f64, neff, 200.0, epsilon = 1e-6);
    }

    /// A charged dimer whose neighbour is an ion 8 Å up the z-axis, interacting through
    /// `nonbonded` — a stateful term, whose group energies are cached. Rotating the dimer swings
    /// its dipole relative to the ion, so the landscape is genuinely orientation-dependent, and a
    /// stale cache shows up as a flat one. A stateless external field cannot expose that.
    fn dimer_and_ion_backend() -> Backend {
        backend_from_str(
            r#"
atoms:
  - {name: P, mass: 1.0, charge: 1.0, sigma: 2.0}
  - {name: N, mass: 1.0, charge: -1.0, sigma: 2.0}
  - {name: I, mass: 1.0, charge: 1.0, sigma: 2.0}
molecules:
  - name: DIMER
    atoms: [P, N]
  - name: ION
    atoms: [I]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy:
    nonbonded:
      default:
        - !CoulombPlain {cutoff: 14.0}
  blocks:
    - molecule: DIMER
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    - molecule: ION
      N: 1
      insert: !Manual [[0.0, 0.0, 8.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
        )
    }

    /// The scan must see the landscape it exists to measure. An orientation-dependent potential
    /// that returns the same energy for all 200 trial orientations has not been evaluated against
    /// the rotated coordinates — ΔU ≡ 0, and every observable downstream is vacuous.
    #[test]
    fn scan_sees_a_nonflat_landscape_under_a_stateful_potential() {
        let ctx = dimer_and_ion_backend();
        let analysis = builder(false, false).build(&ctx, RT_300).unwrap();
        let gi = ctx.resolve_groups(&Selection::parse("molecule DIMER").unwrap())[0];

        let mut trial = ctx.clone();
        let energies = analysis.scan_energies(&mut trial, gi).unwrap();

        let min = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let max = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 1e-3,
            "orientational landscape is flat: every trial orientation gave U = {min} kT"
        );
    }

    /// Each scanned energy must be the energy of the orientation it claims to describe. The
    /// reference is a fresh whole-system evaluation, the one number no cache can serve.
    #[test]
    fn scanned_energies_match_fresh_total_energies() {
        let ctx = dimer_and_ion_backend();
        let analysis = builder(false, false).build(&ctx, RT_300).unwrap();
        let gi = ctx.resolve_groups(&Selection::parse("molecule DIMER").unwrap())[0];

        let mut trial = ctx.clone();
        let scanned = analysis.scan_energies(&mut trial, gi).unwrap();

        // The dimer's intramolecular Coulomb energy is invariant under a rotation about the mass
        // center, so total and group↔environment energies differ by a constant: compare shapes.
        let mut reference = ctx.clone();
        let expected: Vec<f64> = analysis
            .quaternions
            .iter()
            .map(|q| {
                reference.rotate_group(gi, q).unwrap();
                reference.update(&Change::Everything).unwrap();
                let total = reference
                    .hamiltonian()
                    .energy(&reference, &Change::Everything)
                    / RT_300;
                reference.rotate_group(gi, &q.inverse()).unwrap();
                reference.update(&Change::Everything).unwrap();
                total
            })
            .collect();

        let offset = scanned[0] - expected[0];
        for (k, (got, want)) in scanned.iter().zip(&expected).enumerate() {
            assert_approx_eq!(f64, *got - offset, *want, epsilon = 1e-9, ulps = 4);
            assert!(got.is_finite(), "orientation {k} gave a non-finite energy");
        }
    }

    /// Two hard-sphere dimers close enough that some trial orientation of either one overlaps the
    /// other, so the scan necessarily meets `u = +∞`. `weights()` documents that pose as expected
    /// and survivable — but only if the scan leaves no trace of it behind.
    fn clashing_dimers_backend() -> Backend {
        backend_from_str(
            r#"
atoms:
  - {name: P, mass: 1.0, charge: 0.0, sigma: 4.0}
molecules:
  - name: DIMER
    atoms: [P, P]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy:
    nonbonded:
      default:
        - !HardSphere {sigma: 4.0}
  blocks:
    - molecule: DIMER
      N: 2
      insert: !Manual [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 5.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
        )
    }

    /// A forbidden trial pose must not leave the system poisoned for the next molecule.
    ///
    /// The failure this guards against is the one [`PerturbContext::measure`] documents: a
    /// neighbour's cached energy is left NaN, so the second molecule of the loop reads its
    /// reference energy from it and reports `.nan` torque for the whole run.
    #[test]
    fn a_forbidden_pose_does_not_poison_the_next_molecule() {
        let ctx = clashing_dimers_backend();
        let mut analysis = WidomRotationBuilder {
            selection: Selection::parse("molecule DIMER").unwrap(),
            orientations: 64,
            vectors: vec![],
            torque: true,
            dtheta: 0.01,
            stiffness: false,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&ctx, RT_300)
        .unwrap();

        // The scan must actually meet a forbidden pose, or the test proves nothing.
        let gi = ctx.resolve_groups(&Selection::parse("molecule DIMER").unwrap())[0];
        let mut probe = ctx.clone();
        let scanned = analysis.scan_energies(&mut probe, gi).unwrap();
        assert!(
            scanned.iter().any(|u| u.is_infinite()),
            "no trial orientation overlapped the neighbour: the test system is too dilute"
        );

        analysis.sample(&ctx, 1).unwrap();

        let probe = analysis.torque.as_ref().unwrap();
        for (axis, accumulator) in ["x", "y", "z"].iter().zip(&probe.axes) {
            let torque = -accumulator.mean_free_energy() / probe.dtheta;
            assert!(
                !torque.is_nan(),
                "torque about {axis} is NaN: a forbidden pose poisoned a cached group energy"
            );
        }
    }

    /// A trial perturbation must leave the system it borrowed exactly as it found it — the energy
    /// caches included, not merely the coordinates.
    #[test]
    fn a_scan_restores_every_cached_energy_exactly() {
        let ctx = clashing_dimers_backend();
        let analysis = builder(false, false).build(&ctx, RT_300).unwrap();
        let gi = ctx.resolve_groups(&Selection::parse("molecule DIMER").unwrap())[0];

        let group_energy = |c: &Backend, g: usize| {
            c.hamiltonian()
                .energy(c, &Change::SingleGroup(g, GroupChange::RigidBody))
        };

        let mut trial = ctx.clone();
        let before: Vec<f64> = (0..2).map(|g| group_energy(&trial, g)).collect();

        let _ = analysis.scan_energies(&mut trial, gi).unwrap();

        for (g, want) in before.iter().enumerate() {
            let got = group_energy(&trial, g);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "group {g}: cached energy came back as {got}, not {want}"
            );
        }
    }

    /// Torque about at least one axis must be non-zero for a dipole in an ion's field.
    #[test]
    fn torque_is_nonzero_for_a_dipole_near_an_ion() {
        let ctx = dimer_and_ion_backend();
        let mut analysis = builder(true, false).build(&ctx, RT_300).unwrap();

        analysis.sample(&ctx, 1).unwrap();

        let probe = analysis.torque.as_ref().unwrap();
        let torques: Vec<f64> = probe
            .axes
            .iter()
            .map(|a| -a.mean_free_energy() / probe.dtheta)
            .collect();
        assert!(
            torques.iter().any(|t| t.abs() > 1e-3),
            "every axis reported zero torque: {torques:?}"
        );
    }
}
