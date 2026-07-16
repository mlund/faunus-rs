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
//! SO(3). A rotation about the center of mass leaves every intramolecular
//! distance unchanged, so the bonded and intramolecular non-bonded energy are
//! rotation-invariant and cancel.
//!
//! Each snapshot is referenced to its own deepest accessible orientation `u_min`,
//! so any snapshot-constant offset cancels. This matters because a stateful term
//! reports a *whole-system* total for a single-group change — Ewald reciprocal
//! space returns the entire k-space energy, not the tagged molecule's share — and
//! only orientation-dependent differences survive the reference. The reported `W`
//! and mean interaction are therefore excess quantities relative to that well, not
//! absolute one-body potentials of mean force.
//!
//! From the referenced landscape `u(Ω)` each snapshot yields a conditional cage
//! free energy, energy and entropy that satisfy `F_b = U_b − T·S_b` exactly, and
//! we report their averages over snapshots:
//! - `W = ⟨F_b⟩` with `F_b = -RT·ln[M⁻¹ Σₖ exp(-(uₖ-u_min)/RT)]` — the mean local
//!   cage free energy (Akke's per-vector `q̃`);
//! - the mean interaction energy `⟨U_b⟩` relative to the same reference;
//! - the orientational entropy relative to a free rotor,
//!   `S_orient/R = -Σₖ wₖ ln(M wₖ)`, taken straight from the Boltzmann weights
//!   `wₖ`. This is the exact SO(3) entropy, with no model for the shape of the
//!   well; the restriction-of-order/free-energy connection is due to
//!   [Akke et al. (1993)](https://doi.org/10.1021/ja00074a073), whose
//!   per-vector `S² → entropy` mapping we do *not* use;
//! - the ensemble Lipari–Szabo generalized order parameter
//!   `S² = 1.5‖⟨vvᵀ⟩‖² − 0.5` for chosen molecular vectors, formed from the grand
//!   tensor accumulated across snapshots (not the average of per-snapshot squares),
//!   so an axis that is locally locked but wanders isotropically reads `S² → 0`
//!   ([Lipari & Szabo (1982)](https://doi.org/10.1021/ja00381a009));
//! - optionally the RMS instantaneous torque and the local-harmonic stiffness of
//!   the cage. The mean torque of an equilibrium molecule is zero by Haar
//!   invariance, so the informative quantity is `√⟨τ²⟩`; the stiffness is a
//!   small-angle harmonic estimate, undefined for a near-free rotor.
//!
//! `W`, `⟨U_b⟩` and the stiffness are in kJ/mol; the entropy is dimensionless
//! (`≤ 0`, zero for a free rotor), the order parameter is dimensionless, and the
//! torque is in kT per radian.
//!
//! With implicit solvent these energies are free energies relative to pure
//! solvent, not mechanical energies.
//!
//! The analysis emits one set of numbers per run.

use super::{Analyze, Frequency, Sampling};
use crate::auxiliary::{BlockSummary, ColumnWriter, MappingExt, WeightedBlockAverage};
use crate::cell::BoundaryConditions;
use crate::context::{PerturbContext, Perturbation};
use crate::energy::EnergyChange;
use crate::geometry::GyrationTensor;
use crate::selection::{CachedSelection, Groups, Selection};
use crate::ObserveContext;
use crate::{Point, UnitQuaternion};
use anyhow::Result;
use derive_more::Debug;
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Quaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// First super-Fibonacci constant, `√2`.
const PHI: f64 = std::f64::consts::SQRT_2;
/// Second super-Fibonacci constant (root of `ψ³ = ψ + 4`).
const PSI: f64 = 1.533_751_168_755_204_3;

/// Below this effective sample size the weighted orientation cloud is too sparse
/// to define a 3×3 covariance, so per-snapshot stiffness is skipped.
const MIN_NEFF_FOR_STIFFNESS: f64 = 3.0;

/// Above this librational variance (rad²) the cloud is too broad for a harmonic
/// well to describe, so the local stiffness estimate is skipped for that axis.
/// A uniform SO(3) cloud has component variance `π²/9 + 2/3 ≈ 1.76`; the cap of
/// `0.25 rad²` (RMS libration ≈ 29°) keeps only genuinely confined, near-harmonic
/// wells and refuses to report a finite spring constant for a near-free rotor.
const MAX_VARIANCE_FOR_STIFFNESS: f64 = 0.25;

/// Guard against dividing by a vanishing orientational variance.
const VARIANCE_EPSILON: f64 = 1e-12;

/// The six independent components `[xx, yy, zz, xy, xz, yz]` of a symmetric 3×3.
fn tensor_components(tensor: &Matrix3<f64>) -> [f64; 6] {
    [
        tensor[(0, 0)],
        tensor[(1, 1)],
        tensor[(2, 2)],
        tensor[(0, 1)],
        tensor[(0, 2)],
        tensor[(1, 2)],
    ]
}

/// Serialize an optional diagnostic to YAML, or null when it has no value — the
/// shape a per-axis quantity that may lack samples in a given run reports.
fn value_or_null<T: Serialize>(value: Option<T>) -> serde_yml::Value {
    value
        .and_then(|v| serde_yml::to_value(v).ok())
        .unwrap_or(serde_yml::Value::Null)
}

/// `S² = 1.5 Σ_αβ T_αβ² − 0.5` and its gradient `∂S²/∂T_αβ` for one vector's six
/// tensor components `[xx, yy, zz, xy, xz, yz]` (off-diagonals appear twice).
fn order_and_gradient(m: &[f64]) -> (f64, [f64; 6]) {
    let s2 = 1.5 * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
        + 3.0 * (m[3] * m[3] + m[4] * m[4] + m[5] * m[5])
        - 0.5;
    let gradient = [
        3.0 * m[0],
        3.0 * m[1],
        3.0 * m[2],
        6.0 * m[3],
        6.0 * m[4],
        6.0 * m[5],
    ];
    (s2, gradient)
}

/// Joint estimator of the ensemble Lipari–Szabo order parameter for every requested
/// vector.
///
/// Each frame contributes the concatenated six components of every vector's
/// conditional tensor `Σₖ wₖ vₖvₖᵀ`. A weighted mean and co-moment (West's algorithm)
/// give the grand tensor `⟨vvᵀ⟩` and the covariance of *that mean*, so `S²` and its
/// error follow from the delta method `gᵀ·Cov·g`. Tracking the full joint covariance
/// — rather than six independent block averages — is what makes the error respect the
/// exact per-frame trace constraint `T_xx + T_yy + T_zz = 1` (which perfectly
/// anti-correlates the diagonal components) and the correlation between vectors that
/// move together on a rigid body.
#[derive(Clone, Debug)]
struct OrderAccumulator {
    num_vectors: usize,
    sum_w: f64,
    sum_w2: f64,
    mean: DVector<f64>,
    /// `Σ w (x − mean_old)(x − mean_new)ᵀ` over frames (weighted co-moment).
    comoment: DMatrix<f64>,
}

impl OrderAccumulator {
    fn new(num_vectors: usize) -> Self {
        let dim = 6 * num_vectors;
        Self {
            num_vectors,
            sum_w: 0.0,
            sum_w2: 0.0,
            mean: DVector::zeros(dim),
            comoment: DMatrix::zeros(dim, dim),
        }
    }

    /// Whether any frame has contributed. `sum_w2` grows only on a non-zero weight,
    /// so it doubles as the sample count without a separate counter.
    fn is_empty(&self) -> bool {
        self.sum_w2 == 0.0
    }

    /// Fold one frame's concatenated tensor components (length `6·num_vectors`).
    fn add(&mut self, components: &[f64], weight: f64) {
        if self.num_vectors == 0 || weight == 0.0 {
            return;
        }
        let x = DVector::from_row_slice(components);
        self.sum_w += weight;
        self.sum_w2 += weight * weight;
        let delta = &x - &self.mean;
        self.mean.axpy(weight / self.sum_w, &delta, 1.0);
        let delta2 = &x - &self.mean;
        // Weighted rank-1 update in place: comoment += weight · delta · delta2ᵀ.
        self.comoment.ger(weight, &delta, &delta2, 1.0);
    }

    /// Independent-frame covariance of the grand-tensor mean:
    /// `comoment / (Σw · (N_eff − 1))`. `None` for fewer than two effective frames.
    fn covariance_of_mean(&self) -> Option<DMatrix<f64>> {
        if self.is_empty() {
            return None;
        }
        let effective_n = self.sum_w * self.sum_w / self.sum_w2;
        if effective_n <= 1.0 || self.sum_w <= 0.0 {
            return None;
        }
        Some(&self.comoment / (self.sum_w * (effective_n - 1.0)))
    }

    /// Ensemble `S²` point estimate for vector `v`, from its grand-tensor mean.
    fn vector_order(&self, v: usize) -> Option<f64> {
        (!self.is_empty() && v < self.num_vectors)
            .then(|| order_and_gradient(self.mean.rows(6 * v, 6).as_slice()).0)
    }

    /// Ensemble `S²` and its delta-method error for vector `v`.
    fn vector_summary(&self, v: usize) -> Option<BlockSummary> {
        let mean = self.vector_order(v)?;
        let (_, gradient) = order_and_gradient(self.mean.rows(6 * v, 6).as_slice());
        let error = self.covariance_of_mean().map_or(0.0, |cov| {
            let g = DVector::from_row_slice(&gradient);
            let sub = cov.view((6 * v, 6 * v), (6, 6));
            (g.transpose() * sub * &g)[(0, 0)].max(0.0).sqrt()
        });
        Some(BlockSummary { mean, error })
    }

    /// Mean ensemble `S²` point estimate over all vectors (no error), cheap enough
    /// to stream every frame.
    fn mean_order(&self) -> Option<f64> {
        if self.is_empty() || self.num_vectors == 0 {
            return None;
        }
        let sum: f64 = (0..self.num_vectors)
            .filter_map(|v| self.vector_order(v))
            .sum();
        Some(sum / self.num_vectors as f64)
    }

    /// Mean ensemble `S²` over all vectors, with the error propagated through the
    /// full joint covariance so cross-vector correlation is kept.
    fn mean_summary(&self) -> Option<BlockSummary> {
        let mean = self.mean_order()?;
        let n = self.num_vectors as f64;
        let mut full_gradient = DVector::zeros(6 * self.num_vectors);
        for v in 0..self.num_vectors {
            let (_, gradient) = order_and_gradient(self.mean.rows(6 * v, 6).as_slice());
            for (i, g) in gradient.iter().enumerate() {
                full_gradient[6 * v + i] = g / n;
            }
        }
        let error = self.covariance_of_mean().map_or(0.0, |cov| {
            (full_gradient.transpose() * cov * &full_gradient)[(0, 0)]
                .max(0.0)
                .sqrt()
        });
        Some(BlockSummary { mean, error })
    }
}

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
            cage_free_energy: WeightedBlockAverage::new(),
            order: OrderAccumulator::new(self.vectors.len()),
            mean_interaction: WeightedBlockAverage::new(),
            entropy: WeightedBlockAverage::new(),
            neff: WeightedBlockAverage::new(),
            torque: self.torque.then(|| TorqueProbe {
                axes: std::array::from_fn(|_| WeightedBlockAverage::new()),
                dtheta: self.dtheta,
            }),
            stiffness: self
                .stiffness
                .then(|| std::array::from_fn(|_| WeightedBlockAverage::new())),
            sampling: Sampling::new(self.frequency),
            num_blocks: 0,
            num_inaccessible: 0,
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

/// Instantaneous-torque probe: one accumulator of `τ²` per lab axis.
///
/// The mean torque of a molecule at equilibrium is zero by Haar invariance (a
/// rigid rotation is measure-preserving, so `⟨τ⟩ = 0` however anisotropic the
/// cage), so the magnitude `√⟨τ²⟩` is the informative quantity.
#[derive(Clone, Debug, Default)]
struct TorqueProbe {
    /// Block averages of `τ²` about x, y, z (kT²/rad²).
    axes: [WeightedBlockAverage; 3],
    dtheta: f64,
}

/// Widom rotational perturbation analysis. See the module documentation.
#[derive(Debug)]
pub struct WidomRotation {
    selection: CachedSelection<Groups>,
    quaternions: Vec<UnitQuaternion>,
    vectors: Vec<VectorSpec>,
    thermal_energy: f64,
    /// Per-snapshot cage free energy `F_b`, averaged over snapshots → `W = ⟨F_b⟩`.
    cage_free_energy: WeightedBlockAverage,
    /// Joint grand-tensor estimator for the ensemble Lipari–Szabo `S²`.
    order: OrderAccumulator,
    /// Mean interaction relative to the per-snapshot deepest well (kJ/mol).
    mean_interaction: WeightedBlockAverage,
    /// Orientational configurational entropy in units of the gas constant.
    entropy: WeightedBlockAverage,
    neff: WeightedBlockAverage,
    torque: Option<TorqueProbe>,
    /// Principal local-harmonic stiffnesses (kJ/mol/rad²), ascending.
    stiffness: Option<[WeightedBlockAverage; 3]>,
    /// Frequency and frame count, owned by the framework.
    sampling: Sampling,
    /// Accessible molecule scans, one per matching molecule per frame.
    num_blocks: usize,
    /// Molecule scans with no accessible orientation (every trial pose clashes);
    /// excluded from every average so `F = U − TS` stays consistent.
    num_inaccessible: usize,
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
    /// orientational distribution is then undefined and the cage free energy
    /// diverges, so the snapshot is counted as inaccessible and skipped.
    fn weights(energies: &[f64]) -> Option<Vec<f64>> {
        let min = energies.iter().copied().fold(f64::INFINITY, f64::min);
        if !min.is_finite() {
            return None;
        }
        let unnormalized: Vec<f64> = energies.iter().map(|u| (-(u - min)).exp()).collect();
        let sum: f64 = unnormalized.iter().sum();
        (sum > 0.0 && sum.is_finite()).then(|| unnormalized.iter().map(|w| w / sum).collect())
    }

    /// Accumulate all snapshot observables for one molecule, weighted by the
    /// frame's reweighting factor `weight` (`1.0` for an unbiased run).
    ///
    /// Every energy is referenced to the deepest accessible orientation `u_min`,
    /// so a snapshot-constant offset — the whole-system total that a stateful
    /// term's `partial_energy` returns (e.g. Ewald reciprocal space) — cancels
    /// and only orientation-dependent differences enter. `W` and the mean
    /// interaction are therefore excess quantities relative to that reference.
    ///
    /// Returns `true` if the snapshot was accessible and fed the averages, so the
    /// caller can keep the torque probe on exactly the same set of frames.
    fn accumulate(&mut self, energies: &[f64], references: &[Point], weight: f64) -> bool {
        let Some(weights) = Self::weights(energies) else {
            // No accessible orientation: the cage free energy diverges and the
            // conditional distribution is undefined. Record it and leave every
            // average untouched, so the reported F = U − TS stays consistent.
            self.num_inaccessible += 1;
            return false;
        };
        let u_min = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let count = energies.len() as f64;

        // Conditional cage free energy relative to the deepest well (Akke's q̃):
        // F_b = −ln[(1/M) Σₖ exp(−(uₖ−u_min))] in kT. Averaged over snapshots this
        // gives W = ⟨F_b⟩, whose error is the SEM of the same per-block estimator.
        // A forbidden pose has u = +∞, so exp(−(u−u_min)) = 0 and drops out cleanly.
        let partition = energies.iter().map(|&u| (-(u - u_min)).exp()).sum::<f64>() / count;
        self.cage_free_energy.add(-partition.ln(), weight);

        // Mean interaction relative to the same reference (kT → kJ/mol). With the
        // free-rotor entropy below, U_b − T·S_b = F_b holds exactly per snapshot.
        // A forbidden pose carries u = +∞ and w = 0; skip it, or (+∞)·0 = NaN would
        // poison the average.
        let mean_u: f64 = energies
            .iter()
            .zip(&weights)
            .filter(|(u, _)| u.is_finite())
            .map(|(u, w)| (u - u_min) * w)
            .sum();
        self.mean_interaction
            .add(mean_u * self.thermal_energy, weight);

        // Orientational configurational entropy relative to a free rotor.
        let entropy: f64 = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * (count * w).ln())
            .sum();
        self.entropy.add(entropy, weight);

        let neff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        self.neff.add(neff, weight);

        self.accumulate_order(&weights, references, weight);
        if let Some(stiffness) = self.stiffness.as_mut() {
            Self::accumulate_stiffness(
                stiffness,
                &self.quaternions,
                &weights,
                neff,
                self.thermal_energy,
                weight,
            );
        }
        self.num_blocks += 1;
        true
    }

    /// Fold this snapshot's conditional orientation tensors `Σₖ wₖ vₖvₖᵀ` into the
    /// joint grand-tensor estimator.
    ///
    /// The ensemble Lipari–Szabo `S²` is `1.5‖⟨vvᵀ⟩‖² − 0.5` — the square of the
    /// *ensemble-averaged* tensor. Accumulating the tensor here and squaring once
    /// at report time (not the per-snapshot square) is what lets an axis that is
    /// locally locked but wanders isotropically between snapshots read `S² → 0`.
    fn accumulate_order(&mut self, weights: &[f64], references: &[Point], weight: f64) {
        if references.is_empty() {
            return;
        }
        let mut components = Vec::with_capacity(6 * references.len());
        for reference in references {
            let mut tensor = Matrix3::zeros();
            for (q, &w) in self.quaternions.iter().zip(weights) {
                let v = q.transform_vector(reference);
                tensor += w * (v * v.transpose());
            }
            components.extend_from_slice(&tensor_components(&tensor));
        }
        self.order.add(&components, weight);
    }

    /// Local small-angle harmonic stiffness `K = RT·Cov⁻¹` from the weighted
    /// orientation cloud; its principal values are `RT / var` along each principal
    /// libration axis.
    ///
    /// This is a *local* estimate: it is meaningful only for a unimodal cloud
    /// confined to small angles, where the well is approximately harmonic. A broad
    /// or near-free cloud has no single well and no Markley mean, so an axis whose
    /// libration variance exceeds [`MAX_VARIANCE_FOR_STIFFNESS`] is skipped rather
    /// than reported — a free rotor must not read as a finite spring constant.
    ///
    /// Because that guard drops the loose snapshots, the reported value is
    /// conditioned on the confined frames and so biased toward the stiff
    /// sub-ensemble of a cage that breathes between tight and loose. The per-axis
    /// sample count (`n`) shows how many snapshots qualified.
    fn accumulate_stiffness(
        stiffness: &mut [WeightedBlockAverage; 3],
        quaternions: &[UnitQuaternion],
        weights: &[f64],
        neff: f64,
        thermal_energy: f64,
        weight: f64,
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
            // Skip an axis whose harmonic estimate is undefined at either extreme:
            // a vanishing variance means a degenerate (collapsed) cloud and a
            // diverging stiffness, while a variance above the harmonic cap means a
            // broad, near-free cloud that no spring constant describes. Skipping the
            // axis alone keeps the well-defined axes and never poisons the average
            // with a non-finite value.
            if variance.is_finite()
                && variance > VARIANCE_EPSILON
                && variance < MAX_VARIANCE_FOR_STIFFNESS
            {
                accumulator.add(thermal_energy / variance, weight);
            }
        }
    }

    /// Instantaneous torque² about each lab axis, by a central finite difference of
    /// the energy at the molecule's current orientation.
    ///
    /// `τ = −(u(+δ) − u(−δ)) / 2δ` is Ewald-safe: the snapshot-constant offset
    /// cancels between the two poses without any reference subtraction. We
    /// accumulate `τ²`, whose root-mean-square is the reported magnitude — the mean
    /// torque itself vanishes at equilibrium (Haar). An axis is skipped if either
    /// probe pose clashes (`u = +∞`), leaving a well-defined pose's neighbours
    /// unrecorded rather than poisoning the average with `+∞`.
    fn accumulate_torque<T: PerturbContext>(
        &mut self,
        trial: &mut T,
        gi: usize,
        weight: f64,
    ) -> anyhow::Result<()> {
        let Some(probe) = self.torque.as_mut() else {
            return Ok(());
        };
        let axes = [Vector3::x_axis(), Vector3::y_axis(), Vector3::z_axis()];
        for (accumulator, axis) in probe.axes.iter_mut().zip(axes) {
            let forward = UnitQuaternion::from_axis_angle(&axis, probe.dtheta);
            let backward = UnitQuaternion::from_axis_angle(&axis, -probe.dtheta);
            let u_forward = energy_at_orientation(trial, gi, &forward, self.thermal_energy)?;
            let u_backward = energy_at_orientation(trial, gi, &backward, self.thermal_energy)?;
            if u_forward.is_finite() && u_backward.is_finite() {
                let torque = -(u_forward - u_backward) / (2.0 * probe.dtheta);
                accumulator.add(torque * torque, weight);
            }
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

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
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

            let energies = self.scan_energies(&mut trial, gi)?;
            // Keep the torque on exactly the frames that fed W/U/S: an inaccessible
            // scan contributes to neither.
            if self.accumulate(&energies, &references, weight) {
                self.accumulate_torque(&mut trial, gi, weight)?;
            }
        }

        // Skip the row until at least one accessible scan exists, so an all-clash
        // opening frame does not write a NaN line. Read every streamed value before
        // borrowing the stream mutably.
        if self.cage_free_energy.n() > 0 {
            let w = self.cage_free_energy.mean() * self.thermal_energy;
            let mean_s2 = self.order.mean_order().unwrap_or(f64::NAN);
            let neff = self.neff.mean();
            if let Some(stream) = self.stream.as_mut() {
                stream.write_row(&[
                    &step,
                    &format_args!("{w:.6e}"),
                    &format_args!("{mean_s2:.6}"),
                    &format_args!("{neff:.3}"),
                ])?;
            }
        }
        Ok(())
    }

    fn results(&self) -> Option<serde_yml::Value> {
        // A frame in which no molecule matched at all leaves nothing to report.
        if self.num_blocks == 0 && self.num_inaccessible == 0 {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("num_blocks", self.num_blocks)?;
        // Scans with no accessible orientation carry a diverging cage free energy;
        // they are excluded from every average, so surface their count separately.
        if self.num_inaccessible > 0 {
            map.try_insert("num_inaccessible", self.num_inaccessible)?;
        }
        // Every molecule clashed in every pose: nothing finite to average.
        if self.num_blocks == 0 {
            return Some(serde_yml::Value::Mapping(map));
        }

        // W = ⟨F_b⟩ over per-snapshot cage free energies (excess, relative to each
        // snapshot's deepest well); mean and error are the one block estimator.
        map.try_insert("W/kJ/mol", &self.cage_free_energy * self.thermal_energy)?;
        map.try_insert(
            "mean_excess_interaction/kJ/mol",
            self.mean_interaction.summary(),
        )?;
        map.try_insert("orientational_entropy/R", self.entropy.summary())?;
        map.try_insert("N_eff", self.neff.summary())?;

        if !self.vectors.is_empty() {
            let per_vector: Vec<serde_yml::Value> = self
                .vectors
                .iter()
                .enumerate()
                .filter_map(|(v, spec)| {
                    let mut entry = serde_yml::Mapping::new();
                    entry.try_insert("vector", spec.label())?;
                    entry.try_insert("S2", self.order.vector_summary(v)?)?;
                    Some(serde_yml::Value::Mapping(entry))
                })
                .collect();
            map.insert("S2".into(), serde_yml::Value::Sequence(per_vector));
            if let Some(mean_s2) = self.order.mean_summary() {
                map.try_insert("mean_S2", mean_s2)?;
            }
        }

        if let Some(probe) = &self.torque {
            let mut torque = serde_yml::Mapping::new();
            for (axis, accumulator) in ["x", "y", "z"].iter().zip(&probe.axes) {
                // Root-mean-square torque magnitude: the mean torque vanishes at
                // equilibrium, so √⟨τ²⟩ is the informative quantity. An axis whose
                // ±δ probe clashed in every frame has no samples and reports null.
                let value = value_or_null(accumulator.checked_mean().map(f64::sqrt));
                torque.insert(axis.to_string().into(), value);
            }
            map.insert(
                "rms_torque/kT_per_rad".into(),
                serde_yml::Value::Mapping(torque),
            );
        }

        if let Some(stiffness) = &self.stiffness {
            // An axis with no samples was degenerate or too soft in every snapshot;
            // report it as null rather than letting an empty average read as zero
            // stiffness, which would claim free rotation about a locked axis.
            let values: Vec<serde_yml::Value> = stiffness
                .iter()
                .map(|accumulator| value_or_null(accumulator.summary()))
                .collect();
            map.insert(
                "local_harmonic_stiffness/kJ_per_mol_per_rad2".into(),
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
        let mut stiffness = std::array::from_fn(|_| WeightedBlockAverage::new());
        let identical = vec![UnitQuaternion::identity(); 4];
        let weights = vec![0.25; 4];

        WidomRotation::accumulate_stiffness(&mut stiffness, &identical, &weights, 4.0, 2.5, 1.0);

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
        let mut stiffness = std::array::from_fn(|_| WeightedBlockAverage::new());
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

        WidomRotation::accumulate_stiffness(
            &mut stiffness,
            &planar,
            &weights,
            5.0,
            thermal_energy,
            1.0,
        );

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
        for key in [
            "W/kJ/mol",
            "mean_excess_interaction/kJ/mol",
            "N_eff",
            "mean_S2",
        ] {
            assert!(yaml.get(key).is_some(), "missing key {key}");
        }
        assert!(yaml.get("rms_torque/kT_per_rad").is_some());
        assert!(yaml
            .get("local_harmonic_stiffness/kJ_per_mol_per_rad2")
            .is_some());

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
            let rms_torque = accumulator.mean().sqrt();
            assert!(
                !rms_torque.is_nan(),
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
        let torques: Vec<f64> = probe.axes.iter().map(|a| a.mean().sqrt()).collect();
        assert!(
            torques.iter().any(|t| t.abs() > 1e-3),
            "every axis reported zero torque: {torques:?}"
        );
    }

    /// The reported free energy, energy and entropy must satisfy `W = U − TS`
    /// exactly, per snapshot and hence in the mean — the decomposition the module
    /// documents. A pooled `W` would break it by Jensen's inequality.
    #[test]
    fn free_energy_splits_into_energy_and_entropy() {
        let ctx = dimer_backend("");
        let mut analysis = builder(false, false).build(&ctx, RT_300).unwrap();

        // Two distinct orientational landscapes, so the mean is a genuine average.
        analysis.accumulate(&[0.0, 1.0, 2.5, 0.3, 4.0], &[], 1.0);
        analysis.accumulate(&[0.0, 0.2, 0.2, 5.0, 0.1], &[], 1.0);

        let w = analysis.cage_free_energy.mean() * RT_300;
        let u = analysis.mean_interaction.mean();
        let ts = analysis.entropy.mean() * RT_300;
        assert_approx_eq!(f64, w, u - ts, epsilon = 1e-9);
    }

    /// Forbidden trial poses (`u = +∞`, `w = 0`) must not poison the mean
    /// interaction with a `+∞·0 = NaN`, and `F = U − TS` must still hold when some
    /// orientations clash.
    #[test]
    fn a_partial_clash_keeps_observables_finite() {
        let ctx = dimer_backend("");
        let mut analysis = builder(false, false).build(&ctx, RT_300).unwrap();

        let energies = [0.0, 1.0, f64::INFINITY, 2.0, f64::INFINITY];
        assert!(analysis.accumulate(&energies, &[], 1.0));

        let w = analysis.cage_free_energy.mean() * RT_300;
        let u = analysis.mean_interaction.mean();
        let ts = analysis.entropy.mean() * RT_300;
        assert!(u.is_finite(), "mean interaction poisoned to NaN by a clash");
        assert!(w.is_finite());
        assert_approx_eq!(f64, w, u - ts, epsilon = 1e-9);
    }

    /// The ensemble `S²` error must respect the covariance of the tensor
    /// components: with no frame-to-frame fluctuation the grand-tensor mean has no
    /// sampling spread, so the error is exactly zero — not the spurious value an
    /// independent-component quadrature would report.
    #[test]
    fn ensemble_order_error_vanishes_without_fluctuation() {
        let ctx = dimer_backend("");
        let mut analysis = WidomRotationBuilder {
            selection: Selection::parse("molecule DIMER").unwrap(),
            orientations: 128,
            vectors: vec![
                VectorSpec::Body([0.0, 0.0, 1.0]),
                VectorSpec::Body([1.0, 0.0, 0.0]),
            ],
            torque: false,
            dtheta: 0.01,
            stiffness: false,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&ctx, RT_300)
        .unwrap();

        // Identical conditional tensors every frame → zero sampling variance.
        let mut weights = vec![0.0; analysis.quaternions.len()];
        weights[3] = 1.0;
        let references = [Point::new(0.0, 0.0, 1.0), Point::new(1.0, 0.0, 0.0)];
        for _ in 0..16 {
            analysis.accumulate_order(&weights, &references, 1.0);
        }

        for v in 0..2 {
            assert_approx_eq!(f64, analysis.order.vector_summary(v).unwrap().error, 0.0);
        }
        assert_approx_eq!(f64, analysis.order.mean_summary().unwrap().error, 0.0);
    }

    /// A snapshot-constant offset added to every trial energy — the signature of a
    /// whole-system stateful term like Ewald reciprocal space, whose `partial_energy`
    /// returns the total — must leave every observable unchanged. Referencing to the
    /// deepest well cancels it.
    #[test]
    fn a_constant_energy_offset_cancels() {
        let ctx = dimer_backend("");
        let energies = [0.0, 1.0, 2.5, 0.3, 4.0];
        let shifted: Vec<f64> = energies.iter().map(|u| u + 137.0).collect();

        let mut plain = builder(false, false).build(&ctx, RT_300).unwrap();
        let mut offset = builder(false, false).build(&ctx, RT_300).unwrap();
        plain.accumulate(&energies, &[], 1.0);
        offset.accumulate(&shifted, &[], 1.0);

        assert_approx_eq!(
            f64,
            plain.cage_free_energy.mean(),
            offset.cage_free_energy.mean(),
            epsilon = 1e-9
        );
        assert_approx_eq!(
            f64,
            plain.mean_interaction.mean(),
            offset.mean_interaction.mean(),
            epsilon = 1e-9
        );
        assert_approx_eq!(
            f64,
            plain.entropy.mean(),
            offset.entropy.mean(),
            epsilon = 1e-9
        );
        assert_approx_eq!(f64, plain.neff.mean(), offset.neff.mean(), epsilon = 1e-9);
    }

    /// The rerun weight must reach the accumulators: two snapshots with distinct
    /// cage free energies must give a weight-dependent mean.
    #[test]
    fn rerun_weight_reweights_the_free_energy() {
        let ctx = dimer_backend("");
        let flat = [1.0, 1.0, 1.0, 1.0]; // F = 0
        let welled = [0.0, 10.0, 10.0, 10.0]; // one deep well, F > 0

        let mut unbiased = builder(false, false).build(&ctx, RT_300).unwrap();
        unbiased.accumulate(&flat, &[], 1.0);
        unbiased.accumulate(&welled, &[], 1.0);

        let mut reweighted = builder(false, false).build(&ctx, RT_300).unwrap();
        reweighted.accumulate(&flat, &[], 3.0);
        reweighted.accumulate(&welled, &[], 1.0);

        let f_flat = 0.0;
        let f_welled = -((1.0 + 3.0 * (-10.0f64).exp()) / 4.0).ln();
        assert_approx_eq!(
            f64,
            unbiased.cage_free_energy.mean(),
            0.5 * (f_flat + f_welled),
            epsilon = 1e-9
        );
        assert_approx_eq!(
            f64,
            reweighted.cage_free_energy.mean(),
            (3.0 * f_flat + f_welled) / 4.0,
            epsilon = 1e-9
        );
    }

    /// An axis that is locally locked in every snapshot but points isotropically
    /// across snapshots has ensemble `S² → 0` (Lipari–Szabo eq 20), even though its
    /// per-snapshot conditional order is 1. The grand-tensor estimator must report
    /// the ensemble value, not the average of the per-snapshot squares.
    #[test]
    fn wandering_axis_has_zero_ensemble_order() {
        let ctx = dimer_backend("");
        let mut analysis = WidomRotationBuilder {
            selection: Selection::parse("molecule DIMER").unwrap(),
            orientations: 512,
            vectors: vec![VectorSpec::Body([0.0, 0.0, 1.0])],
            torque: false,
            dtheta: 0.01,
            stiffness: false,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&ctx, RT_300)
        .unwrap();

        // Each snapshot locks onto a different trial orientation (one-hot weights),
        // and those orientations tile SO(3) isotropically, so the fixed body vector
        // is carried to isotropically-scattered lab directions.
        let reference = Point::new(0.0, 0.0, 1.0);
        for locked in 0..analysis.quaternions.len() {
            let mut weights = vec![0.0; analysis.quaternions.len()];
            weights[locked] = 1.0;
            analysis.accumulate_order(&weights, std::slice::from_ref(&reference), 1.0);
        }

        let s2 = analysis.order.vector_summary(0).unwrap();
        assert!(
            s2.mean.abs() < 1e-2,
            "wandering axis should give ensemble S² ≈ 0, got {}",
            s2.mean
        );
    }

    /// A free rotor (flat landscape) must not report a finite librational stiffness:
    /// the cloud fills SO(3), no harmonic well exists, and the naive `RT·Cov⁻¹`
    /// would report a spurious `≈0.57 RT/rad²`. Every axis must be skipped.
    #[test]
    fn free_rotor_reports_no_stiffness() {
        let ctx = dimer_backend("");
        let mut analysis = builder(false, true).build(&ctx, RT_300).unwrap();

        // A flat landscape gives uniform weights over the full SO(3) trial set.
        let flat = vec![0.0; analysis.quaternions.len()];
        analysis.accumulate(&flat, &[], 1.0);

        for accumulator in analysis.stiffness.as_ref().unwrap() {
            assert_eq!(
                accumulator.n(),
                0,
                "a free rotor must report no local-harmonic stiffness"
            );
        }
    }

    /// A charged dimer in a uniform field `q·z` is a point dipole, `u(Ω) = a·n_z`
    /// in kT with `a = μE/kT`. Its orientational entropy, effective sample size and
    /// bond-axis order parameter then have closed Langevin forms, so this checks the
    /// quadrature, Boltzmann weights and ensemble tensor against dipole theory — not
    /// just internal consistency.
    #[test]
    fn dipole_in_uniform_field_matches_langevin_theory() {
        let orientations = 4000;
        let ctx = dimer_backend(r#"custom_external: [{selection: "all", function: "q * z"}]"#);
        let mut analysis = WidomRotationBuilder {
            selection: Selection::parse("molecule DIMER").unwrap(),
            orientations,
            vectors: vec![VectorSpec::Pair([0, 1])],
            torque: false,
            dtheta: 0.01,
            stiffness: false,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&ctx, RT_300)
        .unwrap();

        // Coupling a = μE/kT read from the scan: u/kT spans [−a, a] since u ∝ n_z.
        let gi = ctx.resolve_groups(analysis.selection.selection())[0];
        let mut trial = ctx.clone();
        let energies = analysis.scan_energies(&mut trial, gi).unwrap();
        let u_max = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let u_min = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let a = 0.5 * (u_max - u_min);
        assert!(a > 0.5, "field too weak to be a meaningful test: a = {a}");

        analysis.sample(&ctx, 1).unwrap();

        let langevin = |a: f64| 1.0 / a.tanh() - 1.0 / a; // L(a) = coth a − 1/a

        // Orientational entropy S/R = −a·L(a) + ln(sinh a / a).
        let entropy_theory = -a * langevin(a) + (a.sinh() / a).ln();
        assert_approx_eq!(f64, analysis.entropy.mean(), entropy_theory, epsilon = 2e-3);

        // Effective accessible orientations N_eff / M = tanh(a) / a.
        let neff_theory = orientations as f64 * a.tanh() / a;
        assert_approx_eq!(f64, analysis.neff.mean(), neff_theory, epsilon = 4.0);

        // Bond-axis order parameter S² = [1 − 3·L(a)/a]² (Langevin second moment).
        let s2_theory = (1.0 - 3.0 * langevin(a) / a).powi(2);
        let s2 = analysis.order.vector_summary(0).unwrap().mean;
        assert_approx_eq!(f64, s2, s2_theory, epsilon = 2e-3);
    }

    /// The ensemble `S²` of a vector distributed uniformly over a cone of half-angle
    /// β about a lab axis is `[½cosβ(1+cosβ)]²`. Feeding exactly that cap
    /// distribution through the grand-tensor path checks the order-parameter
    /// magnitude at a substantial value, not only the isotropic and locked limits.
    #[test]
    fn cone_distribution_matches_analytic_order_parameter() {
        let ctx = dimer_backend("");
        let mut analysis = WidomRotationBuilder {
            selection: Selection::parse("molecule DIMER").unwrap(),
            orientations: 8000,
            vectors: vec![VectorSpec::Body([0.0, 0.0, 1.0])],
            torque: false,
            dtheta: 0.01,
            stiffness: false,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&ctx, RT_300)
        .unwrap();

        // Uniform over the cap n_z ≥ cos β about +z: the trial set tiles SO(3), so
        // its image of ẑ tiles the sphere and selecting the cap gives a uniform cone.
        let reference = Point::new(0.0, 0.0, 1.0);
        let cos_beta = 0.5_f64; // β = 60°
        let in_cap: Vec<bool> = analysis
            .quaternions
            .iter()
            .map(|q| q.transform_vector(&reference).z >= cos_beta)
            .collect();
        let count = in_cap.iter().filter(|&&c| c).count() as f64;
        let weights: Vec<f64> = in_cap
            .iter()
            .map(|&c| if c { 1.0 / count } else { 0.0 })
            .collect();
        analysis.accumulate_order(&weights, std::slice::from_ref(&reference), 1.0);

        let s2_theory = (0.5 * cos_beta * (1.0 + cos_beta)).powi(2);
        let s2 = analysis.order.vector_summary(0).unwrap().mean;
        assert_approx_eq!(f64, s2, s2_theory, epsilon = 1e-2);
    }
}
