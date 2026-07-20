// Copyright 2025 Mikael Lund
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

//! Multipolar decomposition of the interaction between two molecular species,
//! resolved by center-of-mass separation.
//!
//! For every distinct pair of molecules (one from each selection) the electrostatic
//! interaction is decomposed into ion–ion, ion–dipole, dipole–dipole and ion–quadrupole
//! contributions and compared with the exact, explicitly summed Coulomb energy, all binned
//! by COM separation R. Alongside the energies, orientational correlations are accumulated:
//! ⟨μ̂ₐ·μ̂_b⟩, ⟨P₂(cosθ)⟩, the longitudinal/transverse projection ⟨(μ̂ₐ·R̂)(μ̂_b·R̂)⟩, the
//! Kirkwood factor g_K(R), and a quadrupole orientational correlation.
//!
//! Ported from the C++ Faunus `MultipoleDistribution` (Malmö 2014). Energies are reported in
//! **kJ/mol** (C++ reports kT/λ_B); output is a CSV table.

use super::{Analyze, Frequency, Sampling};
use crate::auxiliary::{ColumnFormat, ColumnWriter, MappingExt, WeightedMean};
use crate::cell::{BoundaryConditions, Shape};
use crate::selection::{CachedSelection, Groups, Selection};
use crate::ObserveContext;
use crate::Point;
use anyhow::Result;
use derive_more::Debug;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use interatomic::coulomb::{Medium, Temperature};

/// Bare-Coulomb multipole pair energies as geometric factors (units e²/Å); the caller scales
/// them to kJ/mol.
///
/// These four pure functions are hand-ported from the C++ Faunus `multipole.h` kernels rather
/// than reused from `coulomb::MultipoleEnergy`: that trait's `quadrupole_potential` subtracts the
/// tensor trace internally (i.e. expects the *primitive* Σqᵢrᵢrᵢᵀ), whereas we already hold the
/// *traceless* Θ from [`geometry::quadrupole_moment`](crate::geometry), and it carries
/// scheme/cutoff state we do not want here. All take the separation vector `R = rₐ − r_b` and are
/// unit-tested against analytical two-body values.
mod kernels {
    use crate::Point;
    use nalgebra::Matrix3;

    /// qₐ q_b / R
    pub fn ion_ion(charge_a: f64, charge_b: f64, r: f64) -> f64 {
        charge_a * charge_b / r
    }

    /// [qₐ (μ_b·R) − q_b (μₐ·R)] / R³, with full dipole vectors.
    ///
    /// The sign follows the potential-of-b-at-a convention: with `R = rₐ − r_b`, a dipole whose
    /// positive end faces the ion gives a repulsive (positive) energy.
    pub fn ion_dipole(
        charge_a: f64,
        dipole_a: &Point,
        charge_b: f64,
        dipole_b: &Point,
        r: &Point,
    ) -> f64 {
        let inv_r3 = r.norm_squared().powf(-1.5);
        (charge_a * dipole_b.dot(r) - charge_b * dipole_a.dot(r)) * inv_r3
    }

    /// (μₐ·μ_b)/R³ − 3(μₐ·R)(μ_b·R)/R⁵
    pub fn dipole_dipole(dipole_a: &Point, dipole_b: &Point, r: &Point) -> f64 {
        let inv_r2 = r.norm_squared().recip();
        let inv_r3 = inv_r2.sqrt() * inv_r2;
        dipole_a.dot(dipole_b) * inv_r3 - 3.0 * dipole_a.dot(r) * dipole_b.dot(r) * inv_r3 * inv_r2
    }

    /// qₐ (RᵀΘ_b R)/R⁵ + q_b (RᵀΘₐ R)/R⁵ using the traceless Θ.
    ///
    /// Equivalent to the C++ primitive-tensor form `3RᵀQR/R⁵ − tr(Q)/R³` because
    /// `RᵀΘR = 3RᵀQR − tr(Q)R²` for `Θ = ½(3P − tr(P)I)`, `Q = ½P`.
    pub fn ion_quadrupole(
        charge_a: f64,
        quadrupole_a: &Matrix3<f64>,
        charge_b: f64,
        quadrupole_b: &Matrix3<f64>,
        r: &Point,
    ) -> f64 {
        let inv_r5 = r.norm_squared().powf(-2.5);
        (charge_a * r.dot(&(quadrupole_b * r)) + charge_b * r.dot(&(quadrupole_a * r))) * inv_r5
    }
}

fn default_resolution() -> f64 {
    1.0
}

fn default_file() -> PathBuf {
    "multipole_dist.csv".into()
}

/// YAML builder for [`MultipoleDistribution`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultipoleDistributionBuilder {
    /// The two molecular species to correlate (COM-bearing groups).
    selections: (Selection, Selection),
    /// Output CSV file.
    #[serde(default = "default_file")]
    file: PathBuf,
    /// Distance resolution along R (Å).
    #[serde(default = "default_resolution", rename = "resolution")]
    dr: f64,
    /// Maximum separation. Defaults to half the shortest box dimension.
    #[serde(default, rename = "max", skip_serializing_if = "Option::is_none")]
    max_r: Option<f64>,
    /// Sampling frequency.
    frequency: Frequency,
}

impl MultipoleDistributionBuilder {
    pub fn apply_output_dir(&mut self, dir: &Path) -> Result<()> {
        crate::analysis::prefix_in_place(&mut self.file, dir)
    }

    pub fn build(
        &self,
        context: &impl ObserveContext,
        medium: Option<&Medium>,
    ) -> Result<MultipoleDistribution> {
        if self.dr <= 0.0 {
            anyhow::bail!("MultipoleDistribution: resolution must be positive");
        }
        if ColumnFormat::from_path(&self.file) != ColumnFormat::Csv {
            anyhow::bail!("MultipoleDistribution: output file must be a .csv file");
        }
        let medium = medium.ok_or_else(|| {
            anyhow::anyhow!("MultipoleDistribution: a medium is required for the Bjerrum length")
        })?;
        let bjerrum_length = medium.bjerrum_length();
        // geometric energy (e²/Å) × λ_B → kT, × RT → kJ/mol
        let energy_scale = bjerrum_length * crate::R_IN_KJ_PER_MOL * medium.temperature();

        let max_r = match self.max_r {
            Some(r) => r,
            None => {
                let bbox = context.cell().bounding_box().ok_or_else(|| {
                    anyhow::anyhow!("MultipoleDistribution: cell has no bounding box; set max")
                })?;
                bbox.x.min(bbox.y).min(bbox.z) / 2.0
            }
        };
        if max_r <= 0.0 {
            anyhow::bail!("MultipoleDistribution: max must be positive");
        }

        // The analysis is defined on molecular COM separations, so every resolved group must
        // carry a center of mass; atomic/reservoir groups do not and are rejected here.
        let mut resolved = [
            context.resolve_groups(&self.selections.0),
            context.resolve_groups(&self.selections.1),
        ];
        for (which, selection, groups) in [
            ("first", &self.selections.0, &resolved[0]),
            ("second", &self.selections.1, &resolved[1]),
        ] {
            if groups
                .iter()
                .any(|&gi| context.groups()[gi].mass_center().is_none())
            {
                anyhow::bail!(
                    "MultipoleDistribution: {which} selection '{}' matches a group without a \
                     center of mass (atomic groups are not supported)",
                    selection.source()
                );
            }
        }

        // Self- vs cross-correlation is decided by the resolved group set, not the raw selection
        // string, so two equivalent expressions for one species are still treated as a
        // self-correlation (and keep the +1 Kirkwood baseline). When both sets are empty — e.g. a
        // GCMC run that starts with none of either species present — the sets carry no information,
        // so fall back to comparing the selection strings.
        resolved[0].sort_unstable();
        resolved[1].sort_unstable();
        let same_selection = if resolved[0].is_empty() && resolved[1].is_empty() {
            self.selections.0.source() == self.selections.1.source()
        } else {
            resolved[0] == resolved[1]
        };
        Ok(MultipoleDistribution {
            selections: (
                CachedSelection::groups(self.selections.0.clone()),
                CachedSelection::groups(self.selections.1.clone()),
            ),
            resolution: self.dr,
            max_r,
            bjerrum_length,
            energy_scale,
            same_selection,
            bins: BTreeMap::new(),
            ref_count_sum: 0.0,
            output_file: self.file.clone(),
            sampling: Sampling::new(self.frequency),
        })
    }
}

/// Reduced electrostatic representation of a single group, precomputed once per frame.
struct GroupMoments {
    charge: f64,
    com: Point,
    /// Full dipole vector μ = Σ qᵢ(rᵢ − com), PBC-aware.
    dipole: Point,
    /// Traceless quadrupole Θ, PBC-aware.
    quadrupole: Matrix3<f64>,
    /// Per-atom (charge, displacement-from-COM), minimum-image. Cached so the exact energy uses
    /// the same periodic image as the multipole moments, and so each group is traversed once.
    atoms: Vec<(f64, Point)>,
}

impl GroupMoments {
    fn from_group(group_index: usize, context: &impl ObserveContext) -> Option<Self> {
        let group = &context.groups()[group_index];
        let com = *group.mass_center()?;
        let atomkinds = context.topology_ref().atomkinds();
        let cell = context.cell();
        // One pass builds the (charge, displacement) list; the moments are then derived from it,
        // matching geometry::{dipole_moment, quadrupole_moment} without three separate traversals.
        let atoms: Vec<(f64, Point)> = group
            .iter_active()
            .map(|i| {
                let charge = atomkinds[context.atom_kind(i).get()].charge();
                (charge, cell.distance(&context.position(i), &com))
            })
            .collect();
        let charge = atoms.iter().map(|&(q, _)| q).sum();
        let dipole = atoms.iter().fold(Point::zeros(), |mu, (q, d)| mu + *q * d);
        let primitive = atoms
            .iter()
            .fold(Matrix3::zeros(), |acc, (q, d)| acc + *q * d * d.transpose());
        let quadrupole = 0.5 * (3.0 * primitive - primitive.trace() * Matrix3::identity());
        Some(Self {
            charge,
            com,
            dipole,
            quadrupole,
            atoms,
        })
    }
}

/// Explicit Coulomb double sum Σᵢ Σⱼ qᵢqⱼ / |rᵢ − rⱼ| over the two groups' active atoms.
///
/// Atom positions are reconstructed as (COM-relative displacement) + `separation`, where
/// `separation` is the minimum-image COM–COM vector. This places both groups in the *same*
/// periodic image the multipole expansion uses, so the exact-vs-expansion comparison is not
/// contaminated by per-atom image flips near the box edge.
fn exact_energy(a: &GroupMoments, b: &GroupMoments, separation: &Point) -> f64 {
    let mut energy = 0.0;
    for (charge_i, d_i) in &a.atoms {
        if *charge_i == 0.0 {
            continue;
        }
        for (charge_j, d_j) in &b.atoms {
            if *charge_j == 0.0 {
                continue;
            }
            let distance = (d_i - d_j + separation).norm();
            // Coincident atoms (overlapping configuration) would divide by zero and poison the
            // running mean with NaN; skip them — such configurations are unphysical anyway.
            if distance < 1e-12 {
                continue;
            }
            energy += charge_i * charge_j / distance;
        }
    }
    energy
}

/// Per-bin running means. Energies are geometric (scaled to kJ/mol at write time); the
/// correlations are dimensionless.
#[derive(Debug, Default)]
struct PairData {
    exact: WeightedMean,
    ion_ion: WeightedMean,
    ion_dipole: WeightedMean,
    dipole_dipole: WeightedMean,
    ion_quadrupole: WeightedMean,
    /// ⟨μ̂ₐ·μ̂_b⟩ (= P₁); also the source of the Kirkwood sum.
    dipole_corr: WeightedMean,
    /// ⟨(3cos²θ − 1)/2⟩
    p2: WeightedMean,
    /// ⟨(μ̂ₐ·R̂)(μ̂_b·R̂)⟩
    longitudinal: WeightedMean,
    /// ⟨Θₐ:Θ_b⟩ (Frobenius inner product)
    quad_corr: WeightedMean,
    /// ⟨Θ̂ₐ:Θ̂_b⟩ ∈ [−1, 1]
    quad_corr_norm: WeightedMean,
}

/// One output row (all columns, energies already in kJ/mol).
struct Row {
    r: f64,
    exact: f64,
    tot: f64,
    ion_ion: f64,
    ion_dipole: f64,
    dipole_dipole: f64,
    ion_quadrupole: f64,
    dipole_corr: f64,
    p2: f64,
    longitudinal: f64,
    kirkwood: f64,
    quad_corr: f64,
    quad_corr_norm: f64,
}

/// Multipolar decomposition and orientational correlations vs. COM separation.
#[derive(Debug)]
pub struct MultipoleDistribution {
    #[debug(skip)]
    selections: (CachedSelection<Groups>, CachedSelection<Groups>),
    resolution: f64,
    max_r: f64,
    bjerrum_length: f64,
    energy_scale: f64,
    same_selection: bool,
    bins: BTreeMap<i64, PairData>,
    /// Σ (number of reference/first-selection groups) × weight, over frames — the Kirkwood
    /// normalization N_ref.
    ref_count_sum: f64,
    #[debug(skip)]
    output_file: PathBuf,
    /// Frequency and frame count, owned by the framework.
    sampling: Sampling,
}

/// Mean of a per-bin accumulator, or 0 when the accumulator is empty (correlations are only added
/// for well-defined dipoles/quadrupoles, so a bin may have fewer correlation samples than pairs).
fn mean_or_zero(accumulator: &WeightedMean) -> f64 {
    if accumulator.is_empty() {
        0.0
    } else {
        accumulator.mean()
    }
}

impl MultipoleDistribution {
    /// Build the output rows, scaling energies to kJ/mol and accumulating the running Kirkwood
    /// factor over bins in ascending R.
    fn rows(&self) -> Vec<Row> {
        let scale = self.energy_scale;
        // Self-correlation stores each unordered pair once, so the Σ_{i≠j} in g_K is twice the
        // accumulated Σ_{i<j}. Cross-correlation already stores every ordered pair.
        let pair_factor = if self.same_selection { 2.0 } else { 1.0 };
        let mut kirkwood_running = 0.0;
        let mut rows = Vec::with_capacity(self.bins.len());
        for (&index, data) in &self.bins {
            let r = (index as f64 + 0.5) * self.resolution;
            let ion_ion = mean_or_zero(&data.ion_ion);
            let ion_dipole = mean_or_zero(&data.ion_dipole);
            let dipole_dipole = mean_or_zero(&data.dipole_dipole);
            let ion_quadrupole = mean_or_zero(&data.ion_quadrupole);
            let exact = mean_or_zero(&data.exact);

            // g_K(R) = 1 + (1/N_ref) Σ_{i≠j, r≤R} ⟨μ̂ᵢ·μ̂ⱼ⟩; the per-bin contribution is
            // pair_factor · Σ(weight·cosθ)/N_ref = pair_factor · sum_weights·mean / ref_count_sum.
            if self.ref_count_sum > 0.0 && !data.dipole_corr.is_empty() {
                kirkwood_running +=
                    pair_factor * data.dipole_corr.sum_weights() * data.dipole_corr.mean()
                        / self.ref_count_sum;
            }
            let kirkwood = if self.same_selection {
                1.0 + kirkwood_running
            } else {
                kirkwood_running
            };

            rows.push(Row {
                r,
                exact: exact * scale,
                tot: (ion_ion + ion_dipole + dipole_dipole + ion_quadrupole) * scale,
                ion_ion: ion_ion * scale,
                ion_dipole: ion_dipole * scale,
                dipole_dipole: dipole_dipole * scale,
                ion_quadrupole: ion_quadrupole * scale,
                dipole_corr: mean_or_zero(&data.dipole_corr),
                p2: mean_or_zero(&data.p2),
                longitudinal: mean_or_zero(&data.longitudinal),
                kirkwood,
                quad_corr: mean_or_zero(&data.quad_corr),
                quad_corr_norm: mean_or_zero(&data.quad_corr_norm),
            });
        }
        rows
    }
}

impl_info!(
    MultipoleDistribution,
    "multipole_distribution",
    "Multipolar decomposition vs. separation"
);

impl<T: ObserveContext> Analyze<T> for MultipoleDistribution {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, _step: usize, weight: f64) -> Result<()> {
        let g1 = self.selections.0.resolve(context).to_vec();
        let moments1: Vec<(usize, GroupMoments)> = g1
            .iter()
            .filter_map(|&gi| Some((gi.get(), GroupMoments::from_group(gi.get(), context)?)))
            .collect();
        let moments2: Option<Vec<(usize, GroupMoments)>> = if self.same_selection {
            None
        } else {
            let g2 = self.selections.1.resolve(context).to_vec();
            Some(
                g2.iter()
                    .filter_map(|&gi| {
                        Some((gi.get(), GroupMoments::from_group(gi.get(), context)?))
                    })
                    .collect(),
            )
        };
        let second: &[(usize, GroupMoments)] = moments2.as_deref().unwrap_or(&moments1);

        // N_ref counts only reference groups with a well-defined dipole; apolar groups never
        // contribute to the Kirkwood sum, so counting them would bias g_K low.
        self.ref_count_sum += moments1
            .iter()
            .filter(|(_, m)| m.dipole.norm() > 1e-9)
            .count() as f64
            * weight;
        let (max_r, dr) = (self.max_r, self.resolution);
        let cell = context.cell();
        for (index, (gi_a, a)) in moments1.iter().enumerate() {
            // Self-correlation visits each unordered pair once (i < j); the means are unaffected
            // and the Kirkwood sum is doubled at write time. Cross-correlation visits every
            // ordered (a, b) across the two distinct selections.
            let partners = if self.same_selection {
                &moments1[index + 1..]
            } else {
                second
            };
            for (gi_b, b) in partners {
                if gi_a == gi_b {
                    continue;
                }
                let r_vec = cell.distance(&a.com, &b.com);
                let r = r_vec.norm();
                if r == 0.0 || r > max_r {
                    continue;
                }
                let data = self.bins.entry((r / dr).floor() as i64).or_default();

                data.exact.add(exact_energy(a, b, &r_vec), weight);
                data.ion_ion
                    .add(kernels::ion_ion(a.charge, b.charge, r), weight);
                data.ion_dipole.add(
                    kernels::ion_dipole(a.charge, &a.dipole, b.charge, &b.dipole, &r_vec),
                    weight,
                );
                data.dipole_dipole
                    .add(kernels::dipole_dipole(&a.dipole, &b.dipole, &r_vec), weight);
                data.ion_quadrupole.add(
                    kernels::ion_quadrupole(
                        a.charge,
                        &a.quadrupole,
                        b.charge,
                        &b.quadrupole,
                        &r_vec,
                    ),
                    weight,
                );

                let (len_a, len_b) = (a.dipole.norm(), b.dipole.norm());
                if len_a > 1e-9 && len_b > 1e-9 {
                    let (unit_a, unit_b) = (a.dipole / len_a, b.dipole / len_b);
                    let cos = unit_a.dot(&unit_b);
                    data.dipole_corr.add(cos, weight);
                    data.p2.add(0.5 * (3.0 * cos * cos - 1.0), weight);
                    let r_hat = r_vec / r;
                    data.longitudinal
                        .add(unit_a.dot(&r_hat) * unit_b.dot(&r_hat), weight);
                }

                let frobenius = a.quadrupole.dot(&b.quadrupole);
                data.quad_corr.add(frobenius, weight);
                let (norm_a, norm_b) = (a.quadrupole.norm(), b.quadrupole.norm());
                if norm_a > 0.0 && norm_b > 0.0 {
                    data.quad_corr_norm
                        .add(frobenius / (norm_a * norm_b), weight);
                }
            }
        }
        Ok(())
    }

    fn write_to_disk(&mut self) -> Result<()> {
        if self.sampling.num_samples() == 0 {
            return Ok(());
        }
        let mut writer = ColumnWriter::open(
            &self.output_file,
            &[
                "R",
                "exact",
                "tot",
                "ii",
                "id",
                "dd",
                "iq",
                "mucorr",
                "p2",
                "long",
                "gK",
                "quadcorr",
                "quadcorr_norm",
            ],
        )?;
        for row in self.rows() {
            writer.write_row(&[
                &format_args!("{:.4}", row.r),
                &format_args!("{:.6}", row.exact),
                &format_args!("{:.6}", row.tot),
                &format_args!("{:.6}", row.ion_ion),
                &format_args!("{:.6}", row.ion_dipole),
                &format_args!("{:.6}", row.dipole_dipole),
                &format_args!("{:.6}", row.ion_quadrupole),
                &format_args!("{:.6}", row.dipole_corr),
                &format_args!("{:.6}", row.p2),
                &format_args!("{:.6}", row.longitudinal),
                &format_args!("{:.6}", row.kirkwood),
                &format_args!("{:.6}", row.quad_corr),
                &format_args!("{:.6}", row.quad_corr_norm),
            ])?;
        }
        writer.flush()?;
        Ok(())
    }

    fn results(&self) -> Option<yaml_serde::Value> {
        if self.sampling.num_samples() == 0 {
            return None;
        }
        let mut map = yaml_serde::Mapping::new();
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("num_bins", self.bins.len())?;
        map.try_insert("bjerrum_length/Å", self.bjerrum_length)?;
        if let Some(last) = self.rows().last() {
            map.try_insert("kirkwood_factor", last.kirkwood)?;
        }
        Some(yaml_serde::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use crate::backend::{get_medium_str, Backend};
    use crate::Point;
    use approx::assert_relative_eq;
    use interatomic::coulomb::Temperature;
    use tempfile::NamedTempFile;

    fn backend_from_str(yaml: &str) -> Backend {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        let mut rng = rand::thread_rng();
        Backend::new(tmp.path(), None, &mut rng).unwrap()
    }

    /// kJ/mol scale factor λ_B·RT for the medium encoded in `yaml`.
    fn energy_scale(yaml: &str) -> f64 {
        let medium = get_medium_str(yaml).unwrap();
        medium.bjerrum_length() * crate::R_IN_KJ_PER_MOL * medium.temperature()
    }

    fn builder(sel_a: &str, sel_b: &str, max_r: f64) -> MultipoleDistributionBuilder {
        MultipoleDistributionBuilder {
            selections: (
                Selection::parse(sel_a).unwrap(),
                Selection::parse(sel_b).unwrap(),
            ),
            file: "multipole.csv".into(),
            dr: 1.0,
            max_r: Some(max_r),
            frequency: Frequency::Every(1),
        }
    }

    // --- Pure kernel tests (analytical two-body values) ---

    #[test]
    fn ion_ion_energy() {
        assert_relative_eq!(kernels::ion_ion(1.0, -1.0, 2.0), -0.5);
    }

    #[test]
    fn ion_dipole_sign_and_magnitude() {
        // Neutral dipole μ=+x̂ at the origin, unit ion at (3,0,0): the dipole's + end faces the
        // ion → repulsive (positive). u = q·(μ·R)/R³ = 1·3/27 = 1/9.
        let r = Point::new(3.0, 0.0, 0.0);
        let mu = Point::new(1.0, 0.0, 0.0);
        let u = kernels::ion_dipole(1.0, &Point::zeros(), 0.0, &mu, &r);
        assert_relative_eq!(u, 1.0 / 9.0);
        // Dipole perpendicular to R → zero.
        let mu_perp = Point::new(0.0, 1.0, 0.0);
        assert_relative_eq!(
            kernels::ion_dipole(1.0, &Point::zeros(), 0.0, &mu_perp, &r),
            0.0
        );
    }

    #[test]
    fn dipole_dipole_aligned_configurations() {
        let m = 1.5;
        let r = Point::new(4.0, 0.0, 0.0);
        let along = Point::new(m, 0.0, 0.0);
        let r3 = r.norm().powi(3);
        // Head-to-tail (both along R): attractive, −2μ²/R³.
        assert_relative_eq!(
            kernels::dipole_dipole(&along, &along, &r),
            -2.0 * m * m / r3
        );
        // Antiparallel head-to-tail: +2μ²/R³.
        assert_relative_eq!(
            kernels::dipole_dipole(&along, &(-along), &r),
            2.0 * m * m / r3
        );
        // Parallel side-by-side (both ⊥ R): repulsive, +μ²/R³.
        let side = Point::new(0.0, m, 0.0);
        assert_relative_eq!(kernels::dipole_dipole(&side, &side, &r), m * m / r3);
    }

    /// A linear quadrupole (+q, −2q, +q along z, spacing d) has traceless Θ; the ion–quadrupole
    /// energy on the z-axis must match both the analytical value and the primitive-tensor form.
    #[test]
    fn ion_quadrupole_matches_primitive_form() {
        let (q, d) = (1.0, 1.0);
        // Primitive P = Σ qᵢ rᵢrᵢᵀ about the center (only zz is non-zero: 2·q·d²).
        let mut primitive = Matrix3::zeros();
        for (charge, z) in [(q, d), (-2.0 * q, 0.0), (q, -d)] {
            let ri = Point::new(0.0, 0.0, z);
            primitive += charge * ri * ri.transpose();
        }
        // Traceless Θ = ½(3P − tr(P) I), matching geometry::quadrupole_moment.
        let theta = 0.5 * (3.0 * primitive - primitive.trace() * Matrix3::identity());

        let r = Point::new(0.0, 0.0, 5.0);
        let ion = 1.0;
        let energy = kernels::ion_quadrupole(ion, &Matrix3::zeros(), 0.0, &theta, &r);

        // Primitive form: q·[3 RᵀQR/R⁵ − tr(Q)/R³] with Q = ½P.
        let half = 0.5 * primitive;
        let rn2 = r.norm_squared();
        let expected =
            ion * (3.0 * r.dot(&(half * r)) / rn2.powf(2.5) - half.trace() / rn2.powf(1.5));
        assert_relative_eq!(energy, expected, epsilon = 1e-12);
    }

    // --- Builder / serde ---

    #[test]
    fn deserialize_defaults() {
        // Only the two required fields; dr, max_r, and file all fall back to defaults.
        let yaml = r#"
selections: ["molecule A", "molecule B"]
frequency: !Every 100
"#;
        let builder: MultipoleDistributionBuilder = yaml_serde::from_str(yaml).unwrap();
        assert_relative_eq!(builder.dr, 1.0);
        assert!(builder.max_r.is_none());
        assert_eq!(builder.file, PathBuf::from("multipole_dist.csv"));
    }

    #[test]
    fn unknown_field_is_rejected() {
        let yaml = r#"
selections: ["molecule A", "molecule B"]
file: multipole.csv
frequency: !Every 100
oops: 1
"#;
        assert!(yaml_serde::from_str::<MultipoleDistributionBuilder>(yaml).is_err());
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !MultipoleDistribution
  selections: ["molecule A", "molecule B"]
  file: multipole.csv
  resolution: 0.5
  frequency: !Every 100
"#;
        let builders: Vec<AnalysisBuilder> = yaml_serde::from_str(yaml).unwrap();
        assert!(matches!(
            builders[0],
            AnalysisBuilder::MultipoleDistribution(_)
        ));
    }

    #[test]
    fn non_csv_file_is_rejected() {
        let yaml = TWO_IONS;
        let ctx = backend_from_str(yaml);
        let medium = get_medium_str(yaml).unwrap();
        let mut b = builder("molecule CATION", "molecule ANION", 50.0);
        b.file = "out.dat".into();
        assert!(b.build(&ctx, Some(&medium)).is_err());
    }

    #[test]
    fn missing_medium_is_rejected() {
        let ctx = backend_from_str(TWO_IONS);
        let b = builder("molecule CATION", "molecule ANION", 50.0);
        assert!(b.build(&ctx, None).is_err());
    }

    #[test]
    fn atomic_selection_without_com_is_rejected() {
        let yaml = r#"
atoms:
  - {name: X, mass: 1.0, charge: 1.0, sigma: 1.0}
molecules:
  - name: particle
    atoms: [X]
    atomic: true
system:
  cell: !Cuboid [50.0, 50.0, 50.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: particle
      N: 10
      active: 4
      insert: !RandomAtomPos {}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let ctx = backend_from_str(yaml);
        let medium = get_medium_str(yaml).unwrap();
        let b = builder("molecule particle", "molecule particle", 20.0);
        let err = b.build(&ctx, Some(&medium)).unwrap_err().to_string();
        assert!(err.contains("center of mass"), "unexpected error: {err}");
    }

    // --- Integration tests (real system, one sample) ---

    const TWO_IONS: &str = r#"
atoms:
  - {name: Pos, mass: 1.0, charge: 1.0, sigma: 1.0}
  - {name: Neg, mass: 1.0, charge: -1.0, sigma: 1.0}
molecules:
  - name: CATION
    atoms: [Pos]
  - name: ANION
    atoms: [Neg]
system:
  cell: !Cuboid [100.0, 100.0, 100.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: CATION
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: ANION
      N: 1
      insert: !Manual [[10.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    #[test]
    fn two_ions_ion_ion_equals_exact_in_kj_per_mol() {
        let ctx = backend_from_str(TWO_IONS);
        let medium = get_medium_str(TWO_IONS).unwrap();
        let scale = energy_scale(TWO_IONS);
        let mut analysis = builder("molecule CATION", "molecule ANION", 50.0)
            .build(&ctx, Some(&medium))
            .unwrap();
        analysis.sample(&ctx, 0).unwrap();

        let rows = analysis.rows();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_relative_eq!(row.r, 10.5); // bin center for R=10, dr=1
        let expected = -scale / 10.0; // q_a q_b / R × λ_B·RT
        assert_relative_eq!(row.exact, expected, max_relative = 1e-9);
        assert_relative_eq!(row.ion_ion, expected, max_relative = 1e-9);
        assert_relative_eq!(row.tot, expected, max_relative = 1e-9);
        // Point ions carry no dipole or quadrupole.
        assert!(row.ion_dipole.abs() < 1e-9);
        assert!(row.dipole_dipole.abs() < 1e-9);
        assert!(row.ion_quadrupole.abs() < 1e-9);
    }

    #[test]
    fn ion_dipole_dominates_for_neutral_dipole() {
        // ION (q=+1) at origin; neutral DIP with μ = 0.2 x̂ centered at (10,0,0).
        let yaml = r#"
atoms:
  - {name: I, mass: 1.0, charge: 1.0, sigma: 1.0}
  - {name: P, mass: 1.0, charge: 1.0, sigma: 1.0}
  - {name: M, mass: 1.0, charge: -1.0, sigma: 1.0}
molecules:
  - name: ION
    atoms: [I]
  - name: DIP
    atoms: [P, M]
system:
  cell: !Cuboid [100.0, 100.0, 100.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: ION
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: DIP
      N: 1
      insert: !Manual [[10.1, 0.0, 0.0], [9.9, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let ctx = backend_from_str(yaml);
        let medium = get_medium_str(yaml).unwrap();
        let scale = energy_scale(yaml);
        let mut analysis = builder("molecule ION", "molecule DIP", 50.0)
            .build(&ctx, Some(&medium))
            .unwrap();
        analysis.sample(&ctx, 0).unwrap();

        let row = &analysis.rows()[0];
        // r = com_ion − com_dip = (−10,0,0); u_id = q_a(μ_b·r)/r³ = 1·(0.2·−10)/1000 = −0.002.
        assert_relative_eq!(row.ion_dipole, -0.002 * scale, max_relative = 1e-6);
        assert!(row.ion_ion.abs() < 1e-9); // DIP is neutral
                                           // Exact ≈ ion-dipole at this separation (higher multipoles are tiny).
        assert_relative_eq!(row.exact, row.ion_dipole, max_relative = 1e-3);
    }

    #[test]
    fn pbc_minimum_image_bins_across_boundary() {
        // Ions near opposite faces of a 10 Å box: min-image separation is 1 Å, not 9 Å.
        let yaml = r#"
atoms:
  - {name: Pos, mass: 1.0, charge: 1.0, sigma: 1.0}
  - {name: Neg, mass: 1.0, charge: -1.0, sigma: 1.0}
molecules:
  - name: CATION
    atoms: [Pos]
  - name: ANION
    atoms: [Neg]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: CATION
      N: 1
      insert: !Manual [[0.5, 5.0, 5.0]]
    - molecule: ANION
      N: 1
      insert: !Manual [[9.5, 5.0, 5.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let ctx = backend_from_str(yaml);
        let medium = get_medium_str(yaml).unwrap();
        let mut analysis = builder("molecule CATION", "molecule ANION", 4.0)
            .build(&ctx, Some(&medium))
            .unwrap();
        analysis.sample(&ctx, 0).unwrap();
        let rows = analysis.rows();
        assert_eq!(rows.len(), 1);
        assert_relative_eq!(rows[0].r, 1.5); // bin center for R=1, dr=1
    }

    #[test]
    fn no_pairs_within_max_r_writes_nothing() {
        let ctx = backend_from_str(TWO_IONS); // ions 10 Å apart
        let medium = get_medium_str(TWO_IONS).unwrap();
        let mut analysis = builder("molecule CATION", "molecule ANION", 2.0) // max_r < 10
            .build(&ctx, Some(&medium))
            .unwrap();
        analysis.sample(&ctx, 0).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 1);
        assert!(analysis.rows().is_empty());
        let yaml = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        assert_eq!(yaml.get("num_bins").unwrap().as_u64().unwrap(), 0);
        assert!(yaml.get("kirkwood_factor").is_none()); // no rows → no final gK
    }

    #[test]
    fn writes_comma_separated_csv() {
        let ctx = backend_from_str(TWO_IONS);
        let medium = get_medium_str(TWO_IONS).unwrap();
        let tmp = NamedTempFile::with_suffix(".csv").unwrap();
        let mut b = builder("molecule CATION", "molecule ANION", 50.0);
        b.file = tmp.path().to_path_buf();
        let mut analysis = b.build(&ctx, Some(&medium)).unwrap();
        analysis.sample(&ctx, 0).unwrap();
        Analyze::<Backend>::write_to_disk(&mut analysis).unwrap();

        let contents = std::fs::read_to_string(tmp.path()).unwrap();
        let mut lines = contents.lines();
        let header = lines.next().unwrap();
        assert!(
            header.starts_with("R,exact,tot,ii,id,dd,iq,mucorr,p2,long,gK,quadcorr,quadcorr_norm")
        );
        assert_eq!(lines.next().unwrap().matches(',').count(), 12); // 13 columns
    }

    // --- Kirkwood running integral ---

    #[test]
    fn kirkwood_factor_accumulates() {
        // Two reference molecules per frame; N_ref = ref_count_sum = 4 over the (implicit) frames.
        let mut bins = BTreeMap::new();
        let mut bin_near = PairData::default();
        bin_near.dipole_corr.add(0.5, 1.0);
        bin_near.dipole_corr.add(0.5, 1.0); // Σ w·cosθ = 1.0
        bins.insert(1, bin_near);
        let mut bin_far = PairData::default();
        bin_far.dipole_corr.add(1.0, 1.0); // Σ w·cosθ = 1.0
        bins.insert(2, bin_far);

        let analysis = MultipoleDistribution {
            selections: (
                CachedSelection::groups(Selection::parse("molecule A").unwrap()),
                CachedSelection::groups(Selection::parse("molecule A").unwrap()),
            ),
            resolution: 1.0,
            max_r: 100.0,
            bjerrum_length: 7.0,
            energy_scale: 1.0,
            same_selection: true,
            bins,
            ref_count_sum: 4.0,
            output_file: "x.csv".into(),
            sampling: Sampling::new(Frequency::Every(1)),
        };
        let rows = analysis.rows();
        // Self case stores unordered pairs, so Σ_{i≠j} = 2·Σ_{i<j}: g_K = 1 + 2·cumulative/N_ref.
        assert_relative_eq!(rows[0].kirkwood, 1.0 + 2.0 * 1.0 / 4.0); // 1.50
        assert_relative_eq!(rows[1].kirkwood, 1.0 + 2.0 * 2.0 / 4.0); // 2.00
    }

    #[test]
    fn kirkwood_cross_case_has_no_baseline() {
        let mut bins = BTreeMap::new();
        let mut bin = PairData::default();
        bin.dipole_corr.add(1.0, 1.0);
        bins.insert(1, bin);
        let analysis = MultipoleDistribution {
            selections: (
                CachedSelection::groups(Selection::parse("molecule A").unwrap()),
                CachedSelection::groups(Selection::parse("molecule B").unwrap()),
            ),
            resolution: 1.0,
            max_r: 100.0,
            bjerrum_length: 7.0,
            energy_scale: 1.0,
            same_selection: false,
            bins,
            ref_count_sum: 2.0,
            output_file: "x.csv".into(),
            sampling: Sampling::new(Frequency::Every(1)),
        };
        // Cross case: bare cumulative, no +1.
        assert_relative_eq!(analysis.rows()[0].kirkwood, 0.5);
    }

    // --- Review-driven fixes ---

    fn point_group(atoms: Vec<(f64, Point)>) -> GroupMoments {
        GroupMoments {
            charge: atoms.iter().map(|(q, _)| q).sum(),
            com: Point::zeros(),
            dipole: Point::zeros(),
            quadrupole: Matrix3::zeros(),
            atoms,
        }
    }

    #[test]
    fn exact_energy_uses_com_consistent_image() {
        // a: unit ion at its COM. b: ±1 at ±0.5 x̂ from its COM. Separation (3,0,0).
        // Atoms are reconstructed as d_i − d_j + separation (no per-atom min image):
        // ion–plus at |0 − 0.5 + 3| = 2.5, ion–minus at |0 + 0.5 + 3| = 3.5.
        let a = point_group(vec![(1.0, Point::zeros())]);
        let b = point_group(vec![
            (1.0, Point::new(0.5, 0.0, 0.0)),
            (-1.0, Point::new(-0.5, 0.0, 0.0)),
        ]);
        let energy = exact_energy(&a, &b, &Point::new(3.0, 0.0, 0.0));
        assert_relative_eq!(energy, 1.0 / 2.5 - 1.0 / 3.5, epsilon = 1e-12);
    }

    #[test]
    fn exact_energy_skips_coincident_atoms() {
        // The single atom pair reconstructs to zero separation (0 − 3 + 3); it must be skipped
        // rather than divide by zero and poison the mean with NaN.
        let a = point_group(vec![(1.0, Point::zeros())]);
        let b = point_group(vec![(1.0, Point::new(3.0, 0.0, 0.0))]);
        let energy = exact_energy(&a, &b, &Point::new(3.0, 0.0, 0.0));
        assert!(energy.is_finite());
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn self_vs_cross_detected_by_resolved_groups() {
        let ctx = backend_from_str(TWO_IONS);
        let medium = get_medium_str(TWO_IONS).unwrap();
        let self_corr = builder("molecule CATION", "molecule CATION", 50.0)
            .build(&ctx, Some(&medium))
            .unwrap();
        assert!(self_corr.same_selection);
        let cross_corr = builder("molecule CATION", "molecule ANION", 50.0)
            .build(&ctx, Some(&medium))
            .unwrap();
        assert!(!cross_corr.same_selection);
    }

    #[test]
    fn empty_selections_fall_back_to_string_comparison() {
        // Both selections match nothing (as a GCMC run starting empty would): the resolved sets
        // carry no information, so self/cross falls back to the selection strings.
        let ctx = backend_from_str(TWO_IONS);
        let medium = get_medium_str(TWO_IONS).unwrap();
        let cross = builder("molecule GHOST_A", "molecule GHOST_B", 50.0)
            .build(&ctx, Some(&medium))
            .unwrap();
        assert!(!cross.same_selection);
        let self_corr = builder("molecule GHOST_A", "molecule GHOST_A", 50.0)
            .build(&ctx, Some(&medium))
            .unwrap();
        assert!(self_corr.same_selection);
    }
}
