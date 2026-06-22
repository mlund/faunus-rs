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

//! Osmotic pressure between two uniformly charged planes (electric double layer).
//!
//! Implements the midplane method of Guldbrand, Jönsson, Wennerström & Linse,
//! [*J. Chem. Phys.* **80**, 2221 (1984)](https://doi.org/10.1063/1.446912). For two
//! equally, uniformly charged walls with explicit point-charge counterions in a
//! [`Slit`](crate::cell::Slit) (walls at `z = ±Lz/2`, **midplane at `z = 0`**), the
//! osmotic pressure is (their Eq. 5)
//!
//! ```text
//! P_osm = kT·Σ_i C_i(0)            // entropic: ion density at the midplane
//!         + F_z^AB / area          // cross-midplane electrostatic force
//! ```
//!
//! The cross-midplane force splits into the configuration-dependent ion–ion term
//! (the correlation/attraction that the Poisson–Boltzmann treatment misses) and the
//! analytic wall contribution. The latter, summed over the two electroneutral halves,
//! collapses to the constant Maxwell stress `-σ²/(2ε_rε_0) = -2π·l_B·kT·σ²` — the same
//! term as in Guldbrand's contact-theorem expression (their Eq. 2). The surface charge
//! density `σ` is fixed by electroneutrality from the counterion charges.
//!
//! Hardcoded to a slit geometry and to electrostatics (point charges); short-range
//! interactions are not part of the cross-midplane stress as long as ions do not
//! overlap the midplane.

use super::{Analyze, Frequency};
use crate::auxiliary::{BlockAverage, ColumnWriter, MappingExt};
use crate::cell::{BoundaryConditions, Shape};
use crate::energy::slab_potential::square_sheet_factor;
use crate::selection::Selection;
use crate::{Context, Point};
use anyhow::Result;
use derive_more::Debug;
use interatomic::coulomb::Temperature;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// YAML builder for [`DoubleLayerPressure`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleLayerPressureBuilder {
    /// Selection of the mobile counterions (e.g. `"atomtype Na Ca"`).
    selection: Selection,
    /// Half-width `w` (Å) of the inner midplane density window; ρ(0) is found from the
    /// even-profile quadratic extrapolation `(4·ρ̄(w) − ρ̄(2w))/3` of the windows `w` and
    /// `2w`. Defaults to `gap/12` so `2w` stays inside the flat midplane region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    midplane_halfwidth: Option<f64>,
    /// Number of z-bins for the laterally-averaged charge profile that drives the
    /// self-consistent long-range correction `F_iPB`. Default 50.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    density_bins: Option<usize>,
    /// Output file path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file: Option<PathBuf>,
    /// Sampling frequency.
    frequency: Frequency,
}

impl DoubleLayerPressureBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        crate::analysis::prefix_opt(&mut self.file, dir)
    }

    /// Build the analysis.
    ///
    /// Needs `context` to read the cell geometry and to fix the surface charge density
    /// from electroneutrality, and `medium` for the thermal energy and permittivity.
    pub fn build(
        &self,
        context: &impl Context,
        medium: Option<&interatomic::coulomb::Medium>,
    ) -> Result<DoubleLayerPressure> {
        let thermal_energy = medium
            .map(|m| crate::R_IN_KJ_PER_MOL * m.temperature())
            .unwrap_or(crate::R_IN_KJ_PER_MOL * 298.15);

        // Coulomb prefactor in kJ/mol·Å (energy of q_i q_j / r), from the medium permittivity.
        let permittivity = medium.map(|m| m.permittivity()).unwrap_or(1.0);
        let prefactor = interatomic::coulomb::TO_CHEMISTRY_UNIT / permittivity;

        // Hardcoded to a slit: periodic in XY only, hard walls in Z, midplane at z = 0.
        // A Cuboid would pass the bounding-box check but wrap distances in Z and break the
        // wall/midplane assumptions, so guard on the boundary conditions.
        let pbc = context.cell().pbc();
        if pbc != crate::cell::PeriodicDirections::PeriodicXY {
            anyhow::bail!(
                "DoubleLayerPressure requires a Slit cell (periodic in XY only); got {pbc:?}"
            );
        }
        let bbox = context.cell().bounding_box().ok_or_else(|| {
            anyhow::anyhow!("DoubleLayerPressure requires a Slit cell with a finite bounding box")
        })?;
        let area = bbox.x * bbox.y;
        if area <= 0.0 {
            anyhow::bail!("DoubleLayerPressure: non-positive lateral area");
        }

        // The quadratic extrapolation holds only where the profile is still parabolic (near
        // the midplane minimum); beyond that the outer window `2w` would catch the rising wall
        // layer and bias ρ(0), so scale the default to the gap: bbox.z/12 = half_gap/6 ⇒ 2w = half_gap/3.
        let midplane_halfwidth = self.midplane_halfwidth.unwrap_or(bbox.z / 12.0);
        if midplane_halfwidth <= 0.0 {
            anyhow::bail!("DoubleLayerPressure: midplane_halfwidth must be positive");
        }

        let density_bins = self.density_bins.unwrap_or(50);
        if density_bins == 0 {
            anyhow::bail!("DoubleLayerPressure: density_bins must be positive");
        }

        // Surface charge per wall from electroneutrality: two walls neutralize the ions,
        // so |σ| = Σ q_ion / (2·area).
        let ion_charge: f64 = context
            .resolve_atoms_live(&self.selection)
            .iter()
            .map(|&i| context.atom_charge(i))
            .sum();
        let sigma = (ion_charge / (2.0 * area)).abs();

        // Finite charged-sheet geometry for the wall term (Guldbrand Eq. 9).
        // NOTE: ASSUMES A SQUARE lateral box (Lx = Ly); the sheet half-width is then
        // b = Lx/2. Walls sit at z = ±Lz/2 (gap = Lz). Eq. 9 keeps the wall term
        // consistent with the finite-box, minimum-image ion–ion cross force, instead of
        // the infinite-sheet value which over-cancels it.
        let half_box = bbox.x.min(bbox.y) / 2.0;
        let half_gap = bbox.z / 2.0;

        let stream = self
            .file
            .as_deref()
            .map(|p| {
                ColumnWriter::open(
                    p,
                    &["step", "rho_mid/Å⁻³", "p_ideal/mM", "p_corr/mM", "p_osm/mM"],
                )
            })
            .transpose()?;

        Ok(DoubleLayerPressure {
            selection: self.selection.clone(),
            thermal_energy,
            prefactor,
            area,
            midplane_halfwidth,
            sigma,
            half_box,
            half_gap,
            density: vec![0.0; density_bins],
            density_samples: 0,
            rho_mid: BlockAverage::new(),
            p_ideal: BlockAverage::new(),
            p_corr: BlockAverage::new(),
            p_osm: BlockAverage::new(),
            num_samples: 0,
            stream,
            frequency: self.frequency,
        })
    }
}

/// Double layer osmotic-pressure analysis (Guldbrand midplane method).
#[derive(Debug)]
pub struct DoubleLayerPressure {
    /// Mobile counterion selection.
    selection: Selection,
    /// Thermal energy R*T in kJ/mol.
    thermal_energy: f64,
    /// Coulomb prefactor in kJ/mol·Å (energy of `q_i q_j / r`).
    prefactor: f64,
    /// Lateral area Lx·Ly in Å².
    area: f64,
    /// Half-width `w` (Å) of the inner midplane density window; ρ(0) is the even-profile
    /// quadratic extrapolation `(4·ρ̄(w) − ρ̄(2w))/3` of the nested windows `w` and `2w`.
    midplane_halfwidth: f64,
    /// Surface charge density magnitude |σ| per wall, e·Å⁻² (from electroneutrality).
    sigma: f64,
    /// Half-width b of the (square) charged sheet, Å.
    half_box: f64,
    /// Half-gap Lz/2 (walls at ±half_gap; the gap Lz = 2·half_gap), Å.
    half_gap: f64,
    /// Accumulated counterion charge per z-bin (summed over samples) → the σ_ion(z)
    /// profile for the self-consistent `F_iPB` long-range correction.
    density: Vec<f64>,
    /// Number of samples accumulated into `density`.
    density_samples: usize,
    /// Ion number density at the midplane, Å⁻³.
    rho_mid: BlockAverage,
    /// Entropic term kT·ρ(0), kJ/mol·Å⁻³.
    p_ideal: BlockAverage,
    /// Configurational term (ion–ion cross force + Maxwell), kJ/mol·Å⁻³.
    p_corr: BlockAverage,
    /// Total osmotic pressure, kJ/mol·Å⁻³.
    p_osm: BlockAverage,
    /// Number of samples taken.
    num_samples: usize,
    /// Optional streaming output.
    #[debug(skip)]
    stream: Option<ColumnWriter>,
    /// Sampling frequency.
    frequency: Frequency,
}

/// Net `z`-force (without the Coulomb prefactor) that the lower half (`z < 0`) exerts on
/// the upper half (`z ≥ 0`): `Σ_{i: z_i<0} Σ_{j: z_j≥0} q_i q_j (z_j − z_i) / r_ij³`,
/// with `r_ij` the minimum-image distance. Positive = repulsion (pushing the halves apart).
fn cross_force_z(positions: &[Point], charges: &[f64], cell: &impl BoundaryConditions) -> f64 {
    let below: Vec<usize> = (0..positions.len())
        .filter(|&i| positions[i].z < 0.0)
        .collect();
    let above: Vec<usize> = (0..positions.len())
        .filter(|&i| positions[i].z >= 0.0)
        .collect();
    let mut fz = 0.0;
    for &a in &below {
        for &b in &above {
            let d = cell.distance(&positions[a], &positions[b]); // pa − pb; min-image in xy, raw in z
            let r2 = d.norm_squared();
            if r2 <= f64::EPSILON {
                continue;
            }
            // -d.z = z_b − z_a > 0
            fz += charges[a] * charges[b] * (-d.z) * r2.powf(-1.5);
        }
    }
    fz
}

/// Number of particles within `±slab` of the midplane (`z = 0`).
fn midplane_count(positions: &[Point], slab: f64) -> usize {
    positions.iter().filter(|p| p.z.abs() < slab).count()
}

/// Midplane number density ρ(0) (Å⁻³), matching Guldbrand's `C_i(0)`.
///
/// The profile is EVEN about the midplane (a symmetric minimum, dρ/dz = 0 at z = 0), so a
/// symmetric window average carries a QUADRATIC bias: `ρ̄(|z|<w) = ρ(0) + ρ″·w²/6`. The
/// two-window quadratic extrapolation `(4·ρ̄(w) − ρ̄(2w)) / 3` cancels it and is exact for a
/// parabolic profile. (A linear `2·ρ̄(w) − ρ̄(2w)` is correct only for a one-sided contact
/// ramp; on a symmetric minimum it under-counts by ρ″·w²/3 — the bug it replaces.) Keep `w`
/// well inside the flat midplane region so `2w` does not reach the rising wall layer.
fn midplane_density(positions: &[Point], w: f64, area: f64) -> f64 {
    let rho_w = midplane_count(positions, w) as f64 / (2.0 * w * area);
    let rho_2w = midplane_count(positions, 2.0 * w) as f64 / (4.0 * w * area);
    // ρ(0) ≥ 0; sampling noise can push the extrapolation slightly negative, so guard it.
    ((4.0 * rho_w - rho_2w) / 3.0).max(0.0)
}

/// Wall contribution to the cross-midplane force (without the Coulomb prefactor),
/// using the finite charged sheet (Eq. 9) so it cancels the finite-box ion–ion sum
/// consistently. Each ion is attracted to the OPPOSITE wall (distance `half_gap+|z|`);
/// the two charged sheets repel across the gap. `sigma` is the magnitude |σ|.
fn wall_cross_raw(
    positions: &[Point],
    charges: &[f64],
    sigma: f64,
    area: f64,
    half_box: f64,
    half_gap: f64,
) -> f64 {
    let ion_wall: f64 = positions
        .iter()
        .zip(charges)
        .map(|(p, &q)| -2.0 * sigma * q * square_sheet_factor(half_gap + p.z.abs(), half_box))
        .sum();
    // wall–wall: two sheets a full gap (2·half_gap) apart.
    let wall_wall = 2.0 * sigma * sigma * area * square_sheet_factor(2.0 * half_gap, half_box);
    ion_wall + wall_wall
}

impl DoubleLayerPressure {
    /// Convert a pressure in kJ/mol·Å⁻³ to mM (i.e. `P/RT` as a concentration).
    fn to_mm(&self, p: f64) -> f64 {
        (p / self.thermal_energy) * 1e3 / crate::MOLAR_TO_INV_ANGSTROM3
    }

    /// Convert a pressure in kJ/mol·Å⁻³ to Pascal.
    fn to_pa(&self, p: f64) -> f64 {
        p * 1e6 / crate::MOLAR_TO_INV_ANGSTROM3
    }

    /// `{mean, error}` mapping for a block average plus a constant `offset`, converted to mM.
    fn block_mm_yaml(&self, b: &BlockAverage, offset: f64) -> Option<serde_yml::Value> {
        let mut m = serde_yml::Mapping::new();
        m.try_insert("mean", self.to_mm(b.mean() + offset))?;
        m.try_insert("error", self.to_mm(b.error()))?;
        Some(serde_yml::Value::Mapping(m))
    }

    /// Self-consistent long-range correction `F_iPB` (Guldbrand), as a pressure
    /// (kJ/mol·Å⁻³). It is the `(∞ − finite-sheet)` cross-midplane force of the **neutral**
    /// laterally-averaged charge `σ(z) = σ_ion(z) − σ·δ(z±half_gap)` — the charge beyond the
    /// minimum-image box that `F_ii`/`F_iw` miss. Reconciles the min-image ion–ion sum with
    /// the sheet-based wall term, so the mean field cancels and only the correlation remains.
    /// Uses the accumulated `σ_ion(z)` profile (mixture-aware via per-atom charge).
    fn fipb_pressure(&self) -> f64 {
        if self.density_samples == 0 {
            return 0.0;
        }
        let n = self.density.len();
        let dz = 2.0 * self.half_gap / n as f64;
        let norm = self.density_samples as f64 * dz * self.area;
        let lambda: Vec<f64> = self.density.iter().map(|s| s / norm).collect();
        let z = |k: usize| -self.half_gap + (k as f64 + 0.5) * dz;
        let tail = |d: f64| std::f64::consts::PI - square_sheet_factor(d, self.half_box);

        // ion–ion across the midplane (below × above)
        let mut sum = 0.0;
        for a in 0..n {
            if z(a) >= 0.0 {
                continue;
            }
            for b in 0..n {
                if z(b) < 0.0 {
                    continue;
                }
                sum += lambda[a] * lambda[b] * tail(z(b) - z(a)) * dz * dz;
            }
        }
        // each ion ↔ its opposite wall (distance half_gap + |z|)
        for k in 0..n {
            sum -= self.sigma * lambda[k] * tail(self.half_gap + z(k).abs()) * dz;
        }
        // wall ↔ wall (two sheets a full gap apart)
        sum += self.sigma * self.sigma * tail(2.0 * self.half_gap);
        2.0 * self.prefactor * sum
    }

    /// Build the YAML results mapping (inherent so it is callable without choosing a
    /// `Context` type; the [`Analyze`] trait method delegates here).
    fn report(&self) -> Option<serde_yml::Value> {
        let fipb = self.fipb_pressure();
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.num_samples)?;
        map.insert("rho_mid/Å⁻³".into(), self.rho_mid.to_yaml()?);
        map.insert("p_ideal/mM".into(), self.block_mm_yaml(&self.p_ideal, 0.0)?);
        map.insert("p_corr/mM".into(), self.block_mm_yaml(&self.p_corr, fipb)?);
        map.insert("p_osm/mM".into(), self.block_mm_yaml(&self.p_osm, fipb)?);
        map.try_insert("F_iPB/mM", self.to_mm(fipb))?;
        map.try_insert("p_osm/Pa", self.to_pa(self.p_osm.mean() + fipb))?;
        Some(serde_yml::Value::Mapping(map))
    }
}

impl crate::Info for DoubleLayerPressure {
    fn short_name(&self) -> Option<&'static str> {
        Some("doublelayerpressure")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Osmotic pressure between two charged planes (Guldbrand midplane method)")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("doi:10.1063/1.446912")
    }
}

impl<T: Context> Analyze<T> for DoubleLayerPressure {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, step: usize, _weight: f64) -> Result<()> {
        let ions = context.resolve_atoms_live(&self.selection);
        let positions: Vec<Point> = ions.iter().map(|&i| context.position(i)).collect();
        let charges: Vec<f64> = ions.iter().map(|&i| context.atom_charge(i)).collect();

        let rho = midplane_density(&positions, self.midplane_halfwidth, self.area);
        let fz = cross_force_z(&positions, &charges, context.cell());
        let wall = wall_cross_raw(
            &positions,
            &charges,
            self.sigma,
            self.area,
            self.half_box,
            self.half_gap,
        );

        let p_ideal = self.thermal_energy * rho;
        let p_corr = self.prefactor * (fz + wall) / self.area;
        let p_osm = p_ideal + p_corr;

        self.rho_mid.add(rho);
        self.p_ideal.add(p_ideal);
        self.p_corr.add(p_corr);
        self.p_osm.add(p_osm);
        self.num_samples += 1;

        // Accumulate the laterally-averaged counterion charge profile σ_ion(z) (each ion
        // weighted by its own charge → mixture-aware) for the F_iPB correction at report time.
        let n = self.density.len();
        let dz = 2.0 * self.half_gap / n as f64;
        for (p, &q) in positions.iter().zip(&charges) {
            let k = ((p.z + self.half_gap) / dz)
                .floor()
                .clamp(0.0, (n - 1) as f64) as usize;
            self.density[k] += q;
        }
        self.density_samples += 1;

        let (ideal_mm, corr_mm, osm_mm) =
            (self.to_mm(p_ideal), self.to_mm(p_corr), self.to_mm(p_osm));
        if let Some(stream) = self.stream.as_mut() {
            stream.write_row(&[
                &step,
                &format_args!("{rho:.6e}"),
                &format_args!("{ideal_mm:.4}"),
                &format_args!("{corr_mm:.4}"),
                &format_args!("{osm_mm:.4}"),
            ])?;
        }
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        self.report()
    }
}

impl<T: Context> From<DoubleLayerPressure> for Box<dyn Analyze<T>> {
    fn from(analysis: DoubleLayerPressure) -> Self {
        Box::new(analysis)
    }
}

impl std::fmt::Display for DoubleLayerPressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Double Layer Pressure Analysis:")?;
        writeln!(f, "  Samples:   {}", self.num_samples)?;
        if self.num_samples > 0 {
            let p_osm = self.p_osm.mean() + self.fipb_pressure();
            writeln!(f, "  <P_osm>:   {:.4} mM", self.to_mm(p_osm))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Slit;

    #[test]
    fn cross_force_two_like_charges_is_repulsive() {
        // One unit charge below the midplane, one above, 2 Å apart along z.
        let cell = Slit::new(100.0, 100.0, 100.0);
        let positions = [Point::new(0.0, 0.0, -1.0), Point::new(0.0, 0.0, 1.0)];
        let charges = [1.0, 1.0];
        // q_i q_j (z_j − z_i) / r³ = 1·1·2 / 8 = 0.25 (repulsive ⇒ positive).
        let fz = cross_force_z(&positions, &charges, &cell);
        assert!((fz - 0.25).abs() < 1e-12, "fz = {fz}");
    }

    #[test]
    fn cross_force_opposite_charges_is_attractive() {
        let cell = Slit::new(100.0, 100.0, 100.0);
        let positions = [Point::new(0.0, 0.0, -1.0), Point::new(0.0, 0.0, 1.0)];
        let charges = [1.0, -1.0];
        let fz = cross_force_z(&positions, &charges, &cell);
        assert!((fz + 0.25).abs() < 1e-12, "fz = {fz}");
    }

    #[test]
    fn cross_force_same_side_pairs_excluded() {
        // Both charges below the midplane: no cross pair, zero force.
        let cell = Slit::new(100.0, 100.0, 100.0);
        let positions = [Point::new(0.0, 0.0, -1.0), Point::new(0.0, 0.0, -3.0)];
        let charges = [1.0, 1.0];
        assert_eq!(cross_force_z(&positions, &charges, &cell), 0.0);
    }

    #[test]
    fn cross_force_uses_minimum_image_in_xy() {
        // Two charges separated by 90 Å in x in a 100 Å box ⇒ min-image image is 10 Å.
        let cell = Slit::new(100.0, 100.0, 100.0);
        let positions = [
            Point::new(-45.0, 0.0, -0.0001),
            Point::new(45.0, 0.0, 0.0001),
        ];
        let charges = [1.0, 1.0];
        let d = 0.0002_f64; // tiny z separation
        let r2 = 10.0_f64 * 10.0 + d * d;
        let expected = 1.0 * 1.0 * d * r2.powf(-1.5);
        let fz = cross_force_z(&positions, &charges, &cell);
        assert!(
            (fz - expected).abs() < 1e-9,
            "fz = {fz}, expected = {expected}"
        );
    }

    #[test]
    fn midplane_count_within_slab() {
        let positions = [
            Point::new(0.0, 0.0, -1.0),
            Point::new(0.0, 0.0, 0.5),
            Point::new(0.0, 0.0, 3.0),
        ];
        assert_eq!(midplane_count(&positions, 2.0), 2);
        assert_eq!(midplane_count(&positions, 0.4), 0);
    }

    #[test]
    fn midplane_density_quadratic_extrapolation() {
        // 10 ions in |z|<w, 20 more in w<|z|<2w; area=2, w=1 ⇒ N(w)=10, N(2w)=30.
        // (8·N(w) − N(2w))/(12·w·area) = (80−30)/24 = 2.0833…
        let mut positions = vec![Point::new(0.0, 0.0, 0.5); 10];
        positions.extend(vec![Point::new(0.0, 0.0, 1.5); 20]);
        let rho = midplane_density(&positions, 1.0, 2.0);
        assert!((rho - 50.0 / 24.0).abs() < 1e-12, "rho = {rho}");
    }

    #[test]
    fn midplane_density_uniform_profile_unbiased() {
        // A flat profile (same density in both windows) extrapolates to itself:
        // (4·ρ − ρ)/3 = ρ.
        let positions: Vec<Point> = (-50..=50)
            .map(|i| Point::new(0.0, 0.0, i as f64 * 0.1))
            .collect();
        let area = 1.0;
        let w = 2.0;
        let n_in = midplane_count(&positions, w) as f64;
        let uniform = n_in / (2.0 * w * area);
        let rho = midplane_density(&positions, w, area);
        assert!(
            (rho - uniform).abs() < 0.2,
            "rho = {rho}, uniform ≈ {uniform}"
        );
    }

    #[test]
    fn midplane_density_recovers_parabolic_minimum() {
        // Even, symmetric-minimum profile ρ(z) = a + b·z² (the midplane shape). Lay down
        // `c·ρ(z)` ions per slice of width dz (c large enough to resolve the curvature) and
        // check the quadratic estimator returns ρ(0)=a — the property the old linear
        // estimator failed (on a convex minimum it under-counts to a − 2b·w²/3).
        let (a, b, w) = (2.0_f64, 0.5_f64, 1.0_f64);
        let (c, dz) = (100.0_f64, 0.005_f64);
        let area = c / dz; // so N(w)/(2w·area) recovers ρ̄(w)
        let mut positions = Vec::new();
        let mut z = -2.0 * w;
        while z <= 2.0 * w {
            for _ in 0..((a + b * z * z) * c).round() as usize {
                positions.push(Point::new(0.0, 0.0, z));
            }
            z += dz;
        }
        // Quadratic extrapolation recovers ρ(0)=a; the old linear `2ρ̄(w)−ρ̄(2w)` would
        // land near a − 2b·w²/3 = 1.667 on this convex profile.
        let rho = midplane_density(&positions, w, area);
        assert!((rho - a).abs() < 0.02, "rho = {rho}, expected a = {a}");
    }

    /// Construct an analysis directly for testing the reporting helpers.
    fn dummy(thermal_energy: f64) -> DoubleLayerPressure {
        DoubleLayerPressure {
            selection: Selection::parse("all").unwrap(),
            thermal_energy,
            prefactor: 1.0,
            area: 1.0,
            midplane_halfwidth: 1.0,
            sigma: 0.0,
            half_box: 1.0,
            half_gap: 1.0,
            density: vec![0.0; 20],
            density_samples: 0,
            rho_mid: BlockAverage::new(),
            p_ideal: BlockAverage::new(),
            p_corr: BlockAverage::new(),
            p_osm: BlockAverage::new(),
            num_samples: 0,
            stream: None,
            frequency: Frequency::Every(1),
        }
    }

    #[test]
    fn unit_conversions_match_concentration() {
        let rt = crate::R_IN_KJ_PER_MOL * 298.15;
        let a = dummy(rt);
        // P = RT·(1 Å⁻³) ⇒ concentration 1 Å⁻³ ≈ 1.66e6 mM.
        let mm = a.to_mm(rt);
        assert!(mm > 1.6e6 && mm < 1.7e6, "mm = {mm}");
        let pa = a.to_pa(rt);
        assert!(pa > 4.0e9 && pa < 4.2e9, "pa = {pa}");
    }

    #[test]
    fn yaml_reports_mean_and_error() {
        let mut a = dummy(crate::R_IN_KJ_PER_MOL * 298.15);
        a.p_osm.add(1.0);
        a.p_osm.add(3.0);
        a.num_samples = 2;
        let yaml = a.report().unwrap();
        let osm = &yaml["p_osm/mM"];
        assert!(osm.get("mean").is_some());
        assert!(osm.get("error").is_some());
        assert!(yaml.get("F_iPB/mM").is_some());
    }

    #[test]
    fn fipb_zero_without_samples() {
        let a = dummy(crate::R_IN_KJ_PER_MOL * 298.15);
        assert_eq!(a.fipb_pressure(), 0.0);
    }

    #[test]
    fn fipb_vanishes_for_infinite_sheet() {
        // tail(d) = π − g(d, b) → 0 as b → ∞, so the long-range correction vanishes.
        let mut a = dummy(crate::R_IN_KJ_PER_MOL * 298.15);
        a.sigma = 0.01;
        a.half_box = 1e7;
        a.density = vec![1.0; 20];
        a.density_samples = 1;
        assert!(
            a.fipb_pressure().abs() < 1e-3,
            "fipb = {}",
            a.fipb_pressure()
        );
    }

    #[test]
    fn fipb_nonzero_for_finite_box() {
        let mut a = dummy(crate::R_IN_KJ_PER_MOL * 298.15);
        a.sigma = 0.01;
        a.half_box = 5.0;
        a.density = vec![1.0; 20];
        a.density_samples = 1;
        assert!(
            a.fipb_pressure().abs() > 1e-9,
            "fipb = {}",
            a.fipb_pressure()
        );
    }

    #[test]
    fn info_trait() {
        let a = dummy(crate::R_IN_KJ_PER_MOL * 298.15);
        use crate::Info;
        assert_eq!(a.short_name(), Some("doublelayerpressure"));
        assert!(a.citation().unwrap().starts_with("doi:"));
    }

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
- !DoubleLayerPressure
  selection: "atomtype Na Ca"
  midplane_halfwidth: 2.0
  file: pressure.csv
  frequency: !Every 10
"#;
        let builders: Vec<crate::analysis::AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        match &builders[0] {
            crate::analysis::AnalysisBuilder::DoubleLayerPressure(b) => {
                assert_eq!(b.midplane_halfwidth, Some(2.0));
            }
            _ => panic!("expected DoubleLayerPressure variant"),
        }
    }

    #[test]
    fn deserialize_from_fixture() {
        let yaml = std::fs::read_to_string("tests/files/double_layer_pressure.yaml").unwrap();
        let builders: Vec<crate::analysis::AnalysisBuilder> = serde_yml::from_str(&yaml).unwrap();
        assert_eq!(builders.len(), 2);
        // Second entry omits midplane_halfwidth/file: they default.
        match &builders[1] {
            crate::analysis::AnalysisBuilder::DoubleLayerPressure(b) => {
                assert_eq!(b.midplane_halfwidth, None);
            }
            _ => panic!("expected DoubleLayerPressure variant"),
        }
    }
}
