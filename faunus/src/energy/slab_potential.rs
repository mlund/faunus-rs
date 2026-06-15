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

//! Laterally-averaged charge on a z-grid and the screened slab potential it generates.
//!
//! For an implicit-salt (Debye–Hückel / Yukawa) electrolyte in a slab geometry the
//! electrostatics reduce from three dimensions to one: integrating the screened point
//! potential `e^(−κr)/r` over an infinite plane of areal charge density σ collapses to an
//! exponential, `φ(z) = (2π·l_B/κ)·σ·e^(−κ|z−z′|)` (units kT/e). Screening makes the lateral
//! integral convergent, so — unlike the bare-Coulomb Åkesson construction — no
//! infinite-slab correction term is needed.
//!
//! This module is the shared core for the
//! [`ElectricPotentialProfile`](crate::analysis::ElectricPotentialProfile) analysis and any
//! future Åkesson-style external energy term: slab-geometry detection, the per-slab kernel,
//! and the potential convolution live here so both consumers reuse the same physics.

use crate::cell::{Cell, Shape};
use anyhow::Result;
use std::f64::consts::PI;

/// Lateral box dimension, in Debye lengths, below which the infinite-plane kernel is flagged.
const MIN_LATERAL_DEBYE_LENGTHS: f64 = 3.0;

/// Relative tolerance for the square-base requirement `Lx == Ly`.
const SQUARE_TOLERANCE: f64 = 1e-6;

/// Lateral cross-section shape of a slab cell, which sets the finite-box correction form.
#[derive(Clone, Copy, Debug, PartialEq)]
enum LateralShape {
    /// Square base of half-width `a` (cuboid / slit).
    Square,
    /// Disk of radius `R` (cylinder).
    Disk,
}

/// Slab dimensions detected from a simulation cell (private; built only by [`SlabGrid::from_cell`]).
struct SlabDimensions {
    /// Lateral cross-sectional area (Å²).
    area: f64,
    /// Smallest lateral box dimension (Å) — the full width `2a` or the diameter `2R`.
    lateral_extent: f64,
    /// Half the box length along z (Å); the slab spans `[−half_length_z, +half_length_z]`.
    half_length_z: f64,
    shape: LateralShape,
}

/// Detect the slab dimensions and lateral shape of `cell`.
///
/// The screened-slab reduction assumes lateral homogeneity, so the cross-section must be a
/// square cuboid/slit (`Lx = Ly`) or a cylinder (a disk, where `Lx = Ly` holds by
/// construction). Other cells cannot be laterally averaged this way and are rejected.
fn slab_dimensions(cell: &Cell) -> Result<SlabDimensions> {
    let bbox = cell
        .bounding_box()
        .ok_or_else(|| anyhow::anyhow!("a finite cell is required (got an endless cell)"))?;
    let (area, lateral_extent, shape) = match cell {
        Cell::Cuboid(_) | Cell::Slit(_) => {
            if (bbox.x - bbox.y).abs() > SQUARE_TOLERANCE * bbox.x.max(bbox.y) {
                anyhow::bail!(
                    "a square base is required (Lx = Ly); got Lx = {:.3}, Ly = {:.3}",
                    bbox.x,
                    bbox.y
                );
            }
            (bbox.x * bbox.y, bbox.x.min(bbox.y), LateralShape::Square)
        }
        // The bounding box of a cylinder is (2R, 2R, height), so the disk area is π·R².
        Cell::Cylinder(_) => {
            let radius = 0.5 * bbox.x;
            (PI * radius * radius, bbox.x, LateralShape::Disk)
        }
        other => {
            anyhow::bail!("only cuboid, slit, and cylinder cells are supported; got {other:?}")
        }
    };
    Ok(SlabDimensions {
        area,
        lateral_extent,
        half_length_z: 0.5 * bbox.z,
        shape,
    })
}

/// Geometry factor of a uniformly charged **square** sheet of half-width `b`, on the axis at
/// perpendicular distance `x`: `arcsin[(b⁴ − x⁴ − 2b²x²)/(b²+x²)²] + π/2`. It is `π` at contact
/// (`x = 0`, the infinite-sheet limit) and decays to `0` as `x → ∞`.
pub(crate) fn square_sheet_factor(x: f64, b: f64) -> f64 {
    let (b2, x2) = (b * b, x * x);
    let ratio = (b2 * b2 - x2 * x2 - 2.0 * b2 * x2) / (b2 + x2).powi(2);
    ratio.clamp(-1.0, 1.0).asin() + std::f64::consts::FRAC_PI_2
}

/// Per-slab Green's function: the potential at axial separation `dz` from a plane of unit
/// areal charge density (units: kT/e per e·Å⁻²).
#[derive(Clone, Copy, Debug)]
pub(crate) enum SlabKernel {
    /// Screened (Debye–Hückel / Yukawa) plane: `(2π·l_B/κ)·e^(−κ|dz|)`.
    Screened { prefactor: f64, kappa: f64 },
    /// Bare-Coulomb (Greberg/Åkesson) plane: `−2π·l_B·|dz|`, the 1-D Poisson Green's function
    /// of an infinite charged sheet (meaningful for an electroneutral slab).
    Unscreened { bjerrum_length: f64 },
}

impl SlabKernel {
    /// Screened kernel from the Bjerrum length `l_B` (Å) and inverse Debye length `κ` (Å⁻¹).
    pub(crate) fn screened(bjerrum_length: f64, kappa: f64) -> Self {
        Self::Screened {
            prefactor: 2.0 * PI * bjerrum_length / kappa,
            kappa,
        }
    }

    /// Bare-Coulomb (Greberg/Åkesson) kernel from the Bjerrum length `l_B` (Å).
    pub(crate) fn unscreened(bjerrum_length: f64) -> Self {
        Self::Unscreened { bjerrum_length }
    }

    /// Potential per unit areal charge density at axial separation `dz`.
    ///
    /// The screened path in [`SlabGrid`] inlines this via a fast separable recurrence; the
    /// non-separable unscreened kernel is convolved through it directly.
    pub(crate) fn potential(&self, dz: f64) -> f64 {
        match *self {
            Self::Screened { prefactor, kappa } => prefactor * (-kappa * dz.abs()).exp(),
            Self::Unscreened { bjerrum_length } => -2.0 * PI * bjerrum_length * dz.abs(),
        }
    }
}

/// Uniform z-grid over `[−half_length_z, +half_length_z]` plus the slab kernel: turns
/// per-slab charges into a laterally-averaged potential profile. Detecting the slab geometry,
/// validating it, and the screened convolution all live behind this one type.
#[derive(Clone, Debug)]
pub(crate) struct SlabGrid {
    half_length_z: f64,
    bin_width: f64,
    n_bins: usize,
    area: f64,
    lateral_extent: f64,
    shape: LateralShape,
    kernel: SlabKernel,
    /// Finite-box correction φ_ext sampled at slab separations `0, Δz, 2Δz, …`, present only
    /// when the correction is enabled (see [`with_finite_box_correction`](Self::with_finite_box_correction)).
    correction: Option<Vec<f64>>,
}

impl SlabGrid {
    /// Detect the slab geometry from `cell` and lay out a z-grid of spacing ≈ `resolution`.
    ///
    /// Fails for cells that cannot be laterally averaged (see [`slab_dimensions`]).
    pub(crate) fn from_cell(cell: &Cell, resolution: f64, kernel: SlabKernel) -> Result<Self> {
        let SlabDimensions {
            area,
            lateral_extent,
            half_length_z,
            shape,
        } = slab_dimensions(cell)?;
        // Pick the bin count tiling the slab closest to `resolution`, but never zero bins.
        let n_bins = ((2.0 * half_length_z / resolution).round() as usize).max(1);
        Ok(Self {
            half_length_z,
            bin_width: 2.0 * half_length_z / n_bins as f64,
            n_bins,
            area,
            lateral_extent,
            shape,
            kernel,
            correction: None,
        })
    }

    /// Enable the finite-box correction: the profile then reports φ_box = φ_∞ − φ_ext, the
    /// potential of the finite minimum-image cross-section rather than of an infinite plane.
    /// Use when the simulation has no Åkesson external term of its own.
    ///
    /// Errors for an **unscreened cylinder**: the bare-Coulomb (Greberg) correction is defined
    /// for the square minimum-image box, not a disk.
    pub(crate) fn with_finite_box_correction(mut self) -> Result<Self> {
        if matches!(
            (self.kernel, self.shape),
            (SlabKernel::Unscreened { .. }, LateralShape::Disk)
        ) {
            anyhow::bail!(
                "the unscreened (Greberg) finite-box correction requires a square base, not a cylinder"
            );
        }
        let table = (0..self.n_bins)
            .map(|k| self.finite_box_correction(k as f64 * self.bin_width))
            .collect();
        self.correction = Some(table);
        Ok(self)
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.n_bins
    }

    pub(crate) fn bin_width(&self) -> f64 {
        self.bin_width
    }

    /// Lateral cross-sectional area (Å²).
    pub(crate) fn area(&self) -> f64 {
        self.area
    }

    /// Smallest lateral box dimension expressed in Debye lengths. The infinite-plane kernel
    /// assumes this is large; [`is_laterally_thin`](Self::is_laterally_thin) flags when it is not.
    pub(crate) fn lateral_debye_lengths(&self, debye_length: f64) -> f64 {
        self.lateral_extent / debye_length
    }

    /// Whether the lateral extent is too thin for the infinite-plane kernel to be accurate.
    pub(crate) fn is_laterally_thin(&self, debye_length: f64) -> bool {
        self.lateral_debye_lengths(debye_length) < MIN_LATERAL_DEBYE_LENGTHS
    }

    /// z at the centre of bin `index`.
    pub(crate) fn bin_center(&self, index: usize) -> f64 {
        -self.half_length_z + (index as f64 + 0.5) * self.bin_width
    }

    /// Bin holding axial position `z`, clamped to the grid so boundary positions fold in.
    pub(crate) fn bin_index(&self, z: f64) -> usize {
        let raw = ((z + self.half_length_z) / self.bin_width).floor();
        raw.clamp(0.0, (self.n_bins - 1) as f64) as usize
    }

    /// Potential profile φ(zᵢ) from per-slab total charges `slab_charges[j] = Qⱼ`.
    ///
    /// φ is the convolution of the areal density σⱼ = Qⱼ/area with the kernel. For the
    /// screened kernel the exponential separates into forward/backward cumulative sums,
    /// giving an O(n) evaluation whose every factor is `e^(−κ·Δz) < 1` (so it cannot
    /// overflow). A non-separable kernel would instead use the direct O(n²) sum.
    pub(crate) fn potential_profile(&self, slab_charges: &[f64]) -> Vec<f64> {
        debug_assert_eq!(slab_charges.len(), self.n_bins);
        let mut phi = match self.kernel {
            // Separable exponential ⇒ O(n) recurrence.
            SlabKernel::Screened { prefactor, kappa } => {
                self.screened_profile(slab_charges, prefactor, kappa)
            }
            // `−2π·l_B·|dz|` is not separable ⇒ direct O(n²) convolution.
            SlabKernel::Unscreened { .. } => self.direct_profile(slab_charges),
        };
        // Subtract the finite-box correction φ_box = φ_∞ − φ_ext, when enabled. The correction
        // kernel is not separable, so this is a direct O(n²) convolution over the cached table.
        if let Some(table) = &self.correction {
            for (i, phi_i) in phi.iter_mut().enumerate() {
                // φ_ext depends only on slab separation, so index the table by |i − j|.
                let exterior: f64 = slab_charges
                    .iter()
                    .enumerate()
                    .map(|(j, &charge)| charge / self.area * table[i.abs_diff(j)])
                    .sum();
                *phi_i -= exterior;
            }
        }
        phi
    }

    /// Finite-box external correction φ_ext at axial separation `dz`, per unit areal charge.
    ///
    /// It is the potential of the charge *outside* the finite cross-section, `φ_∞ − φ_box`,
    /// so that [`with_finite_box_correction`](Self::with_finite_box_correction) reports the
    /// finite-patch potential. The disk forms are closed; the screened square is the smooth
    /// Greberg quadrature (the `4/π` and `π/4` domain factor cancel to `prefactor · ∫₀¹`);
    /// the unscreened square is Greberg's `−2π·l_B·|z| − l_B·u_box` via [`square_sheet_factor`].
    fn finite_box_correction(&self, dz: f64) -> f64 {
        let a = 0.5 * self.lateral_extent;
        let z = dz.abs();
        match self.kernel {
            SlabKernel::Screened { prefactor, kappa } => match self.shape {
                LateralShape::Disk => prefactor * (-kappa * (a * a + z * z).sqrt()).exp(),
                LateralShape::Square => {
                    const SAMPLES: usize = 65; // odd ⇒ pure Simpson 1/3 over [0, π/4]
                    let mut samples = [0.0; SAMPLES];
                    for (k, sample) in samples.iter_mut().enumerate() {
                        let theta = std::f64::consts::FRAC_PI_4 * k as f64 / (SAMPLES - 1) as f64;
                        let edge = a / theta.cos();
                        *sample = (-kappa * (edge * edge + z * z).sqrt()).exp();
                    }
                    prefactor * crate::auxiliary::simpson_integrate(&samples)
                }
            },
            // Square-only by construction (`with_finite_box_correction` rejects a cylinder):
            // φ_ext = φ_∞ − φ_box = −2π·l_B·z − l_B·u_box (Greberg Eq. 13), with
            // u_box(z) = 8a·ln[(√(2a²+z²)+a)/√(a²+z²)] − 2z·square_sheet_factor(z, a).
            SlabKernel::Unscreened { bjerrum_length } => {
                let log_term =
                    8.0 * a * (((2.0 * a * a + z * z).sqrt() + a) / (a * a + z * z).sqrt()).ln();
                let u_box = log_term - 2.0 * z * square_sheet_factor(z, a);
                bjerrum_length * (-2.0 * PI * z - u_box)
            }
        }
    }

    /// O(n) screened convolution via separable forward/backward cumulative sums, using a
    /// single output buffer: fill it with the backward sum, then fold in the forward sum and
    /// scale. Every factor is `decay < 1`, so it cannot overflow.
    fn screened_profile(&self, slab_charges: &[f64], prefactor: f64, kappa: f64) -> Vec<f64> {
        let n = self.n_bins;
        let decay = (-kappa * self.bin_width).exp();
        let sigma = |i: usize| slab_charges[i] / self.area; // areal density of slab i

        let mut phi = vec![0.0; n];
        // phi[i] ← backward sum Σ_{j>i} σⱼ·decay^(j−i); phi[n−1] stays 0 (no j > n−1).
        let mut running = 0.0;
        for i in (0..n.saturating_sub(1)).rev() {
            running = decay * (sigma(i + 1) + running);
            phi[i] = running;
        }
        // Fold in the forward sum Σ_{j≤i} σⱼ·decay^(i−j) (self term j = i included) and scale.
        running = 0.0;
        for (i, phi_i) in phi.iter_mut().enumerate() {
            running = sigma(i) + decay * running;
            *phi_i = prefactor * (running + *phi_i);
        }
        phi
    }

    /// Direct O(n²) convolution using the kernel: the evaluator for non-separable kernels
    /// (the unscreened one) and the correctness baseline the screened fast path is tested against.
    pub(crate) fn direct_profile(&self, slab_charges: &[f64]) -> Vec<f64> {
        (0..self.n_bins)
            .map(|i| {
                (0..self.n_bins)
                    .map(|j| {
                        let sigma = slab_charges[j] / self.area;
                        sigma
                            * self
                                .kernel
                                .potential(self.bin_center(i) - self.bin_center(j))
                    })
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{Cuboid, Cylinder, Slit, Sphere};

    // 100 Å cube ⇒ half-length 50, lateral extent 100, area 100·100 = 10⁴ Å².
    fn screened_grid(resolution: f64) -> SlabGrid {
        let kernel = SlabKernel::screened(7.0, 1.0 / 10.0);
        SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(100.0, 100.0, 100.0)),
            resolution,
            kernel,
        )
        .unwrap()
    }

    #[test]
    fn kernel_is_even_in_separation() {
        let (bjerrum, kappa) = (7.0, 0.1);
        let kernel = SlabKernel::screened(bjerrum, kappa);
        assert!((kernel.potential(3.0) - kernel.potential(-3.0)).abs() < 1e-12);
        // At zero separation the kernel equals its prefactor 2π·l_B/κ (e⁰ = 1).
        assert!((kernel.potential(0.0) - 2.0 * PI * bjerrum / kappa).abs() < 1e-12);
    }

    #[test]
    fn fast_profile_matches_naive() {
        let grid = screened_grid(1.0);
        // Deterministic pseudo-random charges (index-varied, no RNG needed).
        let charges: Vec<f64> = (0..grid.n_bins())
            .map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.1)
            .collect();
        let fast = grid.potential_profile(&charges);
        let naive = grid.direct_profile(&charges);
        for (a, b) in fast.iter().zip(&naive) {
            assert!((a - b).abs() < 1e-9, "fast {a} vs naive {b}");
        }
    }

    #[test]
    fn single_sheet_decays_exponentially() {
        let grid = screened_grid(1.0);
        let mid = grid.n_bins() / 2;
        let mut charges = vec![0.0; grid.n_bins()];
        let total_charge = 2.0;
        charges[mid] = total_charge;
        let phi = grid.potential_profile(&charges);

        let (bjerrum, kappa) = (7.0, 0.1);
        let prefactor = 2.0 * PI * bjerrum / kappa;
        let sigma = total_charge / grid.area();
        for i in 0..grid.n_bins() {
            let dz = grid.bin_center(i) - grid.bin_center(mid);
            let expected = prefactor * sigma * (-kappa * dz.abs()).exp();
            assert!(
                (phi[i] - expected).abs() < 1e-9,
                "bin {i}: {} vs {expected}",
                phi[i]
            );
        }
    }

    #[test]
    fn obeys_linearized_poisson_boltzmann() {
        // The screened kernel is the Green's function of the 1-D linearized Poisson–Boltzmann
        // (Debye–Hückel) equation φ″ − κ²φ = −4π·l_B·ρ. For the exponential kernel this holds
        // *exactly* in discrete form on interior bins (the 3-point recurrence each r^|i−j| obeys):
        //   φ_{i+1} − (r + 1/r)·φ_i + φ_{i−1} = −A·(1/r − r)·σ_i,   r = e^(−κΔz), A = 2π·l_B/κ.
        let (bjerrum, kappa) = (7.0, 0.1);
        let grid = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(100.0, 100.0, 100.0)),
            0.25,
            SlabKernel::screened(bjerrum, kappa),
        )
        .unwrap();
        let n = grid.n_bins();
        // A smooth, interior-localised charge so boundary truncation never reaches the test band.
        let charges: Vec<f64> = (0..n)
            .map(|i| {
                let z = grid.bin_center(i);
                (-(z / 8.0).powi(2)).exp() * (z / 8.0)
            })
            .collect();
        let phi = grid.potential_profile(&charges);

        let r = (-kappa * grid.bin_width()).exp();
        let prefactor = 2.0 * PI * bjerrum / kappa;
        for i in 1..n - 1 {
            let lhs = phi[i + 1] - (r + 1.0 / r) * phi[i] + phi[i - 1];
            let sigma = charges[i] / grid.area();
            let rhs = -prefactor * (1.0 / r - r) * sigma;
            assert!((lhs - rhs).abs() < 1e-9, "bin {i}: lhs {lhs} vs rhs {rhs}");
        }
    }

    #[test]
    fn uniform_charge_reaches_debye_huckel_plateau() {
        // A uniform volume charge density ρ has no curvature, so deep inside the slab the DH
        // equation gives the plateau φ = 4π·l_B·ρ/κ² — the screened analogue of a bulk potential.
        let (bjerrum, kappa) = (7.0, 0.2);
        let grid = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(100.0, 100.0, 200.0)),
            0.25,
            SlabKernel::screened(bjerrum, kappa),
        )
        .unwrap();
        let n = grid.n_bins();
        let charge_per_slab = 0.01;
        let charges = vec![charge_per_slab; n];
        let phi = grid.potential_profile(&charges);

        let rho = charge_per_slab / (grid.area() * grid.bin_width()); // e·Å⁻³
        let plateau = 4.0 * PI * bjerrum * rho / (kappa * kappa);
        // Sample the centre, far from the edges where the finite slab cuts the kernel tails.
        assert!(
            (phi[n / 2] - plateau).abs() / plateau < 1e-3,
            "centre {} vs DH plateau {plateau}",
            phi[n / 2]
        );
    }

    #[test]
    fn antisymmetric_charge_gives_zero_midplane_potential() {
        // +Q and −Q placed symmetrically about the centre ⇒ φ = 0 at the midplane by symmetry.
        let grid = screened_grid(1.0); // even bin count, midplane between bins n/2−1 and n/2
        let n = grid.n_bins();
        let mut charges = vec![0.0; n];
        charges[n / 2 - 3] = 1.0;
        charges[n / 2 + 2] = -1.0; // mirror of n/2−3 about the centre
        let phi = grid.potential_profile(&charges);
        assert!((phi[n / 2 - 1] + phi[n / 2]).abs() < 1e-12); // antisymmetric about the centre
    }

    #[test]
    fn finite_box_correction_square_matches_exterior_integral() {
        // Validate the Greberg quadrature φ_ext = φ_∞ − φ_box against a brute-force 2-D
        // integral of the screened kernel over the square base [−a, a]².
        let (bjerrum, kappa, a, dz) = (7.0, 0.1, 10.0, 2.0);
        let grid = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(2.0 * a, 2.0 * a, 100.0)),
            1.0,
            SlabKernel::screened(bjerrum, kappa),
        )
        .unwrap();
        let prefactor = 2.0 * PI * bjerrum / kappa;
        let m = 800;
        let h = 2.0 * a / m as f64;
        let mut sum = 0.0;
        for ix in 0..m {
            let x = -a + (ix as f64 + 0.5) * h;
            for iy in 0..m {
                let y = -a + (iy as f64 + 0.5) * h;
                let r = (x * x + y * y + dz * dz).sqrt();
                sum += (-kappa * r).exp() / r;
            }
        }
        let phi_box = bjerrum * sum * h * h;
        let expected = prefactor * (-kappa * dz).exp() - phi_box;
        let got = grid.finite_box_correction(dz);
        assert!(
            (got - expected).abs() / expected < 1e-2,
            "got {got} vs {expected}"
        );
    }

    #[test]
    fn finite_box_correction_disk_matches_radial_integral() {
        // Disk closed form φ_ext = prefactor·e^(−κ√(R²+dz²)) vs a direct radial integral.
        let (bjerrum, kappa, radius, dz) = (7.0, 0.1, 10.0, 2.0);
        let grid = SlabGrid::from_cell(
            &Cell::Cylinder(Cylinder::new(radius, 100.0)),
            1.0,
            SlabKernel::screened(bjerrum, kappa),
        )
        .unwrap();
        let prefactor = 2.0 * PI * bjerrum / kappa;
        let m = 4000;
        let h = radius / m as f64;
        let mut sum = 0.0;
        for k in 0..m {
            let s = (k as f64 + 0.5) * h;
            let r = (s * s + dz * dz).sqrt();
            sum += (-kappa * r).exp() / r * 2.0 * PI * s;
        }
        let phi_disk = bjerrum * sum * h;
        let expected = prefactor * (-kappa * dz).exp() - phi_disk;
        let got = grid.finite_box_correction(dz);
        assert!(
            (got - expected).abs() / expected < 1e-3,
            "got {got} vs {expected}"
        );
    }

    #[test]
    fn finite_box_correction_vanishes_for_large_box() {
        // κ·a = 30 ⇒ the exterior charge is exponentially negligible.
        let grid = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(600.0, 600.0, 100.0)),
            1.0,
            SlabKernel::screened(7.0, 0.1),
        )
        .unwrap();
        assert!(grid.finite_box_correction(0.0) < 1e-6);
    }

    #[test]
    fn correction_lowers_potential_of_positive_charges() {
        // φ_box = φ_∞ − φ_ext with φ_ext > 0, so a positive charge profile drops everywhere.
        let base = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(30.0, 30.0, 60.0)),
            1.0,
            SlabKernel::screened(7.0, 0.1),
        )
        .unwrap();
        let corrected = base.clone().with_finite_box_correction().unwrap();
        let charges = vec![1.0; base.n_bins()];
        let phi = base.potential_profile(&charges);
        let phi_box = corrected.potential_profile(&charges);
        for (uncorrected, boxed) in phi.iter().zip(&phi_box) {
            assert!(
                *boxed < *uncorrected && *boxed > 0.0,
                "{boxed} vs {uncorrected}"
            );
        }
    }

    #[test]
    fn unscreened_kernel_obeys_poisson() {
        // The bare-Coulomb kernel −2π·l_B·|z| is the 1-D Poisson Green's function, so the
        // discrete second difference recovers the source exactly: φ_{i+1} − 2φ_i + φ_{i−1} =
        // −4π·l_B·Δz·σ_i for every interior bin (no neutrality needed for the discrete identity).
        let bjerrum = 7.0;
        let grid = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(80.0, 80.0, 80.0)),
            1.0,
            SlabKernel::unscreened(bjerrum),
        )
        .unwrap();
        let n = grid.n_bins();
        let charges: Vec<f64> = (0..n)
            .map(|i| ((i * 53 % 13) as f64 - 6.0) * 0.05)
            .collect();
        let phi = grid.potential_profile(&charges);
        let dz = grid.bin_width();
        for i in 1..n - 1 {
            let lhs = phi[i + 1] - 2.0 * phi[i] + phi[i - 1];
            let rhs = -4.0 * PI * bjerrum * dz * charges[i] / grid.area();
            assert!((lhs - rhs).abs() < 1e-6, "bin {i}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn unscreened_finite_box_square_matches_2d_integral() {
        // Greberg φ_ext = φ_∞ − φ_box = −2π·l_B·z − l_B·∫∫_box 1/r, validating u_box (and the
        // reused square_sheet_factor) against a brute-force 2-D integral of the bare kernel.
        let (bjerrum, a, dz) = (7.0, 10.0, 2.0);
        let grid = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(2.0 * a, 2.0 * a, 100.0)),
            1.0,
            SlabKernel::unscreened(bjerrum),
        )
        .unwrap();
        let m = 800;
        let h = 2.0 * a / m as f64;
        let mut sum = 0.0;
        for ix in 0..m {
            let x = -a + (ix as f64 + 0.5) * h;
            for iy in 0..m {
                let y = -a + (iy as f64 + 0.5) * h;
                sum += 1.0 / (x * x + y * y + dz * dz).sqrt();
            }
        }
        let phi_box = bjerrum * sum * h * h;
        let expected = -2.0 * PI * bjerrum * dz - phi_box;
        let got = grid.finite_box_correction(dz);
        assert!(
            (got - expected).abs() / expected.abs() < 1e-2,
            "got {got} vs {expected}"
        );
    }

    #[test]
    fn unscreened_finite_box_correction_requires_square_base() {
        // Greberg's bare-Coulomb correction is a square-box construction: accepted for a
        // cuboid/slit, rejected for a cylinder (a disk has no minimum-image square patch).
        let square = SlabGrid::from_cell(
            &Cell::Cuboid(Cuboid::new(20.0, 20.0, 100.0)),
            1.0,
            SlabKernel::unscreened(7.0),
        )
        .unwrap();
        assert!(square.with_finite_box_correction().is_ok());

        let cylinder = SlabGrid::from_cell(
            &Cell::Cylinder(Cylinder::new(10.0, 100.0)),
            1.0,
            SlabKernel::unscreened(7.0),
        )
        .unwrap();
        assert!(cylinder.with_finite_box_correction().is_err());
    }

    #[test]
    fn screened_correction_reduces_to_unscreened_as_kappa_vanishes() {
        // Coulomb limit: φ_ext^screened(z) = 2π·l_B/κ − 2π·l_B·z − l_B·u_box + O(κ), so removing
        // the regularized constant 2π·l_B/κ recovers the bare-Coulomb φ_ext (Greberg).
        let (bjerrum, a, dz, kappa) = (7.0, 10.0, 2.0, 1e-4);
        let cell = Cell::Cuboid(Cuboid::new(2.0 * a, 2.0 * a, 100.0));
        let screened =
            SlabGrid::from_cell(&cell, 1.0, SlabKernel::screened(bjerrum, kappa)).unwrap();
        let unscreened = SlabGrid::from_cell(&cell, 1.0, SlabKernel::unscreened(bjerrum)).unwrap();
        let regularized = screened.finite_box_correction(dz) - 2.0 * PI * bjerrum / kappa;
        let bare = unscreened.finite_box_correction(dz);
        assert!(
            (regularized - bare).abs() / bare.abs() < 1e-2,
            "{regularized} vs {bare}"
        );
    }

    #[test]
    fn bin_index_and_center_roundtrip() {
        let grid = screened_grid(1.0);
        for i in 0..grid.n_bins() {
            assert_eq!(grid.bin_index(grid.bin_center(i)), i);
        }
        // Out-of-range positions clamp to the end bins.
        assert_eq!(grid.bin_index(-1e3), 0);
        assert_eq!(grid.bin_index(1e3), grid.n_bins() - 1);
    }

    fn grid_from(cell: Cell) -> Result<SlabGrid> {
        SlabGrid::from_cell(&cell, 1.0, SlabKernel::screened(7.0, 0.1))
    }

    #[test]
    fn geometry_cuboid_requires_square_base() {
        let grid = grid_from(Cell::Cuboid(Cuboid::new(30.0, 30.0, 80.0))).unwrap();
        assert!((grid.area() - 900.0).abs() < 1e-9);
        // 80 Å along z over Δz ≈ 1 ⇒ 80 bins.
        assert_eq!(grid.n_bins(), 80);

        assert!(grid_from(Cell::Cuboid(Cuboid::new(30.0, 40.0, 80.0))).is_err());
    }

    #[test]
    fn geometry_slit_is_supported() {
        let grid = grid_from(Cell::Slit(Slit::new(30.0, 30.0, 80.0))).unwrap();
        assert!((grid.area() - 900.0).abs() < 1e-9);
    }

    #[test]
    fn geometry_cylinder_uses_disk_area() {
        let grid = grid_from(Cell::Cylinder(Cylinder::new(10.0, 80.0))).unwrap();
        assert!((grid.area() - PI * 100.0).abs() < 1e-9);
    }

    #[test]
    fn geometry_rejects_unsupported_cell() {
        assert!(grid_from(Cell::Sphere(Sphere::new(20.0))).is_err());
    }

    #[test]
    fn thinness_judged_in_debye_lengths() {
        // 30 Å lateral extent: thin below ~10 Å Debye length (< 3 Debye lengths).
        let grid = grid_from(Cell::Cuboid(Cuboid::new(30.0, 30.0, 80.0))).unwrap();
        assert!(grid.is_laterally_thin(15.0));
        assert!(!grid.is_laterally_thin(5.0));
        assert!((grid.lateral_debye_lengths(10.0) - 3.0).abs() < 1e-9);
    }
}
