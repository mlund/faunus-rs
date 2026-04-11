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

//! # Polymer depletion many-body interaction
//!
//! Implements the Forsman & Woodward many-body Hamiltonian for colloids in an
//! ideal polymer fluid. See [`PolymerDepletion`] for details.

use crate::{Change, Context, Point};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Yukawa-type kernel k_0(x) = exp(-x) / x used in eq 17 of the paper.
#[inline]
fn k0(x: f64) -> f64 {
    (-x).exp() / x
}

/// Modified spherical Bessel function i₀(x) = sinh(x)/x.
///
/// Regular (non-singular) radial solution to the modified Helmholtz equation
/// in spherical coordinates. Appears in the interior field of each colloid.
#[inline]
fn i0(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        // Taylor: sinh(x)/x = 1 + x²/6 + ... avoids 0/0
        1.0
    } else {
        x.sinh() / x
    }
}

/// Derivative i₀'(x) = cosh(x)/x - sinh(x)/x².
///
/// Needed for the Robin BC matching at the colloid surface, where the
/// radial derivative of the interior solution enters the calligraphic
/// functions K₀ and I₀.
#[inline]
fn i0_prime(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        // i₀ is even ⇒ i₀'(0) = 0; avoids cancellation in cosh/x - sinh/x²
        0.0
    } else {
        x.cosh() / x - x.sinh() / x.powi(2)
    }
}

/// Derivative k₀'(x) = -exp(-x)·(1+x)/x².
#[inline]
fn k0_prime(x: f64) -> f64 {
    -(-x).exp() * (1.0 + x) / x.powi(2)
}

/// Robin-modified k₀: K₀(x) = ε·k₀(x) + λ·k₀'(x).
///
/// Arises from applying the Robin BC (∂g/∂r + ε·g = 0) to the exterior
/// Yukawa solution k₀. The combination ε·k₀ + λ·k₀' vanishes when the
/// BC is satisfied, so it appears in denominators of the monopole strength.
#[inline]
fn cal_k0(x: f64, eps: f64, lam: f64) -> f64 {
    eps * k0(x) + lam * k0_prime(x)
}

/// Robin-modified i₀: I₀(x) = ε·i₀(x) + λ·i₀'(x).
///
/// Interior counterpart of [`cal_k0`]; same Robin BC applied to the
/// regular solution i₀. Enters the surface density ĝ_S (Eq. 18).
#[inline]
fn cal_i0(x: f64, eps: f64, lam: f64) -> f64 {
    eps * i0(x) + lam * i0_prime(x)
}

/// Renormalizing surface chemical-potential shift from the steric adsorption model.
///
/// βδμ = ln(1 - 1/g₀²) - 2/(g₀² - 1)
///
/// This offset ensures zero excess adsorption at ε₀' = 0 in the reduced
/// continuum parametrization used by the Huy-BC model.
#[inline]
fn beta_delta_mu(g0: f64) -> f64 {
    let g0_sq = g0.powi(2);
    // Use ln_1p for better accuracy when 1/g0² is small.
    (-1.0 / g0_sq).ln_1p() - 2.0 / (g0_sq - 1.0)
}

/// Self-consistent ε_eff from surface density (Eq. 14).
///
/// ε_eff = ε₀' + ln(1 - ĝ_S²/g₀²) - 2ĝ_S²/(g₀² - ĝ_S²) - βδμ
///
/// The ln and rational terms are the steric free-energy penalty for
/// crowding adsorbed chains: they drive ε_eff → -∞ as ĝ_S → g₀,
/// which weakens the effective adsorption and prevents divergence. The
/// renormalizing shift βδμ matches the thesis/manuscript form of the
/// steric adsorption boundary condition.
/// Caller must ensure |gs| < g0 to keep the log argument positive.
#[inline]
fn epsilon_eff_from_gs(eps0p: f64, gs: f64, g0: f64, delta_mu: f64) -> f64 {
    let ratio_sq = (gs / g0).powi(2);
    eps0p + (1.0 - ratio_sq).ln() - 2.0 * ratio_sq / (1.0 - ratio_sq) - delta_mu
}

/// Per-colloid information cached from the context.
#[derive(Debug, Clone)]
struct ColloidInfo {
    group_index: usize,
    com: Point,
    radius: f64,
}

/// Robin boundary condition amplitude suppression factor f(σ, h̃) = h̃ / ((1+σ) + h̃).
///
/// Modifies the Dirichlet monopole amplitude A_D to A_R = A_D · f,
/// where h̃ = R_c / b is the dimensionless inverse extrapolation length.
/// Returns 1.0 (Dirichlet limit) when h̃ is None (infinite).
///
/// - h̃ > 0: depletion (f ∈ (0,1), weaker than Dirichlet)
/// - h̃ = 0: Neumann / neutral (f = 0, no polymer-mediated interaction)
/// - -(1+σ) < h̃ < 0: adsorption (f < 0, sign-inverted interactions)
/// - h̃ ≤ -(1+σ): divergent; monopole approximation breaks down
#[inline]
fn robin_f(sigma: f64, h_tilde: Option<f64>) -> f64 {
    h_tilde.map_or(1.0, |ht| ht / ((1.0 + sigma) + ht))
}

/// Many-body polymer depletion interaction for colloids in an ideal polymer
/// fluid.
///
/// Implements the Hamiltonian of
/// [Forsman & Woodward, Soft Matter, 2012, 8, 2121](https://doi.org/10.1039/c2sm06737d)
/// (eq 17), generalised with Robin boundary conditions for tunable
/// polymer-surface affinity. The Robin parameter h̃ = R_c / b interpolates
/// between full depletion (h̃ → ∞, Dirichlet) and neutral adsorption
/// (h̃ = 0, Neumann). The single-particle insertion term scales as f(σ,h̃)
/// and the many-body pairwise term as f², where f = h̃/((1+σ)+h̃).
///
/// Rigid macromolecules are treated as neutral spheres via their
/// center of mass and bounding sphere radius. The potential decomposes into
/// pairwise sums at O(N_c²) cost.
#[derive(Debug, Clone)]
pub struct PolymerDepletion {
    /// Polymer radius of gyration (Å)
    rg: f64,
    /// Reduced polymer reservoir density rho_P* = rho_P * R_g^3 (dimensionless)
    rho_star: f64,
    /// Schulz-Flory order kappa = n + 1
    kappa: f64,
    /// Molecule type IDs treated as colloids
    colloid_molecule_ids: Vec<usize>,
    /// Molecule type names (for reporting)
    colloid_molecule_names: Vec<String>,
    /// Optional fixed colloid radius override (Å); else use bounding_radius
    fixed_radius: Option<f64>,
    /// Multiplicative scaling of the effective colloid radius
    radius_scaling: f64,
    /// Dimensionless Robin BC parameter h̃ = R_c / b; None means Dirichlet (h̃ → ∞)
    h_tilde: Option<f64>,
    /// Thermal energy kT (kJ/mol)
    thermal_energy: f64,
    /// Cached colloid positions and radii
    colloids: Vec<ColloidInfo>,
    /// Cached total energy (kJ/mol)
    cached_energy: f64,
    /// Pre-allocated backup buffer for MC undo (reused across steps)
    backup_colloids: Vec<ColloidInfo>,
    backup_energy: f64,
    has_backup: bool,
    /// Self-consistent steric adsorption state
    steric: Option<StericAdsorptionState>,
    /// Pre-allocated backup buffers for steric g_s / h_tilde_eff vectors.
    /// Stored separately (not as `Option<StericAdsorptionState>`) to avoid
    /// heap-allocating new Vecs on every MC trial; capacity is reused via
    /// clear + extend_from_slice, matching the backup_colloids pattern.
    backup_g_s: Vec<f64>,
    backup_h_tilde_eff: Vec<f64>,
}

impl PolymerDepletion {
    /// Compute the dimensionless free energy beta*dw (eq 17 from the paper).
    fn compute_beta_energy(
        &self,
        colloids: &[ColloidInfo],
        cell: &impl crate::cell::BoundaryConditions,
    ) -> f64 {
        if colloids.is_empty() {
            return 0.0;
        }
        let lambda = self.kappa.sqrt() / self.rg;
        let kappa_inv_3_2 = self.kappa.powf(-1.5);
        let prefactor = 4.0 * PI * self.rho_star;

        let mut energy = 0.0;
        for (i, ci) in colloids.iter().enumerate() {
            let sigma = lambda * ci.radius;
            let f = match &self.steric {
                Some(s) => {
                    debug_assert_eq!(s.h_tilde_eff.len(), colloids.len());
                    robin_f(sigma, Some(s.h_tilde_eff[i]))
                }
                None => robin_f(sigma, self.h_tilde),
            };

            let sigma2 = sigma.powi(2);

            // Single-particle insertion term (positive, unfavorable), scaled by f
            energy += f * kappa_inv_3_2 * (sigma + sigma2 + sigma2 * sigma / 3.0);

            // Many-body pairwise sum (negative, attractive depletion), scaled by f²
            let k0_sum: f64 = colloids
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, cj)| k0(lambda * cell.distance(&ci.com, &cj.com).norm()))
                .sum();

            let e2s = (2.0 * sigma).exp();
            let denom = 1.0 + (e2s - 1.0) * k0_sum / 2.0;
            energy -= f * f * sigma2 * e2s * kappa_inv_3_2 * k0_sum / denom;
        }

        prefactor * energy
    }

    /// Rebuild the colloid list from the current context, reusing existing capacity.
    fn rebuild_colloids(&mut self, context: &impl Context) {
        self.colloids.clear();
        for (gi, g) in context.groups().iter().enumerate() {
            if !self.colloid_molecule_ids.contains(&g.molecule()) {
                continue;
            }
            if let Some(&com) = g.mass_center() {
                let radius =
                    self.fixed_radius.or(g.bounding_radius()).unwrap_or(0.0) * self.radius_scaling;
                self.colloids.push(ColloidInfo {
                    group_index: gi,
                    com,
                    radius,
                });
            }
        }
    }

    /// Update a single colloid's COM and radius from the context.
    fn update_single_colloid(&mut self, group_index: usize, context: &impl Context) {
        let groups = context.groups();
        if let Some(colloid) = self
            .colloids
            .iter_mut()
            .find(|c| c.group_index == group_index)
        {
            if let Some(&com) = groups[group_index].mass_center() {
                colloid.com = com;
                colloid.radius = self
                    .fixed_radius
                    .or(groups[group_index].bounding_radius())
                    .unwrap_or(0.0)
                    * self.radius_scaling;
            }
        }
    }

    /// Check whether a change involves any colloid group.
    fn change_involves_colloids(&self, change: &Change) -> bool {
        match change {
            Change::Everything | Change::Volume(_, _) => true,
            Change::None => false,
            Change::SingleGroup(gi, _) => self.colloids.iter().any(|c| c.group_index == *gi),
            Change::Groups(groups) => groups
                .iter()
                .any(|(gi, _)| self.colloids.iter().any(|c| c.group_index == *gi)),
        }
    }

    /// Update internal state after a system change.
    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::SingleGroup(gi, _) if self.change_involves_colloids(change) => {
                self.update_single_colloid(*gi, context);
            }
            Change::Everything | Change::Volume(_, _) => {
                self.rebuild_colloids(context);
            }
            Change::Groups(groups) if self.change_involves_colloids(change) => {
                for &(gi, _) in groups {
                    if self.colloids.iter().any(|c| c.group_index == gi) {
                        self.update_single_colloid(gi, context);
                    }
                }
            }
            _ => return Ok(()),
        }
        // Re-solve self-consistency before energy evaluation: colloid positions
        // changed, so the neighbor sums and effective Robin parameters must update.
        if let Some(steric) = &mut self.steric {
            let lambda = self.kappa.sqrt() / self.rg;
            steric.solve(&self.colloids, lambda, context.cell());
        }
        let beta_energy = self.compute_beta_energy(&self.colloids, context.cell());
        self.cached_energy = beta_energy * self.thermal_energy;
        Ok(())
    }

    /// Compute the energy relevant to a change (kJ/mol).
    pub fn energy(&self, _context: &impl Context, change: &Change) -> f64 {
        if !self.change_involves_colloids(change) {
            return 0.0;
        }
        self.cached_energy
    }

    pub(super) fn save_backup(&mut self) {
        assert!(!self.has_backup, "backup already exists");
        self.backup_colloids.clear();
        self.backup_colloids.extend_from_slice(&self.colloids);
        self.backup_energy = self.cached_energy;
        if let Some(steric) = &self.steric {
            self.backup_g_s.clear();
            self.backup_g_s.extend_from_slice(&steric.g_s);
            self.backup_h_tilde_eff.clear();
            self.backup_h_tilde_eff
                .extend_from_slice(&steric.h_tilde_eff);
        }
        self.has_backup = true;
    }

    pub(super) fn undo(&mut self) {
        assert!(self.has_backup, "undo called without backup");
        std::mem::swap(&mut self.colloids, &mut self.backup_colloids);
        self.cached_energy = self.backup_energy;
        if let Some(steric) = &mut self.steric {
            std::mem::swap(&mut steric.g_s, &mut self.backup_g_s);
            std::mem::swap(&mut steric.h_tilde_eff, &mut self.backup_h_tilde_eff);
        }
        self.has_backup = false;
    }

    pub(super) fn discard_backup(&mut self) {
        self.has_backup = false;
    }

    /// Report key parameters and per-colloid bounding spheres as YAML.
    pub(super) fn to_yaml(&self) -> serde_yml::Value {
        let mut map = serde_yml::Mapping::new();
        map.insert("polymer_rg".into(), self.rg.into());
        map.insert("polymer_density".into(), self.rho_star.into());
        map.insert("kappa".into(), self.kappa.into());
        let molecules: Vec<serde_yml::Value> = self
            .colloid_molecule_names
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        map.insert("molecules".into(), serde_yml::Value::Sequence(molecules));
        if self.radius_scaling != 1.0 {
            map.insert("colloid_radius_scaling".into(), self.radius_scaling.into());
        }
        if let Some(r) = self.fixed_radius {
            map.insert("colloid_radius".into(), r.into());
        }
        if let Some(ht) = self.h_tilde {
            map.insert("h_tilde".into(), ht.into());
        }

        if let Some(steric) = &self.steric {
            let mut steric_map = serde_yml::Mapping::new();
            steric_map.insert("epsilon0_prime".into(), steric.config.epsilon0_prime.into());
            steric_map.insert("g0".into(), steric.config.g0.into());
            let gs_vals: Vec<serde_yml::Value> = steric.g_s.iter().map(|&v| v.into()).collect();
            steric_map.insert("g_s".into(), serde_yml::Value::Sequence(gs_vals));
            let ht_vals: Vec<serde_yml::Value> =
                steric.h_tilde_eff.iter().map(|&v| v.into()).collect();
            steric_map.insert("h_tilde_eff".into(), serde_yml::Value::Sequence(ht_vals));
            map.insert(
                "steric_adsorption".into(),
                serde_yml::Value::Mapping(steric_map),
            );
        }

        let colloids: Vec<serde_yml::Value> = self
            .colloids
            .iter()
            .map(|c| {
                let mut m = serde_yml::Mapping::new();
                m.insert("group".into(), (c.group_index as u64).into());
                m.insert("radius".into(), c.radius.into());
                serde_yml::Value::Mapping(m)
            })
            .collect();
        map.insert("colloids".into(), serde_yml::Value::Sequence(colloids));

        serde_yml::Value::Mapping(map)
    }

    /// Compute COM forces on all colloids (kJ/(mol·Å)).
    ///
    /// Returns `(group_index, force_vector)` for each colloid. The force is the
    /// negative gradient of the many-body energy with respect to the colloid COM:
    ///
    /// **F_m** = kT · 4πρ\* · λ · Σ_{j≠m} (1 + 1/(λR_mj)) · k₀(λR_mj) · û_mj
    ///           · \[A_m/D_m² + A_j/D_j²\]
    ///
    /// where A_i = σ_i² · exp(2σ_i) · κ^{-3/2} and
    ///       D_i = 1 + (exp(2σ_i) - 1) · S_i / 2.
    #[allow(dead_code)]
    pub fn forces(&self, cell: &impl crate::cell::BoundaryConditions) -> Vec<(usize, Point)> {
        // Configuration-dependent h̃_eff makes ∂f/∂R non-trivial; the force
        // expression needs additional ∂ε_eff/∂R terms not yet derived.
        if self.steric.is_some() {
            log::warn!("Analytical forces not implemented for steric adsorption; returning zeros");
            return self
                .colloids
                .iter()
                .map(|c| (c.group_index, Point::zeros()))
                .collect();
        }
        let colloids = &self.colloids;
        if colloids.is_empty() {
            return Vec::new();
        }
        let lambda = self.kappa.sqrt() / self.rg;
        let kappa_inv_3_2 = self.kappa.powf(-1.5);
        let prefactor = 4.0 * PI * self.rho_star * lambda * self.thermal_energy;

        // Pre-compute per-colloid quantities: A_i, S_i, D_i
        let per_colloid: Vec<(f64, f64)> = colloids
            .iter()
            .enumerate()
            .map(|(i, ci)| {
                let sigma = lambda * ci.radius;
                let f = robin_f(sigma, self.h_tilde);
                let sigma2 = sigma.powi(2);
                let e2s = (2.0 * sigma).exp();
                let a = f * f * sigma2 * e2s * kappa_inv_3_2;
                let k0_sum: f64 = colloids
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, cj)| k0(lambda * cell.distance(&ci.com, &cj.com).norm()))
                    .sum();
                let d = 1.0 + (e2s - 1.0) * k0_sum / 2.0;
                (a, d)
            })
            .collect();

        colloids
            .iter()
            .enumerate()
            .map(|(m, cm)| {
                let (a_m, d_m) = per_colloid[m];
                let mut force = Point::zeros();
                let d_m_sq = d_m.powi(2);
                for (j, cj) in colloids.iter().enumerate() {
                    if j == m {
                        continue;
                    }
                    let dvec = cell.distance(&cj.com, &cm.com);
                    let r = dvec.norm();
                    let lr = lambda * r;
                    let k0_val = k0(lr);
                    let (a_j, d_j) = per_colloid[j];
                    // dk₀/dx = -(1 + 1/x)·k₀(x); force uses the positive product
                    let weight = (1.0 + 1.0 / lr) * k0_val * (a_m / d_m_sq + a_j / d_j.powi(2));
                    // û_mj = dvec / r points from m toward j (attractive)
                    force += dvec * (weight / r);
                }
                (cm.group_index, force * prefactor)
            })
            .collect()
    }
}

/// YAML-deserializable builder for [`PolymerDepletion`].
///
/// # Example
/// ```yaml
/// energy:
///   polymer_depletion:
///     polymer_rg: 10.0
///     polymer_density: 0.5
///     kappa: 1.0
///     molecules: [Colloid]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolymerDepletionBuilder {
    /// Polymer radius of gyration R_g (Å)
    pub polymer_rg: f64,
    /// Reduced polymer reservoir density rho_P* (dimensionless)
    pub polymer_density: f64,
    /// Schulz-Flory order kappa = n + 1 (default: 1.0 for equilibrium polymers)
    #[serde(default = "default_kappa")]
    pub kappa: f64,
    /// Molecule type names treated as colloids
    pub molecules: Vec<String>,
    /// Optional fixed colloid radius (Å); default: bounding sphere radius
    #[serde(default)]
    pub colloid_radius: Option<f64>,
    /// Scaling factor for the effective colloid radius (default: 1.0)
    #[serde(default = "default_one")]
    pub colloid_radius_scaling: f64,
    /// Dimensionless Robin BC parameter h̃ = R_c / b, where b is the de Gennes
    /// extrapolation length. Controls polymer-surface affinity: large values give
    /// full depletion (Dirichlet limit); small positive values weaken it; omit for
    /// the original non-adsorbing model. Negative values model polymer adsorption
    /// onto the colloid surface.
    #[serde(default, alias = "h̃")]
    pub h_tilde: Option<f64>,
    /// Self-consistent steric adsorption configuration.
    /// Mutually exclusive with `h_tilde`.
    #[serde(default)]
    pub steric_adsorption: Option<StericAdsorptionConfig>,
}

fn default_kappa() -> f64 {
    1.0
}

fn default_one() -> f64 {
    1.0
}

fn default_picard_mixing() -> f64 {
    0.3
}

fn default_max_iterations() -> usize {
    50
}

fn default_tolerance() -> f64 {
    1e-8
}

/// Configuration for self-consistent steric adsorption boundary conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StericAdsorptionConfig {
    /// Bare adsorption parameter ε₀' (dimensionless, > 0)
    pub epsilon0_prime: f64,
    /// Saturation surface density g₀ (must be > 1.0)
    pub g0: f64,
    /// Picard mixing parameter α ∈ (0, 1]
    #[serde(default = "default_picard_mixing")]
    pub picard_mixing: f64,
    /// Maximum number of self-consistency iterations
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// Convergence tolerance for ĝ_S
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

/// Runtime state for self-consistent steric adsorption.
#[derive(Debug, Clone)]
struct StericAdsorptionState {
    config: StericAdsorptionConfig,
    /// Per-colloid surface density ĝ_S(i)
    g_s: Vec<f64>,
    /// Per-colloid effective Robin parameter h̃_eff(i) = -ε_eff(i)·R_c
    h_tilde_eff: Vec<f64>,
}

impl StericAdsorptionState {
    fn new(config: StericAdsorptionConfig, n_colloids: usize) -> Self {
        Self {
            config,
            g_s: vec![0.0; n_colloids],
            h_tilde_eff: vec![0.0; n_colloids],
        }
    }

    /// Run self-consistent iteration to convergence (Eqs. 14, 18, 19).
    fn solve(
        &mut self,
        colloids: &[ColloidInfo],
        lambda: f64,
        cell: &impl crate::cell::BoundaryConditions,
    ) {
        let n = colloids.len();
        if n == 0 {
            self.g_s.clear();
            self.h_tilde_eff.clear();
            return;
        }

        // Resize but preserve existing g_s values: after an MC move only one
        // colloid shifts, so the previous solution is a good initial guess
        // and the Picard iteration converges in very few steps.
        self.g_s.resize(n, 0.0);
        self.h_tilde_eff.resize(n, 0.0);

        let sqrt_4pi = (4.0 * PI).sqrt();
        let alpha = self.config.picard_mixing;
        // The renormalizing shift depends only on g0, so compute it once per solve
        // instead of inside the per-particle Picard updates.
        let delta_mu = beta_delta_mu(self.config.g0);
        // Clamp g_s strictly below g0 to keep epsilon_eff_from_gs's log finite;
        // without this, aggressive Picard mixing can overshoot past g0.
        let gs_max = self.config.g0 * (1.0 - 1e-12);

        // Neighbor sums depend only on colloid geometry (not on g_s), so
        // precompute once to avoid O(max_iter × N²) distance evaluations.
        let k0_sums: Vec<f64> = (0..n)
            .map(|i| {
                colloids
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, cj)| k0(lambda * cell.distance(&colloids[i].com, &cj.com).norm()))
                    .sum()
            })
            .collect();

        for _iter in 0..self.config.max_iterations {
            let mut max_change = 0.0_f64;

            for i in 0..n {
                let r_s = colloids[i].radius;
                let x_s = lambda * r_s;

                // ε_eff from current g_s (Eq. 14)
                let eps_eff = epsilon_eff_from_gs(
                    self.config.epsilon0_prime,
                    self.g_s[i],
                    self.config.g0,
                    delta_mu,
                );

                // Calligraphic functions at surface
                let cal_k = cal_k0(x_s, eps_eff, lambda);
                let cal_i = cal_i0(x_s, eps_eff, lambda);

                // Γ_S₀(i) from Eq. 19
                let gamma_s0 = -sqrt_4pi * eps_eff / (cal_k + cal_i * k0_sums[i]);

                // ĝ_S₀(i) from Eq. 18
                let gs_new = sqrt_4pi * lambda * i0_prime(x_s) / cal_i
                    + gamma_s0 / (cal_i * lambda * r_s.powi(2));

                // Picard mixing with saturation guard
                let gs_mixed =
                    (alpha * gs_new + (1.0 - alpha) * self.g_s[i]).clamp(-gs_max, gs_max);
                max_change = max_change.max((gs_mixed - self.g_s[i]).abs());
                self.g_s[i] = gs_mixed;

                // Recompute eps_eff from the mixed g_s so that h_tilde_eff stays
                // consistent with the current state (not the pre-mixing value).
                let eps_eff_updated = epsilon_eff_from_gs(
                    self.config.epsilon0_prime,
                    gs_mixed,
                    self.config.g0,
                    delta_mu,
                );
                self.h_tilde_eff[i] = -eps_eff_updated * r_s;
            }

            if max_change < self.config.tolerance {
                log::debug!(
                    "Steric adsorption converged in {} iterations (max_change={:.2e})",
                    _iter + 1,
                    max_change
                );
                break;
            }
        }
    }
}

impl PolymerDepletionBuilder {
    /// Build a [`PolymerDepletion`] energy term, resolving molecule names from the context.
    pub fn build(
        &self,
        context: &impl Context,
        thermal_energy: f64,
    ) -> anyhow::Result<PolymerDepletion> {
        let topology = context.topology();
        let colloid_molecule_ids: Vec<usize> = self
            .molecules
            .iter()
            .map(|name| {
                topology
                    .moleculekinds()
                    .iter()
                    .position(|m| m.name() == name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Molecule '{}' in polymer_depletion energy term does not exist",
                            name
                        )
                    })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Validate mutual exclusivity of h_tilde and steric_adsorption
        anyhow::ensure!(
            !(self.h_tilde.is_some() && self.steric_adsorption.is_some()),
            "h_tilde and steric_adsorption are mutually exclusive"
        );

        // Validate steric adsorption parameters
        if let Some(ref steric) = self.steric_adsorption {
            anyhow::ensure!(
                steric.g0 > 1.0,
                "steric_adsorption.g0 ({}) must be > 1.0",
                steric.g0
            );
            anyhow::ensure!(
                steric.epsilon0_prime > 0.0,
                "steric_adsorption.epsilon0_prime ({}) must be > 0",
                steric.epsilon0_prime
            );
        }

        // Validate Robin BC parameter: h̃ > -(1+σ) required for convergence
        if let Some(ht) = self.h_tilde {
            let lambda = self.kappa.sqrt() / self.polymer_rg;
            let rc = self.colloid_radius.unwrap_or(0.0) * self.colloid_radius_scaling;
            let sigma = lambda * rc;
            anyhow::ensure!(
                ht > -(1.0 + sigma),
                "h_tilde ({ht}) must be > -(1+σ) = {}, otherwise the monopole diverges",
                -(1.0 + sigma)
            );
        }

        let mut pm = PolymerDepletion {
            rg: self.polymer_rg,
            rho_star: self.polymer_density,
            kappa: self.kappa,
            colloid_molecule_ids,
            colloid_molecule_names: self.molecules.clone(),
            fixed_radius: self.colloid_radius,
            radius_scaling: self.colloid_radius_scaling,
            h_tilde: self.h_tilde,
            thermal_energy,
            colloids: Vec::new(),
            cached_energy: 0.0,
            backup_colloids: Vec::new(),
            backup_energy: 0.0,
            has_backup: false,
            steric: None,
            backup_g_s: Vec::new(),
            backup_h_tilde_eff: Vec::new(),
        };

        // Initialize colloids from current context state
        pm.rebuild_colloids(context);

        // Initialize steric adsorption if configured
        if let Some(ref config) = self.steric_adsorption {
            let mut state = StericAdsorptionState::new(config.clone(), pm.colloids.len());
            let lambda = self.kappa.sqrt() / self.polymer_rg;
            state.solve(&pm.colloids, lambda, context.cell());
            pm.steric = Some(state);
        }

        let beta_energy = pm.compute_beta_energy(&pm.colloids, context.cell());
        pm.cached_energy = beta_energy * pm.thermal_energy;

        log::info!(
            "Polymer depletion energy: R_g={}, rho*={}, kappa={}, h_tilde={}, {} colloid type(s), initial energy={:.4} kJ/mol",
            self.polymer_rg,
            self.polymer_density,
            self.kappa,
            match self.h_tilde { Some(h) => format!("{h}"), None => "inf".into() },
            self.molecules.len(),
            pm.cached_energy
        );

        Ok(pm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    /// Create a test instance with given physics parameters and colloid list.
    fn make_test_instance(
        rg: f64,
        rho_star: f64,
        kappa: f64,
        rc: f64,
        colloids: Vec<ColloidInfo>,
    ) -> PolymerDepletion {
        make_test_instance_robin(rg, rho_star, kappa, rc, None, colloids)
    }

    fn make_test_instance_robin(
        rg: f64,
        rho_star: f64,
        kappa: f64,
        rc: f64,
        h_tilde: Option<f64>,
        colloids: Vec<ColloidInfo>,
    ) -> PolymerDepletion {
        PolymerDepletion {
            rg,
            rho_star,
            kappa,
            colloid_molecule_ids: vec![0],
            colloid_molecule_names: vec!["Test".to_string()],
            fixed_radius: Some(rc),
            radius_scaling: 1.0,
            h_tilde,
            thermal_energy: 1.0,
            colloids,
            cached_energy: 0.0,
            backup_colloids: Vec::new(),
            backup_energy: 0.0,
            has_backup: false,
            steric: None,
            backup_g_s: Vec::new(),
            backup_h_tilde_eff: Vec::new(),
        }
    }

    fn make_test_instance_steric(
        rg: f64,
        rho_star: f64,
        kappa: f64,
        rc: f64,
        steric_config: StericAdsorptionConfig,
        colloids: Vec<ColloidInfo>,
    ) -> PolymerDepletion {
        let lambda = kappa.sqrt() / rg;
        let mut state = StericAdsorptionState::new(steric_config, colloids.len());
        let cell = crate::cell::Endless;
        state.solve(&colloids, lambda, &cell);
        let mut pm = make_test_instance_robin(rg, rho_star, kappa, rc, None, colloids);
        pm.steric = Some(state);
        pm
    }

    #[test]
    fn test_k0_kernel() {
        // k_0(1) = exp(-1)/1
        assert_approx_eq!(f64, k0(1.0), (-1.0_f64).exp(), epsilon = 1e-15);
        // k_0(2) = exp(-2)/2
        assert_approx_eq!(f64, k0(2.0), (-2.0_f64).exp() / 2.0, epsilon = 1e-15);
        // Large x -> 0
        assert!(k0(100.0) < 1e-40);
    }

    #[test]
    fn test_single_colloid() {
        // N_c=1: pairwise sum = 0, energy = single-particle insertion only
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);

        let pm = make_test_instance(
            rg,
            rho_star,
            kappa,
            rc,
            vec![ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            }],
        );

        let cell = crate::cell::Endless;
        let beta_e = pm.compute_beta_energy(&pm.colloids, &cell);

        let lambda = kappa.sqrt() / rg;
        let sigma = lambda * rc;
        let expected =
            4.0 * PI * rho_star * kappa.powf(-1.5) * (sigma + sigma * sigma + sigma.powi(3) / 3.0);

        assert_approx_eq!(f64, beta_e, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_two_colloids_large_separation() {
        // At large R, k_0(lambda*R) -> 0, so energy -> 2x single-particle
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let r_large = 1000.0_f64;

        let colloids = vec![
            ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 1,
                com: Point::new(r_large, 0.0, 0.0),
                radius: rc,
            },
        ];

        let pm = make_test_instance(rg, rho_star, kappa, rc, colloids.clone());

        let cell = crate::cell::Endless;
        let beta_e = pm.compute_beta_energy(&colloids, &cell);

        // Single-particle energy for one colloid
        let lambda = kappa.sqrt() / rg;
        let sigma = lambda * rc;
        let single =
            4.0 * PI * rho_star * kappa.powf(-1.5) * (sigma + sigma * sigma + sigma.powi(3) / 3.0);

        // At large separation, total should be ~2x single
        assert_approx_eq!(f64, beta_e, 2.0 * single, epsilon = 1e-6);
    }

    #[test]
    fn test_two_colloids_vs_pair_approximation() {
        // For kappa=1, compare pairwise interaction with eq 19:
        // beta*dw_pair(R) = -8*pi*rho_P* * (Rc/Rg)^2 * exp(-(R-2Rc)/Rg) / (R/Rg)
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);

        let cell = crate::cell::Endless;
        let pm = make_test_instance(rg, rho_star, kappa, rc, Vec::new());

        // Single-particle energy
        let lambda = kappa.sqrt() / rg;
        let sigma = lambda * rc;
        let single =
            4.0 * PI * rho_star * kappa.powf(-1.5) * (sigma + sigma * sigma + sigma.powi(3) / 3.0);

        // Test at several separations
        for &r in &[12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0] {
            let colloids = vec![
                ColloidInfo {
                    group_index: 0,
                    com: Point::new(0.0, 0.0, 0.0),
                    radius: rc,
                },
                ColloidInfo {
                    group_index: 1,
                    com: Point::new(r, 0.0, 0.0),
                    radius: rc,
                },
            ];

            let beta_total = pm.compute_beta_energy(&colloids, &cell);
            let beta_pair = beta_total - 2.0 * single;

            // Eq 19 pair approximation (leading order, valid when k_0 << 1)
            let eq19 =
                -8.0 * PI * rho_star * (rc / rg).powi(2) * (-(r - 2.0 * rc) / rg).exp() / (r / rg);

            // Pair approximation is leading-order; agreement improves at
            // larger separations where k_0(lambda*R) << 1.
            if r >= 40.0 {
                assert_approx_eq!(f64, beta_pair, eq19, epsilon = eq19.abs() * 0.02);
            }
            // At all distances the pair interaction should be attractive (negative)
            assert!(
                beta_pair < 0.0,
                "pair interaction should be attractive at R={r}"
            );
            // and the pair approximation should also be negative
            assert!(eq19 < 0.0, "eq19 should be negative at R={r}");
        }
    }

    #[test]
    fn test_yaml_roundtrip() {
        let yaml = r#"
polymer_rg: 10.0
polymer_density: 0.5
kappa: 2.0
molecules: [Colloid, Sphere]
colloid_radius: 5.0
"#;
        let builder: PolymerDepletionBuilder = serde_yml::from_str(yaml).unwrap();
        assert_approx_eq!(f64, builder.polymer_rg, 10.0);
        assert_approx_eq!(f64, builder.polymer_density, 0.5);
        assert_approx_eq!(f64, builder.kappa, 2.0);
        assert_eq!(builder.molecules, vec!["Colloid", "Sphere"]);
        assert_approx_eq!(f64, builder.colloid_radius.unwrap(), 5.0);

        // Default kappa
        let yaml2 = r#"
polymer_rg: 10.0
polymer_density: 0.5
molecules: [Colloid]
"#;
        let builder2: PolymerDepletionBuilder = serde_yml::from_str(yaml2).unwrap();
        assert_approx_eq!(f64, builder2.kappa, 1.0);
        assert!(builder2.colloid_radius.is_none());
    }

    /// Check analytical forces against central finite differences.
    fn assert_forces_match_finite_differences(pm: &PolymerDepletion, colloids: &[ColloidInfo]) {
        let cell = crate::cell::Endless;
        let forces = pm.forces(&cell);
        let h = 1e-5;
        for (idx, (group_index, analytical_force)) in forces.iter().enumerate() {
            assert_eq!(*group_index, colloids[idx].group_index);
            for axis in 0..3 {
                let mut colloids_plus = colloids.to_vec();
                let mut colloids_minus = colloids.to_vec();
                colloids_plus[idx].com[axis] += h;
                colloids_minus[idx].com[axis] -= h;

                let e_plus = pm.compute_beta_energy(&colloids_plus, &cell) * pm.thermal_energy;
                let e_minus = pm.compute_beta_energy(&colloids_minus, &cell) * pm.thermal_energy;
                let numerical_force = -(e_plus - e_minus) / (2.0 * h);

                assert_approx_eq!(
                    f64,
                    analytical_force[axis],
                    numerical_force,
                    epsilon = 1e-5 * analytical_force[axis].abs().max(1e-10)
                );
            }
        }
    }

    /// Verify analytical COM forces against central finite differences for a
    /// 3-body system with asymmetric geometry (breaks any pairwise cancellation).
    #[test]
    fn test_forces_vs_finite_difference_3body() {
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let colloids = vec![
            ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 1,
                com: Point::new(15.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 2,
                com: Point::new(5.0, 12.0, 3.0),
                radius: rc,
            },
        ];

        let pm = make_test_instance(rg, rho_star, kappa, rc, colloids.clone());
        assert_forces_match_finite_differences(&pm, &colloids);
    }

    /// Forces should be zero for a single colloid (no pairwise interactions).
    #[test]
    fn test_forces_single_colloid_zero() {
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let pm = make_test_instance(
            rg,
            rho_star,
            kappa,
            rc,
            vec![ColloidInfo {
                group_index: 0,
                com: Point::new(1.0, 2.0, 3.0),
                radius: rc,
            }],
        );
        let forces = pm.forces(&crate::cell::Endless);
        assert_eq!(forces.len(), 1);
        assert_approx_eq!(f64, forces[0].1.norm(), 0.0, epsilon = 1e-15);
    }

    /// Verify forces with kappa != 1 (Schulz-Flory polydispersity).
    #[test]
    fn test_forces_vs_finite_difference_kappa3() {
        let (rg, rho_star, kappa, rc) = (8.0, 0.3, 3.0, 6.0);
        let colloids = vec![
            ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 1,
                com: Point::new(14.0, 3.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 2,
                com: Point::new(-2.0, 10.0, 5.0),
                radius: rc,
            },
        ];

        let pm = make_test_instance(rg, rho_star, kappa, rc, colloids.clone());
        assert_forces_match_finite_differences(&pm, &colloids);
    }

    #[test]
    fn test_robin_f_limits() {
        // Dirichlet limit: h_tilde = None -> f = 1
        assert_approx_eq!(f64, robin_f(1.0, None), 1.0);
        // Neumann limit: h_tilde = 0 -> f = 0
        assert_approx_eq!(f64, robin_f(1.0, Some(0.0)), 0.0);
        // Large h_tilde -> f ~ 1
        assert_approx_eq!(f64, robin_f(1.0, Some(1e6)), 1.0, epsilon = 1e-5);
        // Analytical check: f(σ=1, h̃=3) = 3 / (2 + 3) = 0.6
        assert_approx_eq!(f64, robin_f(1.0, Some(3.0)), 0.6);
        // Protein limit σ→0: f -> h̃/(1+h̃)
        assert_approx_eq!(f64, robin_f(1e-10, Some(1.0)), 0.5, epsilon = 1e-9);
    }

    #[test]
    fn test_robin_none_matches_dirichlet() {
        // Energy with h_tilde=None must match the original Dirichlet model exactly
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let colloids = vec![
            ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 1,
                com: Point::new(15.0, 0.0, 0.0),
                radius: rc,
            },
        ];
        let pm_dirichlet = make_test_instance(rg, rho_star, kappa, rc, colloids.clone());
        let pm_robin_none =
            make_test_instance_robin(rg, rho_star, kappa, rc, None, colloids.clone());
        let cell = crate::cell::Endless;
        assert_approx_eq!(
            f64,
            pm_dirichlet.compute_beta_energy(&colloids, &cell),
            pm_robin_none.compute_beta_energy(&colloids, &cell),
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_robin_suppresses_energy() {
        // Finite h_tilde must reduce the magnitude of both insertion and pair terms
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let colloids = vec![
            ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 1,
                com: Point::new(15.0, 0.0, 0.0),
                radius: rc,
            },
        ];
        let cell = crate::cell::Endless;
        let e_dirichlet = make_test_instance(rg, rho_star, kappa, rc, colloids.clone())
            .compute_beta_energy(&colloids, &cell);
        let e_robin =
            make_test_instance_robin(rg, rho_star, kappa, rc, Some(3.0), colloids.clone())
                .compute_beta_energy(&colloids, &cell);
        // Robin energy should be smaller in magnitude (closer to zero)
        assert!(
            e_robin.abs() < e_dirichlet.abs(),
            "Robin energy {e_robin} should be smaller than Dirichlet {e_dirichlet}"
        );
        assert!(
            e_robin > 0.0,
            "Robin energy should remain positive (insertion dominates)"
        );
    }

    #[test]
    fn test_robin_analytical_single_particle() {
        // Single colloid: energy = 4π·ρ*·κ^{-3/2}·f·(σ + σ² + σ³/3)
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let ht = 3.0;
        let pm = make_test_instance_robin(
            rg,
            rho_star,
            kappa,
            rc,
            Some(ht),
            vec![ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            }],
        );
        let cell = crate::cell::Endless;
        let beta_e = pm.compute_beta_energy(&pm.colloids, &cell);

        let lambda = kappa.sqrt() / rg;
        let sigma = lambda * rc;
        let f = ht / ((1.0 + sigma) + ht);
        let expected = 4.0
            * PI
            * rho_star
            * kappa.powf(-1.5)
            * f
            * (sigma + sigma * sigma + sigma.powi(3) / 3.0);
        assert_approx_eq!(f64, beta_e, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_robin_forces_vs_finite_difference() {
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let colloids = vec![
            ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 1,
                com: Point::new(15.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 2,
                com: Point::new(5.0, 12.0, 3.0),
                radius: rc,
            },
        ];
        let pm = make_test_instance_robin(rg, rho_star, kappa, rc, Some(3.0), colloids.clone());
        assert_forces_match_finite_differences(&pm, &colloids);
    }

    #[test]
    fn test_robin_yaml_roundtrip() {
        let yaml = r#"
polymer_rg: 10.0
polymer_density: 0.5
kappa: 2.0
molecules: [Colloid]
h_tilde: 5.0
"#;
        let builder: PolymerDepletionBuilder = serde_yml::from_str(yaml).unwrap();
        assert_approx_eq!(f64, builder.h_tilde.unwrap(), 5.0);

        // Without h_tilde -> None (Dirichlet)
        let yaml2 = r#"
polymer_rg: 10.0
polymer_density: 0.5
molecules: [Colloid]
"#;
        let builder2: PolymerDepletionBuilder = serde_yml::from_str(yaml2).unwrap();
        assert!(builder2.h_tilde.is_none());
    }

    #[test]
    fn test_i0_and_derivatives() {
        // i₀(1) = sinh(1)/1
        assert_approx_eq!(f64, i0(1.0), 1.0_f64.sinh(), epsilon = 1e-14);
        // i₀(2) = sinh(2)/2
        assert_approx_eq!(f64, i0(2.0), 2.0_f64.sinh() / 2.0, epsilon = 1e-14);
        // i₀(0) -> 1
        assert_approx_eq!(f64, i0(1e-15), 1.0, epsilon = 1e-10);

        // i₀'(1) = cosh(1)/1 - sinh(1)/1
        let expected_i0p = 1.0_f64.cosh() - 1.0_f64.sinh();
        assert_approx_eq!(f64, i0_prime(1.0), expected_i0p, epsilon = 1e-14);
        // i₀'(0) -> 0
        assert_approx_eq!(f64, i0_prime(1e-15), 0.0, epsilon = 1e-10);

        // k₀'(1) = -exp(-1)·2/1 = -2·exp(-1)
        let expected_k0p = -(-1.0_f64).exp() * 2.0;
        assert_approx_eq!(f64, k0_prime(1.0), expected_k0p, epsilon = 1e-14);
        // k₀'(2) = -exp(-2)·3/4
        let expected_k0p2 = -(-2.0_f64).exp() * 3.0 / 4.0;
        assert_approx_eq!(f64, k0_prime(2.0), expected_k0p2, epsilon = 1e-14);

        // Verify k₀' numerically
        let h = 1e-7;
        let numerical_k0p = (k0(1.0 + h) - k0(1.0 - h)) / (2.0 * h);
        assert_approx_eq!(f64, k0_prime(1.0), numerical_k0p, epsilon = 1e-6);

        // Verify i₀' numerically
        let numerical_i0p = (i0(1.0 + h) - i0(1.0 - h)) / (2.0 * h);
        assert_approx_eq!(f64, i0_prime(1.0), numerical_i0p, epsilon = 1e-6);
    }

    #[test]
    fn test_epsilon_eff_limits() {
        let delta_mu = beta_delta_mu(10.0);

        // ĝ_S → 0 gives ε₀' - βδμ
        assert_approx_eq!(
            f64,
            epsilon_eff_from_gs(0.02, 0.0, 10.0, delta_mu),
            0.02 - delta_mu,
            epsilon = 1e-14
        );

        // ĝ_S → g₀ gives -∞
        let eps = epsilon_eff_from_gs(0.02, 9.999, 10.0, delta_mu);
        assert!(
            eps < -100.0,
            "ε_eff should diverge to -∞ near g₀, got {eps}"
        );

        // Intermediate value: check sign and magnitude
        let eps_mid = epsilon_eff_from_gs(0.02, 5.0, 10.0, delta_mu);
        let expected = 0.02 + (0.75_f64).ln() - 2.0 * 0.25 / 0.75 - delta_mu;
        assert_approx_eq!(f64, eps_mid, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_steric_single_particle_convergence() {
        let (rg, _rho_star, kappa, rc) = (10.0_f64, 0.5, 1.0, 5.0);
        let config = StericAdsorptionConfig {
            epsilon0_prime: 0.02,
            g0: 10.0,
            picard_mixing: 0.3,
            max_iterations: 200,
            tolerance: 1e-12,
        };

        let colloids = vec![ColloidInfo {
            group_index: 0,
            com: Point::new(0.0, 0.0, 0.0),
            radius: rc,
        }];

        let lambda = f64::sqrt(kappa) / rg;
        let mut state = StericAdsorptionState::new(config, 1);
        let cell = crate::cell::Endless;
        state.solve(&colloids, lambda, &cell);

        // Should converge to a finite value
        assert!(
            state.g_s[0].is_finite(),
            "g_s should be finite, got {}",
            state.g_s[0]
        );
        assert!(
            state.h_tilde_eff[0].is_finite(),
            "h_tilde_eff should be finite, got {}",
            state.h_tilde_eff[0]
        );

        // For small ε₀', g_s should be small
        assert!(
            state.g_s[0].abs() < 5.0,
            "g_s should be moderate for small ε₀', got {}",
            state.g_s[0]
        );
    }

    #[test]
    fn test_steric_prevents_divergence() {
        // Large ε₀' with steric should produce finite energy
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let config = StericAdsorptionConfig {
            epsilon0_prime: 5.0,
            g0: 10.0,
            picard_mixing: 0.1,
            max_iterations: 500,
            tolerance: 1e-10,
        };

        let colloids = vec![
            ColloidInfo {
                group_index: 0,
                com: Point::new(0.0, 0.0, 0.0),
                radius: rc,
            },
            ColloidInfo {
                group_index: 1,
                com: Point::new(15.0, 0.0, 0.0),
                radius: rc,
            },
        ];

        let pm = make_test_instance_steric(rg, rho_star, kappa, rc, config, colloids.clone());
        let cell = crate::cell::Endless;
        let energy = pm.compute_beta_energy(&colloids, &cell);

        assert!(
            energy.is_finite(),
            "Steric energy should be finite even for large ε₀', got {energy}"
        );
    }

    #[test]
    fn test_steric_energy_finite_various_eps0p() {
        // Self-consistent steric scheme should produce finite energies
        // across a range of ε₀' values
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let cell = crate::cell::Endless;

        let colloids = vec![ColloidInfo {
            group_index: 0,
            com: Point::new(0.0, 0.0, 0.0),
            radius: rc,
        }];

        for &eps0p in &[0.001, 0.01, 0.1, 1.0, 5.0] {
            let config = StericAdsorptionConfig {
                epsilon0_prime: eps0p,
                g0: 10.0,
                picard_mixing: 0.2,
                max_iterations: 500,
                tolerance: 1e-10,
            };
            let pm = make_test_instance_steric(rg, rho_star, kappa, rc, config, colloids.clone());
            let e = pm.compute_beta_energy(&colloids, &cell);
            assert!(
                e.is_finite(),
                "Energy should be finite for ε₀'={eps0p}, got {e}"
            );
            // h_tilde_eff should also be finite
            assert!(
                pm.steric.as_ref().unwrap().h_tilde_eff[0].is_finite(),
                "h_tilde_eff should be finite for ε₀'={eps0p}"
            );
        }
    }

    #[test]
    fn test_steric_backup_undo() {
        let (rg, rho_star, kappa, rc) = (10.0, 0.5, 1.0, 5.0);
        let config = StericAdsorptionConfig {
            epsilon0_prime: 0.02,
            g0: 10.0,
            picard_mixing: 0.3,
            max_iterations: 50,
            tolerance: 1e-8,
        };

        let colloids = vec![ColloidInfo {
            group_index: 0,
            com: Point::new(0.0, 0.0, 0.0),
            radius: rc,
        }];

        let mut pm = make_test_instance_steric(rg, rho_star, kappa, rc, config, colloids);

        // Save state
        let original_gs = pm.steric.as_ref().unwrap().g_s.clone();
        let original_ht = pm.steric.as_ref().unwrap().h_tilde_eff.clone();
        pm.save_backup();

        // Modify state (simulate an update)
        if let Some(steric) = &mut pm.steric {
            steric.g_s[0] = 999.0;
            steric.h_tilde_eff[0] = 999.0;
        }

        // Undo should restore
        pm.undo();
        assert_approx_eq!(
            f64,
            pm.steric.as_ref().unwrap().g_s[0],
            original_gs[0],
            epsilon = 1e-15
        );
        assert_approx_eq!(
            f64,
            pm.steric.as_ref().unwrap().h_tilde_eff[0],
            original_ht[0],
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_steric_yaml_roundtrip() {
        let yaml = r#"
polymer_rg: 100.0
polymer_density: 1.0
kappa: 1.0
molecules: [Colloid]
colloid_radius: 5.0
steric_adsorption:
  epsilon0_prime: 0.02
  g0: 10.0
"#;
        let builder: PolymerDepletionBuilder = serde_yml::from_str(yaml).unwrap();
        let steric = builder.steric_adsorption.as_ref().unwrap();
        assert_approx_eq!(f64, steric.epsilon0_prime, 0.02);
        assert_approx_eq!(f64, steric.g0, 10.0);
        assert_approx_eq!(f64, steric.picard_mixing, 0.3);
        assert_eq!(steric.max_iterations, 50);
        assert!(builder.h_tilde.is_none());
    }

    #[test]
    fn test_steric_mutual_exclusion() {
        let yaml = r#"
polymer_rg: 100.0
polymer_density: 1.0
kappa: 1.0
molecules: [Colloid]
colloid_radius: 5.0
h_tilde: 3.0
steric_adsorption:
  epsilon0_prime: 0.02
  g0: 10.0
"#;
        let builder: PolymerDepletionBuilder = serde_yml::from_str(yaml).unwrap();
        // Can't test build() without a Context, but verify both are set
        assert!(builder.h_tilde.is_some());
        assert!(builder.steric_adsorption.is_some());
        // The build() method would reject this with an error
    }
}
