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

/// Per-colloid information cached from the context.
#[derive(Debug, Clone)]
struct ColloidInfo {
    group_index: usize,
    com: Point,
    radius: f64,
}

/// Many-body polymer depletion interaction for colloids in an ideal polymer
/// fluid.
///
/// Implements the Hamiltonian of
/// [Forsman & Woodward, Soft Matter, 2012, 8, 2121](https://doi.org/10.1039/c2sm06737d)
/// (eq 17). Rigid macromolecules are treated as neutral spheres via their
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

            let sigma2 = sigma.powi(2);

            // Single-particle insertion term (positive, unfavorable)
            energy += kappa_inv_3_2 * (sigma + sigma2 + sigma2 * sigma / 3.0);

            // Many-body pairwise sum (negative, attractive depletion)
            let k0_sum: f64 = colloids
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, cj)| k0(lambda * cell.distance(&ci.com, &cj.com).norm()))
                .sum();

            let e2s = (2.0 * sigma).exp();
            let denom = 1.0 + (e2s - 1.0) * k0_sum / 2.0;
            energy -= sigma2 * e2s * kappa_inv_3_2 * k0_sum / denom;
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
        self.has_backup = true;
    }

    pub(super) fn undo(&mut self) {
        assert!(self.has_backup, "undo called without backup");
        std::mem::swap(&mut self.colloids, &mut self.backup_colloids);
        self.cached_energy = self.backup_energy;
        self.has_backup = false;
    }

    pub(super) fn discard_backup(&mut self) {
        self.has_backup = false;
    }

    /// Report key parameters and per-colloid bounding spheres as YAML.
    pub(super) fn to_yaml(&self) -> serde_yaml::Value {
        let mut map = serde_yaml::Mapping::new();
        map.insert("polymer_rg".into(), self.rg.into());
        map.insert("polymer_density".into(), self.rho_star.into());
        map.insert("kappa".into(), self.kappa.into());
        let molecules: Vec<serde_yaml::Value> = self
            .colloid_molecule_names
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        map.insert("molecules".into(), serde_yaml::Value::Sequence(molecules));
        if self.radius_scaling != 1.0 {
            map.insert("colloid_radius_scaling".into(), self.radius_scaling.into());
        }
        if let Some(r) = self.fixed_radius {
            map.insert("colloid_radius".into(), r.into());
        }

        let colloids: Vec<serde_yaml::Value> = self
            .colloids
            .iter()
            .map(|c| {
                let mut m = serde_yaml::Mapping::new();
                m.insert("group".into(), (c.group_index as u64).into());
                m.insert("radius".into(), c.radius.into());
                serde_yaml::Value::Mapping(m)
            })
            .collect();
        map.insert("colloids".into(), serde_yaml::Value::Sequence(colloids));

        serde_yaml::Value::Mapping(map)
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
                let sigma2 = sigma.powi(2);
                let e2s = (2.0 * sigma).exp();
                let a = sigma2 * e2s * kappa_inv_3_2;
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
}

fn default_kappa() -> f64 {
    1.0
}

fn default_one() -> f64 {
    1.0
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

        let mut pm = PolymerDepletion {
            rg: self.polymer_rg,
            rho_star: self.polymer_density,
            kappa: self.kappa,
            colloid_molecule_ids,
            colloid_molecule_names: self.molecules.clone(),
            fixed_radius: self.colloid_radius,
            radius_scaling: self.colloid_radius_scaling,
            thermal_energy,
            colloids: Vec::new(),
            cached_energy: 0.0,
            backup_colloids: Vec::new(),
            backup_energy: 0.0,
            has_backup: false,
        };

        // Initialize colloids from current context state
        pm.rebuild_colloids(context);
        let beta_energy = pm.compute_beta_energy(&pm.colloids, context.cell());
        pm.cached_energy = beta_energy * pm.thermal_energy;

        log::info!(
            "Polymer depletion energy: R_g={}, rho*={}, kappa={}, {} colloid type(s), initial energy={:.4} kJ/mol",
            self.polymer_rg,
            self.polymer_density,
            self.kappa,
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
        PolymerDepletion {
            rg,
            rho_star,
            kappa,
            colloid_molecule_ids: vec![0],
            colloid_molecule_names: vec!["Test".to_string()],
            fixed_radius: Some(rc),
            radius_scaling: 1.0,
            thermal_energy: 1.0,
            colloids,
            cached_energy: 0.0,
            backup_colloids: Vec::new(),
            backup_energy: 0.0,
            has_backup: false,
        }
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
        let builder: PolymerDepletionBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder2: PolymerDepletionBuilder = serde_yaml::from_str(yaml2).unwrap();
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
}
