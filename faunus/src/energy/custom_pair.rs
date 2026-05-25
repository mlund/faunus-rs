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

//! User-defined energy between two rigid-body centers of mass.
//!
//! Evaluates `f(r, dx, dy, dz)` between the COMs of two `selection`s and
//! contributes both energy (consumed by Metropolis MC) and forces (consumed
//! by LD via the Hamiltonian's `forces()` path). The force on each rigid body
//! is `-dU/dr · d̂`, distributed to its atoms by mass fraction so the COM
//! force is exact and the torque about the COM is identically zero — pure
//! translation for the rigid integrator.
//!
//! For v1, each selection must resolve to exactly one rigid molecule
//! (`degrees_of_freedom: Rigid`); other selection patterns are rejected
//! at build time.

use exmex::{Express, FlatEx, FlatExVal, Val};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::cell::BoundaryConditions;
use crate::selection::Selection;
use crate::{Change, Context};

use super::expr_helpers::substitute_constants;
use super::EnergyTerm;

/// Numerical step (Å) for central-difference dU/dr. Two evaluations per call.
const DEFAULT_GRADIENT_H: f64 = 1.0e-5;

/// Variables exposed to the expression, alphabetical (exmex convention).
const ALLOWED_VARS: [&str; 4] = ["dx", "dy", "dz", "r"];

/// COMs closer than this are treated as overlapping; energy and force are zero
/// (the unit vector d/r is undefined). 1 fm is far below any sensible step.
const MIN_SEPARATION: f64 = 1.0e-12;

/// Massless or near-massless groups are skipped to avoid division-by-zero
/// when computing the per-atom mass fraction.
const MIN_MASS: f64 = 1.0e-30;

/// YAML deserializer for a single `custompair` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomPairBuilder {
    selection1: Selection,
    selection2: Selection,
    function: String,
    #[serde(default)]
    constants: HashMap<String, f64>,
    #[serde(default = "default_gradient_h")]
    gradient_h: f64,
}

const fn default_gradient_h() -> f64 {
    DEFAULT_GRADIENT_H
}

impl CustomPairBuilder {
    pub(crate) fn build(&self, context: &impl Context) -> anyhow::Result<CustomPair> {
        let group1 = resolve_unique_rigid_group(context, &self.selection1, "selection1")?;
        let group2 = resolve_unique_rigid_group(context, &self.selection2, "selection2")?;
        if group1 == group2 {
            anyhow::bail!(
                "custompair: selection1 and selection2 resolved to the same group ({group1})"
            );
        }

        let substituted = substitute_constants(&self.function, &self.constants);
        let (expression, var_names) = parse_expression(&substituted)?;
        let var_indices = map_var_indices(&var_names)?;

        log::info!(
            "Custom pair potential: '{}' (selection1='{}', selection2='{}')",
            self.function,
            self.selection1,
            self.selection2
        );

        Ok(CustomPair {
            expression: Arc::new(expression),
            function: self.function.clone(),
            var_indices,
            group1,
            group2,
            gradient_h: self.gradient_h,
            selection1: self.selection1.clone(),
            selection2: self.selection2.clone(),
        })
    }
}

/// Resolve a selection to exactly one rigid-body group, or fail.
fn resolve_unique_rigid_group(
    context: &impl Context,
    selection: &Selection,
    label: &str,
) -> anyhow::Result<usize> {
    let groups = context.resolve_groups_live(selection);
    if groups.len() != 1 {
        anyhow::bail!(
            "custompair: {label} '{selection}' must resolve to exactly one group, got {} ({:?})",
            groups.len(),
            groups
        );
    }
    let gi = groups[0];
    let mol_id = context.groups()[gi].molecule();
    let topology = context.topology_ref();
    let dof = topology.moleculekinds()[mol_id].degrees_of_freedom();
    if !dof.is_rigid() {
        anyhow::bail!(
            "custompair: {label} '{selection}' resolved to molecule '{}' with {:?} degrees of freedom; \
             only Rigid molecules are supported",
            topology.moleculekinds()[mol_id].name(),
            dof
        );
    }
    Ok(gi)
}

/// Compiled expression backend.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum Expression {
    Float(FlatEx<f64>),
    Val(FlatExVal<i32, f64>),
}

fn parse_expression(substituted: &str) -> anyhow::Result<(Expression, Vec<String>)> {
    let has_conditionals = substituted.contains(" if ") || substituted.contains(" else ");
    if has_conditionals {
        let expr: FlatExVal<i32, f64> = exmex::parse_val(substituted)
            .map_err(|e| anyhow::anyhow!("custompair expression parse error: {e}"))?;
        let names = expr.var_names().to_vec();
        Ok((Expression::Val(expr), names))
    } else {
        let expr: FlatEx<f64> = FlatEx::parse(substituted)
            .map_err(|e| anyhow::anyhow!("custompair expression parse error: {e}"))?;
        let names = expr.var_names().to_vec();
        Ok((Expression::Float(expr), names))
    }
}

/// Map each exmex variable slot (in alphabetical order) to its index in `ALLOWED_VARS`.
fn map_var_indices(var_names: &[String]) -> anyhow::Result<Vec<usize>> {
    let bad: Vec<_> = var_names
        .iter()
        .filter(|v| !ALLOWED_VARS.contains(&v.as_str()))
        .cloned()
        .collect();
    if !bad.is_empty() {
        anyhow::bail!(
            "unresolved variables in custompair: {} (allowed: r, dx, dy, dz)",
            bad.join(", ")
        );
    }
    Ok(var_names
        .iter()
        .map(|name| {
            ALLOWED_VARS
                .iter()
                .position(|&a| a == name)
                .expect("var_names already filtered against ALLOWED_VARS")
        })
        .collect())
}

/// Compiled custom pair-COM energy term.
#[derive(Debug, Clone)]
pub struct CustomPair {
    expression: Arc<Expression>,
    function: String,
    var_indices: Vec<usize>,
    group1: usize,
    group2: usize,
    gradient_h: f64,
    selection1: Selection,
    selection2: Selection,
}

impl CustomPair {
    /// Energy U = f(dx, dy, dz, r) at the current COM-COM separation.
    /// MC consumes this; LD never calls it (uses `forces()` instead).
    pub(crate) fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        if matches!(change, Change::None) {
            return 0.0;
        }
        let (d, r) = self.com_separation(context);
        if r < MIN_SEPARATION {
            return 0.0;
        }
        self.eval_at(d.x, d.y, d.z, r)
    }

    /// Per-atom forces from this term, indexed by absolute particle index.
    /// Mass-weighted distribution gives the rigid-body integrator pure translation
    /// (zero torque about each COM) — see distribute_force().
    pub(crate) fn forces(&self, context: &impl Context) -> Vec<crate::Point> {
        let n_atoms = context
            .groups()
            .iter()
            .map(|g| g.iter_active().end)
            .max()
            .unwrap_or(0);
        let mut forces = vec![crate::Point::zeros(); n_atoms];

        let (d, r) = self.com_separation(context);
        if r < MIN_SEPARATION {
            return forces;
        }
        let dudr = self.dudr_numerical(r, d.x, d.y, d.z);
        let f_on_g1 = -dudr * d / r;
        distribute_force(context, self.group1, f_on_g1, &mut forces);
        distribute_force(context, self.group2, -f_on_g1, &mut forces);
        forces
    }

    pub(super) fn to_yaml(&self) -> serde_yml::Value {
        let mut map = serde_yml::Mapping::new();
        map.insert("function".into(), self.function.clone().into());
        map.insert("selection1".into(), self.selection1.to_string().into());
        map.insert("selection2".into(), self.selection2.to_string().into());
        serde_yml::Value::Mapping(map)
    }

    /// Minimum-image vector com1 − com2 and its norm.
    fn com_separation(&self, context: &impl Context) -> (crate::Point, f64) {
        let com1 = group_com(context, self.group1);
        let com2 = group_com(context, self.group2);
        let d = context.cell().distance(&com1, &com2);
        let r = d.norm();
        (d, r)
    }

    /// Evaluate the parsed expression at given (dx, dy, dz, r).
    fn eval_at(&self, dx: f64, dy: f64, dz: f64, r: f64) -> f64 {
        let all = [dx, dy, dz, r];
        let n = self.var_indices.len();
        match self.expression.as_ref() {
            Expression::Float(expr) => {
                let mut vals = [0.0_f64; 4];
                for (i, &vi) in self.var_indices.iter().enumerate() {
                    vals[i] = all[vi];
                }
                expr.eval(&vals[..n]).unwrap_or(f64::NAN)
            }
            Expression::Val(expr) => {
                let mut vals: [Val<i32, f64>; 4] = Default::default();
                for (i, &vi) in self.var_indices.iter().enumerate() {
                    vals[i] = Val::Float(all[vi]);
                }
                expr.eval(&vals[..n])
                    .and_then(|v| v.to_float())
                    .unwrap_or(f64::NAN)
            }
        }
    }

    /// dU/dr by central difference. Scales (dx, dy, dz) proportionally to r±h
    /// so the perturbation is purely radial — correct even when the expression
    /// depends on signed components.
    fn dudr_numerical(&self, r: f64, dx: f64, dy: f64, dz: f64) -> f64 {
        let h = self.gradient_h;
        let scale_p = (r + h) / r;
        let scale_m = (r - h) / r;
        let u_p = self.eval_at(dx * scale_p, dy * scale_p, dz * scale_p, r + h);
        let u_m = self.eval_at(dx * scale_m, dy * scale_m, dz * scale_m, r - h);
        (u_p - u_m) / (2.0 * h)
    }
}

/// Mass-weighted COM of a group (uses the cached COM if available).
fn group_com(context: &impl Context, group_idx: usize) -> crate::Point {
    if let Some(&com) = context.groups()[group_idx].mass_center() {
        return com;
    }
    let atoms: Vec<usize> = context.groups()[group_idx].iter_active().collect();
    context.mass_center(&atoms)
}

/// Distribute a COM-acting force over the atoms of a group by mass fraction.
///
/// With Fᵢ = (mᵢ/M)·F_com, ΣFᵢ = F_com (correct net translation force) and
/// Σ(rᵢ−r_com)×Fᵢ = (1/M)[Σmᵢ(rᵢ−r_com)]×F_com = 0 by definition of the COM.
/// The rigid integrator therefore sees pure translation with zero spurious torque.
fn distribute_force(
    context: &impl Context,
    group_idx: usize,
    f_com: crate::Point,
    forces: &mut [crate::Point],
) {
    let group = &context.groups()[group_idx];
    let total_mass: f64 = group.iter_active().map(|i| context.atom_mass(i)).sum();
    if total_mass < MIN_MASS {
        return;
    }
    for i in group.iter_active() {
        if i < forces.len() {
            let w = context.atom_mass(i) / total_mass;
            forces[i] += w * f_com;
        }
    }
}

impl From<CustomPair> for EnergyTerm {
    fn from(cp: CustomPair) -> Self {
        Self::CustomPair(cp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal CustomPair without going through Context resolution.
    fn make_harmonic(k: f64, r0: f64) -> CustomPair {
        let mut constants = HashMap::new();
        constants.insert("k".to_string(), k);
        constants.insert("r0".to_string(), r0);
        let function = "0.5 * k * (r - r0)^2".to_string();
        let substituted = substitute_constants(&function, &constants);
        let (expression, var_names) = parse_expression(&substituted).unwrap();
        let var_indices = map_var_indices(&var_names).unwrap();
        CustomPair {
            expression: Arc::new(expression),
            function,
            var_indices,
            group1: 0,
            group2: 1,
            gradient_h: DEFAULT_GRADIENT_H,
            selection1: Selection::parse("molecule a").unwrap(),
            selection2: Selection::parse("molecule b").unwrap(),
        }
    }

    #[test]
    fn eval_harmonic_at_r() {
        let cp = make_harmonic(100.0, 30.0);
        // U = 0.5 * k * (r - r0)^2 — depends on r only, signed components ignored
        assert!((cp.eval_at(0.0, 0.0, 0.0, 35.0) - 1250.0).abs() < 1e-9);
        assert!((cp.eval_at(0.0, 0.0, 0.0, 30.0) - 0.0).abs() < 1e-12);
        assert!((cp.eval_at(0.0, 0.0, 0.0, 25.0) - 1250.0).abs() < 1e-9);
    }

    #[test]
    fn force_direction_reciprocal_and_signed() {
        // Verify the algebra of the force step inside forces() without needing a Context.
        // Take d = com1 − com2 along +x, harmonic with r0 = 30. At r = 35, dU/dr = +500,
        // so F_on_g1 = -dU/dr * d̂ = -500 * (+x̂) ⇒ pulls g1 toward g2 (good — Hooke restoring).
        let cp = make_harmonic(100.0, 30.0);
        let d = crate::Point::new(35.0, 0.0, 0.0);
        let r = d.norm();
        let dudr = cp.dudr_numerical(r, d.x, d.y, d.z);
        let f_g1 = -dudr * d / r;
        let f_g2 = -f_g1;
        // Newton III
        assert!((f_g1 + f_g2).norm() < 1e-12);
        // Magnitude matches |dU/dr|
        assert!((f_g1.norm() - 500.0).abs() < 1e-6);
        // Direction: at r > r0, force on g1 points back toward g2 (i.e. −x̂)
        assert!(f_g1.x < 0.0);
    }

    #[test]
    fn dudr_matches_analytic_for_harmonic() {
        let cp = make_harmonic(100.0, 30.0);
        // dU/dr = k(r - r0). At r=35: 500. At r=27: -300.
        assert!((cp.dudr_numerical(35.0, 5.0, 0.0, 0.0) - 500.0).abs() < 1e-6);
        assert!((cp.dudr_numerical(27.0, -3.0, 0.0, 0.0) - (-300.0)).abs() < 1e-6);
        assert!((cp.dudr_numerical(30.0, 0.0, 0.0, 30.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn dudr_constant_force_is_constant() {
        // U = f0 * r ⇒ dU/dr = f0
        let mut constants = HashMap::new();
        constants.insert("f0".to_string(), 10.0);
        let substituted = substitute_constants("f0 * r", &constants);
        let (expression, var_names) = parse_expression(&substituted).unwrap();
        let cp = CustomPair {
            expression: Arc::new(expression),
            function: "f0 * r".to_string(),
            var_indices: map_var_indices(&var_names).unwrap(),
            group1: 0,
            group2: 1,
            gradient_h: DEFAULT_GRADIENT_H,
            selection1: Selection::parse("molecule a").unwrap(),
            selection2: Selection::parse("molecule b").unwrap(),
        };
        for &r in &[5.0, 30.0, 100.0] {
            assert!((cp.dudr_numerical(r, r, 0.0, 0.0) - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn parse_rejects_unknown_variable() {
        let (_, var_names) = parse_expression("a * r + b").unwrap();
        let err = map_var_indices(&var_names).unwrap_err();
        assert!(err.to_string().contains("unresolved variables"));
    }

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
selection1: "molecule a"
selection2: "molecule b"
function: "0.5 * k * (r - r0)^2"
constants:
  k: 100.0
  r0: 30.0
"#;
        let b: CustomPairBuilder = serde_yml::from_str(yaml).unwrap();
        assert_eq!(b.gradient_h, DEFAULT_GRADIENT_H);
    }
}
