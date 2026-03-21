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

//! Custom external potential energy term.
//!
//! Applies a user-defined mathematical expression as an external potential
//! to selected atoms or molecular mass centers. Variables `q`, `x`, `y`, `z`
//! are available in the expression (evaluated in alphabetical order per exmex convention).

use crate::change::GroupChange;
use crate::selection::{Selection, SelectionCache};
use crate::{Change, Context};
use exmex::{Express, FlatEx, FlatExVal, Val};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use super::EnergyTerm;

/// Builder for deserializing a `CustomExternal` entry from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomExternalBuilder {
    /// Selection expression for atoms/molecules to act on.
    selection: Selection,
    /// Math expression for the external potential (kJ/mol).
    function: String,
    /// Apply to molecular mass center instead of individual atoms.
    #[serde(default)]
    com: bool,
    /// User-defined constants substituted into the expression.
    #[serde(default)]
    constants: HashMap<String, f64>,
}

impl CustomExternalBuilder {
    /// Build a [`CustomExternal`] energy term.
    ///
    /// Substitutes user constants into the expression string, parses with exmex,
    /// and validates that only `q`, `x`, `y`, `z` remain as variables.
    pub fn build(&self) -> anyhow::Result<CustomExternal> {
        // Try preset name first (pure Rust, no parsing overhead)
        if let Some(preset) = find_preset(&self.function) {
            log::info!(
                "Custom external potential: preset '{}' (com={}, selection='{}')",
                self.function,
                self.com,
                self.selection
            );
            return Ok(CustomExternal {
                expression: Arc::new(Expression::Preset(preset)),
                function: self.function.clone(),
                var_indices: vec![0, 1, 2, 3], // q, x, y, z — all passed to preset
                selection: self.selection.clone(),
                com: self.com,
                group_cache: RefCell::default(),
            });
        }

        let substituted = substitute_constants(&self.function, &self.constants);

        // FlatExVal supports if/else conditionals but has ~8× overhead from Val
        // enum dispatch and heap allocation. Use fast FlatEx<f64> when possible.
        let has_conditionals = substituted.contains(" if ") || substituted.contains(" else ");
        let (expression, var_names) = if has_conditionals {
            let expr: FlatExVal<i32, f64> = exmex::parse_val(&substituted)
                .map_err(|e| anyhow::anyhow!("expression parse error: {e}"))?;
            let names = expr.var_names().to_vec();
            (Expression::Val(expr), names)
        } else {
            let expr: FlatEx<f64> = FlatEx::parse(&substituted)
                .map_err(|e| anyhow::anyhow!("expression parse error: {e}"))?;
            let names = expr.var_names().to_vec();
            (Expression::Float(expr), names)
        };

        let allowed = ["q", "x", "y", "z"];
        let bad_vars: Vec<_> = var_names
            .iter()
            .filter(|v| !allowed.contains(&v.as_str()))
            .cloned()
            .collect();
        if !bad_vars.is_empty() {
            anyhow::bail!(
                "unresolved variables in custom external: {}",
                bad_vars.join(", ")
            );
        }

        log::info!(
            "Custom external potential: '{}' (com={}, selection='{}')",
            self.function,
            self.com,
            self.selection
        );

        let var_indices: Vec<usize> = var_names
            .iter()
            .map(|name| allowed.iter().position(|&a| a == name).unwrap())
            .collect();

        Ok(CustomExternal {
            expression: Arc::new(expression),
            function: self.function.clone(),
            var_indices,
            selection: self.selection.clone(),
            com: self.com,
            group_cache: RefCell::default(),
        })
    }
}

/// Return group indices affected by a change that also match a cached selection.
fn affected_groups(
    change: &Change,
    cache: &RefCell<SelectionCache>,
    selection: &Selection,
    context: &impl Context,
) -> Vec<usize> {
    if matches!(change, Change::None) {
        return vec![];
    }
    let gen = context.group_lists_generation();
    let mut cache = cache.borrow_mut();
    let selected = cache.get_or_resolve(gen, || context.resolve_groups_live(selection));
    match change {
        Change::Everything | Change::Volume(..) => selected.to_vec(),
        Change::SingleGroup(gi, gc) => {
            if !matches!(gc, GroupChange::None) && selected.contains(gi) {
                vec![*gi]
            } else {
                vec![]
            }
        }
        Change::Groups(changes) => changes
            .iter()
            .filter(|(_, gc)| !matches!(gc, GroupChange::None))
            .filter_map(|(gi, _)| selected.contains(gi).then_some(*gi))
            .collect(),
        Change::None => unreachable!(),
    }
}

/// Substitute named constants into an expression string using word-boundary matching.
///
/// Only replaces occurrences where the constant name is not part of a longer
/// identifier (e.g. constant `c` won't clobber `cos` or `rc`).
fn substitute_constants(expression: &str, constants: &HashMap<String, f64>) -> String {
    let mut sorted: Vec<_> = constants.iter().collect();
    sorted.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));

    let mut result = expression.to_string();
    for (name, value) in sorted {
        result = replace_whole_word(&result, name, &format!("({value:.17})"));
    }
    result
}

/// Replace all whole-word occurrences of `word` in `text` with `replacement`.
///
/// A match is "whole word" when the characters immediately before and after
/// are not alphanumeric or underscore (i.e. not part of an identifier).
fn replace_whole_word(text: &str, word: &str, replacement: &str) -> String {
    fn is_ident_char(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(word) {
        let before_ok = pos == 0 || !rest[..pos].ends_with(is_ident_char);
        let end = pos + word.len();
        let after_ok = end >= rest.len() || !rest[end..].starts_with(is_ident_char);
        result.push_str(&rest[..pos]);
        if before_ok && after_ok {
            result.push_str(replacement);
        } else {
            result.push_str(word);
        }
        rest = &rest[end..];
    }
    result.push_str(rest);
    result
}

/// Preset potential functions implemented in pure Rust for performance.
/// Avoids the overhead of expression parsing and Val enum dispatch.
type PresetFn = fn(q: f64, x: f64, y: f64, z: f64) -> f64;

/// Compiled expression: `FlatEx<f64>` for pure arithmetic (fast),
/// `FlatExVal` when conditionals (`if`/`else`) are present,
/// or a hardcoded `Preset` for known analytical surfaces.
/// Always stored behind `Arc`, so the large enum size is irrelevant.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum Expression {
    Float(FlatEx<f64>),
    Val(FlatExVal<i32, f64>),
    Preset(PresetFn),
}

/// Look up a preset potential by name.
fn find_preset(name: &str) -> Option<PresetFn> {
    match name {
        // Piecewise 2D surface from Frenkel & Smit, Ch. 7.
        // u(x,y) = m(x) × (1 + sin(2πx) + cos(2πy))
        // where m(x) is a staircase: 1,2,3,4,5 across five x-regions.
        "staircase-sincos" => Some(|_q, x, y, _z| {
            use std::f64::consts::TAU;
            let s = 1.0 + (TAU * x).sin() + (TAU * y).cos();
            let m = if x >= 1.75 {
                5.0
            } else if x >= 0.75 {
                4.0
            } else if x >= -0.25 {
                3.0
            } else if x >= -1.25 {
                2.0
            } else {
                1.0
            };
            m * s
        }),
        _ => None,
    }
}

/// Custom external potential energy term.
///
/// Evaluates a mathematical expression at each selected particle position
/// (or molecular mass center). The expression can use any subset of
/// `q` (charge), `x`, `y`, `z` (position). Supports Python-style
/// conditionals via exmex's `value` feature (e.g. `1.0 if x > 0 else 2.0`).
#[derive(Debug, Clone)]
pub struct CustomExternal {
    expression: Arc<Expression>,
    /// Original function string for reporting.
    function: String,
    /// Maps each exmex variable slot to index in [q, x, y, z].
    var_indices: Vec<usize>,
    selection: Selection,
    com: bool,
    /// RefCell because energy() takes &self
    group_cache: RefCell<SelectionCache>,
}

impl CustomExternal {
    /// Evaluate the expression for a single point with given charge and position.
    fn eval_at(&self, q: f64, x: f64, y: f64, z: f64) -> f64 {
        let all = [q, x, y, z];
        let n = self.var_indices.len();
        match self.expression.as_ref() {
            Expression::Preset(f) => f(q, x, y, z),
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

    /// Sum external potential over all active particles in a group.
    fn energy_for_group(&self, context: &impl Context, group_index: usize) -> f64 {
        let group = &context.groups()[group_index];
        let topology = context.topology_ref();
        let atomkinds = topology.atomkinds();

        if self.com {
            if let Some(&com) = group.mass_center() {
                let net_charge: f64 = group
                    .iter_active()
                    .map(|i| atomkinds[context.atom_kind(i)].charge())
                    .sum();
                self.eval_at(net_charge, com.x, com.y, com.z)
            } else {
                0.0
            }
        } else {
            group
                .iter_active()
                .map(|i| {
                    let pos = context.position(i);
                    let q = atomkinds[context.atom_kind(i)].charge();
                    self.eval_at(q, pos.x, pos.y, pos.z)
                })
                .sum()
        }
    }

    /// Compute energy for a given change.
    pub fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        affected_groups(change, &self.group_cache, &self.selection, context)
            .iter()
            .map(|&gi| self.energy_for_group(context, gi))
            .sum()
    }

    /// Report custom external parameters as YAML.
    pub(super) fn to_yaml(&self) -> serde_yml::Value {
        let mut map = serde_yml::Mapping::new();
        map.insert("function".into(), self.function.clone().into());
        map.insert("com".into(), self.com.into());
        map.insert("selection".into(), self.selection.to_string().into());
        serde_yml::Value::Mapping(map)
    }
}

impl From<CustomExternal> for EnergyTerm {
    fn from(ce: CustomExternal) -> Self {
        Self::CustomExternal(ce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
selection: "all"
function: "0.5 * k * (x^2 + y^2 + z^2)"
constants:
  k: 100.0
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        assert!(builder.build().is_ok());
        assert!(!builder.com);
    }

    #[test]
    fn deserialize_builder_with_com() {
        let yaml = r#"
selection: "all"
function: "q * 0.1 * z"
com: true
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        assert!(ext.com);
    }

    #[test]
    fn unresolved_variable_error() {
        let yaml = r#"
selection: "all"
function: "a * x + b"
constants:
  a: 1.0
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let err = builder.build().unwrap_err();
        assert!(err.to_string().contains("unresolved variables"));
        assert!(err.to_string().contains("b"));
    }

    #[test]
    fn constant_substitution() {
        let mut constants = HashMap::new();
        constants.insert("sigma".to_string(), 2.0);
        constants.insert("sig".to_string(), 999.0);
        let result = substitute_constants("sigma + sig", &constants);
        assert!(!result.contains("sigma"));
        assert!(result.contains("999"));
    }

    #[test]
    fn constant_substitution_word_boundary() {
        // Single-letter constant `c` must not clobber `cos` or `rc`
        let mut constants = HashMap::new();
        constants.insert("c".to_string(), 3.0);
        let result = substitute_constants("c * cos(x) + c", &constants);
        assert!(result.contains("cos"), "cos was clobbered: {result}");
        assert!(
            !result.contains(" c "),
            "standalone c not replaced: {result}"
        );
    }

    #[test]
    fn eval_simple_expression() {
        let yaml = r#"
selection: "all"
function: "0.5 * (x^2 + y^2 + z^2)"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        let energy = ext.eval_at(0.0, 1.0, 2.0, 3.0);
        assert!((energy - 7.0).abs() < 1e-10); // 0.5 * (1 + 4 + 9) = 7
    }

    #[test]
    fn eval_conditional_expression() {
        let yaml = r#"
selection: "all"
function: "10.0 if x > 0 else -5.0"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        assert!((ext.eval_at(0.0, 1.0, 0.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((ext.eval_at(0.0, -1.0, 0.0, 0.0) - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn eval_chained_conditional_with_trig() {
        let yaml = r#"
selection: "all"
function: "(1 if x < -1.25 else 2 if x < -0.25 else 3 if x < 0.75 else 4 if x < 1.75 else 5) * (1 + sin(TAU * x) + cos(TAU * y))"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        assert!((ext.eval_at(0.0, 0.0, 0.0, 0.0) - 6.0).abs() < 1e-10); // 3*(1+0+1)
        assert!((ext.eval_at(0.0, -1.0, 0.0, 0.0) - 4.0).abs() < 1e-10); // 2*(1+0+1)
        assert!((ext.eval_at(0.0, 1.0, 0.0, 0.0) - 8.0).abs() < 1e-10); // 4*(1+0+1)
        assert!((ext.eval_at(0.0, 0.0, 0.5, 0.0) - 0.0).abs() < 1e-10); // 3*(1+0-1)
    }

    #[test]
    fn eval_preset_matches_conditional() {
        // Preset must produce identical values to the if/else expression
        let preset_yaml = "selection: \"all\"\nfunction: staircase-sincos\n";
        let expr_yaml = r#"
selection: "all"
function: "(1 if x < -1.25 else 2 if x < -0.25 else 3 if x < 0.75 else 4 if x < 1.75 else 5) * (1 + sin(TAU * x) + cos(TAU * y))"
"#;
        let preset: CustomExternalBuilder = serde_yml::from_str(preset_yaml).unwrap();
        let expr: CustomExternalBuilder = serde_yml::from_str(expr_yaml).unwrap();
        let p = preset.build().unwrap();
        let e = expr.build().unwrap();
        for &x in &[-1.9, -1.0, 0.0, 0.5, 1.0, 1.5, 1.9] {
            for &y in &[-1.5, 0.0, 0.5, 1.5] {
                let ep = p.eval_at(0.0, x, y, 0.0);
                let ee = e.eval_at(0.0, x, y, 0.0);
                assert!((ep - ee).abs() < 1e-10, "mismatch at x={x}, y={y}");
            }
        }
    }

    #[test]
    fn eval_charge_expression() {
        let yaml = r#"
selection: "all"
function: "q * z"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        let energy = ext.eval_at(2.0, 0.0, 0.0, 3.0);
        assert!((energy - 6.0).abs() < 1e-10); // 2.0 * 3.0 = 6
    }

    #[test]
    fn deserialize_list_of_builders() {
        let yaml = r#"
- selection: "all"
  function: "x^2"
- selection: "all"
  function: "q * z"
  com: true
"#;
        let builders: Vec<CustomExternalBuilder> = serde_yml::from_str(yaml).unwrap();
        assert_eq!(builders.len(), 2);
        assert!(!builders[0].com);
        assert!(builders[1].com);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::backend::Backend;
    use std::path::Path;

    fn make_context() -> Backend {
        let mut rng = rand::thread_rng();
        Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap()
    }

    #[test]
    fn zero_potential_gives_zero_energy() {
        let ctx = make_context();
        let yaml = r#"
selection: "all"
function: "0 * x"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        let energy = ext.energy(&ctx, &Change::Everything);
        assert!((energy).abs() < 1e-10);
    }

    #[test]
    fn harmonic_confinement_positive_energy() {
        let ctx = make_context();
        let yaml = r#"
selection: "all"
function: "0.5 * (x^2 + y^2 + z^2)"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        let energy = ext.energy(&ctx, &Change::Everything);
        // With particles at various positions, energy should be > 0
        assert!(energy > 0.0);
    }

    #[test]
    fn no_energy_on_no_change() {
        let ctx = make_context();
        let yaml = r#"
selection: "all"
function: "100 * (x^2 + y^2 + z^2)"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        assert_eq!(ext.energy(&ctx, &Change::None), 0.0);
    }

    #[test]
    fn single_group_change() {
        let ctx = make_context();
        let yaml = r#"
selection: "all"
function: "0.5 * (x^2 + y^2 + z^2)"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();

        // Energy for single group change should equal energy_for_group
        let group0_energy = ext.energy_for_group(&ctx, 0);
        let change = Change::SingleGroup(0, GroupChange::RigidBody);
        let energy = ext.energy(&ctx, &change);
        assert!((energy - group0_energy).abs() < 1e-10);
    }

    #[test]
    fn com_mode_uses_mass_center() {
        let ctx = make_context();
        let yaml = r#"
selection: "all"
function: "x^2 + y^2 + z^2"
com: true
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        let energy = ext.energy(&ctx, &Change::Everything);
        // With com mode, should get energy from mass centers
        assert!(energy >= 0.0);
    }
}
