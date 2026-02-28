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
use crate::selection::Selection;
use crate::{Change, Context};
use exmex::{Express, FlatEx};
use serde::{Deserialize, Serialize};
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
        let substituted = substitute_constants(&self.function, &self.constants);

        let expression: FlatEx<f64> = FlatEx::parse(&substituted)
            .map_err(|e| anyhow::anyhow!("expression parse error: {e}"))?;

        let allowed = ["q", "x", "y", "z"];
        let var_names = expression.var_names();
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

        // Map each exmex variable slot to its index in [q, x, y, z]
        let var_indices: Vec<usize> = var_names
            .iter()
            .map(|name| allowed.iter().position(|&a| a == name).unwrap())
            .collect();

        Ok(CustomExternal {
            expression: Arc::new(expression),
            var_indices,
            selection: self.selection.clone(),
            com: self.com,
        })
    }
}

/// Return group indices affected by a change that also match a selection.
fn affected_groups(change: &Change, selection: &Selection, context: &impl Context) -> Vec<usize> {
    match change {
        Change::None => vec![],
        Change::Everything | Change::Volume(..) => {
            selection.resolve_groups(context.topology_ref(), context.groups())
        }
        Change::SingleGroup(gi, gc) => {
            if matches!(gc, GroupChange::None) {
                return vec![];
            }
            let selected = selection.resolve_groups(context.topology_ref(), context.groups());
            if selected.contains(gi) {
                vec![*gi]
            } else {
                vec![]
            }
        }
        Change::Groups(changes) => {
            let selected = selection.resolve_groups(context.topology_ref(), context.groups());
            changes
                .iter()
                .filter(|(_, gc)| !matches!(gc, GroupChange::None))
                .filter_map(|(gi, _)| selected.contains(gi).then_some(*gi))
                .collect()
        }
    }
}

/// Substitute named constants into an expression string.
///
/// Sorts by name length (longest first) to avoid substring collisions,
/// following the same pattern as `interatomic::twobody::custom`.
fn substitute_constants(expression: &str, constants: &HashMap<String, f64>) -> String {
    let mut sorted: Vec<_> = constants.iter().collect();
    sorted.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));

    let mut result = expression.to_string();
    for (name, value) in sorted {
        result = result.replace(name.as_str(), &format!("({value:.17})"));
    }
    result
}

/// Custom external potential energy term.
///
/// Evaluates a mathematical expression at each selected particle position
/// (or molecular mass center). The expression can use any subset of
/// `q` (charge), `x`, `y`, `z` (position).
#[derive(Debug, Clone)]
pub struct CustomExternal {
    /// Arc avoids copying ~19KB FlatEx on clone.
    expression: Arc<FlatEx<f64>>,
    /// Maps each exmex variable slot to index in [q, x, y, z].
    var_indices: Vec<usize>,
    selection: Selection,
    com: bool,
}

impl CustomExternal {
    /// Evaluate the expression for a single point with given charge and position.
    fn eval_at(&self, q: f64, x: f64, y: f64, z: f64) -> f64 {
        let all = [q, x, y, z];
        let mut vals = [0.0_f64; 4];
        for (i, &vi) in self.var_indices.iter().enumerate() {
            vals[i] = all[vi];
        }
        self.expression
            .eval(&vals[..self.var_indices.len()])
            .unwrap_or(f64::NAN)
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
                    .map(|i| atomkinds[context.get_atomkind(i)].charge())
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
                    let q = atomkinds[context.get_atomkind(i)].charge();
                    self.eval_at(q, pos.x, pos.y, pos.z)
                })
                .sum()
        }
    }

    /// Compute energy for a given change.
    pub fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        let affected = affected_groups(change, &self.selection, context);
        affected
            .iter()
            .map(|&gi| self.energy_for_group(context, gi))
            .sum()
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        // "sigma" (longer) should be substituted first
        assert!(!result.contains("sigma"));
        assert!(result.contains("999"));
    }

    #[test]
    fn eval_simple_expression() {
        let yaml = r#"
selection: "all"
function: "0.5 * (x^2 + y^2 + z^2)"
"#;
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        let energy = ext.eval_at(0.0, 1.0, 2.0, 3.0);
        assert!((energy - 7.0).abs() < 1e-10); // 0.5 * (1 + 4 + 9) = 7
    }

    #[test]
    fn eval_charge_expression() {
        let yaml = r#"
selection: "all"
function: "q * z"
"#;
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builders: Vec<CustomExternalBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(builders.len(), 2);
        assert!(!builders[0].com);
        assert!(builders[1].com);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::platform::reference::ReferencePlatform;
    use std::path::Path;

    fn make_context() -> ReferencePlatform {
        let mut rng = rand::thread_rng();
        ReferencePlatform::new(
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: CustomExternalBuilder = serde_yaml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        let energy = ext.energy(&ctx, &Change::Everything);
        // With com mode, should get energy from mass centers
        assert!(energy >= 0.0);
    }
}
