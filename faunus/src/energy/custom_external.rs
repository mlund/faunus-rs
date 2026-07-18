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
use crate::group::{AbsIndex, GroupIndex};
use crate::selection::{Atoms, CachedSelection, ComSelection, Groups, Selection, Target};
use crate::Change;
use crate::ObserveContext;
use exmex::{Express, FlatEx, FlatExVal, Val};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use super::expr_helpers::substitute_constants;

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
    use_com: bool,
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
                "Custom external potential: preset '{}' (use_com={}, selection='{}')",
                self.function,
                self.use_com,
                self.selection
            );
            return Ok(CustomExternal {
                expression: Arc::new(Expression::Preset(preset)),
                function: self.function.clone(),
                var_indices: vec![0, 1, 2, 3], // q, x, y, z — all passed to preset
                use_com: self.use_com,
                selection_cache: RefCell::new(self.cached_selection()),
                warned_empty: std::cell::Cell::new(false),
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
            "Custom external potential: '{}' (use_com={}, selection='{}')",
            self.function,
            self.use_com,
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
            use_com: self.use_com,
            selection_cache: RefCell::new(self.cached_selection()),
            warned_empty: std::cell::Cell::new(false),
        })
    }

    fn cached_selection(&self) -> ComSelection {
        ComSelection::new(self.selection.clone(), self.use_com)
    }
}

/// A selection target `custom_external` can evaluate the potential at.
///
/// Lets [`affected`] filter a change against either index space without duplicating the walk over
/// `Change::Groups`.
trait ExternalTarget: Target {
    /// What the selection matched, for the "matched nothing" warning.
    const NOUN: &'static str;

    /// Whether a change to `group_index` touches the selected `index`.
    fn touched_by(index: Self::Index, group_index: usize, groups: &[crate::group::Group]) -> bool;
}

impl ExternalTarget for Atoms {
    const NOUN: &'static str = "atoms";
    fn touched_by(index: AbsIndex, group_index: usize, groups: &[crate::group::Group]) -> bool {
        groups[group_index].contains(index.get())
    }
}

impl ExternalTarget for Groups {
    const NOUN: &'static str = "groups";
    fn touched_by(index: GroupIndex, group_index: usize, _: &[crate::group::Group]) -> bool {
        index.get() == group_index
    }
}

/// Return the selected indices a change touches, in the index space the selection resolves to.
fn affected<T: ExternalTarget>(
    change: &Change,
    cache: &mut CachedSelection<T>,
    context: &impl ObserveContext,
    warned_empty: &std::cell::Cell<bool>,
) -> Vec<T::Index> {
    if matches!(change, Change::None) {
        return vec![];
    }
    let (selection, selected) = cache.resolve_with_selection(context);
    if selected.is_empty() && !warned_empty.replace(true) {
        log::warn!(
            "custom_external: selection '{selection}' matched no {} — energy will always be zero",
            T::NOUN
        );
    }
    let touched = |group_index: usize| {
        let groups = context.groups();
        selected
            .iter()
            .copied()
            .filter(move |&index| T::touched_by(index, group_index, groups))
    };
    match change {
        Change::Everything | Change::Volume(..) => selected.to_vec(),
        Change::SingleGroup(group_index, group_change) => {
            if matches!(group_change, GroupChange::None) {
                vec![]
            } else {
                touched(*group_index).collect()
            }
        }
        Change::Groups(changes) => changes
            .iter()
            .filter(|(_, group_change)| !matches!(group_change, GroupChange::None))
            .flat_map(|(group_index, _)| touched(*group_index))
            .collect(),
        Change::None => unreachable!(),
    }
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
    use_com: bool,
    /// Owns the selection, its resolved indices, and its cache key. RefCell because energy()
    /// takes &self.
    selection_cache: RefCell<ComSelection>,
    /// Guards the "matched nothing" warning, which can only be discovered once a context exists.
    warned_empty: std::cell::Cell<bool>,
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

    /// Evaluate the potential at a single atom.
    fn energy_for_atom(&self, context: &impl ObserveContext, atom_idx: usize) -> f64 {
        let topology = context.topology_ref();
        let pos = context.position(atom_idx);
        let q = topology.atomkind(context.atom_kind(atom_idx)).charge();
        self.eval_at(q, pos.x, pos.y, pos.z)
    }

    /// Evaluate the potential at the center of mass of a group (COM mode).
    fn energy_for_com(&self, context: &impl ObserveContext, group_index: GroupIndex) -> f64 {
        let group = context.group(group_index);
        let topology = context.topology_ref();
        let atomkinds = topology.atomkinds();
        if let Some(&com) = group.mass_center() {
            let net_charge: f64 = group
                .iter_active()
                .map(|i| atomkinds[context.atom_kind(i).get()].charge())
                .sum();
            self.eval_at(net_charge, com.x, com.y, com.z)
        } else {
            0.0
        }
    }

    /// Compute energy for a given change.
    pub(crate) fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        match &mut *self.selection_cache.borrow_mut() {
            ComSelection::Atoms(cache) => affected(change, cache, context, &self.warned_empty)
                .into_iter()
                .map(|atom| self.energy_for_atom(context, atom.get()))
                .sum(),
            ComSelection::Groups(cache) => affected(change, cache, context, &self.warned_empty)
                .into_iter()
                .map(|group| self.energy_for_com(context, group))
                .sum(),
        }
    }

    /// Report custom external parameters as YAML.
    pub(super) fn to_yaml(&self) -> serde_yml::Value {
        let selection = self.selection_cache.borrow().selection().to_string();
        yaml_map! {
            "function" => self.function.clone(),
            "use_com" => self.use_com,
            "selection" => selection,
        }
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
        assert!(!builder.use_com);
    }

    #[test]
    fn deserialize_builder_with_com() {
        let yaml = r#"
selection: "all"
function: "q * 0.1 * z"
use_com: true
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let ext = builder.build().unwrap();
        assert!(ext.use_com);
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

    /// A titration or speciation move swaps an atom's kind in place, leaving every group's active
    /// count — and hence the group-list generation — untouched. An energy term selecting on atom
    /// type must still follow the swap, or it silently keeps scoring the pre-swap system while the
    /// drift check compares one stale energy against another and reports no problem.
    #[test]
    fn atom_kind_swap_changes_the_selected_set() {
        use crate::group::{GroupCollection, GroupCollectionMut};
        use crate::WithTopology;
        let mut context = make_context();
        let yaml = r#"
selection: "atomtype HW"
function: "z"
"#;
        let builder: CustomExternalBuilder = serde_yml::from_str(yaml).unwrap();
        let term = builder.build().unwrap();
        let before = term.energy(&context, &Change::Everything);

        let kinds = context.topology().atomkinds().to_vec();
        let hydrogen =
            crate::group::AtomKindId::new(kinds.iter().position(|k| k.name() == "HW").unwrap());
        let oxygen =
            crate::group::AtomKindId::new(kinds.iter().position(|k| k.name() == "OW").unwrap());
        let atom = (0..context.num_particles())
            .find(|&i| context.atom_kind(i) == oxygen)
            .expect("an OW atom");
        // A non-zero z, else the swap could not change the energy at all.
        assert!(context.position(atom).z.abs() > 1e-9);

        context.set_atom_kind(atom, hydrogen);
        let after = term.energy(&context, &Change::Everything);

        // The swapped atom now matches `atomtype HW` and must contribute.
        assert!(
            (after - before).abs() > 1e-9,
            "energy unchanged after swap: {before} -> {after}"
        );
        // The only trustworthy oracle: a term built fresh against the post-swap system.
        let fresh = builder
            .build()
            .unwrap()
            .energy(&context, &Change::Everything);
        assert!(
            (after - fresh).abs() < 1e-12,
            "cached term {after} disagrees with a freshly built one {fresh}"
        );
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

        // Sum of per-group energies must equal the full evaluation
        let total = ext.energy(&ctx, &Change::Everything);
        use crate::group::GroupCollection;
        let n_groups = ctx.groups().len();
        let sum_partials: f64 = (0..n_groups)
            .map(|gi| ext.energy(&ctx, &Change::SingleGroup(gi, GroupChange::RigidBody)))
            .sum();
        assert!((sum_partials - total).abs() < 1e-10);
    }
}
