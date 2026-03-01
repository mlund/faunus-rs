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

//! Constrain energy term for collective variables.
//!
//! Supports hard constraints (infinite energy outside range) and
//! soft harmonic constraints (quadratic penalty around equilibrium).

use crate::collective_variable::{CollectiveVariableBuilder, ConcreteCollectiveVariable};
use crate::{Change, Context};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Harmonic constraint parameters.
///
/// Applies a quadratic penalty: `0.5 * force_constant * (equilibrium - value)²`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarmonicConstraint {
    pub force_constant: f64,
    pub equilibrium: f64,
}

/// Builder for deserializing a single constrain entry from YAML.
///
/// Flattens the CV builder fields together with an optional `harmonic` section.
// Note: `deny_unknown_fields` is intentionally omitted here because serde
// does not support it in combination with `flatten`. Unknown fields are
// still rejected by the flattened `CollectiveVariableBuilder` which has
// `deny_unknown_fields`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainBuilder {
    #[serde(flatten)]
    pub cv: CollectiveVariableBuilder,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harmonic: Option<HarmonicConstraint>,
}

impl ConstrainBuilder {
    /// Build a [`Constrain`] energy term by resolving selections against the context.
    pub fn build(&self, context: &impl Context) -> Result<Constrain> {
        let cv = self.cv.build_concrete(context)?;
        Ok(Constrain {
            cv,
            harmonic: self.harmonic.clone(),
        })
    }
}

/// Constrains a collective variable to a range.
///
/// - **Hard constraint** (no `harmonic`): returns `f64::INFINITY` if the CV
///   value falls outside its axis range, otherwise 0.
/// - **Soft harmonic constraint**: returns `0.5 * k * (eq - value)²`.
#[derive(Debug, Clone)]
pub struct Constrain {
    cv: ConcreteCollectiveVariable,
    harmonic: Option<HarmonicConstraint>,
}

impl Constrain {
    pub fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        if matches!(change, Change::None) {
            return 0.0;
        }
        let value = self.cv.evaluate(context);
        #[allow(clippy::option_if_let_else)] // if-let-else with else-if is clearer here
        if let Some(h) = &self.harmonic {
            let delta = h.equilibrium - value;
            0.5 * h.force_constant * delta * delta
        } else if self.cv.axis().in_range(value) {
            0.0
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_hard_constraint() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
"#;
        let builder: ConstrainBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(builder.harmonic.is_none());
        assert_eq!(builder.cv.range, (1000.0, 5000.0));
    }

    #[test]
    fn deserialize_harmonic_constraint() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
harmonic:
  force_constant: 100.0
  equilibrium: 3000.0
"#;
        let builder: ConstrainBuilder = serde_yaml::from_str(yaml).unwrap();
        let h = builder.harmonic.unwrap();
        assert!((h.force_constant - 100.0).abs() < 1e-10);
        assert!((h.equilibrium - 3000.0).abs() < 1e-10);
    }

    #[test]
    fn deserialize_list_of_constraints() {
        let yaml = r#"
- property: volume
  range: [1000.0, 5000.0]
- property: volume
  range: [1000.0, 5000.0]
  harmonic:
    force_constant: 100.0
    equilibrium: 3000.0
"#;
        let builders: Vec<ConstrainBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(builders.len(), 2);
        assert!(builders[0].harmonic.is_none());
        assert!(builders[1].harmonic.is_some());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cell::Shape;
    use crate::context::WithCell;
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
    fn hard_constraint_volume_in_range() {
        let ctx = make_context();
        let volume = ctx.cell().volume().unwrap();
        let builder: ConstrainBuilder = serde_yaml::from_str(&format!(
            "property: volume\nrange: [{}, {}]",
            volume - 1.0,
            volume + 1.0
        ))
        .unwrap();
        let constrain = builder.build(&ctx).unwrap();
        let energy = constrain.energy(&ctx, &Change::Everything);
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn hard_constraint_volume_out_of_range() {
        let ctx = make_context();
        let builder: ConstrainBuilder =
            serde_yaml::from_str("property: volume\nrange: [0.0, 1.0]").unwrap();
        let constrain = builder.build(&ctx).unwrap();
        let energy = constrain.energy(&ctx, &Change::Everything);
        assert_eq!(energy, f64::INFINITY);
    }

    #[test]
    fn harmonic_constraint_volume() {
        let ctx = make_context();
        let volume = ctx.cell().volume().unwrap();
        let eq = volume + 10.0;
        let k = 50.0;
        let yaml = format!(
            "property: volume\nrange: [0.0, 1e10]\nharmonic:\n  force_constant: {k}\n  equilibrium: {eq}"
        );
        let builder: ConstrainBuilder = serde_yaml::from_str(&yaml).unwrap();
        let constrain = builder.build(&ctx).unwrap();
        let energy = constrain.energy(&ctx, &Change::Everything);
        let expected = 0.5 * k * (eq - volume) * (eq - volume);
        assert!((energy - expected).abs() < 1e-6);
    }

    #[test]
    fn no_energy_on_no_change() {
        let ctx = make_context();
        let builder: ConstrainBuilder =
            serde_yaml::from_str("property: volume\nrange: [0.0, 1.0]").unwrap();
        let constrain = builder.build(&ctx).unwrap();
        // Even though volume is out of range, Change::None returns 0
        assert_eq!(constrain.energy(&ctx, &Change::None), 0.0);
    }
}
