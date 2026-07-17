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
//! A [`Restraint`] restrains a collective variable, either with a hard wall
//! (infinite energy outside the allowed region) or a quadratic penalty.

use crate::collective_variable::{
    CollectiveVariable, CvKindBuilder, Finite, ForceConstant, Interval,
};
use crate::Change;
use crate::ObserveContext;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// How a collective variable is restrained.
///
/// A single `restraint:` YAML tag selects one form, so the old ambiguity —
/// giving both `range` and `harmonic`, or neither — is unrepresentable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Restraint {
    /// Hard one-sided wall: `0` for `x ≤ max`, `∞` above.
    Below(Finite),
    /// Hard one-sided wall: `0` for `x ≥ min`, `∞` below.
    Above(Finite),
    /// Hard two-sided wall: `0` inside the interval, `∞` outside.
    Between(Interval),
    /// Quadratic penalty about a point: `½k(x₀ − x)²`.
    Harmonic {
        force_constant: ForceConstant,
        equilibrium: Finite,
    },
    /// Flat-bottomed well: `0` inside the interval, `½k·d²` beyond the nearest
    /// edge (`d` is the distance to that edge). The soft counterpart of `Between`.
    HarmonicWall {
        interval: Interval,
        force_constant: ForceConstant,
    },
}

impl Restraint {
    /// Restraining energy at collective-variable value `x`.
    fn energy(&self, x: f64) -> f64 {
        // `∞` for a hard wall breached, `0.0` otherwise.
        let wall = |inside: bool| if inside { 0.0 } else { f64::INFINITY };
        match self {
            Self::Below(max) => wall(x <= max.get()),
            Self::Above(min) => wall(x >= min.get()),
            Self::Between(interval) => wall(interval.contains(x)),
            Self::Harmonic {
                force_constant,
                equilibrium,
            } => 0.5 * force_constant.get() * (equilibrium.get() - x).powi(2),
            Self::HarmonicWall {
                interval,
                force_constant,
            } => {
                // Distance to the nearest edge, or 0 inside the interval.
                let d = (interval.min() - x).max(x - interval.max()).max(0.0);
                0.5 * force_constant.get() * d * d
            }
        }
    }
}

/// Builder for deserializing a single constrain entry from YAML.
///
/// Flattens the CV kind builder together with the `restraint`. `deny_unknown_fields`
/// is impossible next to `flatten`; strictness comes from the concrete
/// `CvKindBuilder`, which rejects whatever the flatten leaves over.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainBuilder {
    #[serde(flatten)]
    pub cv: Box<dyn CvKindBuilder>,
    pub restraint: Restraint,
}

impl ConstrainBuilder {
    /// Build a [`Constrain`] energy term by resolving selections against the context.
    pub(crate) fn build(&self, context: &impl ObserveContext) -> Result<Constrain> {
        Ok(Constrain {
            cv: self.cv.build_cv(context)?,
            restraint: self.restraint.clone(),
        })
    }
}

/// Restrains a collective variable according to its [`Restraint`].
#[derive(Debug, Clone)]
pub struct Constrain {
    cv: CollectiveVariable,
    restraint: Restraint,
}

impl Constrain {
    pub(crate) fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        if matches!(change, Change::None) {
            return 0.0;
        }
        self.restraint.energy(self.cv.evaluate(context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_between_and_harmonic() {
        let between: ConstrainBuilder =
            serde_yml::from_str("property: volume\nrestraint: !Between [1000.0, 5000.0]").unwrap();
        assert!(matches!(between.restraint, Restraint::Between(_)));

        let harmonic: ConstrainBuilder = serde_yml::from_str(
            "property: volume\nrestraint: !Harmonic {force_constant: 100.0, equilibrium: 3000.0}",
        )
        .unwrap();
        assert!(matches!(
            harmonic.restraint,
            Restraint::Harmonic { force_constant, equilibrium }
                if force_constant.get() == 100.0 && equilibrium.get() == 3000.0
        ));
    }

    #[test]
    fn missing_restraint_is_rejected() {
        // #73: a constrain entry must name a restraint. The old silent no-op —
        // neither `range` nor `harmonic`, so `range` defaulted to (-∞,∞) and the
        // constraint never fired — is now a parse error.
        let err = serde_yml::from_str::<ConstrainBuilder>("property: volume")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("restraint"),
            "should name the missing field: {err}"
        );
    }

    #[test]
    fn unknown_field_is_rejected() {
        // The former `range` key now lands in the flattened kind builder, which
        // denies it — a typo can no longer disable the constraint silently.
        assert!(serde_yml::from_str::<ConstrainBuilder>(
            "property: volume\nrestraint: !Between [0.0, 1.0]\nrange: [2.0, 3.0]"
        )
        .is_err());
        // A typo'd key inside a restraint tag is denied too.
        assert!(serde_yml::from_str::<ConstrainBuilder>(
            "property: volume\nrestraint: !Harmonic {force_constant: 100.0, equilibrium: 0.0, k: 5.0}"
        )
        .is_err());
    }

    #[test]
    fn non_finite_bound_is_rejected() {
        // `!Below .inf` would make `x <= ∞` always true — a silent no-op, the very
        // failure this fix removes. Non-finite bounds and equilibria are rejected.
        for yaml in [
            "property: volume\nrestraint: !Below .inf",
            "property: volume\nrestraint: !Above .nan",
            "property: volume\nrestraint: !Harmonic {force_constant: 1.0, equilibrium: .inf}",
        ] {
            assert!(
                serde_yml::from_str::<ConstrainBuilder>(yaml).is_err(),
                "non-finite scalar must be rejected: {yaml}"
            );
        }
    }

    #[test]
    fn non_positive_force_constant_is_rejected() {
        // A negative k makes ½k·d² reward leaving the well — an anti-confining
        // "restraint". Rejected at parse, like a zero-width bin.
        for k in ["-100.0", "0.0"] {
            let yaml = format!(
                "property: volume\nrestraint: !Harmonic {{force_constant: {k}, equilibrium: 0.0}}"
            );
            assert!(
                serde_yml::from_str::<ConstrainBuilder>(&yaml).is_err(),
                "force_constant {k} must be rejected"
            );
        }
    }

    // --- Restraint::energy physics (no context needed) ---

    #[test]
    fn between_is_a_hard_wall() {
        let r = Restraint::Between(Interval::new(0.0, 10.0).unwrap());
        assert_eq!(r.energy(0.0), 0.0);
        assert_eq!(r.energy(5.0), 0.0);
        assert_eq!(r.energy(10.0), 0.0);
        assert_eq!(r.energy(-0.1), f64::INFINITY);
        assert_eq!(r.energy(10.1), f64::INFINITY);
    }

    #[test]
    fn below_and_above_are_one_sided_walls() {
        let below = Restraint::Below(Finite::new(50.0).unwrap());
        assert_eq!(below.energy(50.0), 0.0);
        assert_eq!(below.energy(51.0), f64::INFINITY);
        let above = Restraint::Above(Finite::new(50.0).unwrap());
        assert_eq!(above.energy(50.0), 0.0);
        assert_eq!(above.energy(49.0), f64::INFINITY);
    }

    #[test]
    fn harmonic_wall_is_flat_inside_and_quadratic_outside() {
        // The flat-bottom the docs always described (range + harmonic) but the
        // code never implemented. Interior is exactly flat; beyond each edge the
        // energy is ½k·d² in the distance d to that edge, continuous at the edge.
        let k = 4.0;
        let r = Restraint::HarmonicWall {
            interval: Interval::new(-50.0, 50.0).unwrap(),
            force_constant: ForceConstant::new(k).unwrap(),
        };
        assert_eq!(r.energy(0.0), 0.0);
        assert_eq!(r.energy(-50.0), 0.0);
        assert_eq!(r.energy(50.0), 0.0);
        assert!(r.energy(50.0 + 1e-6) < 1e-9, "continuous at the edge");
        assert!((r.energy(53.0) - 0.5 * k * 9.0).abs() < 1e-12); // d = 3 above max
        assert!((r.energy(-54.0) - 0.5 * k * 16.0).abs() < 1e-12); // d = 4 below min
    }

    #[test]
    fn harmonic_wall_approaches_hard_wall_as_k_grows() {
        // As k → ∞ the soft wall matches the hard wall: 0 inside, diverging outside.
        let interval = Interval::new(0.0, 10.0).unwrap();
        let soft = Restraint::HarmonicWall {
            interval,
            force_constant: ForceConstant::new(1e30).unwrap(),
        };
        assert_eq!(soft.energy(5.0), Restraint::Between(interval).energy(5.0)); // 0 inside
        assert!(soft.energy(11.0) > 1e20, "huge outside, like the hard wall");
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::backend::Backend;
    use crate::cell::Shape;
    use crate::context::WithSimulationCell;
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
    fn hard_constraint_volume_in_range() {
        let ctx = make_context();
        let volume = ctx.cell().volume().unwrap();
        let builder: ConstrainBuilder = serde_yml::from_str(&format!(
            "property: volume\nrestraint: !Between [{}, {}]",
            volume - 1.0,
            volume + 1.0
        ))
        .unwrap();
        let constrain = builder.build(&ctx).unwrap();
        assert_eq!(constrain.energy(&ctx, &Change::Everything), 0.0);
    }

    #[test]
    fn hard_constraint_volume_out_of_range() {
        let ctx = make_context();
        let builder: ConstrainBuilder =
            serde_yml::from_str("property: volume\nrestraint: !Between [0.0, 1.0]").unwrap();
        let constrain = builder.build(&ctx).unwrap();
        assert_eq!(constrain.energy(&ctx, &Change::Everything), f64::INFINITY);
    }

    #[test]
    fn harmonic_constraint_volume() {
        let ctx = make_context();
        let volume = ctx.cell().volume().unwrap();
        let eq = volume + 10.0;
        let k = 50.0;
        let yaml = format!(
            "property: volume\nrestraint: !Harmonic {{force_constant: {k}, equilibrium: {eq}}}"
        );
        let builder: ConstrainBuilder = serde_yml::from_str(&yaml).unwrap();
        let constrain = builder.build(&ctx).unwrap();
        let energy = constrain.energy(&ctx, &Change::Everything);
        let expected = 0.5 * k * (eq - volume) * (eq - volume);
        assert!((energy - expected).abs() < 1e-6);
    }

    #[test]
    fn no_energy_on_no_change() {
        let ctx = make_context();
        let builder: ConstrainBuilder =
            serde_yml::from_str("property: volume\nrestraint: !Between [0.0, 1.0]").unwrap();
        let constrain = builder.build(&ctx).unwrap();
        // Volume is out of range, but Change::None short-circuits to 0.
        assert_eq!(constrain.energy(&ctx, &Change::None), 0.0);
    }

    #[test]
    fn flat_bottom_constrains_volume_softly() {
        // The documented `range` + `harmonic` intent, now real: a hard-walled
        // interval around the current volume gives 0; shrinking the window below
        // the volume produces a finite quadratic penalty, not INFINITY.
        let ctx = make_context();
        let volume = ctx.cell().volume().unwrap();
        let k = 2.0;
        let hi = volume - 5.0; // volume sits 5 above the upper edge
        let yaml = format!(
            "property: volume\nrestraint: !HarmonicWall {{interval: [0.0, {hi}], force_constant: {k}}}"
        );
        let energy = serde_yml::from_str::<ConstrainBuilder>(&yaml)
            .unwrap()
            .build(&ctx)
            .unwrap()
            .energy(&ctx, &Change::Everything);
        assert!((energy - 0.5 * k * 25.0).abs() < 1e-6, "½k·d² with d = 5");
    }
}
