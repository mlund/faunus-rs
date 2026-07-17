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

//! Bounded-value vocabulary shared by the collective-variable consumers.
//!
//! A collective variable's value is always finite, so a range with an infinite
//! bound was never a physical constraint — only a sentinel for "unbounded this
//! side", which the [`Restraint`](crate::energy::Restraint) enum now expresses
//! directly. [`Interval`] and [`BinWidth`] make the invalid states
//! (min ≥ max, non-finite bound, non-positive width) unrepresentable, so the
//! silent no-ops they used to cause cannot occur.

use super::{CollectiveVariable, CvKindBuilder};
use serde::{Deserialize, Serialize};

/// A closed interval `[min, max]` with `min < max` and both bounds finite.
///
/// Deserializes from a two-element sequence, e.g. `[1000.0, 5000.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "(f64, f64)", into = "(f64, f64)")]
pub struct Interval {
    min: f64,
    max: f64,
}

impl Interval {
    /// Construct an interval, rejecting non-finite bounds and `min ≥ max`.
    pub fn new(min: f64, max: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            min.is_finite() && max.is_finite(),
            "interval bounds must be finite, got [{min}, {max}]"
        );
        anyhow::ensure!(min < max, "interval requires min < max, got [{min}, {max}]");
        Ok(Self { min, max })
    }

    /// True if `value` lies within `[min, max]`, inclusive.
    pub fn contains(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }

    /// Width of the interval, `max − min`.
    pub fn span(&self) -> f64 {
        self.max - self.min
    }
}

impl TryFrom<(f64, f64)> for Interval {
    type Error = String;
    fn try_from((min, max): (f64, f64)) -> Result<Self, String> {
        Self::new(min, max).map_err(|e| e.to_string())
    }
}

impl From<Interval> for (f64, f64) {
    fn from(i: Interval) -> Self {
        (i.min, i.max)
    }
}

/// Defines a newtype wrapping a strictly positive, finite `f64` that
/// deserializes from (and serializes to) a bare scalar, rejecting non-positive
/// and non-finite values at parse.
macro_rules! positive_finite_f64 {
    ($(#[$meta:meta])* $name:ident, $what:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
        #[serde(try_from = "f64", into = "f64")]
        pub struct $name(f64);

        impl $name {
            #[doc = concat!("Construct a ", $what, ", rejecting non-positive and non-finite values.")]
            pub fn new(value: f64) -> anyhow::Result<Self> {
                anyhow::ensure!(
                    value.is_finite() && value > 0.0,
                    concat!($what, " must be finite and positive, got {}"),
                    value
                );
                Ok(Self(value))
            }

            pub fn get(&self) -> f64 {
                self.0
            }
        }

        impl TryFrom<f64> for $name {
            type Error = String;
            fn try_from(value: f64) -> Result<Self, String> {
                Self::new(value).map_err(|e| e.to_string())
            }
        }

        impl From<$name> for f64 {
            fn from(x: $name) -> Self {
                x.0
            }
        }
    };
}

positive_finite_f64!(
    /// A strictly positive, finite histogram bin width. YAML: a bare scalar, e.g. `0.5`.
    BinWidth,
    "bin width"
);

positive_finite_f64!(
    /// A strictly positive, finite harmonic force constant. A non-positive `k`
    /// would make `½k·d²` reward leaving the restrained region — an anti-confining
    /// "restraint" — so it is rejected at parse.
    ForceConstant,
    "force constant"
);

/// A collective variable together with the uniform grid it is histogrammed on.
///
/// Both bounds and the bin width are required and validated at parse time, so a
/// Wang-Landau grid can never be built from an unbounded range or a zero width.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HistogrammedCv {
    range: Interval,
    resolution: BinWidth,
    #[serde(flatten)]
    cv: Box<dyn CvKindBuilder>,
}

impl HistogrammedCv {
    pub fn range(&self) -> Interval {
        self.range
    }

    pub fn resolution(&self) -> BinWidth {
        self.resolution
    }

    /// Resolve selections against context into a runtime collective variable.
    pub fn build_cv(
        &self,
        context: &impl crate::ObserveContext,
    ) -> anyhow::Result<CollectiveVariable> {
        self.cv.build_cv(context)
    }
}

/// A collective variable binned on demand at a fixed width, with no bounded
/// range. Used by `MeanAlongCoordinate`, which needs a bin width but no bounds.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BinnedCv {
    resolution: BinWidth,
    #[serde(flatten)]
    cv: Box<dyn CvKindBuilder>,
}

impl BinnedCv {
    pub fn resolution(&self) -> BinWidth {
        self.resolution
    }

    /// Resolve selections against context into a runtime collective variable.
    pub fn build_cv(
        &self,
        context: &impl crate::ObserveContext,
    ) -> anyhow::Result<CollectiveVariable> {
        self.cv.build_cv(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_contains() {
        let i = Interval::new(0.0, 10.0).unwrap();
        assert!(i.contains(0.0));
        assert!(i.contains(5.0));
        assert!(i.contains(10.0));
        assert!(!i.contains(-0.1));
        assert!(!i.contains(10.1));
        assert_eq!(i.span(), 10.0);
    }

    #[test]
    fn interval_rejects_degenerate_and_infinite() {
        assert!(Interval::new(5.0, 5.0).is_err(), "min == max");
        assert!(Interval::new(5.0, 1.0).is_err(), "min > max");
        assert!(Interval::new(f64::NEG_INFINITY, 1.0).is_err(), "−∞");
        assert!(Interval::new(0.0, f64::INFINITY).is_err(), "+∞");
        assert!(Interval::new(0.0, f64::NAN).is_err(), "NaN");
    }

    #[test]
    fn interval_yaml_roundtrip() {
        let i: Interval = serde_yml::from_str("[1.0, 2.0]").unwrap();
        assert_eq!(i, Interval::new(1.0, 2.0).unwrap());
        assert_eq!(
            serde_yml::to_string(&i).unwrap().trim(),
            "- 1.0\n- 2.0".trim()
        );
        // An infinite bound is rejected at parse, not silently accepted.
        assert!(serde_yml::from_str::<Interval>("[-.inf, 50.0]").is_err());
    }

    #[test]
    fn bin_width_rejects_nonpositive_and_infinite() {
        assert!(BinWidth::new(0.5).is_ok());
        assert!(BinWidth::new(0.0).is_err());
        assert!(BinWidth::new(-1.0).is_err());
        assert!(BinWidth::new(f64::INFINITY).is_err());
        assert!(BinWidth::new(f64::NAN).is_err());
    }

    #[test]
    fn bin_width_yaml_is_a_bare_scalar() {
        let w: BinWidth = serde_yml::from_str("0.5").unwrap();
        assert_eq!(w.get(), 0.5);
        assert_eq!(serde_yml::to_string(&w).unwrap().trim(), "0.5");
        assert!(serde_yml::from_str::<BinWidth>("0.0").is_err());
    }

    #[test]
    fn force_constant_rejects_nonpositive_and_infinite() {
        assert!(ForceConstant::new(100.0).is_ok());
        assert!(ForceConstant::new(0.0).is_err());
        assert!(ForceConstant::new(-1.0).is_err());
        assert!(ForceConstant::new(f64::INFINITY).is_err());
        assert!(ForceConstant::new(f64::NAN).is_err());
    }

    #[test]
    fn histogrammed_cv_requires_finite_range_and_resolution() {
        // #73: Wang-Landau's grid needs finite bounds and a positive width. Both
        // are required now, so the old silent defaults — (-∞,∞) range that
        // allocated usize::MAX bins, and a resolution that fell back to 1.0 —
        // become parse errors.
        assert!(serde_yml::from_str::<HistogrammedCv>(
            "property: volume\nrange: [0.0, 10.0]\nresolution: 0.5"
        )
        .is_ok());
        assert!(
            serde_yml::from_str::<HistogrammedCv>("property: volume\nrange: [0.0, 10.0]").is_err(),
            "missing resolution must be rejected"
        );
        assert!(
            serde_yml::from_str::<HistogrammedCv>("property: volume\nresolution: 0.5").is_err(),
            "missing range must be rejected"
        );
        assert!(
            serde_yml::from_str::<HistogrammedCv>(
                "property: volume\nrange: [-.inf, .inf]\nresolution: 0.5"
            )
            .is_err(),
            "infinite range must be rejected"
        );
    }
}
