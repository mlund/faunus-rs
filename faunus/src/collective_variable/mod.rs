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

//! Collective variables for enhanced sampling, constraints, and analysis.
//!
//! A collective variable (CV) maps the simulation state to a single scalar value.
//! The runtime [`CollectiveVariable`] wraps a [`CvKind`] trait object that
//! evaluates the CV. Range and bin width are not part of the CV — each consumer
//! that needs them owns a validated [`Interval`], bin width, or
//! [`crate::energy::Restraint`].
//!
//! Each CV type is defined in its own submodule and registered via `typetag`.

mod atom;
mod axis;
mod cell;
mod dynamic;
pub(crate) mod group;

pub use axis::{BinnedCv, Finite, ForceConstant, HistogrammedCv, Interval};

use anyhow::Result;

// Re-export CV types for convenience

/// Trait for collective variable evaluation.
///
/// Implementors provide the `evaluate` method that computes a scalar from context.
/// Use `#[typetag::serde(name = "snake_case")]` to register for YAML deserialization.
#[typetag::serde(tag = "type")]
pub trait CvKind: Send + Sync + std::fmt::Debug + dyn_clone::DynClone {
    /// Evaluate the collective variable given the current simulation state.
    fn evaluate(&self, context: &dyn EvalContext) -> f64;

    /// Return the name of this CV kind for axis labeling.
    fn name(&self) -> &'static str;
}

dyn_clone::clone_trait_object!(CvKind);

/// Minimal context trait for CV evaluation (object-safe subset of Context).
pub trait EvalContext:
    crate::group::GroupCollection + crate::context::WithSimulationCell + crate::context::WithTopology
{
    fn get_distance(&self, i: usize, j: usize) -> crate::Point;
}

/// A scalar observable of the simulation state: a resolved [`CvKind`] together
/// with an optional human-readable description of the selections it bound to.
///
/// Range and resolution are *not* here — they belong to whichever consumer
/// histograms or restrains the CV, and each owns them (see [`Interval`],
/// [`BinWidth`], and [`crate::energy::Restraint`]).
#[derive(Debug, Clone)]
pub struct CollectiveVariable {
    kind: Box<dyn CvKind>,
    description: Option<String>,
}

impl CollectiveVariable {
    /// Create a collective variable from a resolved kind and description.
    pub fn new(kind: Box<dyn CvKind>, description: Option<String>) -> Self {
        Self { kind, description }
    }

    pub fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        self.kind.evaluate(context)
    }

    /// The CV kind's name, e.g. `"Volume"`.
    pub fn name(&self) -> &'static str {
        self.kind.name()
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl dyn CvKindBuilder {
    /// Resolve selections against context into a runtime [`CollectiveVariable`],
    /// pairing the built kind with its description. This is the whole job the
    /// old `CollectiveVariableBuilder` wrapper existed to do.
    pub fn build_cv(&self, context: &impl crate::ObserveContext) -> Result<CollectiveVariable> {
        Ok(CollectiveVariable::new(
            self.build(context)?,
            self.description(),
        ))
    }
}

/// Trait for CV kind builders that resolve selections into indices.
#[typetag::serde(tag = "property")]
pub trait CvKindBuilder: Send + Sync + std::fmt::Debug + dyn_clone::DynClone {
    /// Build the CV kind by resolving selections against context.
    fn build(&self, context: &dyn EvalContext) -> Result<Box<dyn CvKind>>;

    /// Human-readable description of what this CV operates on (selections, projection, etc.).
    fn description(&self) -> Option<String> {
        None
    }
}

dyn_clone::clone_trait_object!(CvKindBuilder);

// Blanket impl: every observable context is an EvalContext. The supertraits above are exactly
// `ObserveContext`'s own, restated so `dyn EvalContext` stays object-safe.
impl<T: crate::context::ObserveContext> EvalContext for T {
    fn get_distance(&self, i: usize, j: usize) -> crate::Point {
        <T as crate::context::ObserveContext>::get_distance(self, i, j)
    }
}

/// Check that a resolved group has COM tracking enabled.
///
/// Used by CV builder macros to reject atomic groups at build time.
pub fn require_group_com(
    group_index: usize,
    context: &dyn EvalContext,
    cv_name: &str,
) -> Result<()> {
    let group = &context.groups()[group_index];
    let mol_kind = context.topology_ref().moleculekind(group.molecule());
    if !mol_kind.has_com() {
        anyhow::bail!(
            "{cv_name}: group '{}' (molecule '{}') has no center of mass",
            group_index,
            mol_kind.name()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Macros for reducing boilerplate
// ---------------------------------------------------------------------------

/// Implements `CvKindBuilder` for a self-building CV (no build-time resolution needed).
///
/// Use this for CVs where the deserialized struct is identical to the evaluated struct.
///
/// # Example
/// ```ignore
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// pub struct Volume;
///
/// #[typetag::serde(name = "volume")]
/// impl CvKind for Volume { /* ... */ }
///
/// impl_self_building_cv!(Volume, "volume");
/// ```
#[macro_export]
macro_rules! impl_self_building_cv {
    ($ty:ty, $name:literal) => {
        #[typetag::serde(name = $name)]
        impl $crate::collective_variable::CvKindBuilder for $ty {
            fn build(
                &self,
                _context: &dyn $crate::collective_variable::EvalContext,
            ) -> anyhow::Result<Box<dyn $crate::collective_variable::CvKind>> {
                Ok(Box::new(self.clone()))
            }
        }
    };
    ($ty:ty, $name:literal, |$s:ident| $desc:expr) => {
        #[typetag::serde(name = $name)]
        impl $crate::collective_variable::CvKindBuilder for $ty {
            fn build(
                &self,
                _context: &dyn $crate::collective_variable::EvalContext,
            ) -> anyhow::Result<Box<dyn $crate::collective_variable::CvKind>> {
                Ok(Box::new(self.clone()))
            }
            fn description(&self) -> Option<String> {
                let $s = self;
                $desc
            }
        }
    };
}

/// Defines a builder that resolves a single group selection.
///
/// Generates `{Name}Builder` struct with `selection: Selection` field.
///
/// # Example
/// ```ignore
/// // Resolved CV struct (you define this + CvKind impl)
/// pub struct Size { group: usize }
///
/// // Generates SizeBuilder with selection field
/// impl_single_group_builder!(Size, "size", |group| Size { group });
/// ```
#[macro_export]
macro_rules! impl_single_group_builder {
    ($cv:ident, $name:literal, |$group:ident| $construct:expr) => {
        ::paste::paste! {
            #[doc = "Builder for " $cv " CV."]
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            pub struct [<$cv Builder>] {
                pub selection: $crate::selection::Selection,
            }

            #[typetag::serde(name = $name)]
            impl $crate::collective_variable::CvKindBuilder for [<$cv Builder>] {
                fn build(
                    &self,
                    context: &dyn $crate::collective_variable::EvalContext,
                ) -> anyhow::Result<Box<dyn $crate::collective_variable::CvKind>> {
                    let indices = self.selection.resolve_groups(
                        context.topology_ref(),
                        context.groups(),
                        &|i| context.atom_kind(i),
                    );
                    if indices.len() != 1 {
                        anyhow::bail!(
                            "{}: selection '{}' must match exactly one group, found {}",
                            stringify!($cv),
                            self.selection,
                            indices.len()
                        );
                    }
                    let $group = indices[0];
                    Ok(Box::new($construct))
                }
                fn description(&self) -> Option<String> {
                    Some(format!("selection: {}", self.selection))
                }
            }
        }
    };
}

/// Defines a builder that resolves a single group selection with projection.
///
/// Generates `{Name}Builder` struct with `selection` and `projection` fields.
///
/// # Example
/// ```ignore
/// pub struct EndToEnd { projection: Axes, group: usize }
///
/// impl_single_group_with_dim_builder!(EndToEnd, "end_to_end",
///     |projection, group| EndToEnd { projection, group });
/// ```
#[macro_export]
macro_rules! impl_single_group_with_dim_builder {
    ($cv:ident, $name:literal, |$dim:ident, $group:ident| $construct:expr, requires_com) => {
        $crate::impl_single_group_with_dim_builder!(@inner $cv, $name, |$dim, $group| $construct, true);
    };
    ($cv:ident, $name:literal, |$dim:ident, $group:ident| $construct:expr) => {
        $crate::impl_single_group_with_dim_builder!(@inner $cv, $name, |$dim, $group| $construct, false);
    };
    (@inner $cv:ident, $name:literal, |$dim:ident, $group:ident| $construct:expr, $check_com:expr) => {
        ::paste::paste! {
            #[doc = "Builder for " $cv " CV."]
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            pub struct [<$cv Builder>] {
                pub selection: $crate::selection::Selection,
                #[serde(default, alias = "dimension")]
                pub projection: $crate::axes::Axes,
            }

            #[typetag::serde(name = $name)]
            impl $crate::collective_variable::CvKindBuilder for [<$cv Builder>] {
                fn build(
                    &self,
                    context: &dyn $crate::collective_variable::EvalContext,
                ) -> anyhow::Result<Box<dyn $crate::collective_variable::CvKind>> {
                    let indices = self.selection.resolve_groups(
                        context.topology_ref(),
                        context.groups(),
                        &|i| context.atom_kind(i),
                    );
                    if indices.len() != 1 {
                        anyhow::bail!(
                            "{}: selection '{}' must match exactly one group, found {}",
                            stringify!($cv),
                            self.selection,
                            indices.len()
                        );
                    }
                    if $check_com {
                        $crate::collective_variable::require_group_com(
                            indices[0], context, stringify!($cv)
                        )?;
                    }
                    let $dim = self.projection;
                    let $group = indices[0];
                    Ok(Box::new($construct))
                }
                fn description(&self) -> Option<String> {
                    Some(format!("selection: {}, projection: {:?}", self.selection, self.projection))
                }
            }
        }
    };
}

/// Defines a builder that resolves two group selections with projection.
///
/// Generates `{Name}Builder` struct with `selection`, `selection2`, and `projection` fields.
///
/// # Example
/// ```ignore
/// pub struct MassCenterSeparation { projection: Axes, group1: usize, group2: usize }
///
/// impl_two_group_with_dim_builder!(MassCenterSeparation, "mass_center_separation",
///     |projection, group1, group2| MassCenterSeparation { projection, group1, group2 });
/// ```
#[macro_export]
macro_rules! impl_two_group_with_dim_builder {
    ($cv:ident, $name:literal, |$dim:ident, $g1:ident, $g2:ident| $construct:expr, requires_com) => {
        $crate::impl_two_group_with_dim_builder!(@inner $cv, $name, |$dim, $g1, $g2| $construct, true);
    };
    ($cv:ident, $name:literal, |$dim:ident, $g1:ident, $g2:ident| $construct:expr) => {
        $crate::impl_two_group_with_dim_builder!(@inner $cv, $name, |$dim, $g1, $g2| $construct, false);
    };
    (@inner $cv:ident, $name:literal, |$dim:ident, $g1:ident, $g2:ident| $construct:expr, $check_com:expr) => {
        ::paste::paste! {
            #[doc = "Builder for " $cv " CV."]
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            pub struct [<$cv Builder>] {
                pub selection: $crate::selection::Selection,
                pub selection2: $crate::selection::Selection,
                #[serde(default, alias = "dimension")]
                pub projection: $crate::axes::Axes,
            }

            #[typetag::serde(name = $name)]
            impl $crate::collective_variable::CvKindBuilder for [<$cv Builder>] {
                fn build(
                    &self,
                    context: &dyn $crate::collective_variable::EvalContext,
                ) -> anyhow::Result<Box<dyn $crate::collective_variable::CvKind>> {
                    let indices1 = self.selection.resolve_groups(
                        context.topology_ref(),
                        context.groups(),
                        &|i| context.atom_kind(i),
                    );
                    if indices1.len() != 1 {
                        anyhow::bail!(
                            "{}: selection '{}' must match exactly one group, found {}",
                            stringify!($cv),
                            self.selection,
                            indices1.len()
                        );
                    }
                    let indices2 = self.selection2.resolve_groups(
                        context.topology_ref(),
                        context.groups(),
                        &|i| context.atom_kind(i),
                    );
                    if indices2.len() != 1 {
                        anyhow::bail!(
                            "{}: selection2 '{}' must match exactly one group, found {}",
                            stringify!($cv),
                            self.selection2,
                            indices2.len()
                        );
                    }
                    if $check_com {
                        $crate::collective_variable::require_group_com(
                            indices1[0], context, stringify!($cv)
                        )?;
                        $crate::collective_variable::require_group_com(
                            indices2[0], context, stringify!($cv)
                        )?;
                    }
                    let $dim = self.projection;
                    let $g1 = indices1[0];
                    let $g2 = indices2[0];
                    Ok(Box::new($construct))
                }
                fn description(&self) -> Option<String> {
                    Some(format!(
                        "selection: {}, selection2: {}, projection: {:?}",
                        self.selection, self.selection2, self.projection
                    ))
                }
            }
        }
    };
}

// Re-export for use in submodules
pub(crate) use impl_self_building_cv;
pub(crate) use impl_single_group_builder;
pub(crate) use impl_single_group_with_dim_builder;
pub(crate) use impl_two_group_with_dim_builder;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_builder_rejects_unknown_field() {
        // #73: each concrete CvKindBuilder now denies unknown fields. typetag
        // strips the `property` tag before the builder sees the map, so a genuine
        // typo is caught rather than swallowed.
        assert!(
            serde_yml::from_str::<Box<dyn CvKindBuilder>>("property: volume\nbogus: 3").is_err(),
            "a single-field builder must reject an unknown key",
        );
        assert!(
            serde_yml::from_str::<Box<dyn CvKindBuilder>>(
                "property: mass_center_separation\nselection: all\nselection2: all\nbogus: 3"
            )
            .is_err(),
            "a two-group builder must reject an unknown key",
        );
        // The valid form still parses.
        assert!(serde_yml::from_str::<Box<dyn CvKindBuilder>>("property: volume").is_ok());
    }
}
