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
//! The [`CollectiveVariable`] struct pairs an [`AxisDescriptor`] (range, resolution)
//! with a [`CvKind`] trait object that evaluates the CV.
//!
//! Each CV type is defined in its own submodule and registered via `typetag`.

mod atom;
mod cell;
mod dynamic;
mod group;

use crate::Context;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// Re-export CV types for convenience
pub use atom::AtomPosition;
pub use cell::{BoxLength, Volume};
pub use dynamic::{Charge, Count};
pub use group::{EndToEnd, MassCenterPosition, MassCenterSeparation, Size};

/// Metadata for one axis of a collective variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisDescriptor {
    pub name: String,
    pub min: f64,
    pub max: f64,
    /// Only needed for Penalty (Wang-Landau) histogramming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<f64>,
}

impl AxisDescriptor {
    pub fn in_range(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }
}

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
    crate::group::GroupCollection + crate::context::WithCell + crate::context::WithTopology
{
    fn get_distance(&self, i: usize, j: usize) -> crate::Point;
    fn get_atomkind(&self, index: usize) -> usize;
}

/// A scalar observable of the simulation state.
///
/// Wraps a [`CvKind`] trait object with axis metadata (range, resolution).
#[derive(Debug, Clone, Deserialize)]
pub struct CollectiveVariable {
    #[serde(flatten)]
    axis: AxisDescriptor,
    #[serde(flatten)]
    kind: Box<dyn CvKind>,
}

impl CollectiveVariable {
    /// Create a new collective variable from a kind and axis descriptor.
    pub fn new(kind: Box<dyn CvKind>, axis: AxisDescriptor) -> Self {
        Self { axis, kind }
    }

    pub fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        self.kind.evaluate(context)
    }

    pub fn axis(&self) -> &AxisDescriptor {
        &self.axis
    }

    pub fn in_range(&self, value: f64) -> bool {
        self.axis.in_range(value)
    }
}

// ---------------------------------------------------------------------------
// Builder support for backward compatibility
// ---------------------------------------------------------------------------

fn default_range() -> (f64, f64) {
    (f64::NEG_INFINITY, f64::INFINITY)
}

/// Builder for constructing a collective variable from YAML.
///
/// This provides a two-phase construction: first deserialize the builder,
/// then call `build()` with context to resolve selections into indices.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CollectiveVariableBuilder {
    #[serde(default = "default_range")]
    pub range: (f64, f64),
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<f64>,
    #[serde(flatten)]
    kind_builder: Box<dyn CvKindBuilder>,
}

impl CollectiveVariableBuilder {
    /// Resolve selections and construct a [`CollectiveVariable`].
    pub fn build(&self, context: &impl Context) -> Result<CollectiveVariable> {
        let kind = self.kind_builder.build(context)?;
        let axis = AxisDescriptor {
            name: kind.name().to_string(),
            min: self.range.0,
            max: self.range.1,
            resolution: self.resolution,
        };
        Ok(CollectiveVariable { axis, kind })
    }
}

/// Trait for CV kind builders that resolve selections into indices.
#[typetag::serde(tag = "property")]
pub trait CvKindBuilder: Send + Sync + std::fmt::Debug + dyn_clone::DynClone {
    /// Build the CV kind by resolving selections against context.
    fn build(&self, context: &dyn EvalContext) -> Result<Box<dyn CvKind>>;
}

dyn_clone::clone_trait_object!(CvKindBuilder);

// Blanket impl: any Context that implements the required traits also implements EvalContext
impl<T> EvalContext for T
where
    T: crate::group::GroupCollection
        + crate::context::WithCell
        + crate::context::WithTopology
        + crate::context::ParticleSystem,
{
    fn get_distance(&self, i: usize, j: usize) -> crate::Point {
        <T as crate::context::ParticleSystem>::get_distance(self, i, j)
    }

    fn get_atomkind(&self, index: usize) -> usize {
        <T as crate::context::ParticleSystem>::get_atomkind(self, index)
    }
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
            }
        }
    };
}

/// Defines a builder that resolves a single group selection with dimension.
///
/// Generates `{Name}Builder` struct with `selection` and `dimension` fields.
///
/// # Example
/// ```ignore
/// pub struct EndToEnd { dimension: Dimension, group: usize }
///
/// impl_single_group_with_dim_builder!(EndToEnd, "end_to_end",
///     |dimension, group| EndToEnd { dimension, group });
/// ```
#[macro_export]
macro_rules! impl_single_group_with_dim_builder {
    ($cv:ident, $name:literal, |$dim:ident, $group:ident| $construct:expr) => {
        ::paste::paste! {
            #[doc = "Builder for " $cv " CV."]
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            pub struct [<$cv Builder>] {
                pub selection: $crate::selection::Selection,
                #[serde(default)]
                pub dimension: $crate::dimension::Dimension,
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
                    );
                    if indices.len() != 1 {
                        anyhow::bail!(
                            "{}: selection '{}' must match exactly one group, found {}",
                            stringify!($cv),
                            self.selection,
                            indices.len()
                        );
                    }
                    let $dim = self.dimension;
                    let $group = indices[0];
                    Ok(Box::new($construct))
                }
            }
        }
    };
}

/// Defines a builder that resolves a single atom selection with dimension.
///
/// Generates `{Name}Builder` struct with `selection` and `dimension` fields.
///
/// # Example
/// ```ignore
/// pub struct AtomPosition { dimension: Dimension, index: usize }
///
/// impl_single_atom_with_dim_builder!(AtomPosition, "atom_position",
///     |dimension, index| AtomPosition { dimension, index });
/// ```
#[macro_export]
macro_rules! impl_single_atom_with_dim_builder {
    ($cv:ident, $name:literal, |$dim:ident, $index:ident| $construct:expr) => {
        ::paste::paste! {
            #[doc = "Builder for " $cv " CV."]
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
            pub struct [<$cv Builder>] {
                pub selection: $crate::selection::Selection,
                #[serde(default)]
                pub dimension: $crate::dimension::Dimension,
            }

            #[typetag::serde(name = $name)]
            impl $crate::collective_variable::CvKindBuilder for [<$cv Builder>] {
                fn build(
                    &self,
                    context: &dyn $crate::collective_variable::EvalContext,
                ) -> anyhow::Result<Box<dyn $crate::collective_variable::CvKind>> {
                    let indices = self.selection.resolve_atoms(
                        context.topology_ref(),
                        context.groups(),
                    );
                    if indices.len() != 1 {
                        anyhow::bail!(
                            "{}: selection '{}' must match exactly one atom, found {}",
                            stringify!($cv),
                            self.selection,
                            indices.len()
                        );
                    }
                    let $dim = self.dimension;
                    let $index = indices[0];
                    Ok(Box::new($construct))
                }
            }
        }
    };
}

// Re-export for use in submodules
pub use impl_self_building_cv;
pub use impl_single_atom_with_dim_builder;
pub use impl_single_group_builder;
pub use impl_single_group_with_dim_builder;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_descriptor_in_range() {
        let axis = AxisDescriptor {
            name: "test".to_string(),
            min: 0.0,
            max: 10.0,
            resolution: None,
        };
        assert!(axis.in_range(0.0));
        assert!(axis.in_range(5.0));
        assert!(axis.in_range(10.0));
        assert!(!axis.in_range(-0.1));
        assert!(!axis.in_range(10.1));
    }
}
