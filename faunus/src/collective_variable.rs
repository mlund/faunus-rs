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
//! with a [`CvKind`] discriminant that holds resolved data for evaluation.
//!
//! To add a new CV, add a single entry to the [`define_cv_kinds!`] invocation.
//!
//! The [`Dimension`] enum controls projection of vector quantities onto axes,
//! and the [`Selection`] language targets atoms or groups.

use crate::cell::{BoundaryConditions, Shape};
use crate::dimension::Dimension;
use crate::selection::Selection;
use crate::Context;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

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

// ---------------------------------------------------------------------------
// Macro: co-locates Property variant, CvKind variant, build, and evaluate
// ---------------------------------------------------------------------------

/// Generates [`Property`] (YAML enum), [`CvKind`] (resolved data),
/// and the [`from_builder`](CvKind::from_builder)/[`evaluate`](CvKind::evaluate) match arms.
///
/// Each entry defines one collective variable:
/// ```text
/// PropertyName => VariantName(field: Type, ...) {
///     build(builder, context) { ... -> Result<CvKind> }
///     eval(context) { ... -> f64 }    // fields from the variant are in scope
/// }
/// ```
macro_rules! define_cv_kinds {
    ($(
        $(#[$meta:meta])*
        $prop:ident => $variant:ident $(( $($field:ident : $ty:ty),* $(,)? ))? {
            build($builder:ident, $bctx:ident) $build_body:block
            eval($ectx:ident) $eval_body:block
        }
    )*) => {
        /// Supported collective variable properties (YAML `property:` field).
        ///
        /// The variant determines which builder fields are required;
        /// see [`CollectiveVariableBuilder`].
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum Property {
            $( $(#[$meta])* $prop, )*
        }

        /// Resolved data for each CV type.
        #[derive(Debug, Clone)]
        enum CvKind {
            $( $variant $({ $($field: $ty),* })?, )*
        }

        impl CvKind {
            fn from_builder(
                builder: &CollectiveVariableBuilder,
                context: &impl Context,
            ) -> Result<Self> {
                #[allow(unused_variables)]
                match &builder.property {
                    $( Property::$prop => {
                        let $builder = builder;
                        let $bctx = context;
                        $build_body
                    } )*
                }
            }

            fn evaluate(&self, context: &impl Context) -> f64 {
                match self {
                    $( Self::$variant $({ $($field),* })? => {
                        #[allow(unused_variables)]
                        let $ectx = context;
                        $eval_body
                    } )*
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// CV definitions — add new CVs here
// ---------------------------------------------------------------------------

define_cv_kinds! {
    Volume => Volume {
        build(b, _c) {
            reject_selection(b)?;
            Ok(CvKind::Volume)
        }
        eval(c) {
            c.cell().volume().unwrap_or(f64::INFINITY)
        }
    }

    BoxLength => BoxLength(component: usize) {
        build(b, _c) {
            reject_selection(b)?;
            Ok(CvKind::BoxLength {
                component: single_component(&b.dimension)?,
            })
        }
        eval(c) {
            c.cell()
                .bounding_box()
                .map(|bb| 2.0 * bb[*component])
                .unwrap_or(f64::INFINITY)
        }
    }

    AtomPosition => AtomPosition(dimension: Dimension, index: usize) {
        build(b, c) {
            Ok(CvKind::AtomPosition {
                dimension: b.dimension,
                index: resolve_one_atom(b, c)?,
            })
        }
        eval(c) {
            dimension.filter(c.position(*index)).norm()
        }
    }

    Size => Size(group: usize) {
        build(b, c) {
            Ok(CvKind::Size {
                group: resolve_one_group(b, c)?,
            })
        }
        eval(c) {
            c.groups()[*group].len() as f64
        }
    }

    EndToEnd => EndToEnd(dimension: Dimension, group: usize) {
        build(b, c) {
            Ok(CvKind::EndToEnd {
                dimension: b.dimension,
                group: resolve_one_group(b, c)?,
            })
        }
        eval(c) {
            let g = &c.groups()[*group];
            let active = g.iter_active();
            let first = active.start;
            let last = active.end.saturating_sub(1);
            if first >= active.end {
                return 0.0;
            }
            dimension.filter(c.get_distance(first, last)).norm()
        }
    }

    MassCenterPosition => MassCenterPosition(dimension: Dimension, group: usize) {
        build(b, c) {
            Ok(CvKind::MassCenterPosition {
                dimension: b.dimension,
                group: resolve_one_group(b, c)?,
            })
        }
        eval(c) {
            c.groups()[*group]
                .mass_center()
                .map(|com| dimension.filter(*com).norm())
                .unwrap_or(0.0)
        }
    }

    MassCenterSeparation => MassCenterSeparation(
        dimension: Dimension, group1: usize, group2: usize,
    ) {
        build(b, c) {
            let (group1, group2) = resolve_two_groups(b, c)?;
            Ok(CvKind::MassCenterSeparation {
                dimension: b.dimension,
                group1,
                group2,
            })
        }
        eval(c) {
            let groups = c.groups();
            match (groups[*group1].mass_center(), groups[*group2].mass_center()) {
                (Some(a), Some(b)) => dimension.filter(c.cell().distance(a, b)).norm(),
                _ => 0.0,
            }
        }
    }

    /// Number of active atoms matching a selection (re-resolves each evaluation).
    Count => Count(selection: Selection) {
        build(b, _c) {
            Ok(CvKind::Count {
                selection: require_selection(b)?,
            })
        }
        eval(c) {
            selection
                .resolve_atoms_live(c.topology_ref(), c.groups(), &|i| c.get_atomkind(i))
                .len() as f64
        }
    }

    /// Sum of charges of active atoms matching a selection (re-resolves each evaluation).
    Charge => Charge(selection: Selection) {
        build(b, _c) {
            Ok(CvKind::Charge {
                selection: require_selection(b)?,
            })
        }
        eval(c) {
            let indices = selection.resolve_atoms_live(
                c.topology_ref(),
                c.groups(),
                &|i| c.get_atomkind(i),
            );
            let atomkinds = c.topology_ref().atomkinds();
            indices
                .iter()
                .map(|&i| atomkinds[c.get_atomkind(i)].charge())
                .sum()
        }
    }
}

// ---------------------------------------------------------------------------
// CollectiveVariable — public wrapper
// ---------------------------------------------------------------------------

/// A scalar observable of the simulation state.
#[derive(Debug, Clone)]
pub struct CollectiveVariable {
    axis: AxisDescriptor,
    kind: CvKind,
}

impl CollectiveVariable {
    pub fn evaluate(&self, context: &impl Context) -> f64 {
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
// Builder (YAML deserialization)
// ---------------------------------------------------------------------------

fn default_range() -> (f64, f64) {
    (f64::NEG_INFINITY, f64::INFINITY)
}

/// Builder for constructing a collective variable from YAML or code.
///
/// Which fields are required depends on `property`; see [`Property`] docs.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveVariableBuilder {
    pub property: Property,
    #[serde(default = "default_range")]
    pub range: (f64, f64),
    #[serde(default)]
    pub dimension: Dimension,
    /// Only needed for Penalty (Wang-Landau) histogramming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection: Option<Selection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection2: Option<Selection>,
}

impl Default for CollectiveVariableBuilder {
    fn default() -> Self {
        Self {
            property: Property::Volume,
            range: (f64::NEG_INFINITY, f64::INFINITY),
            dimension: Dimension::XYZ,
            resolution: None,
            selection: None,
            selection2: None,
        }
    }
}

impl CollectiveVariableBuilder {
    /// Resolve selections and construct a [`CollectiveVariable`].
    pub fn build(&self, context: &impl Context) -> Result<CollectiveVariable> {
        let axis = AxisDescriptor {
            name: format!("{:?}", self.property),
            min: self.range.0,
            max: self.range.1,
            resolution: self.resolution,
        };
        Ok(CollectiveVariable {
            kind: CvKind::from_builder(self, context)?,
            axis,
        })
    }
}

// ---------------------------------------------------------------------------
// Selection resolution helpers
// ---------------------------------------------------------------------------

fn reject_selection(builder: &CollectiveVariableBuilder) -> Result<()> {
    if builder.selection.is_some() || builder.selection2.is_some() {
        bail!("{:?} does not use selections", builder.property);
    }
    Ok(())
}

/// Map dimension to a bounding_box index (0/1/2). Rejects multi-axis dimensions
/// because bounding_box returns half-widths per axis, not a combined length.
fn single_component(dim: &Dimension) -> Result<usize> {
    match dim {
        Dimension::X => Ok(0),
        Dimension::Y => Ok(1),
        Dimension::Z => Ok(2),
        _ => bail!("BoxLength requires dimension x, y, or z"),
    }
}

fn resolve_one_atom<T: Context>(builder: &CollectiveVariableBuilder, context: &T) -> Result<usize> {
    let sel = builder
        .selection
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("{:?} requires a 'selection' field", builder.property))?;
    let indices = sel.resolve_atoms(context.topology_ref(), context.groups());
    if indices.len() != 1 {
        bail!(
            "{:?}: selection '{}' must match exactly one atom, found {}",
            builder.property,
            sel,
            indices.len()
        );
    }
    Ok(indices[0])
}

fn resolve_one_group<T: Context>(
    builder: &CollectiveVariableBuilder,
    context: &T,
) -> Result<usize> {
    let sel = builder
        .selection
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("{:?} requires a 'selection' field", builder.property))?;
    let indices = sel.resolve_groups(context.topology_ref(), context.groups());
    if indices.len() != 1 {
        bail!(
            "{:?}: selection '{}' must match exactly one group, found {}",
            builder.property,
            sel,
            indices.len()
        );
    }
    Ok(indices[0])
}

fn resolve_two_groups<T: Context>(
    builder: &CollectiveVariableBuilder,
    context: &T,
) -> Result<(usize, usize)> {
    let group1 = resolve_one_group(builder, context)?;

    let sel2 = builder
        .selection2
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("{:?} requires a 'selection2' field", builder.property))?;
    let indices2 = sel2.resolve_groups(context.topology_ref(), context.groups());
    if indices2.len() != 1 {
        bail!(
            "{:?}: selection2 '{}' must match exactly one group, found {}",
            builder.property,
            sel2,
            indices2.len()
        );
    }
    Ok((group1, indices2[0]))
}

fn require_selection(builder: &CollectiveVariableBuilder) -> Result<Selection> {
    builder
        .selection
        .clone()
        .ok_or_else(|| anyhow::anyhow!("{:?} requires a 'selection' field", builder.property))
}

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

    #[test]
    fn deserialize_volume_builder() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder.property, Property::Volume));
        assert_eq!(builder.range, (1000.0, 5000.0));
        assert!(builder.resolution.is_none());
        assert!(builder.selection.is_none());
    }

    #[test]
    fn deserialize_volume_with_resolution() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
resolution: 0.5
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(builder.resolution, Some(0.5));
    }

    #[test]
    fn deserialize_box_length() {
        let yaml = r#"
property: box_length
dimension: z
range: [10.0, 100.0]
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder.property, Property::BoxLength));
        assert_eq!(builder.dimension, Dimension::Z);
    }

    #[test]
    fn deserialize_atom_position() {
        let yaml = r#"
property: atom_position
selection: "name CA and resid 1"
dimension: z
range: [-50.0, 50.0]
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder.property, Property::AtomPosition));
        assert_eq!(
            builder.selection.as_ref().unwrap().source(),
            "name CA and resid 1"
        );
        assert_eq!(builder.dimension, Dimension::Z);
    }

    #[test]
    fn deserialize_mass_center_separation() {
        let yaml = r#"
property: mass_center_separation
selection: "molecule MOL"
selection2: "molecule ION"
dimension: xyz
range: [0.0, 50.0]
resolution: 0.2
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder.property, Property::MassCenterSeparation));
        assert_eq!(builder.selection.as_ref().unwrap().source(), "molecule MOL");
        assert_eq!(
            builder.selection2.as_ref().unwrap().source(),
            "molecule ION"
        );
        assert_eq!(builder.resolution, Some(0.2));
    }

    #[test]
    fn deserialize_list_of_builders() {
        let yaml = r#"
- property: volume
  range: [1000.0, 5000.0]
  resolution: 0.5
- property: mass_center_separation
  selection: "molecule MOL"
  selection2: "molecule ION"
  range: [0.0, 50.0]
  resolution: 0.2
"#;
        let builders: Vec<CollectiveVariableBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(builders.len(), 2);
        assert!(matches!(builders[0].property, Property::Volume));
        assert!(matches!(
            builders[1].property,
            Property::MassCenterSeparation
        ));
    }

    #[test]
    fn deserialize_count() {
        let yaml = r#"
property: count
selection: "molecule M"
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder.property, Property::Count));
        assert_eq!(builder.selection.as_ref().unwrap().source(), "molecule M");
    }

    #[test]
    fn count_requires_selection() {
        let yaml = r#"
property: count
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder.property, Property::Count));
        assert!(builder.selection.is_none());
    }

    #[test]
    fn single_component_validation() {
        assert_eq!(single_component(&Dimension::X).unwrap(), 0);
        assert_eq!(single_component(&Dimension::Y).unwrap(), 1);
        assert_eq!(single_component(&Dimension::Z).unwrap(), 2);
        assert!(single_component(&Dimension::XY).is_err());
        assert!(single_component(&Dimension::XYZ).is_err());
    }

    #[test]
    fn roundtrip_serialize_builder() {
        let yaml = r#"
property: volume
range: [1000.0, 5000.0]
resolution: 0.5
"#;
        let builder: CollectiveVariableBuilder = serde_yaml::from_str(yaml).unwrap();
        let serialized = serde_yaml::to_string(&builder).unwrap();
        let roundtrip: CollectiveVariableBuilder = serde_yaml::from_str(&serialized).unwrap();
        assert!(matches!(roundtrip.property, Property::Volume));
        assert_eq!(roundtrip.range, (1000.0, 5000.0));
        assert_eq!(roundtrip.resolution, Some(0.5));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::backend::Backend;
    use crate::cell::Shape;
    use crate::context::{WithCell, WithTopology};
    use crate::group::GroupCollection;
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

    fn builder(property: Property) -> CollectiveVariableBuilder {
        CollectiveVariableBuilder {
            property,
            range: (-1e10, 1e10),
            ..Default::default()
        }
    }

    /// Find a molecule kind that has exactly one group instance in the test context.
    fn single_group_selection(ctx: &Backend) -> Option<(Selection, usize)> {
        ctx.topology_ref().moleculekinds().iter().find_map(|mk| {
            let sel = Selection::parse(&format!("molecule {}", mk.name())).ok()?;
            let groups = sel.resolve_groups(ctx.topology_ref(), ctx.groups());
            (groups.len() == 1).then(|| (sel, groups[0]))
        })
    }

    #[test]
    fn build_volume_cv() {
        let ctx = make_context();
        let cv = builder(Property::Volume).build(&ctx).unwrap();
        let expected = ctx.cell().volume().unwrap();
        assert!((cv.evaluate(&ctx) - expected).abs() < 1e-10);
    }

    #[test]
    fn build_box_length_cv() {
        let ctx = make_context();
        for (dim, idx) in [(Dimension::X, 0), (Dimension::Y, 1), (Dimension::Z, 2)] {
            let mut b = builder(Property::BoxLength);
            b.dimension = dim;
            let cv = b.build(&ctx).unwrap();
            let expected = 2.0 * ctx.cell().bounding_box().unwrap()[idx];
            assert!((cv.evaluate(&ctx) - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn build_box_length_rejects_multi_axis() {
        let ctx = make_context();
        let mut b = builder(Property::BoxLength);
        b.dimension = Dimension::XY;
        assert!(b.build(&ctx).is_err());
    }

    #[test]
    fn build_volume_rejects_selection() {
        let ctx = make_context();
        let mut b = builder(Property::Volume);
        b.selection = Some(Selection::parse("all").unwrap());
        assert!(b.build(&ctx).is_err());
    }

    #[test]
    fn build_size_cv() {
        let ctx = make_context();
        if let Some((sel, group_idx)) = single_group_selection(&ctx) {
            let expected = ctx.groups()[group_idx].len() as f64;
            let mut b = builder(Property::Size);
            b.selection = Some(sel);
            let cv = b.build(&ctx).unwrap();
            assert!((cv.evaluate(&ctx) - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn build_mass_center_position_cv() {
        let ctx = make_context();
        if let Some((sel, group_idx)) = single_group_selection(&ctx) {
            let mut b = builder(Property::MassCenterPosition);
            b.dimension = Dimension::Z;
            b.selection = Some(sel);
            let cv = b.build(&ctx).unwrap();

            let com = ctx.groups()[group_idx].mass_center().unwrap();
            let expected = Dimension::Z.filter(*com).norm();
            assert!((cv.evaluate(&ctx) - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn cv_in_range() {
        let ctx = make_context();
        let mut b = builder(Property::Volume);
        b.range = (0.0, 1.0);
        let cv = b.build(&ctx).unwrap();
        let value = cv.evaluate(&ctx);
        // Cell volume >> 1.0 for any real system
        assert!(!cv.in_range(value));
        assert!(cv.in_range(0.5));
    }

    #[test]
    fn cv_clone() {
        let ctx = make_context();
        let cv = builder(Property::Volume).build(&ctx).unwrap();
        let cloned = cv.clone();
        assert!((cv.evaluate(&ctx) - cloned.evaluate(&ctx)).abs() < 1e-10);
    }

    #[test]
    fn build_count_cv() {
        let ctx = make_context();
        let sel = Selection::parse("all").unwrap();
        let mut b = builder(Property::Count);
        b.selection = Some(sel);
        let cv = b.build(&ctx).unwrap();
        let expected: usize = ctx.groups().iter().map(|g| g.len()).sum();
        assert_eq!(cv.evaluate(&ctx) as usize, expected);
    }

    #[test]
    fn count_cv_with_active_inactive_groups() {
        let mut rng = rand::thread_rng();
        let ctx = Backend::new("tests/files/speciation_test.yaml", None, &mut rng).unwrap();
        let sel = Selection::parse("molecule M").unwrap();
        let mut b = builder(Property::Count);
        b.selection = Some(sel);
        let cv = b.build(&ctx).unwrap();
        // speciation_test.yaml: M has N=10, active=5 (single-atom molecules)
        assert_eq!(cv.evaluate(&ctx) as usize, 5);
    }

    #[test]
    fn count_cv_requires_selection() {
        let ctx = make_context();
        let b = builder(Property::Count);
        assert!(b.build(&ctx).is_err());
    }
}
