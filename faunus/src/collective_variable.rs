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
//! Multiple CVs can be composed into multidimensional coordinates by consumers
//! (e.g. Penalty, Constrain, Analysis) holding `Vec<Box<dyn CollectiveVariable<T>>>`.
//!
//! The [`Dimension`] enum controls projection of vector quantities onto axes,
//! and the [`Selection`] language targets atoms or groups.

use crate::cell::{BoundaryConditions, Shape};
use crate::dimension::{default_dimension, Dimension};
use crate::selection::Selection;
use crate::Context;
use anyhow::{bail, Result};
use dyn_clone::DynClone;
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

/// A scalar observable of the simulation state.
///
/// Each CV is 1D. Consumers compose multi-dimensional coordinates
/// by holding `Vec<Box<dyn CollectiveVariable<T>>>`.
pub trait CollectiveVariable<T: Context>: std::fmt::Debug + crate::Info + DynClone {
    fn evaluate(&self, context: &T) -> f64;
    fn axis(&self) -> &AxisDescriptor;
    fn in_range(&self, value: f64) -> bool {
        self.axis().in_range(value)
    }
}

// Enables `Box<dyn CollectiveVariable<T>>` cloning, needed for
// future Penalty/Constrain energy terms where EnergyTerm derives Clone.
dyn_clone::clone_trait_object!(<T> CollectiveVariable<T> where T: Context);

// ---------------------------------------------------------------------------
// Property enum
// ---------------------------------------------------------------------------

/// Supported collective variable properties.
///
/// The variant determines which builder fields are required:
/// - No selection: `Volume`, `BoxLength`
/// - One atom (`selection`): `AtomPosition`
/// - One group (`selection`): `Size`, `EndToEnd`, `MassCenterPosition`
/// - Two groups (`selection` + `selection2`): `MassCenterSeparation`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Property {
    Volume,
    BoxLength,
    AtomPosition,
    Size,
    EndToEnd,
    MassCenterPosition,
    MassCenterSeparation,
}

// ---------------------------------------------------------------------------
// Builder
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
    #[serde(default = "default_dimension")]
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
    /// Resolve selections and construct a type-erased CV.
    ///
    /// Takes `context` (not just topology) because selection resolution
    /// needs live group state to determine which groups are active.
    pub fn build<T: Context>(&self, context: &T) -> Result<Box<dyn CollectiveVariable<T>>> {
        self.build_concrete(context).map(|cv| cv.into_boxed())
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

// ---------------------------------------------------------------------------
// Concrete CV types (private; users see Box<dyn CollectiveVariable<T>>)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct VolumeCV {
    pub(crate) axis: AxisDescriptor,
}

impl crate::Info for VolumeCV {
    fn short_name(&self) -> Option<&'static str> {
        Some("volume")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Simulation cell volume")
    }
}

impl<T: Context> CollectiveVariable<T> for VolumeCV {
    fn evaluate(&self, context: &T) -> f64 {
        context.cell().volume().unwrap_or(f64::INFINITY)
    }
    fn axis(&self) -> &AxisDescriptor {
        &self.axis
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BoxLengthCV {
    /// Index into `bounding_box()` Point: 0=x, 1=y, 2=z.
    component: usize,
    axis: AxisDescriptor,
}

impl crate::Info for BoxLengthCV {
    fn short_name(&self) -> Option<&'static str> {
        Some("box_length")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Simulation cell side length")
    }
}

impl<T: Context> CollectiveVariable<T> for BoxLengthCV {
    fn evaluate(&self, context: &T) -> f64 {
        context
            .cell()
            .bounding_box()
            .map(|bb| 2.0 * bb[self.component]) // bounding_box returns half-widths
            .unwrap_or(f64::INFINITY)
    }
    fn axis(&self) -> &AxisDescriptor {
        &self.axis
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AtomPositionCV {
    dimension: Dimension,
    index: usize,
    axis: AxisDescriptor,
}

impl crate::Info for AtomPositionCV {
    fn short_name(&self) -> Option<&'static str> {
        Some("atom_position")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Atom position projected onto dimension")
    }
}

impl<T: Context> CollectiveVariable<T> for AtomPositionCV {
    fn evaluate(&self, context: &T) -> f64 {
        let pos = context.position(self.index);
        self.dimension.filter(pos).norm()
    }
    fn axis(&self) -> &AxisDescriptor {
        &self.axis
    }
}

/// Discriminant for [`GroupCV`] — all share the same resolved fields.
#[derive(Debug, Clone)]
pub(crate) enum GroupProperty {
    Size,
    EndToEnd,
    MassCenterPosition,
}

#[derive(Debug, Clone)]
pub(crate) struct GroupCV {
    property: GroupProperty,
    dimension: Dimension,
    group: usize,
    axis: AxisDescriptor,
}

impl crate::Info for GroupCV {
    fn short_name(&self) -> Option<&'static str> {
        match self.property {
            GroupProperty::Size => Some("size"),
            GroupProperty::EndToEnd => Some("end_to_end"),
            GroupProperty::MassCenterPosition => Some("mass_center_position"),
        }
    }
    fn long_name(&self) -> Option<&'static str> {
        match self.property {
            GroupProperty::Size => Some("Number of active particles in group"),
            GroupProperty::EndToEnd => Some("End-to-end distance of a molecular group"),
            GroupProperty::MassCenterPosition => Some("Mass center position of a molecular group"),
        }
    }
}

impl<T: Context> CollectiveVariable<T> for GroupCV {
    fn evaluate(&self, context: &T) -> f64 {
        match self.property {
            GroupProperty::Size => context.groups()[self.group].len() as f64,
            GroupProperty::EndToEnd => {
                let group = &context.groups()[self.group];
                let active = group.iter_active();
                let first = active.start;
                let last = active.end.saturating_sub(1);
                if first >= active.end {
                    return 0.0;
                }
                let dist = context.get_distance(first, last);
                self.dimension.filter(dist).norm()
            }
            GroupProperty::MassCenterPosition => context.groups()[self.group]
                .mass_center()
                .map(|com| self.dimension.filter(*com).norm())
                .unwrap_or(0.0),
        }
    }
    fn axis(&self) -> &AxisDescriptor {
        &self.axis
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MassCenterSeparationCV {
    dimension: Dimension,
    group1: usize,
    group2: usize,
    axis: AxisDescriptor,
}

impl crate::Info for MassCenterSeparationCV {
    fn short_name(&self) -> Option<&'static str> {
        Some("mass_center_separation")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Distance between two group mass centers")
    }
}

impl<T: Context> CollectiveVariable<T> for MassCenterSeparationCV {
    fn evaluate(&self, context: &T) -> f64 {
        let groups = context.groups();
        let com1 = groups[self.group1].mass_center();
        let com2 = groups[self.group2].mass_center();
        match (com1, com2) {
            (Some(a), Some(b)) => {
                let dist = context.cell().distance(a, b);
                self.dimension.filter(dist).norm()
            }
            _ => 0.0,
        }
    }
    fn axis(&self) -> &AxisDescriptor {
        &self.axis
    }
}

// ---------------------------------------------------------------------------
// ConcreteCollectiveVariable — non-generic wrapper for energy terms
// ---------------------------------------------------------------------------

/// Energy terms like `Constrain` need to store a CV without a generic type
/// parameter because `EnergyTerm` is a concrete enum. This wraps each CV
/// type so it can be held directly.
#[derive(Debug, Clone)]
pub(crate) enum ConcreteCollectiveVariable {
    Volume(VolumeCV),
    BoxLength(BoxLengthCV),
    AtomPosition(AtomPositionCV),
    Group(GroupCV),
    MassCenterSeparation(MassCenterSeparationCV),
}

impl ConcreteCollectiveVariable {
    /// Convert into a type-erased trait object.
    fn into_boxed<T: Context>(self) -> Box<dyn CollectiveVariable<T>> {
        match self {
            Self::Volume(cv) => Box::new(cv),
            Self::BoxLength(cv) => Box::new(cv),
            Self::AtomPosition(cv) => Box::new(cv),
            Self::Group(cv) => Box::new(cv),
            Self::MassCenterSeparation(cv) => Box::new(cv),
        }
    }

    pub fn evaluate(&self, context: &impl Context) -> f64 {
        match self {
            Self::Volume(cv) => cv.evaluate(context),
            Self::BoxLength(cv) => cv.evaluate(context),
            Self::AtomPosition(cv) => cv.evaluate(context),
            Self::Group(cv) => cv.evaluate(context),
            Self::MassCenterSeparation(cv) => cv.evaluate(context),
        }
    }

    pub const fn axis(&self) -> &AxisDescriptor {
        match self {
            Self::Volume(cv) => &cv.axis,
            Self::BoxLength(cv) => &cv.axis,
            Self::AtomPosition(cv) => &cv.axis,
            Self::Group(cv) => &cv.axis,
            Self::MassCenterSeparation(cv) => &cv.axis,
        }
    }
}

impl CollectiveVariableBuilder {
    /// Resolve selections and construct a [`ConcreteCollectiveVariable`].
    ///
    /// This is the primary construction method. [`build`](Self::build) delegates
    /// here and wraps the result into a trait object.
    pub(crate) fn build_concrete(
        &self,
        context: &impl Context,
    ) -> Result<ConcreteCollectiveVariable> {
        let axis = AxisDescriptor {
            name: format!("{:?}", self.property),
            min: self.range.0,
            max: self.range.1,
            resolution: self.resolution,
        };

        match &self.property {
            Property::Volume => {
                reject_selection(self)?;
                Ok(ConcreteCollectiveVariable::Volume(VolumeCV { axis }))
            }
            Property::BoxLength => {
                reject_selection(self)?;
                let component = single_component(&self.dimension)?;
                Ok(ConcreteCollectiveVariable::BoxLength(BoxLengthCV {
                    component,
                    axis,
                }))
            }
            Property::AtomPosition => {
                let index = resolve_one_atom(self, context)?;
                Ok(ConcreteCollectiveVariable::AtomPosition(AtomPositionCV {
                    dimension: self.dimension,
                    index,
                    axis,
                }))
            }
            Property::Size | Property::EndToEnd | Property::MassCenterPosition => {
                let group = resolve_one_group(self, context)?;
                let property = match self.property {
                    Property::Size => GroupProperty::Size,
                    Property::EndToEnd => GroupProperty::EndToEnd,
                    Property::MassCenterPosition => GroupProperty::MassCenterPosition,
                    _ => unreachable!(),
                };
                Ok(ConcreteCollectiveVariable::Group(GroupCV {
                    property,
                    dimension: self.dimension,
                    group,
                    axis,
                }))
            }
            Property::MassCenterSeparation => {
                let (group1, group2) = resolve_two_groups(self, context)?;
                Ok(ConcreteCollectiveVariable::MassCenterSeparation(
                    MassCenterSeparationCV {
                        dimension: self.dimension,
                        group1,
                        group2,
                        axis,
                    },
                ))
            }
        }
    }
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

#[cfg(all(test, feature = "chemfiles"))]
mod integration_tests {
    use super::*;
    use crate::cell::Shape;
    use crate::context::{WithCell, WithTopology};
    use crate::group::GroupCollection;
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

    fn builder(property: Property) -> CollectiveVariableBuilder {
        CollectiveVariableBuilder {
            property,
            range: (-1e10, 1e10),
            ..Default::default()
        }
    }

    /// Find a molecule kind that has exactly one group instance in the test context.
    fn single_group_selection(ctx: &ReferencePlatform) -> Option<(Selection, usize)> {
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
}
