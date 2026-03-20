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

//! Group-based collective variables: Size, EndToEnd, GyrationRadius, DipoleMoment,
//! DipoleProduct, MassCenterPosition, MassCenterSeparation.
//!
//! These CVs resolve group indices at build time and evaluate against those fixed indices.

use super::{
    impl_single_group_builder, impl_single_group_with_dim_builder, impl_two_group_with_dim_builder,
    CvKind, EvalContext,
};
use crate::axes::Axes;
use crate::cell::BoundaryConditions;
use crate::geometry::{self, GyrationTensor};
use crate::group::GroupCollection;
use serde::{Deserialize, Serialize};

/// Compute the dipole moment vector of a group, PBC-aware relative to its COM.
fn group_dipole_moment(group_index: usize, context: &dyn EvalContext) -> Option<crate::Point> {
    let group = &context.groups()[group_index];
    let com = group.mass_center()?;
    let atomkinds = context.topology_ref().atomkinds();
    let charges_positions = group.iter_active().map(|i| {
        let charge = atomkinds[context.get_atomkind(i)].charge();
        (charge, GroupCollection::position(context, i))
    });
    Some(geometry::dipole_moment(
        charges_positions,
        com,
        context.cell(),
    ))
}

// ---------------------------------------------------------------------------
// Size
// ---------------------------------------------------------------------------

/// Number of active atoms in a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Size {
    group: usize,
}

#[typetag::serde(name = "size")]
impl CvKind for Size {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        context.groups()[self.group].len() as f64
    }

    fn name(&self) -> &'static str {
        "Size"
    }
}

impl_single_group_builder!(Size, "size", |group| Size { group });

// ---------------------------------------------------------------------------
// EndToEnd
// ---------------------------------------------------------------------------

/// Distance between first and last active atoms of a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndToEnd {
    #[serde(alias = "dimension")]
    projection: Axes,
    group: usize,
}

#[typetag::serde(name = "end_to_end")]
impl CvKind for EndToEnd {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let g = &context.groups()[self.group];
        let active = g.iter_active();
        let first = active.start;
        let last = active.end.saturating_sub(1);
        if first >= active.end {
            return 0.0;
        }
        self.projection
            .project(context.get_distance(first, last))
            .norm()
    }

    fn name(&self) -> &'static str {
        "EndToEnd"
    }
}

impl_single_group_with_dim_builder!(EndToEnd, "end_to_end", |projection, group| EndToEnd {
    projection,
    group
});

// ---------------------------------------------------------------------------
// GyrationRadius
// ---------------------------------------------------------------------------

/// Radius of gyration of a molecular group.
///
/// With default projection (`XYZ`), returns Rg = sqrt(trace(S)).
/// With a single axis (e.g. `X`), returns sqrt(Sxx) — the mass-weighted
/// spread along that axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GyrationRadius {
    #[serde(alias = "dimension")]
    projection: Axes,
    group: usize,
}

impl GyrationRadius {
    /// Compute mass-weighted gyration tensor for the group.
    fn gyration_tensor(&self, context: &dyn EvalContext) -> Option<GyrationTensor> {
        let group = &context.groups()[self.group];
        let com = group.mass_center()?;
        if group.len() < 2 {
            return None;
        }
        let atomkinds = context.topology_ref().atomkinds();
        let positions_masses = group.iter_active().map(|i| {
            let pos = GroupCollection::position(context, i);
            let mass = atomkinds[context.get_atomkind(i)].mass();
            (pos, mass)
        });
        GyrationTensor::from_positions_masses_com(positions_masses, com, context.cell())
    }
}

#[typetag::serde(name = "gyration_radius")]
impl CvKind for GyrationRadius {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        self.gyration_tensor(context)
            .map(|gt| {
                let diag =
                    crate::Point::new(gt.tensor[(0, 0)], gt.tensor[(1, 1)], gt.tensor[(2, 2)]);
                self.projection.project(diag).iter().sum::<f64>().sqrt()
            })
            .unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "GyrationRadius"
    }
}

impl_single_group_with_dim_builder!(
    GyrationRadius,
    "gyration_radius",
    |projection, group| { GyrationRadius { projection, group } },
    requires_com
);

// ---------------------------------------------------------------------------
// DipoleMoment
// ---------------------------------------------------------------------------

/// Electric dipole moment of a molecular group.
///
/// Computed as **μ** = Σ qᵢ · (**rᵢ** − **r_cm**) with PBC-aware distances.
/// With default projection (`xyz`), returns |**μ**|; with a single axis, returns
/// the signed component along that axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DipoleMoment {
    #[serde(alias = "dimension")]
    projection: Axes,
    group: usize,
}

#[typetag::serde(name = "dipole_moment")]
impl CvKind for DipoleMoment {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        group_dipole_moment(self.group, context)
            .map(|mu| {
                let filtered = self.projection.project(mu);
                if self.projection.dimension() == 1 {
                    // Single axis: return signed component
                    filtered.x + filtered.y + filtered.z
                } else {
                    filtered.norm()
                }
            })
            .unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "DipoleMoment"
    }
}

impl_single_group_with_dim_builder!(
    DipoleMoment,
    "dipole_moment",
    |projection, group| { DipoleMoment { projection, group } },
    requires_com
);

// ---------------------------------------------------------------------------
// MassCenterPosition
// ---------------------------------------------------------------------------

/// Position of the mass center of a group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassCenterPosition {
    #[serde(alias = "dimension")]
    projection: Axes,
    group: usize,
}

#[typetag::serde(name = "mass_center_position")]
impl CvKind for MassCenterPosition {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        context.groups()[self.group]
            .mass_center()
            .map(|com| self.projection.project(*com).norm())
            .unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "MassCenterPosition"
    }
}

impl_single_group_with_dim_builder!(
    MassCenterPosition,
    "mass_center_position",
    |projection, group| MassCenterPosition { projection, group },
    requires_com
);

// ---------------------------------------------------------------------------
// MassCenterSeparation (two groups)
// ---------------------------------------------------------------------------

/// Distance between mass centers of two groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassCenterSeparation {
    #[serde(alias = "dimension")]
    projection: Axes,
    group1: usize,
    group2: usize,
}

#[typetag::serde(name = "mass_center_separation")]
impl CvKind for MassCenterSeparation {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        let groups = context.groups();
        match (
            groups[self.group1].mass_center(),
            groups[self.group2].mass_center(),
        ) {
            (Some(a), Some(b)) => self
                .projection
                .project(context.cell().distance(a, b))
                .norm(),
            _ => 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "MassCenterSeparation"
    }
}

impl_two_group_with_dim_builder!(
    MassCenterSeparation,
    "mass_center_separation",
    |projection, group1, group2| MassCenterSeparation {
        projection,
        group1,
        group2
    },
    requires_com
);

// ---------------------------------------------------------------------------
// DipoleProduct (two groups)
// ---------------------------------------------------------------------------

/// Scalar product of normalized dipole moment vectors: **μ̂₁ · μ̂₂** = cos(θ).
///
/// Tracks orientational correlations between molecular dipoles.
/// Returns values in \[−1, 1\]; +1 = parallel, −1 = antiparallel.
/// With a single axis, normalizes then returns the component-wise product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DipoleProduct {
    #[serde(alias = "dimension")]
    projection: Axes,
    group1: usize,
    group2: usize,
}

#[typetag::serde(name = "dipole_product")]
impl CvKind for DipoleProduct {
    fn evaluate(&self, context: &dyn EvalContext) -> f64 {
        match (
            group_dipole_moment(self.group1, context),
            group_dipole_moment(self.group2, context),
        ) {
            (Some(mu1), Some(mu2)) => {
                // Filter before normalizing so single-axis mode compares
                // only the selected component(s)
                let a = self.projection.project(mu1);
                let b = self.projection.project(mu2);
                let norm_a = a.norm();
                let norm_b = b.norm();
                // Guard against zero-magnitude dipoles (e.g. net-neutral group)
                if norm_a > 0.0 && norm_b > 0.0 {
                    a.dot(&b) / (norm_a * norm_b)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn name(&self) -> &'static str {
        "DipoleProduct"
    }
}

impl_two_group_with_dim_builder!(
    DipoleProduct,
    "dipole_product",
    |projection, group1, group2| DipoleProduct {
        projection,
        group1,
        group2
    },
    requires_com
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::context::WithTopology;
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

    /// Find the first two MOL2 group indices (they have COM enabled).
    fn mol2_group_indices(ctx: &Backend) -> (usize, usize) {
        let topo = ctx.topology_ref();
        let groups = ctx.groups();
        let mol2_id = topo
            .moleculekinds()
            .iter()
            .position(|m| m.name() == "MOL2")
            .expect("MOL2 not found");
        let mut indices = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.molecule() == mol2_id)
            .map(|(i, _)| i);
        (indices.next().unwrap(), indices.next().unwrap())
    }

    #[test]
    fn dipole_product_same_group() {
        let ctx = make_context();
        let (g, _) = mol2_group_indices(&ctx);
        let cv = DipoleProduct {
            projection: Axes::default(),
            group1: g,
            group2: g,
        };
        // Same group → μ̂·μ̂ = cos(0) = 1
        let value = cv.evaluate(&ctx);
        assert!(
            (value - 1.0).abs() < 1e-10,
            "μ̂·μ̂ should be 1.0, got {value}"
        );
    }

    #[test]
    fn dipole_product_in_range() {
        let ctx = make_context();
        let (g1, g2) = mol2_group_indices(&ctx);
        let cv = DipoleProduct {
            projection: Axes::default(),
            group1: g1,
            group2: g2,
        };
        let value = cv.evaluate(&ctx);
        assert!(
            (-1.0 - 1e-10..=1.0 + 1e-10).contains(&value),
            "normalized dot product must be in [-1, 1], got {value}"
        );
    }

    #[test]
    fn dipole_product_no_com_returns_zero() {
        let ctx = make_context();
        // MOL groups (index 0) have has_com: false
        let cv = DipoleProduct {
            projection: Axes::default(),
            group1: 0,
            group2: 0,
        };
        assert_eq!(cv.evaluate(&ctx), 0.0);
    }

    #[test]
    fn dipole_product_serde_roundtrip() {
        let cv = DipoleProduct {
            projection: Axes::default(),
            group1: 0,
            group2: 1,
        };
        let yaml = serde_yml::to_string(&cv as &dyn CvKind).unwrap();
        let _: Box<dyn CvKind> = serde_yml::from_str(&yaml).unwrap();
    }
}
