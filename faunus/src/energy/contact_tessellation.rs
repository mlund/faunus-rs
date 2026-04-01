// Copyright 2026 Mikael Lund
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

//! # Contact tessellation energy between rigid bodies.
//!
//! For each pair of nearby rigid bodies, a radical tessellation of their combined
//! atoms yields inter-body contact areas. Each contact is weighted by a
//! geometric-mean surface tension: `γ_ab = sqrt(γ_a × γ_b)` when both signs agree,
//! zero otherwise.

use crate::cell::{BoundaryConditions, Shape};
use crate::group::Group;
use crate::topology::AtomKind;
use crate::{Change, Context};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use voronota_ltr::{compute_contacts_only, Ball, PeriodicBox};

use super::nonbonded::cache::GroupEnergyCache;

/// Precomputed `γ_ab` matrix indexed by atom kind pairs.
///
/// `γ_ab = sqrt(|γ_a × γ_b|) × sign` when both γ have the same sign, zero otherwise.
#[derive(Debug, Clone, Default)]
struct GammaMatrix {
    data: Vec<f64>,
    n_kinds: usize,
}

impl GammaMatrix {
    fn new(atomkinds: &[AtomKind]) -> Self {
        let n_kinds = atomkinds.len();
        let mut data = vec![0.0; n_kinds * n_kinds];
        for (i, aki) in atomkinds.iter().enumerate() {
            let gamma_i = aki.surface_tension().unwrap_or(0.0);
            for (j, akj) in atomkinds.iter().enumerate() {
                let gamma_j = akj.surface_tension().unwrap_or(0.0);
                data[i * n_kinds + j] = combine_gamma(gamma_i, gamma_j);
            }
        }
        Self { data, n_kinds }
    }

    #[inline]
    fn get(&self, kind_a: usize, kind_b: usize) -> f64 {
        self.data[kind_a * self.n_kinds + kind_b]
    }
}

/// Geometric-mean combining rule: `sqrt(|γ_a × γ_b|)` with the common sign,
/// or zero if signs differ or either is zero.
fn combine_gamma(ga: f64, gb: f64) -> f64 {
    let product = ga * gb;
    if product <= 0.0 {
        // Zero if either is zero or signs differ (hydrophobic-hydrophilic)
        0.0
    } else {
        product.sqrt().copysign(ga)
    }
}

/// Convert a simulation cell to a voronota-ltr `PeriodicBox`.
/// Returns `None` for non-periodic cells.
fn make_periodic_box(cell: &crate::cell::Cell) -> Option<PeriodicBox> {
    cell.pbc()
        .is_some()
        .then(|| cell.bounding_box())
        .flatten()
        .map(|bb| {
            let h = bb / 2.0;
            PeriodicBox::from_corners((-h.x, -h.y, -h.z), (h.x, h.y, h.z))
        })
}

/// Check if two groups are beyond contact range using bounding spheres.
///
/// The bounding radius is the max distance from the mass center to any atom center.
/// For tessellation contact, two atoms can interact when the distance between their
/// centers is less than `sigma_i/2 + sigma_j/2 + 2*probe`. The `max_sigma` parameter
/// is a global upper bound on atom diameter across all atom kinds in the topology.
fn groups_beyond_cutoff(
    gi: &Group,
    gj: &Group,
    probe_radius: f64,
    max_sigma: f64,
    cell: &crate::cell::Cell,
) -> bool {
    match (
        gi.mass_center(),
        gj.mass_center(),
        gi.bounding_radius(),
        gj.bounding_radius(),
    ) {
        (Some(ci), Some(cj), Some(ri), Some(rj)) => {
            // bounding_radius covers atom centers; max_sigma + 2*probe accounts for
            // the largest possible expanded (radius + probe) overlap between any two atoms
            let threshold = ri + rj + max_sigma + 2.0 * probe_radius;
            cell.distance_squared(ci, cj) > threshold * threshold
        }
        // Conservative: if bounding info is missing, assume groups could interact
        _ => false,
    }
}

/// Append balls and atom kind indices for a group's active atoms.
fn append_group_balls(
    context: &impl Context,
    group: &Group,
    atomkinds: &[AtomKind],
    balls: &mut Vec<Ball>,
    atom_kinds: &mut Vec<usize>,
) {
    for i in group.iter_active() {
        let pos = context.position(i);
        let kind = context.atom_kind(i);
        let radius = atomkinds[kind].sigma().map(|s| s / 2.0).unwrap_or(0.0);
        balls.push(Ball::new(pos.x, pos.y, pos.z, radius));
        atom_kinds.push(kind);
    }
}

/// Compute contact energy for a single body pair (unscaled).
fn pair_contact_energy(
    context: &impl Context,
    gi: &Group,
    gj: &Group,
    gamma: &GammaMatrix,
    probe_radius: f64,
    periodic_box: Option<&PeriodicBox>,
) -> f64 {
    let atomkinds = context.topology_ref().atomkinds();
    let atoms_i = gi.iter_active().len();
    let atoms_j = gj.iter_active().len();
    let mut balls = Vec::with_capacity(atoms_i + atoms_j);
    let mut atom_kinds = Vec::with_capacity(atoms_i + atoms_j);

    append_group_balls(context, gi, atomkinds, &mut balls, &mut atom_kinds);
    append_group_balls(context, gj, atomkinds, &mut balls, &mut atom_kinds);

    // voronota-ltr groups parameter: 0 = body i, 1 = body j.
    // Only inter-group contacts are computed; intra-body contacts are skipped.
    let mut groups = vec![0i32; atoms_i];
    groups.resize(atoms_i + atoms_j, 1);

    let contacts = compute_contacts_only(&balls, probe_radius, periodic_box, Some(&groups));

    contacts
        .iter()
        .map(|c| gamma.get(atom_kinds[c.id_a], atom_kinds[c.id_b]) * c.area)
        .sum()
}

/// Compute all pairwise energies and group energy sums.
fn compute_all_pairwise(
    context: &impl Context,
    groups: &[Group],
    probe_radius: f64,
    scaling: f64,
    max_sigma: f64,
    gamma: &GammaMatrix,
    periodic_box: Option<&PeriodicBox>,
) -> (Vec<f64>, Vec<f64>) {
    let n_groups = groups.len();
    let mut pairwise = vec![0.0; n_groups * n_groups];
    let mut group_energies = vec![0.0; n_groups];
    for i in 0..n_groups {
        for j in (i + 1)..n_groups {
            if groups_beyond_cutoff(
                &groups[i],
                &groups[j],
                probe_radius,
                max_sigma,
                context.cell(),
            ) {
                continue;
            }
            let pair_energy = scaling
                * pair_contact_energy(
                    context,
                    &groups[i],
                    &groups[j],
                    gamma,
                    probe_radius,
                    periodic_box,
                );
            pairwise[i * n_groups + j] = pair_energy;
            pairwise[j * n_groups + i] = pair_energy;
            group_energies[i] += pair_energy;
            group_energies[j] += pair_energy;
        }
    }
    (pairwise, group_energies)
}

#[derive(Debug, Clone, Builder)]
#[builder(derive(Deserialize, Serialize, Debug))]
#[builder_struct_attr(serde(deny_unknown_fields))]
pub struct ContactTessellationEnergy {
    /// Probe radius for tessellation (Å)
    probe_radius: f64,

    /// Global scaling factor for the total energy
    #[builder(default = "1.0")]
    scaling: f64,

    /// Precomputed gamma combining rule matrix
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    gamma: GammaMatrix,

    /// Pairwise energy cache with O(1) group energy lookup and symmetric backup/undo
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    cache: GroupEnergyCache,

    /// Max atom diameter across all atom kinds (for bounding sphere cutoff)
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    max_sigma: f64,

    /// Cached periodic box (rebuilt on volume change)
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    periodic_box: Option<PeriodicBox>,
}

impl ContactTessellationEnergy {
    /// Construct from builder and live context.
    pub(super) fn from_builder(
        builder: &ContactTessellationEnergyBuilder,
        context: &impl Context,
    ) -> anyhow::Result<Self> {
        let probe_radius = builder
            .probe_radius
            .ok_or_else(|| anyhow::anyhow!("contact_tessellation: probe_radius is required"))?;
        let scaling = builder.scaling.unwrap_or(1.0);
        let topology = context.topology();
        let atomkinds = topology.atomkinds();
        let gamma = GammaMatrix::new(atomkinds);
        let max_sigma = atomkinds
            .iter()
            .filter_map(|ak| ak.sigma())
            .fold(0.0_f64, f64::max);
        let periodic_box = make_periodic_box(context.cell());

        let (pairwise, group_energies) = compute_all_pairwise(
            context,
            context.groups(),
            probe_radius,
            scaling,
            max_sigma,
            &gamma,
            periodic_box.as_ref(),
        );
        let n_groups = context.groups().len();

        log::info!(
            "Contact tessellation: {} groups, probe={}, scaling={}, initial E={:.4} kJ/mol",
            n_groups,
            probe_radius,
            scaling,
            group_energies.iter().sum::<f64>() / 2.0,
        );

        Ok(Self {
            probe_radius,
            scaling,
            gamma,
            max_sigma,
            cache: GroupEnergyCache::new(pairwise, group_energies, n_groups),
            periodic_box,
        })
    }

    fn recompute_group_row(&mut self, context: &impl Context, k: usize) {
        let groups = context.groups();
        let n_groups = self.cache.n_groups;

        // Snapshot before overwriting — needed for O(n) delta update below
        let old_row: Vec<f64> = (0..n_groups)
            .map(|j| self.cache.pairwise[k * n_groups + j])
            .collect();

        for j in 0..n_groups {
            if j == k {
                continue;
            }
            let pair_energy = if groups_beyond_cutoff(
                &groups[k],
                &groups[j],
                self.probe_radius,
                self.max_sigma,
                context.cell(),
            ) {
                0.0
            } else {
                self.scaling
                    * pair_contact_energy(
                        context,
                        &groups[k],
                        &groups[j],
                        &self.gamma,
                        self.probe_radius,
                        self.periodic_box.as_ref(),
                    )
            };
            self.cache.pairwise[k * n_groups + j] = pair_energy;
            self.cache.pairwise[j * n_groups + k] = pair_energy;
        }

        // O(n) delta update instead of O(n²) full recompute:
        // group k gets a fresh sum; all others adjust by the per-pair difference
        self.cache.group_energies[k] = (0..n_groups)
            .filter(|&j| j != k)
            .map(|j| self.cache.pairwise[k * n_groups + j])
            .sum();
        for (j, &old_energy) in old_row.iter().enumerate() {
            if j != k {
                self.cache.group_energies[j] += self.cache.pairwise[k * n_groups + j] - old_energy;
            }
        }
    }

    fn rebuild_all(&mut self, context: &impl Context) {
        let (pairwise, group_energies) = compute_all_pairwise(
            context,
            context.groups(),
            self.probe_radius,
            self.scaling,
            self.max_sigma,
            &self.gamma,
            self.periodic_box.as_ref(),
        );
        self.cache.pairwise = pairwise;
        self.cache.group_energies = group_energies;
        self.cache.n_groups = context.groups().len();
    }

    pub(super) fn energy(&self, _context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(..) => {
                // Each pair counted in both group_energies[i] and [j], so halve
                self.cache.group_energies.iter().sum::<f64>() / 2.0
            }
            Change::SingleGroup(k, _) => self.cache.group_energies[*k],
            Change::Groups(changes) => {
                // Sum pair energies involving changed groups, counting shared pairs once
                let n_groups = self.cache.n_groups;
                let is_changed = |k: usize| changes.iter().any(|(idx, _)| *idx == k);
                let mut total = 0.0;
                for gi in 0..n_groups {
                    if !is_changed(gi) {
                        continue;
                    }
                    for gj in 0..n_groups {
                        if gj == gi {
                            continue;
                        }
                        // When both gi and gj changed, count only once (gi < gj)
                        if is_changed(gj) && gj < gi {
                            continue;
                        }
                        total += self.cache.pairwise[gi * n_groups + gj];
                    }
                }
                total
            }
            Change::None => 0.0,
        }
    }

    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything | Change::Volume(..) => {
                // Volume change may alter periodic box dimensions
                self.periodic_box = make_periodic_box(context.cell());
                self.rebuild_all(context);
            }
            Change::SingleGroup(k, _) => {
                self.recompute_group_row(context, *k);
            }
            Change::Groups(changes) => {
                for &(k, _) in changes {
                    self.recompute_group_row(context, k);
                }
            }
            Change::None => {}
        }
        Ok(())
    }

    pub(super) fn save_backup(&mut self, change: &Change) {
        if let Change::SingleGroup(k, _) = change {
            self.cache.save_backup(*k);
        }
    }

    pub(super) fn undo(&mut self) {
        self.cache.undo();
    }

    pub(super) fn discard_backup(&mut self) {
        self.cache.discard_backup();
    }

    pub(super) fn to_yaml(&self) -> serde_yml::Value {
        let mut map = serde_yml::Mapping::new();
        map.insert("probe_radius".into(), self.probe_radius.into());
        map.insert("scaling".into(), self.scaling.into());
        let total = self.cache.group_energies.iter().sum::<f64>() / 2.0;
        map.insert("total_energy".into(), total.into());
        serde_yml::Value::Mapping(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_same_sign_negative() {
        let g = combine_gamma(-0.5, -0.8);
        assert!(g < 0.0);
        assert!((g.abs() - (0.5_f64 * 0.8).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn gamma_same_sign_positive() {
        let g = combine_gamma(0.5, 0.8);
        assert!(g > 0.0);
        assert!((g - (0.5_f64 * 0.8).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn gamma_mixed_signs_is_zero() {
        assert_eq!(combine_gamma(-0.5, 0.8), 0.0);
        assert_eq!(combine_gamma(0.5, -0.8), 0.0);
    }

    #[test]
    fn gamma_zero_input_is_zero() {
        assert_eq!(combine_gamma(0.0, 0.8), 0.0);
        assert_eq!(combine_gamma(-0.5, 0.0), 0.0);
        assert_eq!(combine_gamma(0.0, 0.0), 0.0);
    }

    /// Integration test: two molecules with known geometry and surface tensions.
    /// Reference energy computed via standalone voronota-ltr compute_contacts_only.
    #[test]
    fn contact_tessellation_energy() {
        use crate::{backend::Backend, energy::EnergyChange, WithHamiltonian};
        let context = Backend::new(
            "tests/files/contact_tessellation.yaml",
            None,
            &mut rand::thread_rng(),
        )
        .unwrap();
        let energy = context
            .hamiltonian()
            .energy(&context, &crate::Change::Everything);
        // Reference: 1 contact (atoms 1,2), area=23.434317700371359
        // gamma_AB = -sqrt(0.5*0.8) = -0.632455532033676
        // energy = gamma_AB * area = -14.821163869034557
        float_cmp::assert_approx_eq!(f64, energy, -14.821163869034557, epsilon = 1e-6);
    }
}
