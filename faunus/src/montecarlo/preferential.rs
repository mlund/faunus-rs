// Copyright 2023-2024 Mikael Lund
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

//! Preferential sampling for distance-biased atom selection.
//!
//! In dilute solution most trial moves displace bulk particles far from the solute. Selecting
//! particles near the solute more often concentrates the sampling where it matters, at the cost
//! of an asymmetric proposal — which the acceptance must then correct for, or the equilibrium
//! distribution shifts.
//!
//! See [Owicki & Scheraga, 1977](https://doi.org/10.1016/0009-2614(77)85051-3), whose eqn 5 gives
//! the acceptance in terms of the underlying transition probabilities, and
//! [Allen & Tildesley, 2017](https://doi.org/10.1093/oso/9780198803195.001.0001) §9.3.1,
//! eqns 9.42–9.44, whose continuous weight function this module implements.

use crate::auxiliary::ColumnWriter;
use crate::cell::BoundaryConditions;
use crate::group::{AtomKindId, GroupIndex, GroupSelection, ParticleSelection};
use crate::histogram::Histogram;
use crate::selection::{CachedSelection, Generation, Groups, Selection};
use crate::ObserveContext;
use crate::Point;
use average::{Estimate, Mean};
use log::{debug, warn};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn default_exponent() -> f64 {
    2.0
}

fn default_offset() -> f64 {
    1.0
}

/// Report only the bounding radii of the reference geometries; the mass centers are internal.
fn serialize_radii<S: serde::Serializer>(
    geometries: &[(Point, f64)],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(geometries.iter().map(|&(_, radius)| radius))
}

/// Acceptance correction for having selected an atom non-uniformly.
///
/// Selecting atom `i` with the normalized weight `W = W'(rᵢ) / Σⱼ W'(rⱼ)` of eqn 9.43 makes the
/// proposal asymmetric, so the acceptance carries the ratio of underlying transition
/// probabilities, eqn 9.44: `α_nm/α_mn = W_n/W_m`. Both are normalized weights *of the moved
/// atom* — `W_m` before the move, `W_n` after — and only that atom's `W'` changes, so the sum
/// shifts by `w_new - w_old`.
///
/// Returned as `ln(W_m/W_n)` rather than `ln(W_n/W_m)` because the acceptance adds the bias to
/// ΔU and applies `exp(-bias)`.
///
/// Arguments are the *unnormalized* weights `W'` of the moved atom before and after the move,
/// and `w_sum = Σⱼ W'(rⱼ)` before it. The sum is not `W`; conflating the two drops the
/// `w_new/w_old` factor, which depletes the neighbourhood of the reference.
fn acceptance_correction(w_old: f64, w_new: f64, w_sum: f64) -> f64 {
    let w_sum_new = w_sum - w_old + w_new;
    ((w_old / w_sum) / (w_new / w_sum_new)).ln()
}

/// The atoms a move may pick from, re-resolved only when the system changes under them.
///
/// Which atoms are eligible depends on group composition and atom identities, never on where the
/// atoms are — so this survives every ordinary Monte Carlo trial and is rebuilt only by a GCMC
/// insertion, an activation change, or a titration swap.
#[derive(Clone, Debug, Default)]
struct Candidates {
    /// Groups the move may draw from.
    groups: GroupSelection,
    /// Restrict to this atom kind, if the move names one.
    atom: Option<AtomKindId>,
    atoms: Vec<usize>,
    /// `None` until first resolved.
    generation: Option<Generation>,
}

impl Candidates {
    fn new(groups: GroupSelection, atom: Option<AtomKindId>) -> Self {
        Self {
            groups,
            atom,
            atoms: Vec::new(),
            generation: None,
        }
    }

    /// The eligible atoms, and whether they were just re-resolved.
    fn resolve(&mut self, context: &impl ObserveContext) -> (&[usize], bool) {
        let generation = Generation {
            groups: context.group_lists_generation(),
            atom_kinds: context.atom_kinds_generation(),
        };
        if self.generation == Some(generation) {
            return (&self.atoms, false);
        }
        let select = self
            .atom
            .map_or(ParticleSelection::Active, ParticleSelection::ById);
        self.atoms = context
            .select(&self.groups)
            .iter()
            .flat_map(|&group| {
                context.groups()[group]
                    .select(&select, context.topology_ref())
                    .expect("Selection should be successful.")
            })
            .collect();
        self.generation = Some(generation);
        (&self.atoms, true)
    }
}

/// Unnormalized weights W'(rⱼ) and their sum, valid for one configuration only.
///
/// Keyed on the coordinates *and* the candidate set, since either can change under them: an
/// ordinary trial moves an atom, a GCMC insertion changes who the candidates are.
#[derive(Clone, Debug)]
struct Weights {
    values: Vec<f64>,
    /// Σⱼ W'(rⱼ) — the denominator of eqn 9.43, never the `W` of eqn 9.44.
    sum: f64,
    positions: u64,
    candidates: Generation,
}

/// Distance-biased atom selection with detailed-balance correction.
///
/// A candidate atom at distance `r` from the nearest reference bounding sphere carries the
/// unnormalized weight `W'(r) = (r + offset)^{-ν}` ([Allen & Tildesley, 2017](https://doi.org/10.1093/oso/9780198803195.001.0001),
/// eqn 9.42; `offset` regularizes the singularity at `r = 0`). Atom `i` is then selected with
/// the *normalized* weight of eqn 9.43,
///
/// ```text
/// W(rᵢ) = W'(rᵢ) / Σⱼ W'(rⱼ)
/// ```
///
/// Selecting non-uniformly makes the proposal asymmetric, so the acceptance must be corrected by
/// the ratio of the underlying transition probabilities, eqn 9.44: `α_nm/α_mn = W_n/W_m`, where
/// `W_n` and `W_m` are the normalized weights of *the moved atom* after and before the move
/// ([Owicki & Scheraga, 1977](https://doi.org/10.1016/0009-2614(77)85051-3), eqn 5). Omitting it
/// depletes the very neighbourhood the bias exists to sample.
///
/// The distinction between `W'` (unnormalized, per atom) and `W` (normalized) is load-bearing:
/// the sum `Σⱼ W'(rⱼ)` is only the *denominator* of eqn 9.43, never the `W` of eqn 9.44. It is
/// named `w_sum` throughout to keep the two apart.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferentialSampling {
    /// Selection expression for reference groups (e.g. "molecule Protein")
    reference: Selection,
    /// Exponent ν in W'(r) = (r + offset)^{-ν}
    #[serde(default = "default_exponent")]
    exponent: f64,
    /// Offset to avoid singularity at r=0 (Angstrom)
    #[serde(default = "default_offset")]
    offset: f64,
    /// Stored ln(W_m / W_n) for the acceptance correction of eqn 9.44
    #[serde(skip)]
    ln_bias: f64,
    /// Cumulative sum of the acceptance correction over all proposed moves
    #[serde(skip_deserializing)]
    sum_bias: f64,
    /// Running mean of |ln(W_m / W_n)| — diagnostic for the size of the correction
    #[serde(skip_deserializing)]
    mean_bias: Mean,
    /// Optional output file for the selection-distance histogram.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file: Option<PathBuf>,
    /// Resolved reference group indices, built from `reference` on first use.
    #[serde(skip)]
    ref_cache: Option<CachedSelection<Groups>>,
    /// (mass_center, bounding_radius) per reference group, re-read on every proposal.
    ///
    /// Reported as `bounding_radii`: the radii are what a reader needs, and deriving them here
    /// keeps reporting off the sampling path.
    #[serde(
        skip_deserializing,
        rename = "bounding_radii",
        serialize_with = "serialize_radii",
        skip_serializing_if = "Vec::is_empty"
    )]
    ref_geometries: Vec<(Point, f64)>,
    /// Histogram of selection distances; only allocated when `file` is set.
    #[serde(skip)]
    distance_histogram: Option<Histogram>,
    /// The atoms this move may pick from, owned here so they cannot disagree with the reference.
    #[serde(skip)]
    candidates: Candidates,
    /// Weights for the current configuration, or `None` until first built.
    #[serde(skip)]
    weights: Option<Weights>,
    /// Index into `candidates` of the atom the last proposal picked.
    ///
    /// The one thing whose weight the next `on_trial_outcome` may have to refresh: if the trial was
    /// accepted, this atom moved and nothing else did.
    #[serde(skip)]
    proposed: Option<usize>,
}

impl PreferentialSampling {
    /// Resolve the reference and take ownership of the atoms the move may pick from.
    ///
    /// The sampler derives the candidates itself, from the move's own group/atom filter, rather
    /// than being handed a list on every proposal. A caller cannot then pass a set the reference
    /// was never checked against, nor one that has drifted from the set the move actually draws
    /// from — the two are the same object.
    pub(super) fn finalize(
        &mut self,
        context: &impl ObserveContext,
        groups: GroupSelection,
        atom: Option<AtomKindId>,
    ) -> anyhow::Result<()> {
        self.candidates = Candidates::new(groups, atom);
        // r ≥ 0, so offset ≤ 0 lets W'(r) = (r + offset)^{-ν} diverge or go complex at r = -offset.
        anyhow::ensure!(
            self.offset > 0.0,
            "PreferentialSampling: offset must be positive, got {}",
            self.offset
        );
        // ν ≤ 0 turns W'(r) = (r + offset)^{-ν} into an increasing function of r, biasing selection
        // toward the atoms *furthest* from the reference — the opposite of the method's purpose.
        anyhow::ensure!(
            self.exponent.is_finite() && self.exponent > 0.0,
            "PreferentialSampling: exponent must be finite and positive, got {}",
            self.exponent
        );
        // Built here, not on first use: the reference selection is known as soon as the move is.
        self.ref_cache = Some(CachedSelection::groups(self.reference.clone()));
        self.refresh_ref_geometries(context);
        anyhow::ensure!(
            !self.ref_geometries.is_empty(),
            "PreferentialSampling: selection '{}' matched no groups with valid mass centers",
            self.reference
        );

        self.ensure_reference_is_disjoint(context)?;

        for (i, &(_, radius)) in self.ref_geometries.iter().enumerate() {
            if radius < f64::EPSILON {
                warn!(
                    "PreferentialSampling: reference group {} has zero bounding radius \
                     (single atom?); bounding-sphere distance reduces to mass-center distance",
                    i
                );
            }
        }
        debug!(
            "PreferentialSampling: '{}' → {} reference group(s)",
            self.reference,
            self.ref_geometries.len()
        );
        if self.file.is_some() {
            self.distance_histogram = Some(Histogram::new(0.0, 200.0, 0.5)?);
        }
        Ok(())
    }

    /// Weight function: `W'(r) = (r + offset)^{-exponent}`
    fn weight(&self, r: f64) -> f64 {
        (r + self.offset).powf(-self.exponent)
    }

    /// The reference must hold none of the atoms this move displaces.
    ///
    /// Moving a reference atom would shift the very sphere the distances are measured from: every
    /// other candidate's weight would change at once, while `acceptance_correction` assumes only
    /// the moved atom's did, and the trial distance would be taken against the sphere as it stood
    /// *before* the move. Both assumptions fail silently, so the overlap is refused instead.
    ///
    /// Re-checked whenever the candidates re-resolve, not only at startup: group selection matches
    /// on any active atom, so a titration swap or a GCMC insertion can make a candidate group
    /// *become* a reference group long after the move was built.
    fn ensure_reference_is_disjoint(
        &mut self,
        context: &impl ObserveContext,
    ) -> anyhow::Result<()> {
        let reference_atoms: HashSet<usize> = self
            .resolve_reference(context)
            .iter()
            .flat_map(|&group| context.groups()[group.get()].iter_active())
            .collect();
        let (candidates, _) = self.candidates.resolve(context);
        if let Some(clash) = candidates.iter().find(|a| reference_atoms.contains(a)) {
            anyhow::bail!(
                "PreferentialSampling: reference '{}' contains atom {} that this move displaces. \
                 The reference must be distinct from the atoms being sampled; restrict the move \
                 with 'molecule' or 'atom'.",
                self.reference,
                clash
            );
        }
        Ok(())
    }

    /// Reference group indices, re-resolved against the current groups and atom kinds.
    fn resolve_reference(&mut self, context: &impl ObserveContext) -> &[GroupIndex] {
        self.ref_cache
            .as_mut()
            .expect("finalize() builds the reference cache before any sampling")
            .resolve(context)
    }

    /// Re-read (mass_center, bounding_radius) from current group state.
    ///
    /// Runs on every proposal, and deliberately logs nothing: the reference is free to move —
    /// Allen & Tildesley note the solute "may be moved as often as desired, with α_nm/α_mn = 1,
    /// without any additional modifications" — so the only requirement is that its geometry be
    /// current when read.
    fn refresh_ref_geometries(&mut self, context: &impl ObserveContext) {
        let groups = context.groups();
        // Assigning `ref_geometries` while `ref_cache` is mutably borrowed is a disjoint-field
        // borrow, so the resolved indices need not be copied out first.
        self.ref_geometries = self
            .ref_cache
            .as_mut()
            .expect("finalize() builds the reference cache before any sampling")
            .resolve(context)
            .iter()
            .filter_map(|group| {
                let g = &groups[group.get()];
                g.mass_center()
                    .map(|&center| (center, g.bounding_radius().unwrap_or(0.0)))
            })
            .collect();
    }

    /// Nearest bounding-sphere distance from a point to any reference group.
    fn distance_to_nearest_reference(&self, pos: &Point, cell: &impl BoundaryConditions) -> f64 {
        self.ref_geometries
            .iter()
            .map(|(cm, radius)| (cell.distance(pos, cm).norm() - radius).max(0.0))
            .reduce(f64::min)
            .unwrap_or(f64::INFINITY)
    }

    /// Weights for the configuration as it now stands, rebuilt only if it moved under them.
    ///
    /// Rebuilding costs one minimum-image distance per candidate, which is the whole cost of the
    /// move at realistic solvent counts. The cache key is the coordinates and the candidate set,
    /// both read from the context — so it cannot be satisfied by a caller that has lost track.
    fn weights(&mut self, context: &impl ObserveContext) -> &Weights {
        self.candidates.resolve(context);
        let key = (
            context.positions_generation(),
            self.candidates
                .generation
                .expect("resolve() just set the generation"),
        );
        if self
            .weights
            .as_ref()
            .is_some_and(|w| (w.positions, w.candidates) == key)
        {
            return self.weights.as_ref().expect("just checked");
        }

        self.refresh_ref_geometries(context);
        let cell = context.cell();
        // Lifted out so the weight closure may borrow `self` immutably; handed straight back.
        let atoms = std::mem::take(&mut self.candidates.atoms);
        let values: Vec<f64> = atoms
            .iter()
            .map(|&atom| {
                let r = self.distance_to_nearest_reference(&context.position(atom), cell);
                self.weight(r)
            })
            .collect();
        self.candidates.atoms = atoms;

        self.weights.insert(Weights {
            sum: values.iter().sum(),
            values,
            positions: key.0,
            candidates: key.1,
        })
    }

    /// Pick a candidate atom for the given displacement and stage its acceptance correction.
    ///
    /// Selection and correction are one step because both read the same configuration: the
    /// normalized weight that selects the atom is the `W_m` that corrects for having selected it.
    ///
    /// The displacement is drawn independently of which atom is picked, so the caller supplies it.
    /// Returns the absolute index of the selected atom; the correction is read via [`Self::ln_bias`].
    pub(super) fn propose(
        &mut self,
        context: &impl ObserveContext,
        displacement: &Point,
        rng: &mut dyn RngCore,
    ) -> Option<usize> {
        let weights = self.weights(context);

        // `WeightedIndex` rejects an empty or all-zero weight vector — the degenerate cases
        // reachable here, when the move has nothing to pick from, every W'(r) underflows, or the
        // reference has left the system so that every distance is infinite. Declining counts the
        // trial as a rejection; carrying on would divide by zero and hand the criterion a NaN bias,
        // which it also rejects, but silently and for every move thereafter.
        let Ok(distribution) = WeightedIndex::new(&weights.values) else {
            debug!("PreferentialSampling: no candidate carries any weight; declining to select");
            return None;
        };
        let selected = distribution.sample(rng);
        let w_old = weights.values[selected]; // computed against this same configuration
        let w_sum = weights.sum;
        let atom = self.candidates.atoms[selected];
        self.proposed = Some(selected);

        let cell = context.cell();
        let old_pos = context.position(atom);
        let r_new = self.distance_to_nearest_reference(&(old_pos + displacement), cell);
        let w_new = self.weight(r_new);

        self.ln_bias = acceptance_correction(w_old, w_new, w_sum);
        self.sum_bias += self.ln_bias;
        self.mean_bias.add(self.ln_bias.abs());

        // The selection distance costs an extra minimum-image evaluation, so only pay for it when
        // the histogram is actually being written.
        if self.distance_histogram.is_some() {
            let r_old = self.distance_to_nearest_reference(&old_pos, cell);
            if let Some(hist) = self.distance_histogram.as_mut() {
                hist.add(r_old);
            }
        }
        Some(atom)
    }

    /// Bring the weights back in step with the configuration the trial settled on.
    ///
    /// This is the one moment the sampler knows the change was *its own*, and so the one moment it
    /// may patch rather than rebuild. An accepted trial moved exactly one candidate — the one it
    /// picked — and left the reference alone, since the two are disjoint. A rejected trial was
    /// rolled back, so the weights already describe the current coordinates. Either way only the
    /// key has to catch up, and the O(N) rebuild is avoided.
    ///
    /// Anything else that moves an atom advances the positions generation without coming through
    /// here, and [`Self::weights`] rebuilds on the next read. Forgetting this hook therefore costs
    /// a rebuild, never correctness.
    pub(super) fn on_trial_outcome(&mut self, context: &impl ObserveContext, accepted: bool) {
        let Some(selected) = self.proposed.take() else {
            return;
        };
        let Some(weights) = self.weights.as_mut() else {
            return;
        };
        if accepted {
            let atom = self.candidates.atoms[selected];
            let r = self
                .ref_geometries
                .iter()
                .map(|(cm, radius)| {
                    (context.cell().distance(&context.position(atom), cm).norm() - radius).max(0.0)
                })
                .reduce(f64::min)
                .unwrap_or(f64::INFINITY);
            let updated = (r + self.offset).powf(-self.exponent);
            weights.sum += updated - weights.values[selected];
            weights.values[selected] = updated;
        }
        weights.positions = context.positions_generation();
    }

    /// The staged acceptance correction `ln(W_m / W_n)` for the last proposal.
    pub(super) fn ln_bias(&self) -> f64 {
        self.ln_bias
    }

    /// Write the distance histogram to the given file path.
    fn write_histogram(histogram: &Histogram, path: &Path) -> anyhow::Result<()> {
        let mut writer = ColumnWriter::open(path, &["distance", "count"])?;
        for (center, count) in histogram.iter() {
            if count > 0.0 {
                writer.write_row(&[&format!("{center:.4}"), &format!("{count:.0}")])?;
            }
        }
        writer.flush()?;
        Ok(())
    }
}

impl Drop for PreferentialSampling {
    fn drop(&mut self) {
        if let (Some(hist), Some(path)) = (&self.distance_histogram, &self.file) {
            if let Err(e) = Self::write_histogram(hist, path) {
                warn!("PreferentialSampling: failed to write histogram: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    /// Distribution of ion–solute distances from an ideal-gas chain, with and without
    /// preferential selection. No energy terms, so every configuration is equally likely.
    ///
    /// Bins are 5 Å wide out to the half-box; counts are returned as fractions.
    fn sample_distance_distribution(preferential: bool) -> [f64; 5] {
        use crate::backend::Backend;
        use crate::context::{WithSimulationCell, WithTopology};
        use crate::group::GroupCollection;
        use crate::montecarlo::{AcceptanceCriterion, TranslateAtom};
        use crate::propagate::MoveRunner;
        use rand::SeedableRng;

        const SWEEPS: usize = 200_000;
        const EQUILIBRATE: usize = 1_000;
        const BIN_WIDTH: f64 = 5.0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut context = Backend::new(
            "tests/files/preferential_ideal_gas/input.yaml",
            None,
            &mut rng,
        )
        .unwrap();

        let yaml = if preferential {
            "{molecule: Ion, max_displacement: 2.0, preferential: \
             {reference: \"molecule Macromolecule\", exponent: 2, offset: 1.0}}"
        } else {
            "{molecule: Ion, max_displacement: 2.0}"
        };
        let mut translate: TranslateAtom = serde_yml::from_str(yaml).unwrap();
        translate.finalize(&context).unwrap();
        let mut runner = MoveRunner::new(Box::new(translate), 1.0, 20);

        // Only the ions move, so the solute stays where it was built.
        let solute = context
            .groups()
            .iter()
            .position(|g| context.topology().moleculekind(g.molecule()).has_com())
            .expect("the input defines one Macromolecule with a mass center");
        let solute_center = *context.groups()[solute].mass_center().unwrap();
        let ions: Vec<usize> = context
            .groups()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != solute)
            .flat_map(|(_, g)| g.iter_active())
            .collect();

        let thermal_energy = crate::R_IN_KJ_PER_MOL * 300.0;
        let criterion = AcceptanceCriterion::MetropolisHastings;
        let mut step = 0;
        let mut counts = [0.0f64; 5];
        for sweep in 0..SWEEPS {
            runner
                .do_move(
                    &mut context,
                    &criterion,
                    thermal_energy,
                    &mut step,
                    &mut rng,
                )
                .unwrap();
            if sweep < EQUILIBRATE {
                continue;
            }
            for &ion in &ions {
                let r = context
                    .cell()
                    .distance(&context.position(ion), &solute_center)
                    .norm();
                let bin = (r / BIN_WIDTH) as usize;
                if bin < counts.len() {
                    counts[bin] += 1.0;
                }
            }
        }
        let total: f64 = counts.iter().sum();
        counts.map(|c| c / total)
    }

    /// Preferential selection must not change what is sampled, only how fast.
    ///
    /// With no energy terms every configuration is equally likely, so the ions are an ideal gas
    /// and the fraction of them in each spherical shell about the solute is just the shell's
    /// share of the volume, `(r₂³ − r₁³)/25³` for shells inside the inscribed sphere. That is the
    /// independent truth here — the chain must reproduce it whether ions are picked uniformly or
    /// with a distance bias.
    ///
    /// If the acceptance correction of eqn 9.44 is wrong, the biased chain samples a different
    /// distribution: dropping the `W'(r_new)/W'(r_old)` factor empties the innermost shell by an
    /// order of magnitude, which is the failure Owicki & Scheraga warn of.
    ///
    /// The unbiased chain is asserted too, as a control on the harness itself.
    #[test]
    fn preferential_selection_leaves_the_ideal_gas_distribution_unchanged() {
        // Shell volume fractions (r₂³ − r₁³)/25³ for the 5 Å shells out to the inscribed sphere.
        const UNIFORM: [f64; 5] = [0.008, 0.056, 0.152, 0.296, 0.488];

        for preferential in [false, true] {
            let sampled = sample_distance_distribution(preferential);
            for (bin, (&observed, &expected)) in sampled.iter().zip(UNIFORM.iter()).enumerate() {
                let deviation = (observed - expected).abs() / expected;
                assert!(
                    deviation < 0.15,
                    "preferential={preferential}, shell {}–{} Å: sampled={observed:.5}, \
                     ideal gas={expected:.5} ({:.0}% off)",
                    bin * 5,
                    (bin + 1) * 5,
                    100.0 * deviation
                );
            }
        }
    }

    /// A reference that moves must drag the bias with it.
    ///
    /// Allen & Tildesley allow the solute to be moved as often as desired; nothing corrects for
    /// it, so the only requirement is that its geometry be read as it currently stands. Here the
    /// solute is displaced by 20 Å between two batches of proposals: the atoms picked afterwards
    /// must cluster about where it now is, not about where it was when the move was built.
    #[test]
    fn selection_follows_a_reference_that_moves() {
        use crate::backend::Backend;
        use crate::context::{WithSimulationCell, WithTopology};
        use crate::group::GroupCollection;
        use crate::propagate::ProposedMove;
        use crate::Context;
        use rand::SeedableRng;

        const PROPOSALS: usize = 2_000;

        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut context = Backend::new(
            "tests/files/preferential_ideal_gas/input.yaml",
            None,
            &mut rng,
        )
        .unwrap();

        let solute = context
            .groups()
            .iter()
            .position(|g| context.topology().moleculekind(g.molecule()).has_com())
            .expect("the input defines one Macromolecule with a mass center");
        let ion_kind = context
            .topology()
            .moleculekinds()
            .iter()
            .position(|m| m.name() == "Ion")
            .map(crate::group::MoleculeId::new)
            .expect("the input defines an Ion molecule");

        let mut sampler = make_sampler(2.0, 1.0);
        sampler.reference = Selection::parse("molecule Macromolecule").unwrap();
        sampler
            .finalize(&context, GroupSelection::ByMoleculeId(ion_kind), None)
            .unwrap();

        // Mean distance from the picked atoms to a given point, over many proposals.
        let mean_pick_distance = |sampler: &mut PreferentialSampling,
                                  context: &Backend,
                                  target: &Point,
                                  rng: &mut rand::rngs::StdRng| {
            let mut mean = Mean::new();
            for _ in 0..PROPOSALS {
                // A null displacement leaves the configuration untouched; only the pick matters.
                let atom = sampler.propose(context, &Point::zeros(), rng).unwrap();
                mean.add(
                    context
                        .cell()
                        .distance(&context.position(atom), target)
                        .norm(),
                );
            }
            mean.mean()
        };

        let before = *context.groups()[solute].mass_center().unwrap();
        let near_old = mean_pick_distance(&mut sampler, &context, &before, &mut rng);

        let shift = Point::new(20.0, 0.0, 0.0);
        let displace = ProposedMove::translate_group(solute, shift);
        context.save_energy_backups(displace.change());
        displace.apply_with_backup(&mut context).unwrap();
        context.update(displace.change()).unwrap();
        context.discard_backup();
        let after = *context.groups()[solute].mass_center().unwrap();
        assert!((after - before).norm() > 15.0, "the solute really moved");

        let near_new = mean_pick_distance(&mut sampler, &context, &after, &mut rng);
        let still_near_old = mean_pick_distance(&mut sampler, &context, &before, &mut rng);

        // Picks now sit closer to where the solute is than to where it was.
        assert!(
            near_new < still_near_old,
            "bias did not follow the reference: {near_new:.1} Å from the new position, \
             {still_near_old:.1} Å from the old one"
        );
        // And they cluster about the new position as tightly as they did about the old one.
        assert!(
            near_new < near_old + 2.0,
            "bias is weaker about the moved reference: {near_new:.1} Å vs {near_old:.1} Å"
        );
    }

    /// Patching one weight after an accepted trial must land exactly where a rebuild would.
    ///
    /// The cache patches the single atom the move displaced instead of recomputing all N weights.
    /// That is only sound if the patch is *identical* to the rebuild — the reference must not have
    /// shifted, and no other candidate can have moved. Compare the two directly, rather than
    /// trusting the reasoning.
    #[test]
    fn patching_an_accepted_trial_agrees_with_a_full_rebuild() {
        use crate::backend::Backend;
        use crate::context::WithTopology;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let mut context = Backend::new(
            "tests/files/preferential_ideal_gas/input.yaml",
            None,
            &mut rng,
        )
        .unwrap();

        let ion_kind = context
            .topology()
            .moleculekinds()
            .iter()
            .position(|m| m.name() == "Ion")
            .map(crate::group::MoleculeId::new)
            .unwrap();
        let reference = || Selection::parse("molecule Macromolecule").unwrap();

        let mut sampler = make_sampler(2.0, 1.0);
        sampler.reference = reference();
        sampler
            .finalize(&context, GroupSelection::ByMoleculeId(ion_kind), None)
            .unwrap();

        // A run of accepted trials, each moving one candidate for real, so the cache is carried
        // forward by patching — never rebuilt, since only this sampler's own atom ever moves.
        for i in 0..25 {
            let displacement = Point::new(1.5, -0.7, 0.3) * (1.0 + i as f64 * 0.1);
            let atom = sampler.propose(&context, &displacement, &mut rng).unwrap();
            context.translate_particles(&[atom], &displacement);
            sampler.on_trial_outcome(&context, true);
        }
        // And one rejected trial: nothing moved, so the weights must simply stand.
        sampler
            .propose(&context, &Point::new(0.5, 0.5, 0.5), &mut rng)
            .unwrap();
        sampler.on_trial_outcome(&context, false);

        let patched = sampler.weights(&context).clone();

        // What a sampler that never patched anything computes from the same configuration.
        let mut fresh = make_sampler(2.0, 1.0);
        fresh.reference = reference();
        fresh
            .finalize(&context, GroupSelection::ByMoleculeId(ion_kind), None)
            .unwrap();
        let rebuilt = fresh.weights(&context).clone();

        assert_eq!(patched.values.len(), rebuilt.values.len());
        for (atom, (&incremental, &full)) in
            patched.values.iter().zip(rebuilt.values.iter()).enumerate()
        {
            assert_approx_eq!(f64, incremental, full, epsilon = 1e-12);
            assert!(full.is_finite(), "candidate {atom} has a non-finite weight");
        }
        // The running sum drifts if a patch ever updates a weight without its own contribution.
        assert_approx_eq!(f64, patched.sum, rebuilt.sum, epsilon = 1e-10);
    }

    fn make_sampler(exponent: f64, offset: f64) -> PreferentialSampling {
        PreferentialSampling {
            reference: Selection::parse("all").unwrap(),
            exponent,
            offset,
            ln_bias: 0.0,
            sum_bias: 0.0,
            mean_bias: Mean::new(),
            file: None,
            ref_cache: None,
            ref_geometries: Vec::new(),
            distance_histogram: None,
            candidates: Candidates::default(),
            weights: None,
            proposed: None,
        }
    }

    #[test]
    fn weight_function_values() {
        let ps = make_sampler(2.0, 1.0);
        assert_approx_eq!(f64, ps.weight(0.0), 1.0, epsilon = 1e-15); // (0+1)^-2
        assert_approx_eq!(f64, ps.weight(1.0), 0.25, epsilon = 1e-15); // (1+1)^-2
        assert_approx_eq!(f64, ps.weight(2.0), 1.0 / 9.0, epsilon = 1e-15); // (2+1)^-2
    }

    #[test]
    fn weight_with_custom_params() {
        let ps = make_sampler(3.0, 0.5);
        let expected = (2.5_f64 + 0.5).powf(-3.0);
        assert_approx_eq!(f64, ps.weight(2.5), expected, epsilon = 1e-15);
    }

    /// The correction of eqn 9.44, against values computed outside this code.
    ///
    /// Three atoms at distances 2, 5, 10 from the reference, with ν = 2 and offset = 1, so the
    /// unnormalized weights of eqn 9.42 are W' = 1/9, 1/36, 1/121 and their sum is 0.147153352.
    /// Atom 0 then moves 2 → 8, giving W'(8) = 1/81 and a new sum of 0.048387920. The normalized
    /// weights of eqn 9.43 for that atom are therefore
    ///
    ///   W_m = (1/9)/0.147153352 = 0.755070203      (before)
    ///   W_n = (1/81)/0.048387920 = 0.255139694     (after)
    ///
    /// and the correction is ln(W_m/W_n) = 1.084999513.
    #[test]
    fn acceptance_correction_matches_eqn_9_44() {
        let ps = make_sampler(2.0, 1.0);
        let (w_old, w_new) = (ps.weight(2.0), ps.weight(8.0));
        let w_sum = ps.weight(2.0) + ps.weight(5.0) + ps.weight(10.0);

        assert_approx_eq!(f64, w_old, 1.0 / 9.0, epsilon = 1e-15);
        assert_approx_eq!(f64, w_new, 1.0 / 81.0, epsilon = 1e-15);
        assert_approx_eq!(f64, w_sum, 0.147_153_351_698_806_25, epsilon = 1e-15);

        assert_approx_eq!(
            f64,
            acceptance_correction(w_old, w_new, w_sum),
            1.084_999_513_014_344,
            epsilon = 1e-12
        );
    }

    /// A lone candidate is picked with certainty wherever it sits, so its normalized weight is 1
    /// before and after and there is nothing to correct. Degenerate, but reachable — a molecular
    /// solvent selected by one atom kind offers a single candidate per molecule.
    #[test]
    fn a_lone_candidate_carries_no_correction() {
        let ps = make_sampler(2.0, 1.0);
        let (w_old, w_new) = (ps.weight(4.0), ps.weight(11.0));
        // The sum runs over that one atom, so it *is* its weight.
        assert_approx_eq!(
            f64,
            acceptance_correction(w_old, w_new, w_old),
            0.0,
            epsilon = 1e-15
        );
    }

    /// An atom that ends up equally far from the reference was equally likely to be picked
    /// either way, so the proposal is symmetric and needs no correction.
    #[test]
    fn unchanged_distance_needs_no_correction() {
        let ps = make_sampler(2.0, 1.0);
        let w = ps.weight(3.0);
        let w_sum = w + ps.weight(7.0);
        assert_approx_eq!(
            f64,
            acceptance_correction(w, w, w_sum),
            0.0,
            epsilon = 1e-15
        );
    }

    /// Selection probability must equal w_i / W for each candidate.
    /// With fixed seed, verify the exact selected index.
    #[test]
    fn selection_probabilities() {
        let ps = make_sampler(2.0, 1.0);
        let distances = [1.0, 5.0, 20.0];
        let weights: Vec<f64> = distances.iter().map(|&r| ps.weight(r)).collect();
        let w_total: f64 = weights.iter().sum();

        // Analytical selection probabilities
        let p: Vec<f64> = weights.iter().map(|w| w / w_total).collect();
        // w(1) = 1/4, w(5) = 1/36, w(20) = 1/441
        // p(0) ≈ 0.25 / 0.280 ≈ 0.893 — closest atom dominates
        assert!(p[0] > 0.85);
        assert!(p[1] < 0.12);
        assert!(p[2] < 0.02);

        // Empirical check: count selections over many draws
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut counts = [0u64; 3];
        let n = 100_000;
        for _ in 0..n {
            let threshold = rng.r#gen::<f64>() * w_total;
            let mut cumulative = 0.0;
            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if cumulative >= threshold {
                    counts[i] += 1;
                    break;
                }
            }
        }

        // Empirical frequencies should match analytical probabilities within ~1%
        for (i, &count) in counts.iter().enumerate() {
            let empirical = count as f64 / n as f64;
            assert!(
                (empirical - p[i]).abs() < 0.01,
                "atom {i}: empirical={empirical:.4}, expected={:.4}",
                p[i]
            );
        }
    }

    /// Moving away from the reference must be *harder* to accept, not easier.
    ///
    /// Close atoms are picked more often than far ones, so an outward move is proposed far more
    /// often than the inward move that reverses it. Detailed balance pays for that by rejecting
    /// outward moves more often. Correction > 0 means `exp(-correction) < 1`, i.e. harder to
    /// accept. Getting this sign backwards is what evacuates the neighbourhood of the reference.
    #[test]
    fn moving_away_from_the_reference_is_harder_to_accept() {
        let ps = make_sampler(2.0, 1.0);
        let (w5, w10, w15) = (ps.weight(5.0), ps.weight(10.0), ps.weight(15.0));

        // An atom at 5 Å moves out to 10 Å; the other candidate sits at 15 Å.
        let outward = acceptance_correction(w5, w10, w5 + w15);
        // The reverse move: the same atom, now at 10 Å, comes back in to 5 Å.
        let inward = acceptance_correction(w10, w5, w10 + w15);

        assert!(outward > 0.0, "outward move should be penalized: {outward}");
        assert!(inward < 0.0, "inward move should be favoured: {inward}");
    }

    /// Nearest-reference distance picks the closest among multiple reference groups.
    #[test]
    fn nearest_reference_distance() {
        use crate::cell::{BoundaryConditions, Cuboid};
        use nalgebra::Vector3;

        let mut ps = make_sampler(2.0, 1.0);
        // Two reference groups: one at (10,0,0) R=2, one at (20,0,0) R=3
        ps.ref_geometries = vec![
            (Point::from(Vector3::new(10.0, 0.0, 0.0)), 2.0),
            (Point::from(Vector3::new(20.0, 0.0, 0.0)), 3.0),
        ];

        let cell = Cuboid::new(100.0, 100.0, 100.0);

        // Helper: compute nearest bounding-sphere distance manually
        let nearest = |pos: &Point| -> f64 {
            ps.ref_geometries
                .iter()
                .map(|(cm, r)| (cell.distance(pos, cm).norm() - r).max(0.0))
                .fold(f64::INFINITY, f64::min)
        };

        // Atom at origin: d1 = 10-2 = 8, d2 = 20-3 = 17 → nearest = 8
        let pos = Point::from(Vector3::new(0.0, 0.0, 0.0));
        assert_approx_eq!(f64, nearest(&pos), 8.0, epsilon = 1e-10);

        // Atom at (18,0,0): d1 = 8-2 = 6, d2 = max(0, 2-3) = 0 → nearest = 0
        let pos2 = Point::from(Vector3::new(18.0, 0.0, 0.0));
        assert_approx_eq!(f64, nearest(&pos2), 0.0, epsilon = 1e-10);

        // Atom at (15,0,0): d1 = 5-2 = 3, d2 = 5-3 = 2 → nearest = 2
        let pos3 = Point::from(Vector3::new(15.0, 0.0, 0.0));
        assert_approx_eq!(f64, nearest(&pos3), 2.0, epsilon = 1e-10);
    }
}
