//! Reaction ensemble (speciation) Monte Carlo move.
//!
//! Performs molecular insertion/deletion and atom-type swaps according to
//! chemical reactions. The acceptance criterion follows Smith & Triska (1994),
//! using `entropy_bias()` for the combinatorial `V^Δν · ∏[N!/(N+ν)!]` factors.
//! For atom swaps, a `N_from/(N_to+1)` combinatorial factor ensures detailed
//! balance, consistent with ESPResSo's reaction ensemble implementation.

use crate::chemistry::reaction::{Direction, Participant, Reaction};
use crate::group::GroupSize;
use crate::montecarlo::{entropy_bias, MoveStatistics, NewOld};
use crate::propagate::{
    default_repeat, default_weight, tagged_yaml, Displacement, MoveProposal, MoveTarget,
    ProposedMove,
};
use crate::transform::{SpeciationAction, Transform};
use crate::{cell::Shape, Change, Context, GroupChange};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Equilibrium constant in different representations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EquilibriumConstant {
    /// Direct equilibrium constant (dimensionless, must be positive).
    K(f64),
    /// Natural logarithm of K.
    #[serde(rename = "lnK")]
    LnK(f64),
    /// Negative log₁₀ of K, i.e. `K = 10⁻ᵖᴷ`.
    #[serde(rename = "pK")]
    Pk(f64),
    /// Molar free energy in kJ/mol; `K = exp(-ΔG / RT)`.
    #[serde(rename = "dG", alias = "ΔG")]
    DeltaG(f64),
}

impl EquilibriumConstant {
    /// Convert to K. The `rt` parameter (kJ/mol) is needed for the `dG` variant.
    fn to_k(&self, thermal_energy: f64) -> f64 {
        match self {
            Self::K(k) => *k,
            Self::LnK(ln_k) => ln_k.exp(),
            Self::Pk(pk) => 10.0_f64.powf(-pk),
            Self::DeltaG(dg) => (-dg / thermal_energy).exp(),
        }
    }
}

/// A reaction entry: `["reaction string", !K value]`, `["...", !pK value]`, `["...", !lnK value]`, or `["...", !dG kJ/mol]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReactionConfig(String, EquilibriumConstant);

/// What to do with a participant during a reaction step.
#[derive(Clone, Debug)]
enum ReactionOp {
    /// Activate a group of this molecule type
    ActivateMolecule(usize),
    /// Deactivate a group of this molecule type
    DeactivateMolecule(usize),
    /// Swap an atom from one kind to another within a molecule
    SwapAtom {
        from_id: usize,
        to_id: usize,
        molecule_id: usize,
    },
    /// Swap a molecule of one kind for another (deactivate from, activate to with copied positions)
    SwapMolecule {
        from_mol_id: usize,
        to_mol_id: usize,
    },
}

/// Reaction with topology IDs resolved.
#[derive(Clone, Debug)]
struct ResolvedReaction {
    /// ln(K_eff) = ln(K) + implicit-species activity contributions
    effective_ln_k: f64,
    /// Operations when running the reaction forward
    forward_ops: Vec<ReactionOp>,
    /// Operations when running the reaction backward
    backward_ops: Vec<ReactionOp>,
}

/// Result of building speciation actions: (actions, group changes, ln_bias).
type ActionBuild = (Vec<SpeciationAction>, Vec<(usize, GroupChange)>, f64);

/// Reaction ensemble Monte Carlo move.
///
/// Supports molecular insertion/deletion and atom-type swaps.
///
/// # YAML example
/// ```yaml
/// - !SpeciationMove
///   temperature: 298.15
///   reactions:
///     - ["= NaCl", !K 100.0]
///     - ["⚛HA = ⚛A + ~H+", !pK 4.24]
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeciationMove {
    /// Reactions to sample from.
    reactions: Vec<ReactionConfig>,
    /// Temperature in Kelvin.
    temperature: f64,
    /// Move selection weight.
    #[serde(skip_serializing, default = "default_weight")]
    pub(crate) weight: f64,
    /// Repeat count.
    #[serde(default = "default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
    /// Thermal energy kT in kJ/mol (computed in finalize).
    #[serde(skip)]
    thermal_energy: f64,
    /// Resolved reactions with topology IDs (populated in finalize).
    #[serde(skip)]
    resolved: Vec<ResolvedReaction>,
    /// Entropy bias from the last `propose_move`, consumed by `bias`.
    #[serde(skip)]
    trial_ln_bias: Option<f64>,
    /// Index of the reaction selected in the last `propose_move`.
    #[serde(skip)]
    trial_reaction_index: Option<usize>,
    /// Per-reaction acceptance statistics.
    #[serde(skip)]
    reaction_statistics: Vec<MoveStatistics>,
}

impl crate::Info for SpeciationMove {
    fn short_name(&self) -> Option<&'static str> {
        Some("speciation")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Reaction ensemble (speciation)")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("doi:10.1063/1.466443")
    }
}

/// Re-export for local callers.
pub(super) use crate::cell::random_point_inside;

/// Look up the activity of an implicit species by name.
///
/// Searches atom kinds first, then molecule kinds.
fn find_implicit_activity(name: &str, topology: &crate::topology::Topology) -> anyhow::Result<f64> {
    let activity = topology
        .atomkinds()
        .iter()
        .find(|a| a.name() == name)
        .and_then(|a| a.activity())
        .or_else(|| {
            topology
                .moleculekinds()
                .iter()
                .find(|m| m.name() == name)
                .and_then(|m| m.activity())
        });
    match activity {
        Some(a) if a > 0.0 => Ok(a),
        Some(_) => anyhow::bail!("Implicit species '{name}' has non-positive activity"),
        None => anyhow::bail!(
            "No activity found for implicit species '{name}'. Define it on an atom or molecule."
        ),
    }
}

/// Look up molecule index by name.
fn find_molecule_index(name: &str, topology: &crate::topology::Topology) -> anyhow::Result<usize> {
    topology
        .moleculekinds()
        .iter()
        .position(|m| m.name() == name)
        .ok_or_else(|| anyhow::anyhow!("Unknown molecule '{name}' in reaction"))
}

/// Extract atom participants as (atom_kind_index, name) pairs.
fn extract_atom_participants<'a>(
    participants: &'a [Participant],
    topology: &crate::topology::Topology,
) -> Vec<(usize, &'a str)> {
    participants
        .iter()
        .filter_map(|p| match p {
            Participant::Atom(name) => {
                let id = topology.atomkinds().iter().position(|a| a.name() == name)?;
                Some((id, name.as_str()))
            }
            _ => None,
        })
        .collect()
}

/// Find atom swap pairs: atoms appearing on both sides form swap operations.
fn resolve_atom_swaps(
    reactants: &[Participant],
    products: &[Participant],
    topology: &crate::topology::Topology,
) -> anyhow::Result<(Vec<ReactionOp>, Vec<ReactionOp>)> {
    let mut forward_ops = Vec::new();
    let mut backward_ops = Vec::new();

    let reactant_atoms = extract_atom_participants(reactants, topology);
    let product_atoms = extract_atom_participants(products, topology);

    // Pair up reactant and product atoms
    for (from, to) in reactant_atoms.iter().zip(product_atoms.iter()) {
        // Prefer molecule containing both atom kinds; fall back to either
        // (titration templates may only list the initial protonation state)
        let molecule_id = topology
            .moleculekinds()
            .iter()
            .position(|m| m.atom_indices().contains(&from.0) && m.atom_indices().contains(&to.0))
            .or_else(|| {
                topology.moleculekinds().iter().position(|m| {
                    m.atom_indices().contains(&from.0) || m.atom_indices().contains(&to.0)
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!("No molecule contains atom '{}' or '{}'", from.1, to.1)
            })?;

        forward_ops.push(ReactionOp::SwapAtom {
            from_id: from.0,
            to_id: to.0,
            molecule_id,
        });
        backward_ops.push(ReactionOp::SwapAtom {
            from_id: to.0,
            to_id: from.0,
            molecule_id,
        });
    }

    Ok((forward_ops, backward_ops))
}

/// Overlay target molecule template onto a source group's orientation.
///
/// Uses gyration tensor principal-axis alignment to map the target group's
/// stored positions (template shape) onto the source group's current pose.
/// For single-atom molecules, simply copies the source position.
fn overlay_swap_positions(
    source_indices: impl Iterator<Item = usize> + Clone,
    target_group: &crate::group::Group,
    context: &impl Context,
    rng: &mut dyn RngCore,
) -> Vec<crate::Point> {
    let topology = context.topology();
    let atomkinds = topology.atomkinds();

    let source_masses: Vec<(crate::Point, f64)> = source_indices
        .clone()
        .map(|i| {
            let mass = atomkinds[context.atom_kind(i)].mass();
            (context.position(i), mass)
        })
        .collect();

    if source_masses.len() < 2 {
        return source_masses.into_iter().map(|(pos, _)| pos).collect();
    }

    let com = context.mass_center(&source_indices.collect::<Vec<_>>());
    let template: Vec<crate::Point> = (0..target_group.capacity())
        .map(|i| context.position(target_group.start() + i))
        .collect();
    let source_positions: Vec<crate::Point> = source_masses.iter().map(|(pos, _)| *pos).collect();

    crate::geometry::overlay_positions(&template, source_masses, &com, context.cell(), rng)
        .unwrap_or(source_positions)
}

/// Resolve a single reaction config into forward/backward ops and effective ln(K).
fn resolve_reaction(
    config: &ReactionConfig,
    thermal_energy: f64,
    topology: &crate::topology::Topology,
) -> anyhow::Result<ResolvedReaction> {
    let k = config.1.to_k(thermal_energy);
    anyhow::ensure!(
        k > 0.0,
        "Equilibrium constant must be positive for reaction '{}'",
        config.0
    );

    let reaction = Reaction::from_reaction(&config.0, k)?;
    let (reactants, products) = reaction.get();

    let mut forward_ops = Vec::new();
    let mut backward_ops = Vec::new();

    // Collect molecular participants
    let mol_ids = |participants: &[Participant]| -> anyhow::Result<Vec<usize>> {
        participants
            .iter()
            .filter_map(|p| match p {
                Participant::Molecule(name) => Some(find_molecule_index(name, topology)),
                _ => None,
            })
            .collect()
    };
    let reactant_mols = mol_ids(reactants)?;
    let product_mols = mol_ids(products)?;

    // Equal-size reactant/product pairs are swaps (position overlay, no insert/delete).
    // Reservoirs are excluded: they use atomic mega-groups, not individual molecular groups,
    // so the swap overlay logic cannot apply.
    let mut swap_reactants: Vec<usize> = Vec::new();
    let mut swap_products: Vec<usize> = Vec::new();
    for (ri, &from_id) in reactant_mols.iter().enumerate() {
        for (pi, &to_id) in product_mols.iter().enumerate() {
            if !swap_products.contains(&pi)
                && from_id != to_id
                && !topology.moleculekinds()[from_id].is_reservoir()
                && !topology.moleculekinds()[to_id].is_reservoir()
                && topology.moleculekinds()[from_id].atom_indices().len()
                    == topology.moleculekinds()[to_id].atom_indices().len()
            {
                forward_ops.push(ReactionOp::SwapMolecule {
                    from_mol_id: from_id,
                    to_mol_id: to_id,
                });
                backward_ops.push(ReactionOp::SwapMolecule {
                    from_mol_id: to_id,
                    to_mol_id: from_id,
                });
                swap_reactants.push(ri);
                swap_products.push(pi);
                break;
            }
        }
    }

    // Remaining unpaired molecules: insert/delete
    for (ri, &mol_id) in reactant_mols.iter().enumerate() {
        if !swap_reactants.contains(&ri) {
            forward_ops.push(ReactionOp::DeactivateMolecule(mol_id));
            backward_ops.push(ReactionOp::ActivateMolecule(mol_id));
        }
    }
    for (pi, &mol_id) in product_mols.iter().enumerate() {
        if !swap_products.contains(&pi) {
            forward_ops.push(ReactionOp::ActivateMolecule(mol_id));
            backward_ops.push(ReactionOp::DeactivateMolecule(mol_id));
        }
    }

    // Atom swaps
    let (swap_fwd, swap_bwd) = resolve_atom_swaps(reactants, products, topology)?;
    forward_ops.extend(swap_fwd);
    backward_ops.extend(swap_bwd);

    // Absorb activities into K so the MC loop only sees a single effective_ln_k.
    // Implicit species (tilde/ghost) contribute their molar activity.
    // Molecular fugacities (GCMC) are divided out because entropy_bias already
    // uses N/(V·c₀), not N/(V·c₀·z). Swaps conserve molecule count → no correction.
    let mut effective_ln_k = k.ln();
    for (participants, sign) in [(reactants, 1.0_f64), (products, -1.0_f64)] {
        for p in participants {
            if let Participant::Implicit(name) = p {
                effective_ln_k += sign * find_implicit_activity(name, topology)?.ln();
            }
        }
    }
    for (mol_ids, swapped, sign) in [
        (&reactant_mols, &swap_reactants, 1.0_f64),
        (&product_mols, &swap_products, -1.0_f64),
    ] {
        for (idx, &id) in mol_ids.iter().enumerate() {
            if !swapped.contains(&idx) {
                if let Some(activity) = topology.moleculekinds()[id].activity() {
                    effective_ln_k -= sign * activity.ln();
                }
            }
        }
    }

    Ok(ResolvedReaction {
        effective_ln_k,
        forward_ops,
        backward_ops,
    })
}

/// Validate that the context has groups for every molecule referenced by resolved reactions.
fn validate_reaction_groups(
    resolved: &[ResolvedReaction],
    context: &impl Context,
    topology: &crate::topology::Topology,
) -> anyhow::Result<()> {
    let has_any = |mol_id: usize| -> bool {
        context.find_molecules(mol_id, GroupSize::Full).is_some()
            || context.find_molecules(mol_id, GroupSize::Empty).is_some()
    };
    for r in resolved {
        for op in r.forward_ops.iter().chain(&r.backward_ops) {
            match op {
                ReactionOp::ActivateMolecule(id) | ReactionOp::DeactivateMolecule(id) => {
                    let name = topology.moleculekinds()[*id].name();
                    anyhow::ensure!(
                        has_any(*id),
                        "No groups found for molecule '{name}' needed by reaction"
                    );
                }
                ReactionOp::SwapAtom { molecule_id, .. } => {
                    let name = topology.moleculekinds()[*molecule_id].name();
                    anyhow::ensure!(
                        context
                            .find_molecules(*molecule_id, GroupSize::Full)
                            .is_some(),
                        "No active groups for molecule '{name}' needed by atom swap"
                    );
                }
                ReactionOp::SwapMolecule {
                    from_mol_id,
                    to_mol_id,
                } => {
                    for &id in [from_mol_id, to_mol_id] {
                        let name = topology.moleculekinds()[id].name();
                        anyhow::ensure!(
                            has_any(id),
                            "No groups found for molecule '{name}' needed by molecular swap"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

impl SpeciationMove {
    /// Resolve reaction strings to topology IDs and validate.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.temperature > 0.0,
            "SpeciationMove: temperature must be positive"
        );
        self.thermal_energy = crate::R_IN_KJ_PER_MOL * self.temperature;

        let topology = context.topology();
        self.resolved = self
            .reactions
            .iter()
            .map(|config| resolve_reaction(config, self.thermal_energy, &topology))
            .collect::<anyhow::Result<_>>()?;

        validate_reaction_groups(&self.resolved, context, &topology)?;
        self.reaction_statistics = vec![MoveStatistics::default(); self.resolved.len()];

        log::info!(
            "SpeciationMove: {} reactions, kT = {:.4} kJ/mol",
            self.resolved.len(),
            self.thermal_energy,
        );
        Ok(())
    }

    /// Deactivate one molecule (atomic: shrink mega-group; molecular: empty a group).
    /// Returns (action, group_change, entropy_bias_delta), or None if infeasible.
    fn deactivate_one(
        mol_id: usize,
        n_old: usize,
        vol: NewOld<f64>,
        context: &impl Context,
        rng: &mut dyn RngCore,
    ) -> Option<(SpeciationAction, (usize, GroupChange), f64)> {
        let molecule = &context.topology_ref().moleculekinds()[mol_id];
        if molecule.atomic() {
            let gi = context.find_atomic_group(mol_id)?;
            if n_old == 0 {
                return None;
            }
            let rel = rng.gen_range(0..n_old);
            let abs = context.groups()[gi].to_absolute_index(rel).unwrap();
            // Reservoirs have zero entropy bias (solid activity = 1; C++ `implicit` convention)
            let bias = if molecule.is_reservoir() {
                0.0
            } else {
                entropy_bias(NewOld::from(n_old - 1, n_old), vol)
            };
            Some((
                SpeciationAction::DeactivateAtom {
                    group_index: gi,
                    abs_index: abs,
                },
                (
                    gi,
                    GroupChange::ResizePartial(GroupSize::Shrink(1), vec![rel]),
                ),
                bias,
            ))
        } else {
            let full = context.find_molecules(mol_id, GroupSize::Full)?;
            if full.is_empty() {
                return None;
            }
            let &gi = full.iter().choose(rng)?;
            let bias = entropy_bias(NewOld::from(n_old.saturating_sub(1), n_old), vol);
            Some((
                SpeciationAction::DeactivateGroup(gi),
                (gi, GroupChange::Resize(GroupSize::Empty)),
                bias,
            ))
        }
    }

    /// Activate one molecule (atomic: expand mega-group; molecular: fill an empty group).
    /// Returns (action, group_change, entropy_bias_delta), or None if infeasible.
    fn activate_one(
        mol_id: usize,
        n_old: usize,
        vol: NewOld<f64>,
        context: &impl Context,
        rng: &mut dyn RngCore,
    ) -> Option<(SpeciationAction, (usize, GroupChange), f64)> {
        let molecule = &context.topology_ref().moleculekinds()[mol_id];
        if molecule.atomic() {
            let gi = context.find_atomic_group(mol_id)?;
            if n_old >= context.groups()[gi].capacity() {
                return None;
            }
            // Reservoirs have zero entropy bias (solid activity = 1; C++ `implicit` convention)
            let bias = if molecule.is_reservoir() {
                0.0
            } else {
                entropy_bias(NewOld::from(n_old + 1, n_old), vol)
            };
            let position = random_point_inside(context.cell(), rng);
            Some((
                SpeciationAction::ActivateAtom {
                    group_index: gi,
                    position,
                },
                (
                    gi,
                    GroupChange::ResizePartial(GroupSize::Expand(1), vec![n_old]),
                ),
                bias,
            ))
        } else {
            let empty = context.find_molecules(mol_id, GroupSize::Empty)?;
            if empty.is_empty() {
                return None;
            }
            let &gi = empty.iter().choose(rng)?;
            let bias = entropy_bias(NewOld::from(n_old + 1, n_old), vol);
            let pos = random_point_inside(context.cell(), rng);
            let positions = vec![pos; context.groups()[gi].capacity()];
            Some((
                SpeciationAction::ActivateGroup {
                    group_index: gi,
                    positions,
                },
                (gi, GroupChange::Resize(GroupSize::Full)),
                bias,
            ))
        }
    }

    /// Look up the running count offset for a molecule id (0 if unseen).
    fn get_offset(offsets: &[(usize, i32)], mol_id: usize) -> i32 {
        offsets
            .iter()
            .find(|(id, _)| *id == mol_id)
            .map_or(0, |(_, v)| *v)
    }

    /// Increment the running count offset for a molecule id.
    fn add_offset(offsets: &mut Vec<(usize, i32)>, mol_id: usize, delta: i32) {
        if let Some(entry) = offsets.iter_mut().find(|(id, _)| *id == mol_id) {
            entry.1 += delta;
        } else {
            offsets.push((mol_id, delta));
        }
    }

    /// Group population for a molecule kind, combining context state with pending offsets.
    ///
    /// Unlike `count_active_molecules` (which excludes reservoirs), this returns the
    /// actual population needed for bookkeeping: bounds checks and random index selection.
    /// Clamped to 0 to guard against underflow when multiple deactivations precede activations.
    fn effective_count(mol_id: usize, offset: i32, context: &impl Context) -> usize {
        // Use `count_active` (not `count_active_molecules`) because reservoirs need
        // a real head-count for bounds checks and random index selection even though
        // they are excluded from physical counts.
        let group_kind = context.topology_ref().moleculekinds()[mol_id].group_kind();
        let base = context.count_active(mol_id, group_kind);
        (base as i32 + offset).max(0) as usize
    }

    /// Swap one molecular group for another (deactivate source, activate target with aligned positions).
    /// Returns (actions, group_changes, ln_bias_delta), or None if infeasible.
    fn swap_molecule_one(
        from_mol_id: usize,
        to_mol_id: usize,
        context: &impl Context,
        rng: &mut dyn RngCore,
    ) -> Option<ActionBuild> {
        let full = context.find_molecules(from_mol_id, GroupSize::Full)?;
        if full.is_empty() {
            return None;
        }
        let &from_gi = full.iter().choose(rng)?;
        let n_from = full.len();

        let empty = context.find_molecules(to_mol_id, GroupSize::Empty)?;
        if empty.is_empty() {
            return None;
        }
        let &to_gi = empty.iter().choose(rng)?;
        let n_to = context.count_molecules(to_mol_id, GroupSize::Full);

        // N_from / (N_to + 1) combinatorial factor for detailed balance
        let ln_bias = (n_from as f64).ln() - ((n_to + 1) as f64).ln();

        let positions = overlay_swap_positions(
            context.groups()[from_gi].iter_active(),
            &context.groups()[to_gi],
            context,
            rng,
        );

        // Intramolecular energy excluded from ΔU — absorbed into K
        let actions = vec![
            SpeciationAction::DeactivateGroup(from_gi),
            SpeciationAction::ActivateGroup {
                group_index: to_gi,
                positions,
            },
        ];
        let changes = vec![
            (from_gi, GroupChange::ResizeExcludeIntra(GroupSize::Empty)),
            (to_gi, GroupChange::ResizeExcludeIntra(GroupSize::Full)),
        ];
        Some((actions, changes, ln_bias))
    }

    /// Swap one atom's type within a molecule.
    /// Returns (action, group_change, ln_bias_delta), or None if infeasible.
    fn swap_atom_one(
        from_id: usize,
        to_id: usize,
        molecule_id: usize,
        context: &impl Context,
        rng: &mut dyn RngCore,
    ) -> Option<(SpeciationAction, (usize, GroupChange), f64)> {
        // Full + partial groups (atomic mega-groups appear as partial)
        let group_indices: Vec<usize> = context
            .find_molecules(molecule_id, GroupSize::Full)
            .into_iter()
            .chain(context.find_molecules(molecule_id, GroupSize::Partial(0)))
            .flat_map(|s| s.iter().copied())
            .collect();
        if group_indices.is_empty() {
            return None;
        }

        let (mut n_from, mut n_to) = (0usize, 0usize);
        let mut from_atoms: Vec<(usize, usize)> = Vec::new();
        for &gi in &group_indices {
            for i in context.groups()[gi].iter_active() {
                let kind = context.atom_kind(i);
                if kind == from_id {
                    n_from += 1;
                    from_atoms.push((gi, i));
                } else if kind == to_id {
                    n_to += 1;
                }
            }
        }
        if n_from == 0 {
            return None;
        }

        let &(gi, abs) = from_atoms.choose(rng)?;
        // N_from / (N_to + 1) for detailed balance (ESPResSo convention)
        let ln_bias = (n_from as f64).ln() - ((n_to + 1) as f64).ln();

        Some((
            SpeciationAction::SwapAtomKind {
                group_index: gi,
                abs_index: abs,
                new_atom_id: to_id,
            },
            (gi, GroupChange::UpdateIdentity(vec![abs])),
            ln_bias,
        ))
    }

    /// Try to build speciation actions for one direction of a resolved reaction.
    fn try_build_actions(
        &self,
        resolved: &ResolvedReaction,
        direction: Direction,
        context: &impl Context,
        rng: &mut dyn RngCore,
    ) -> Option<ActionBuild> {
        let (ops, ln_k) = match direction {
            Direction::Forward => (&resolved.forward_ops, resolved.effective_ln_k),
            Direction::Backward => (&resolved.backward_ops, -resolved.effective_ln_k),
        };

        let volume = context.cell().volume()?;
        let vol = NewOld::from(volume, volume);
        let mut actions = Vec::new();
        let mut group_changes = Vec::new();
        let mut ln_bias = ln_k;

        // When a species appears multiple times (e.g. 2 OH⁻), each successive op must see
        // the updated count to produce the correct Smith & Triska ∏[N!/(N+ν)!] factor.
        // Without this, both OH⁻ activations would use the same N, giving (N+1)² instead of (N+1)(N+2).
        let mut offsets: Vec<(usize, i32)> = Vec::with_capacity(ops.len());

        for op in ops {
            match op {
                ReactionOp::DeactivateMolecule(mol_id) => {
                    let n = Self::effective_count(
                        *mol_id,
                        Self::get_offset(&offsets, *mol_id),
                        context,
                    );
                    let (a, c, b) = Self::deactivate_one(*mol_id, n, vol, context, rng)?;
                    ln_bias -= b;
                    actions.push(a);
                    group_changes.push(c);
                    Self::add_offset(&mut offsets, *mol_id, -1);
                }
                ReactionOp::ActivateMolecule(mol_id) => {
                    let n = Self::effective_count(
                        *mol_id,
                        Self::get_offset(&offsets, *mol_id),
                        context,
                    );
                    let (a, c, b) = Self::activate_one(*mol_id, n, vol, context, rng)?;
                    ln_bias -= b;
                    actions.push(a);
                    group_changes.push(c);
                    Self::add_offset(&mut offsets, *mol_id, 1);
                }
                ReactionOp::SwapMolecule {
                    from_mol_id,
                    to_mol_id,
                } => {
                    let (a, c, b) =
                        Self::swap_molecule_one(*from_mol_id, *to_mol_id, context, rng)?;
                    ln_bias += b;
                    actions.extend(a);
                    group_changes.extend(c);
                }
                ReactionOp::SwapAtom {
                    from_id,
                    to_id,
                    molecule_id,
                } => {
                    let (a, c, b) =
                        Self::swap_atom_one(*from_id, *to_id, *molecule_id, context, rng)?;
                    ln_bias += b;
                    actions.push(a);
                    group_changes.push(c);
                }
            }
        }

        Some((actions, group_changes, ln_bias))
    }
}

impl<T: Context> MoveProposal<T> for SpeciationMove {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        if self.resolved.is_empty() {
            return None;
        }

        // Pick random reaction and direction
        let reaction_index = rng.gen_range(0..self.resolved.len());
        let resolved = &self.resolved[reaction_index];
        let direction = if rng.r#gen::<bool>() {
            Direction::Forward
        } else {
            Direction::Backward
        };

        let (actions, group_changes, ln_bias) =
            self.try_build_actions(resolved, direction, context, rng)?;

        if actions.is_empty() {
            return None;
        }

        self.trial_ln_bias = Some(ln_bias);
        self.trial_reaction_index = Some(reaction_index);

        Some(ProposedMove {
            change: Change::Groups(group_changes),
            displacement: Displacement::None,
            transform: Transform::Speciation(actions),
            target: MoveTarget::System,
        })
    }

    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> crate::montecarlo::Bias {
        if let Some(ln_bias) = self.trial_ln_bias {
            crate::montecarlo::Bias::Energy(-self.thermal_energy * ln_bias)
        } else {
            crate::montecarlo::Bias::None
        }
    }

    fn on_trial_outcome(&mut self, accepted: bool) {
        if let Some(i) = self.trial_reaction_index.take() {
            if accepted {
                self.reaction_statistics[i].accept(0.0, Displacement::None);
            } else {
                self.reaction_statistics[i].reject();
            }
        }
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        let mut value = tagged_yaml("SpeciationMove", self)?;
        // Append per-reaction acceptance ratios
        if let serde_yml::Value::Tagged(ref mut tagged) = value {
            if let serde_yml::Value::Mapping(ref mut map) = tagged.value {
                let per_reaction: Vec<serde_yml::Value> = self
                    .reactions
                    .iter()
                    .zip(self.reaction_statistics.iter())
                    .map(|(config, stats)| {
                        serde_yml::Value::Mapping(serde_yml::Mapping::from_iter([
                            ("reaction".into(), config.0.clone().into()),
                            ("accepted".into(), stats.num_accepted.into()),
                            ("trials".into(), stats.num_trials.into()),
                            (
                                "acceptance_ratio".into(),
                                format!("{:.4}", stats.acceptance_ratio()).into(),
                            ),
                        ]))
                    })
                    .collect();
                map.insert("per_reaction".into(), per_reaction.into());
            }
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::group::GroupCollection;
    use crate::propagate::MoveProposal;
    use crate::WithCell;
    use float_cmp::assert_approx_eq;

    const TEST_YAML: &str = "tests/files/speciation_test.yaml";
    const RT: f64 = crate::R_IN_KJ_PER_MOL * 298.15;

    fn make_context() -> Backend {
        let mut rng = rand::thread_rng();
        Backend::new(TEST_YAML, None, &mut rng).unwrap()
    }

    fn make_move(reaction: &str, k: f64) -> SpeciationMove {
        serde_yml::from_str(&format!(
            "temperature: 298.15\nreactions:\n  - [\"{reaction}\", !K {k}]"
        ))
        .unwrap()
    }

    // --- YAML deserialization ---

    #[test]
    fn reaction_config_yaml_k() {
        let config: ReactionConfig = serde_yml::from_str(r#"["= NaCl", !K 100.0]"#).unwrap();
        assert_eq!(config.0, "= NaCl");
        assert!((config.1.to_k(1.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn reaction_config_yaml_pk() {
        let config: ReactionConfig = serde_yml::from_str(r#"["⚛HA = ⚛A + ~H+", !pK 4.0]"#).unwrap();
        assert!((config.1.to_k(1.0) - 1e-4).abs() < 1e-14);
    }

    #[test]
    fn reaction_config_yaml_lnk() {
        let config: ReactionConfig = serde_yml::from_str(r#"["= M", !lnK -2.302585]"#).unwrap();
        assert!((config.1.to_k(1.0) - 0.1).abs() < 1e-5);
    }

    #[test]
    fn reaction_config_yaml_dg() {
        // dG = 0 => K = 1; dG = -kT·ln(10) => K = 10
        let config: ReactionConfig = serde_yml::from_str(r#"["= Na+ + Cl-", !dG 0.0]"#).unwrap();
        assert!((config.1.to_k(2.479) - 1.0).abs() < 1e-10);

        let rt = 2.479;
        let config: ReactionConfig =
            serde_yml::from_str(&format!(r#"["= M", !dG {}]"#, -rt * 10.0_f64.ln())).unwrap();
        assert!((config.1.to_k(rt) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn speciation_move_yaml() {
        let yaml =
            "temperature: 298.15\nreactions:\n  - [\"= M\", !K 10.0]\n  - [\"⚛A = ⚛B\", !pK 0.0]";
        let mv: SpeciationMove = serde_yml::from_str(yaml).unwrap();
        assert_eq!(mv.temperature, 298.15);
        assert_eq!(mv.reactions.len(), 2);
    }

    #[test]
    fn unknown_field_rejected() {
        let yaml = r#"{ temperature: 300.0, reactions: [], bogus: 42 }"#;
        assert!(serde_yml::from_str::<SpeciationMove>(yaml).is_err());
    }

    // --- Finalize / reaction resolution ---

    #[test]
    fn finalize_resolves_molecular_insertion() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        assert_eq!(mv.resolved.len(), 1);
        let r = &mv.resolved[0];
        // Forward: activate M (product side)
        assert!(matches!(r.forward_ops[0], ReactionOp::ActivateMolecule(0)));
        // Backward: deactivate M
        assert!(matches!(
            r.backward_ops[0],
            ReactionOp::DeactivateMolecule(0)
        ));
        assert_approx_eq!(f64, r.effective_ln_k, 10.0_f64.ln());
    }

    #[test]
    fn finalize_resolves_atom_swap() {
        let context = make_context();
        // A and B are atoms in molecule AB (mol_id=1)
        let mut mv = make_move("⚛A = ⚛B", 1.0);
        mv.finalize(&context).unwrap();

        let r = &mv.resolved[0];
        assert!(matches!(
            r.forward_ops[0],
            ReactionOp::SwapAtom {
                from_id: 0,
                to_id: 1,
                ..
            }
        ));
        assert!(matches!(
            r.backward_ops[0],
            ReactionOp::SwapAtom {
                from_id: 1,
                to_id: 0,
                ..
            }
        ));
    }

    #[test]
    fn finalize_rejects_zero_temperature() {
        let context = make_context();
        let mut mv = make_move("= M", 1.0);
        mv.temperature = 0.0;
        assert!(mv.finalize(&context).is_err());
    }

    #[test]
    fn finalize_rejects_negative_k() {
        let context = make_context();
        let mut mv = make_move("= M", -1.0);
        assert!(mv.finalize(&context).is_err());
    }

    #[test]
    fn finalize_computes_thermal_energy() {
        let context = make_context();
        let mut mv = make_move("= M", 1.0);
        mv.finalize(&context).unwrap();
        let expected_rt = RT;
        assert_approx_eq!(f64, mv.thermal_energy, expected_rt, epsilon = 1e-10);
    }

    // --- Feasibility (try_build_actions) ---

    #[test]
    fn insertion_feasible_with_empty_groups() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        // M has 5 inactive groups -> insertion should be feasible
        let mut rng = rand::thread_rng();
        let result = mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng);
        assert!(result.is_some(), "Insertion should be feasible");
    }

    #[test]
    fn deletion_feasible_with_full_groups() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        // M has 5 active groups -> deletion (backward) should be feasible
        let mut rng = rand::thread_rng();
        let result = mv.try_build_actions(&mv.resolved[0], Direction::Backward, &context, &mut rng);
        assert!(result.is_some(), "Deletion should be feasible");
    }

    #[test]
    fn insertion_infeasible_when_no_empty_groups() {
        let mut context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        // Activate all M groups so none are empty
        let mol_id = 0;
        let empty_groups: Vec<usize> = context
            .find_molecules(mol_id, GroupSize::Empty)
            .map(|gs| gs.to_vec())
            .unwrap_or_default();
        for gi in empty_groups {
            crate::transform::Transform::Activate
                .on_group(gi, &mut context)
                .unwrap();
        }

        let mut rng = rand::thread_rng();
        let result = mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng);
        assert!(
            result.is_none(),
            "Insertion should fail when all groups are active"
        );
    }

    #[test]
    fn deletion_infeasible_when_no_full_groups() {
        let mut context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        // Deactivate all M groups so none are full
        let mol_id = 0;
        let full_groups: Vec<usize> = context
            .find_molecules(mol_id, GroupSize::Full)
            .map(|gs| gs.to_vec())
            .unwrap_or_default();
        for gi in full_groups {
            crate::transform::Transform::Deactivate
                .on_group(gi, &mut context)
                .unwrap();
        }

        let mut rng = rand::thread_rng();
        let result = mv.try_build_actions(&mv.resolved[0], Direction::Backward, &context, &mut rng);
        assert!(
            result.is_none(),
            "Deletion should fail when all groups are inactive"
        );
    }

    // --- Entropy bias ---

    #[test]
    fn insertion_bias_uses_volume_and_count() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        let mut rng = rand::thread_rng();
        let (_, _, ln_bias) = mv
            .try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng)
            .unwrap();

        // ln_bias = ln(K) - entropy_bias(n_new=6, n_old=5, V=1000)
        let volume = context.cell().volume().unwrap();
        let expected_entropy = entropy_bias(NewOld::from(6, 5), NewOld::from(volume, volume));
        let expected_ln_bias = 10.0_f64.ln() - expected_entropy;
        assert_approx_eq!(f64, ln_bias, expected_ln_bias, epsilon = 1e-10);
    }

    #[test]
    fn deletion_bias_uses_volume_and_count() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        let mut rng = rand::thread_rng();
        let (_, _, ln_bias) = mv
            .try_build_actions(&mv.resolved[0], Direction::Backward, &context, &mut rng)
            .unwrap();

        // Backward: ln_bias = -ln(K) - entropy_bias(n_new=4, n_old=5, V=1000)
        let volume = context.cell().volume().unwrap();
        let expected_entropy = entropy_bias(NewOld::from(4, 5), NewOld::from(volume, volume));
        let expected_ln_bias = -(10.0_f64.ln()) - expected_entropy;
        assert_approx_eq!(f64, ln_bias, expected_ln_bias, epsilon = 1e-10);
    }

    // --- Propose move ---

    #[test]
    fn propose_move_returns_system_target() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        let mut rng = rand::thread_rng();
        // Try multiple times since direction is random and might fail
        for _ in 0..20 {
            if let Some(proposed) = mv.propose_move(&context, &mut rng) {
                assert!(matches!(proposed.target, MoveTarget::System));
                assert!(matches!(proposed.transform, Transform::Speciation(_)));
                assert!(mv.trial_ln_bias.is_some());
                return;
            }
        }
        panic!("propose_move should succeed at least once in 20 tries");
    }

    #[test]
    fn propose_move_returns_none_when_empty() {
        let context = make_context();
        let mut mv: SpeciationMove =
            serde_yml::from_str("temperature: 298.15\nreactions: []").unwrap();
        mv.finalize(&context).unwrap();

        let mut rng = rand::thread_rng();
        assert!(mv.propose_move(&context, &mut rng).is_none());
    }

    // --- Bias ---

    #[test]
    fn bias_returns_energy_after_propose() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context).unwrap();

        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            if let Some(proposed) = mv.propose_move(&context, &mut rng) {
                let bias =
                    MoveProposal::<Backend>::bias(&mv, &proposed.change, &NewOld::from(0.0, 0.0));
                assert!(matches!(bias, crate::montecarlo::Bias::Energy(_)));
                return;
            }
        }
        panic!("Should get at least one proposal");
    }

    #[test]
    fn bias_returns_none_without_propose() {
        let mv = make_move("= M", 10.0);
        let bias = MoveProposal::<Backend>::bias(&mv, &Change::Everything, &NewOld::from(0.0, 0.0));
        assert!(matches!(bias, crate::montecarlo::Bias::None));
    }

    // --- Full simulation round-trip ---

    #[test]
    fn speciation_simulation_energy_drift() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let mut rng = rand::thread_rng();
        let context = Backend::new(TEST_YAML, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(TEST_YAML, &context).unwrap();

        let rt = RT;
        let mut mc =
            MarkovChain::new(context, propagate, rt, AnalysisCollection::default()).unwrap();

        let initial_energy = mc.system_energy();

        for step in mc.iter() {
            step.unwrap();
        }

        let drift = mc.energy_drift(initial_energy);
        assert!(
            drift < 1e-6,
            "Energy drift {drift:.6e} exceeds tolerance for speciation"
        );
    }

    // --- Molecular swap: phosphate titration at different pH ---

    /// Exact phosphate species fractions for ideal (non-interacting) system.
    ///
    /// α_i = (Ka1·...·Ka_i · [H⁺]^(3-i)) / D, where
    /// D = [H⁺]³ + Ka1·[H⁺]² + Ka1·Ka2·[H⁺] + Ka1·Ka2·Ka3
    fn phosphate_fractions(ph: f64) -> [f64; 4] {
        let h = 10.0_f64.powf(-ph);
        let ka1 = 10.0_f64.powf(-2.15);
        let ka2 = 10.0_f64.powf(-7.20);
        let ka3 = 10.0_f64.powf(-12.35);
        let d = h.powi(3) + ka1 * h.powi(2) + ka1 * ka2 * h + ka1 * ka2 * ka3;
        [
            h.powi(3) / d,
            ka1 * h.powi(2) / d,
            ka1 * ka2 * h / d,
            ka1 * ka2 * ka3 / d,
        ]
    }

    /// Generate YAML input for phosphate titration at a given pH.
    fn phosphate_yaml(ph: f64, n_molecules: usize, repeat: usize) -> String {
        let activity = 10.0_f64.powf(-ph);
        format!(
            r#"atoms:
  - {{name: P, mass: 31.0, sigma: 3.0}}
  - {{name: O, mass: 16.0, sigma: 2.8}}
  - {{name: H+, mass: 1.0, activity: {activity:.6e}}}
molecules:
  - name: "H3PO4"
    atoms: [P, O, O, O, O]
  - name: "H2PO4"
    atoms: [P, O, O, O, O]
  - name: "HPO4"
    atoms: [P, O, O, O, O]
  - name: "PO4"
    atoms: [P, O, O, O, O]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  energy: {{}}
  blocks:
    - molecule: "H3PO4"
      N: {n_molecules}
      active: 0
      insert: !RandomAtomPos {{}}
    - molecule: "H2PO4"
      N: {n_molecules}
      active: {n_molecules}
      insert: !RandomAtomPos {{}}
    - molecule: "HPO4"
      N: {n_molecules}
      active: 0
      insert: !RandomAtomPos {{}}
    - molecule: "PO4"
      N: {n_molecules}
      active: 0
      insert: !RandomAtomPos {{}}
propagate:
  seed: !Fixed 42
  criterion: Metropolis
  repeat: {repeat}
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          temperature: 298.15
          reactions:
            - ["H3PO4 = H2PO4 + ~H+", !pK 2.15]
            - ["H2PO4 = HPO4 + ~H+", !pK 7.20]
            - ["HPO4 = PO4 + ~H+", !pK 12.35]
"#
        )
    }

    /// Count active molecules of each phosphate species (mol_id 0..4).
    fn count_phosphate_species(context: &Backend) -> [usize; 4] {
        [0, 1, 2, 3].map(|id| context.count_molecules(id, GroupSize::Full))
    }

    #[test]
    fn molswap_phosphate_vs_henderson_hasselbalch() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let n_molecules = 40;
        let repeat = 20_000;
        let equilibrate = 2_000;

        // pH at each pKa and midpoints between them
        for ph in [2.15, 4.675, 7.20, 9.775, 12.35] {
            let yaml = phosphate_yaml(ph, n_molecules, repeat);
            let tmp = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
            let path = tmp.path();

            let mut rng = rand::thread_rng();
            let context = Backend::new(path, None, &mut rng).unwrap();
            let propagate = Propagate::from_file(path, &context).unwrap();
            let rt = RT;
            let mut mc =
                MarkovChain::new(context, propagate, rt, AnalysisCollection::default()).unwrap();

            let mut sums = [0.0_f64; 4];
            let mut n_samples = 0usize;

            for step_num in 0..repeat {
                let running = mc.propagate.propagate(
                    &mut mc.context,
                    mc.thermal_energy,
                    &mut mc.step,
                    &mut mc.analyses,
                );
                assert!(
                    running.unwrap(),
                    "Simulation ended early at step {step_num}"
                );
                if step_num >= equilibrate {
                    let counts = count_phosphate_species(&mc.context);
                    for (s, c) in sums.iter_mut().zip(counts.iter()) {
                        *s += *c as f64;
                    }
                    n_samples += 1;
                }
            }

            let expected = phosphate_fractions(ph);
            let total: f64 = sums.iter().sum();
            let observed: Vec<f64> = sums.iter().map(|s| s / total).collect();

            for (i, (obs, exp)) in observed.iter().zip(expected.iter()).enumerate() {
                let tol = 0.05; // 5% tolerance for stochastic test
                assert!(
                    (obs - exp).abs() < tol,
                    "pH={ph}, species {i}: observed {obs:.4} vs expected {exp:.4} \
                     (diff={:.4}, n_samples={n_samples})",
                    (obs - exp).abs()
                );
            }
        }
    }
}
