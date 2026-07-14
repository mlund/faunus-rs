//! Reaction ensemble (speciation) Monte Carlo move.
//!
//! Performs molecular insertion/deletion and atom-type swaps according to
//! chemical reactions. The acceptance criterion follows Smith & Triska (1994),
//! using `entropy_bias()` for the combinatorial `V^Δν · ∏[N!/(N+ν)!]` factors.
//! For atom swaps, a `N_from/(N_to+1)` combinatorial factor ensures detailed
//! balance, consistent with ESPResSo's reaction ensemble implementation.

use crate::chemistry::reaction::{Direction, Participant, Reaction};
use crate::group::{AbsIndex, AtomKindId, GroupSize, MoleculeId, RelIndex};
use crate::montecarlo::{entropy_bias, MoveStatistics, NewOld};
use crate::propagate::{
    default_repeat, default_weight, tagged_yaml, Displacement, MoveProposal, ProposedMove,
};
use crate::transform::SpeciationAction;
use crate::ObserveContext;
use crate::{cell::Shape, Change, GroupChange};
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
    ActivateMolecule(MoleculeId),
    /// Deactivate a group of this molecule type
    DeactivateMolecule(MoleculeId),
    /// Swap an atom from one kind to another within a molecule
    SwapAtom {
        from_id: AtomKindId,
        to_id: AtomKindId,
        /// Every molecule kind carrying this site. A titratable atom kind may appear in more
        /// than one kind of molecule, and all of them titrate — so `N_from`/`N_to` must be
        /// counted over the whole population, not just the first kind that matched.
        molecule_ids: Vec<MoleculeId>,
    },
    /// Swap a molecule of one kind for another (deactivate from, activate to with copied positions)
    SwapMolecule {
        from_mol_id: MoleculeId,
        to_mol_id: MoleculeId,
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
/// The thermal energy comes from the system medium; the move has no temperature of its own.
///
/// # YAML example
/// ```yaml
/// - !SpeciationMove
///   reactions:
///     - ["= NaCl", !K 100.0]
///     - ["⚛HA = ⚛A + ~H+", !pK 4.24]
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeciationMove {
    /// Reactions to sample from.
    reactions: Vec<ReactionConfig>,
    /// Temperature in Kelvin. Deprecated: the system temperature
    /// (`system.medium.temperature`) is used. Kept only so that existing inputs still
    /// parse; a value disagreeing with the system temperature is an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
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

impl_info!(
    SpeciationMove,
    "speciation",
    "Reaction ensemble (speciation)",
    "doi:10.1063/1.466443"
);

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
fn find_molecule_index(
    name: &str,
    topology: &crate::topology::Topology,
) -> anyhow::Result<MoleculeId> {
    topology
        .moleculekinds()
        .iter()
        .position(|m| m.name() == name)
        .map(MoleculeId::new)
        .ok_or_else(|| anyhow::anyhow!("Unknown molecule '{name}' in reaction"))
}

/// Pick a group from `groups`, skipping any already `claimed` by an earlier op in the same
/// reaction. When nothing is claimed (the usual single-op case) this draws exactly like the
/// plain `iter().choose` it replaces — same RNG consumption, so existing trajectories are
/// byte-for-byte unchanged; the filter only engages for coefficient-≥2 ops.
fn choose_unclaimed(groups: &[usize], claimed: &[usize], rng: &mut dyn RngCore) -> Option<usize> {
    if claimed.is_empty() {
        groups.iter().copied().choose(rng)
    } else {
        groups
            .iter()
            .copied()
            .filter(|g| !claimed.contains(g))
            .choose(rng)
    }
}

/// Extract atom participants as (atom_kind_index, name) pairs.
///
/// An atom name matching no atom kind is an error: dropping it silently leaves the reaction
/// with fewer operations than the user wrote, and the run then samples a different reaction
/// (or none at all) without saying so.
fn extract_atom_participants<'a>(
    participants: &'a [Participant],
    topology: &crate::topology::Topology,
) -> anyhow::Result<Vec<(AtomKindId, &'a str)>> {
    participants
        .iter()
        .filter_map(|p| match p {
            Participant::Atom(name) => Some(name),
            _ => None,
        })
        .map(|name| {
            let id = topology
                .atomkinds()
                .iter()
                .position(|a| a.name() == name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Unknown atom kind '{name}' in reaction; known kinds: {}",
                        topology
                            .atomkinds()
                            .iter()
                            .map(|a| a.name())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            Ok((AtomKindId::new(id), name.as_str()))
        })
        .collect()
}

/// Comma-separated molecule-kind names, for error messages.
fn kind_names(ids: &[MoleculeId], topology: &crate::topology::Topology) -> String {
    ids.iter()
        .map(|id| topology.moleculekind(*id).name().as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Find atom swap pairs: atoms appearing on both sides form swap operations.
fn resolve_atom_swaps(
    reactants: &[Participant],
    products: &[Participant],
    topology: &crate::topology::Topology,
) -> anyhow::Result<(Vec<ReactionOp>, Vec<ReactionOp>)> {
    let mut forward_ops = Vec::new();
    let mut backward_ops = Vec::new();

    let reactant_atoms = extract_atom_participants(reactants, topology)?;
    let product_atoms = extract_atom_participants(products, topology)?;

    // Atoms are paired off one-to-one, so an unbalanced count would be silently truncated by
    // the zip below and the move would run a different reaction than the one written.
    anyhow::ensure!(
        reactant_atoms.len() == product_atoms.len(),
        "Unbalanced atom stoichiometry: {} reactant atom(s) but {} product atom(s). \
         Each swapped atom needs a counterpart on the other side.",
        reactant_atoms.len(),
        product_atoms.len()
    );

    // Pair up reactant and product atoms
    for (from, to) in reactant_atoms.iter().zip(product_atoms.iter()) {
        // Which molecule kinds carry this site. Kinds listing *both* states are unambiguously
        // the titratable species, so if any exist they alone titrate — a kind listing only one
        // state may well be a different species that merely happens to reuse the atom kind
        // (e.g. a monatomic GCMC particle `M = [A]` alongside a titratable `AB = [A, B]`).
        // Only when no kind lists both do we fall back to kinds listing either, which is the
        // usual titration case: the template gives just the initial protonation state.
        //
        // Both tiers return *every* match, not the first: one titratable site may occur in
        // several molecule kinds, and all of them titrate. Binding to the first match freezes
        // every other kind's sites for the whole run.
        let kinds_with = |predicate: &dyn Fn(&[usize]) -> bool| -> Vec<MoleculeId> {
            topology
                .moleculekinds()
                .iter()
                .enumerate()
                .filter(|(_, m)| predicate(m.atom_indices()))
                .map(|(index, _)| MoleculeId::new(index))
                .collect()
        };
        let (from_index, to_index) = (from.0.get(), to.0.get());
        let mut molecule_ids =
            kinds_with(&|atoms: &[usize]| atoms.contains(&from_index) && atoms.contains(&to_index));

        if molecule_ids.is_empty() {
            let with_from = kinds_with(&|atoms: &[usize]| atoms.contains(&from_index));
            let with_to = kinds_with(&|atoms: &[usize]| atoms.contains(&to_index));

            // Kinds hosting one state and kinds hosting the other, with none hosting both,
            // cannot be told apart from a titratable site and an unrelated species that
            // merely reuses the atom kind — a free `A⁻` ion pool beside a titratable `HA`
            // site. Titrating both would convert ions into sites; titrating one silently
            // freezes the other. Neither is defensible, so ask the user to disambiguate.
            anyhow::ensure!(
                with_from.is_empty() || with_to.is_empty(),
                "Ambiguous atom swap '{}' ⇌ '{}': {} contain(s) '{}' while {} contain(s) '{}', \
                 and no molecule lists both. If these are the same titratable site, list both \
                 states in the molecule definition; if they are different species, use \
                 distinct atom names.",
                from.1,
                to.1,
                kind_names(&with_from, topology),
                from.1,
                kind_names(&with_to, topology),
                to.1
            );
            molecule_ids = if with_from.is_empty() {
                with_to
            } else {
                with_from
            };
        }

        anyhow::ensure!(
            !molecule_ids.is_empty(),
            "No molecule contains atom '{}' or '{}'",
            from.1,
            to.1
        );

        forward_ops.push(ReactionOp::SwapAtom {
            from_id: from.0,
            to_id: to.0,
            molecule_ids: molecule_ids.clone(),
        });
        backward_ops.push(ReactionOp::SwapAtom {
            from_id: to.0,
            to_id: from.0,
            molecule_ids,
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
    context: &impl ObserveContext,
    rng: &mut dyn RngCore,
) -> Vec<crate::Point> {
    let topology = context.topology();
    let atomkinds = topology.atomkinds();

    let source_masses: Vec<(crate::Point, f64)> = source_indices
        .clone()
        .map(|i| {
            let mass = atomkinds[context.atom_kind(i).get()].mass();
            (context.position(i), mass)
        })
        .collect();

    let com = context.mass_center(&source_indices.collect::<Vec<_>>());
    let template: Vec<crate::Point> = (0..target_group.capacity())
        .map(|i| context.position(target_group.start() + i))
        .collect();

    // The result is written into the target group's slots, so it must have exactly that many
    // positions — never the source group's count, which may differ. When either molecule has too
    // few atoms to define a principal-axis frame, there is no orientation to align and the
    // template is simply carried over to the outgoing molecule's mass center.
    crate::geometry::overlay_positions(&template, source_masses, &com, context.cell(), rng)
        .unwrap_or_else(|| crate::geometry::place_at(&template, &com, context.cell()))
}

/// Resolve a single reaction config into forward/backward ops and effective ln(K).
fn resolve_reaction(
    config: &ReactionConfig,
    thermal_energy: f64,
    topology: &crate::topology::Topology,
) -> anyhow::Result<ResolvedReaction> {
    let k = config.1.to_k(thermal_energy);
    // `exp` overflows to +∞ past an exponent of ~709 — reachable from a large `lnK`, or from
    // a `dG` given in J/mol where kJ/mol was meant. `k > 0.0` accepts ∞, whereupon the
    // acceptance bias is infinite and every trial is accepted regardless of energy.
    anyhow::ensure!(
        k.is_finite() && k > 0.0,
        "Equilibrium constant must be positive and finite for reaction '{}' (got {k})",
        config.0
    );

    let reaction = Reaction::from_reaction(&config.0, k)?;
    let (reactants, products) = reaction.get();

    let mut forward_ops = Vec::new();
    let mut backward_ops = Vec::new();

    // Collect molecular participants
    let mol_ids = |participants: &[Participant]| -> anyhow::Result<Vec<MoleculeId>> {
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
    // The overlay is only valid for a genuine 1:1 exchange, so the kind must appear exactly
    // once on each side: a coefficient >1 (e.g. `2 Na = Ca`) is a count-changing insertion-
    // deletion, not a position-preserving swap. Reservoirs are excluded: they use atomic
    // mega-groups, not individual molecular groups, so the swap overlay cannot apply.
    let multiplicity = |v: &[MoleculeId], id: MoleculeId| v.iter().filter(|&&x| x == id).count();
    let mut swap_reactants: Vec<usize> = Vec::new();
    let mut swap_products: Vec<usize> = Vec::new();
    for (ri, &from_id) in reactant_mols.iter().enumerate() {
        for (pi, &to_id) in product_mols.iter().enumerate() {
            if !swap_products.contains(&pi)
                && from_id != to_id
                && multiplicity(&reactant_mols, from_id) == 1
                && multiplicity(&product_mols, to_id) == 1
                && !topology.moleculekind(from_id).is_reservoir()
                && !topology.moleculekind(to_id).is_reservoir()
                && topology.moleculekind(from_id).atom_indices().len()
                    == topology.moleculekind(to_id).atom_indices().len()
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
                if let Some(activity) = topology.moleculekind(id).activity() {
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
    context: &impl ObserveContext,
    topology: &crate::topology::Topology,
) -> anyhow::Result<()> {
    // Does this kind have *any* group allocated by a `blocks:` entry? All three sizes count:
    // an insertion needs an Empty group to fill, a deletion a Full one, and an atomic
    // mega-group is Partial for as long as it is neither full nor empty.
    let has_any = |mol_id: MoleculeId| -> bool {
        [GroupSize::Full, GroupSize::Partial(0), GroupSize::Empty]
            .iter()
            .any(|size| context.count_molecules(mol_id, *size) > 0)
    };
    for r in resolved {
        for op in r.forward_ops.iter().chain(&r.backward_ops) {
            match op {
                ReactionOp::ActivateMolecule(id) | ReactionOp::DeactivateMolecule(id) => {
                    let name = topology.moleculekind(*id).name();
                    anyhow::ensure!(
                        has_any(*id),
                        "No groups found for molecule '{name}' needed by reaction"
                    );
                }
                ReactionOp::SwapAtom { molecule_ids, .. } => {
                    // A kind carrying the site must have groups *allocated*, but need not be
                    // populated yet: a kind that starts at `active: 0` may be filled during
                    // the run by another reaction, and its sites titrate from then on. An
                    // empty pool at this instant is a runtime rejection, not a bad input.
                    let names = molecule_ids
                        .iter()
                        .map(|id| topology.moleculekind(*id).name().as_str())
                        .collect::<Vec<_>>();
                    anyhow::ensure!(
                        molecule_ids.iter().copied().any(has_any),
                        "No groups allocated for molecule(s) {} needed by atom swap",
                        names.join(", ")
                    );
                }
                ReactionOp::SwapMolecule {
                    from_mol_id,
                    to_mol_id,
                } => {
                    for &id in [from_mol_id, to_mol_id] {
                        let name = topology.moleculekind(id).name();
                        anyhow::ensure!(
                            has_any(id),
                            "No groups found for molecule '{name}' needed by molecular swap"
                        );
                        // A molecular swap moves a whole group between Full and Empty. Every
                        // member of an atomic kind shares one mega-group, which is Partial
                        // whenever it is neither full nor empty, so the swap would find no
                        // group to act on and the reaction would silently never fire.
                        anyhow::ensure!(
                            !topology.moleculekind(id).atomic(),
                            "Molecule '{name}' is atomic and cannot take part in a molecular \
                             swap ('A = B'). Give it a non-atomic molecule kind, or write the \
                             reaction with explicit stoichiometry so it becomes an \
                             insertion/deletion."
                        );
                    }
                }
            }
        }

        // A molecular swap leaves intramolecular energy out of ΔU, on the grounds that it is
        // absorbed into K. The total energy still counts it, so if the two kinds differ in
        // intramolecular energy the incremental and full energies disagree and the run
        // accumulates drift — the very check that would otherwise catch a bad ΔU. Excluding
        // the intramolecular pairs removes the term from both sides and settles it.
        for op in &r.forward_ops {
            let ReactionOp::SwapMolecule {
                from_mol_id,
                to_mol_id,
            } = op
            else {
                continue;
            };
            let (from, to) = (
                topology.moleculekind(*from_mol_id),
                topology.moleculekind(*to_mol_id),
            );
            // Identical atom kinds means identical intramolecular energy, so nothing to warn
            // about however the pairs are treated.
            if from.atom_indices() == to.atom_indices() {
                continue;
            }
            let n = from.atom_indices().len();
            let all_pairs = n * n.saturating_sub(1) / 2;
            if from.exclusions().len() < all_pairs || to.exclusions().len() < all_pairs {
                log::warn!(
                    "SpeciationMove: '{}' ⇌ '{}' swaps molecules whose atoms differ, but not \
                     every intramolecular pair is excluded. Intramolecular energy is left out \
                     of ΔU (absorbed into K), so any difference between the two states will \
                     show up as energy drift. Add `exclusions` covering all intramolecular \
                     pairs if the equilibrium constant already accounts for them.",
                    from.name(),
                    to.name()
                );
            }
        }

        // All members of an atomic kind share one mega-group, so activating *and*
        // deactivating that kind in one reaction yields an expand and a shrink on the same
        // group. Those cannot be coalesced into the single change entry the energy path
        // requires, so reject the input instead of proposing a corrupt move.
        for ops in [&r.forward_ops, &r.backward_ops] {
            let atomic_targets = |activate: bool| -> Vec<MoleculeId> {
                ops.iter()
                    .filter_map(|op| match op {
                        ReactionOp::ActivateMolecule(id) if activate => Some(*id),
                        ReactionOp::DeactivateMolecule(id) if !activate => Some(*id),
                        _ => None,
                    })
                    .filter(|id| topology.moleculekind(*id).atomic())
                    .collect()
            };
            let (activated, deactivated) = (atomic_targets(true), atomic_targets(false));
            if let Some(id) = activated.iter().find(|id| deactivated.contains(id)) {
                anyhow::bail!(
                    "Atomic molecule '{}' appears on both sides of one reaction. Every atom of \
                     an atomic kind lives in a single group, so it cannot grow and shrink in \
                     the same step; use a non-atomic molecule kind.",
                    topology.moleculekind(*id).name()
                );
            }
        }
    }
    Ok(())
}

impl SpeciationMove {
    /// Resolve reaction strings to topology IDs and validate.
    ///
    /// `thermal_energy` is the system kT (kJ/mol). It is the single source of truth: the
    /// acceptance criterion divides by it, so the reaction bias must be built from it too.
    pub(crate) fn finalize(
        &mut self,
        context: &impl ObserveContext,
        thermal_energy: f64,
    ) -> anyhow::Result<()> {
        self.thermal_energy = thermal_energy;

        // The move used to carry its own temperature, which silently rescaled every
        // equilibrium constant when it disagreed with the system's.
        if let Some(temperature) = self.temperature {
            anyhow::ensure!(
                temperature > 0.0,
                "SpeciationMove: temperature must be positive"
            );
            let system_temperature = thermal_energy / crate::R_IN_KJ_PER_MOL;
            anyhow::ensure!(
                (temperature - system_temperature).abs() <= 1e-6 * system_temperature,
                "SpeciationMove: `temperature: {temperature}` K disagrees with the system \
                 temperature ({system_temperature:.4} K from `system.medium.temperature`). \
                 The key is deprecated and ignored — remove it."
            );
            log::warn!(
                "SpeciationMove: `temperature` is deprecated; the system temperature is used."
            );
        }

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
        mol_id: MoleculeId,
        n_old: usize,
        vol: NewOld<f64>,
        context: &impl ObserveContext,
        rng: &mut dyn RngCore,
        claimed: &[usize],
        claimed_slots: &[(usize, usize)],
    ) -> Option<(SpeciationAction, (usize, GroupChange), f64)> {
        let molecule = context.topology_ref().moleculekind(mol_id);
        if molecule.atomic() {
            let gi = context.find_atomic_group(mol_id)?;
            if n_old == 0 {
                return None;
            }
            // Pick a random slot for deactivation. The transform's swap-and-pop
            // (in `Transform::DeactivateAtom`) keeps the active range contiguous
            // by swapping the chosen atom with the last active slot before
            // shrinking. The change variant carries the pre-shrink size `n_old`
            // so the energy code can distinguish pre- vs. post-transform state
            // (`groups[gi].len() < n_old` means post) — required because the
            // swap leaves a *different* atom at slot `rel` in the new state.
            //
            // Slots claimed by an earlier op in this reaction are off limits: a
            // coefficient-2 deletion (`Ca(OH)₂ = Ca²⁺ + 2 OH⁻`) must remove two *distinct*
            // atoms, or the reaction silently removes one while the bias assumes two.
            // The context is not mutated during proposal, so the true pre-transform size
            // is the group's current length — `n_old` here is the *effective* count, which
            // already accounts for earlier ops and is what the entropy bias needs.
            let num_active = context.groups()[gi].len();
            let rel = if claimed_slots.iter().all(|(group, _)| *group != gi) {
                // Nothing claimed in *this* group: draw exactly as a lone deletion would,
                // so the RNG stream of every single-deletion reaction is unchanged.
                rng.gen_range(0..num_active)
            } else {
                (0..num_active)
                    .filter(|slot| !claimed_slots.contains(&(gi, *slot)))
                    .choose(rng)?
            };
            let rel = RelIndex::new(rel);
            let abs = context.groups()[gi].to_absolute(rel).ok()?.get();
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
                    GroupChange::AtomicShrink {
                        rels: vec![rel],
                        n_old: num_active,
                    },
                ),
                bias,
            ))
        } else {
            // Exclude groups already claimed by earlier ops in this reaction, so a
            // coefficient-2 deletion (e.g. `2 Na = Ca`) empties two DISTINCT groups
            // rather than the same one twice (which would leave charge unbalanced).
            let full = context.find_molecules(mol_id, GroupSize::Full);
            let gi = choose_unclaimed(full, claimed, rng)?;
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
        mol_id: MoleculeId,
        n_old: usize,
        vol: NewOld<f64>,
        context: &impl ObserveContext,
        rng: &mut dyn RngCore,
        claimed: &[usize],
    ) -> Option<(SpeciationAction, (usize, GroupChange), f64)> {
        let molecule = context.topology_ref().moleculekind(mol_id);
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
            let position = context.cell().get_point_inside(rng);
            Some((
                SpeciationAction::ActivateAtom {
                    group_index: gi,
                    position,
                },
                (
                    gi,
                    GroupChange::ResizePartial(GroupSize::Expand(1), vec![RelIndex::new(n_old)]),
                ),
                bias,
            ))
        } else {
            // Exclude groups already claimed by earlier ops in this reaction (distinct
            // groups for a coefficient-2 insertion).
            let empty = context.find_molecules(mol_id, GroupSize::Empty);
            let gi = choose_unclaimed(empty, claimed, rng)?;
            let bias = entropy_bias(NewOld::from(n_old + 1, n_old), vol);
            let com = context.cell().get_point_inside(rng);

            // Place the molecule's template at a random position and orientation. A lone
            // atom has neither internal geometry nor an orientation, so it skips the
            // rotation entirely — and with it the RNG draw, keeping the stream of every
            // monatomic GCMC input unchanged.
            let capacity = context.groups()[gi].capacity();
            let positions = if capacity == 1 {
                vec![com]
            } else {
                let centered = Self::centered_template(mol_id, gi, context);
                crate::topology::InsertionPolicy::place_molecule_at(
                    &centered,
                    &com,
                    true,
                    context.cell(),
                    rng,
                )
            };
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
    fn get_offset(offsets: &[(MoleculeId, i32)], mol_id: MoleculeId) -> i32 {
        offsets
            .iter()
            .find(|(id, _)| *id == mol_id)
            .map_or(0, |(_, v)| *v)
    }

    /// Increment the running count offset for a molecule id.
    fn add_offset(offsets: &mut Vec<(MoleculeId, i32)>, mol_id: MoleculeId, delta: i32) {
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
    fn effective_count(mol_id: MoleculeId, offset: i32, context: &impl ObserveContext) -> usize {
        // Use `count_active` (not `count_active_molecules`) because reservoirs need
        // a real head-count for bounds checks and random index selection even though
        // they are excluded from physical counts.
        let group_kind = context.topology_ref().moleculekind(mol_id).group_kind();
        let base = context.count_active(mol_id, group_kind);
        (base as i32 + offset).max(0) as usize
    }

    /// Swap one molecular group for another (deactivate source, activate target with aligned positions).
    /// Returns (actions, group_changes, ln_bias_delta), or None if infeasible.
    fn swap_molecule_one(
        from_mol_id: MoleculeId,
        to_mol_id: MoleculeId,
        context: &impl ObserveContext,
        rng: &mut dyn RngCore,
        claimed: &[usize],
    ) -> Option<ActionBuild> {
        // Pick unclaimed source/target groups (counts stay the full available totals so the
        // N_from/(N_to+1) detailed-balance factor is unaffected).
        let full = context.find_molecules(from_mol_id, GroupSize::Full);
        let n_from = full.len();
        let from_gi = choose_unclaimed(full, claimed, rng)?;

        let empty = context.find_molecules(to_mol_id, GroupSize::Empty);
        let to_gi = choose_unclaimed(empty, claimed, rng)?;
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
        from_id: AtomKindId,
        to_id: AtomKindId,
        molecule_ids: &[MoleculeId],
        context: &impl ObserveContext,
        rng: &mut dyn RngCore,
        claimed_atoms: &[(usize, AtomKindId)],
    ) -> Option<(SpeciationAction, (usize, GroupChange), f64)> {
        // Every kind carrying the site, so the combinatorial factor is counted over the whole
        // population. Full + partial groups (atomic mega-groups appear as partial).
        let group_indices: Vec<usize> = molecule_ids
            .iter()
            .flat_map(|&mol_id| {
                context
                    .find_molecules(mol_id, GroupSize::Full)
                    .iter()
                    .chain(context.find_molecules(mol_id, GroupSize::Partial(0)))
                    .copied()
            })
            .collect();
        if group_indices.is_empty() {
            return None;
        }

        // The context is not mutated during proposal, so an atom already swapped by an
        // earlier op in this reaction still reads as its *old* kind. Substituting the kind it
        // is becoming is what makes a coefficient-2 titration correct: without it both ops
        // draw from the same pool and use the same N_from/N_to, so one atom can be converted
        // twice while the bias assumes two — charge is not conserved and the acceptance
        // factor is N/(N_to+1)² instead of N(N-1)/((N_to+1)(N_to+2)).
        let (mut n_from, mut n_to) = (0usize, 0usize);
        let mut from_atoms: Vec<(usize, usize)> = Vec::new();
        for &gi in &group_indices {
            for i in context.groups()[gi].iter_active() {
                let kind = claimed_atoms
                    .iter()
                    .find(|(index, _)| *index == i)
                    .map_or_else(|| context.atom_kind(i), |(_, kind)| *kind);
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

        // `SwapAtomKind` addresses the global particle array, but every index-carrying
        // `GroupChange` is group-relative — `energy/ewald.rs` reconstructs the global index as
        // `group.start() + rel`. Emitting the absolute index here would point Ewald at a different
        // particle for any group that does not start at 0.
        let rel = context.groups()[gi].to_relative(AbsIndex::new(abs)).ok()?;

        Some((
            SpeciationAction::SwapAtomKind {
                group_index: gi,
                abs_index: abs,
                new_atom_id: to_id,
            },
            (gi, GroupChange::UpdateIdentity(vec![rel])),
            ln_bias,
        ))
    }

    /// Origin-centred coordinates to insert a molecule of kind `mol_id` with.
    ///
    /// Prefer the kind's reference geometry, but that is only populated by `from_structure`;
    /// a molecule declared with a plain `atoms:` list has none. Fall back to the coordinates
    /// the group itself already holds: `resize_group` only changes the active count, so an
    /// inactive group still carries the conformation it was built with. Returning the
    /// reference positions blindly would yield an *empty* vector for such kinds, and the
    /// transform would then activate the group without writing any coordinates at all —
    /// re-inserting it at stale positions while the bias assumes a uniform random insertion.
    fn centered_template(
        mol_id: MoleculeId,
        group_index: usize,
        context: &impl ObserveContext,
    ) -> Vec<crate::Point> {
        let topology = context.topology_ref();
        let molecule = topology.moleculekind(mol_id);
        if !molecule.reference_positions().is_empty() {
            return crate::topology::InsertionPolicy::centered_reference_positions(
                molecule,
                topology.atomkinds(),
            );
        }

        let group = &context.groups()[group_index];
        let indices: Vec<usize> = (group.start()..group.start() + group.capacity()).collect();
        let mut positions: Vec<crate::Point> =
            indices.iter().map(|&i| context.position(i)).collect();
        let masses: Vec<f64> = indices.iter().map(|&i| context.atom_mass(i)).collect();
        let com = crate::geometry::mass_center(&positions, &masses);
        positions.iter_mut().for_each(|pos| *pos -= com);
        positions
    }

    /// Fold a group change into the list, coalescing with any existing entry for the same
    /// group.
    ///
    /// The energy path pairs *distinct* entries with one another, so a group index must
    /// appear at most once; two entries naming the same group would make it pair that group
    /// with itself and corrupt ΔU. Reactions that touch one atomic mega-group several times
    /// (`Ca(OH)₂ = Ca²⁺ + 2 OH⁻`) therefore merge here into a single entry.
    ///
    /// Returns `None` for a combination that cannot be merged — e.g. activating and
    /// deactivating the same atomic kind in one reaction — which rejects the proposal
    /// rather than producing a corrupt one.
    fn merge_change(
        changes: &mut Vec<(usize, GroupChange)>,
        gi: usize,
        change: GroupChange,
    ) -> Option<()> {
        let Some((_, existing)) = changes.iter_mut().find(|(index, _)| *index == gi) else {
            changes.push((gi, change));
            return Some(());
        };
        match (existing, change) {
            (
                GroupChange::AtomicShrink { rels, n_old },
                GroupChange::AtomicShrink {
                    rels: more,
                    n_old: n,
                },
            ) if *n_old == n => rels.extend(more),
            (
                GroupChange::ResizePartial(GroupSize::Expand(count), rels),
                GroupChange::ResizePartial(GroupSize::Expand(more_count), more),
            ) => {
                *count += more_count;
                rels.extend(more);
            }
            (GroupChange::UpdateIdentity(rels), GroupChange::UpdateIdentity(more)) => {
                rels.extend(more)
            }
            _ => return None,
        }
        Some(())
    }

    /// Try to build speciation actions for one direction of a resolved reaction.
    fn try_build_actions(
        &self,
        resolved: &ResolvedReaction,
        direction: Direction,
        context: &impl ObserveContext,
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
        let mut offsets: Vec<(MoleculeId, i32)> = Vec::with_capacity(ops.len());

        // Group indices already selected by earlier ops in THIS reaction. Non-atomic
        // selection skips these so repeated ops on one kind (e.g. `2 Na = Ca`) hit distinct
        // groups; atomic mega-groups reuse the same group (coordinated via `offsets` and
        // `claimed_slots` instead).
        let mut claimed: Vec<usize> = Vec::new();

        // Atom slots (group index, relative slot) already spoken for by an earlier op, so a
        // repeated deletion from one mega-group cannot pick the same atom twice.
        let mut claimed_slots: Vec<(usize, usize)> = Vec::new();

        // Atoms already swapped by an earlier op, with the kind they are becoming, so a
        // repeated swap sees the counts the first one leaves behind.
        let mut claimed_atoms: Vec<(usize, AtomKindId)> = Vec::new();

        for op in ops {
            match op {
                ReactionOp::DeactivateMolecule(mol_id) => {
                    let n = Self::effective_count(
                        *mol_id,
                        Self::get_offset(&offsets, *mol_id),
                        context,
                    );
                    let (a, c, b) = Self::deactivate_one(
                        *mol_id,
                        n,
                        vol,
                        context,
                        rng,
                        &claimed,
                        &claimed_slots,
                    )?;
                    ln_bias -= b;
                    claimed.push(c.0);
                    if let GroupChange::AtomicShrink { rels, .. } = &c.1 {
                        claimed_slots.extend(rels.iter().map(|rel| (c.0, rel.get())));
                    }
                    actions.push(a);
                    Self::merge_change(&mut group_changes, c.0, c.1)?;
                    Self::add_offset(&mut offsets, *mol_id, -1);
                }
                ReactionOp::ActivateMolecule(mol_id) => {
                    let n = Self::effective_count(
                        *mol_id,
                        Self::get_offset(&offsets, *mol_id),
                        context,
                    );
                    let (a, c, b) = Self::activate_one(*mol_id, n, vol, context, rng, &claimed)?;
                    ln_bias -= b;
                    claimed.push(c.0);
                    actions.push(a);
                    Self::merge_change(&mut group_changes, c.0, c.1)?;
                    Self::add_offset(&mut offsets, *mol_id, 1);
                }
                ReactionOp::SwapMolecule {
                    from_mol_id,
                    to_mol_id,
                } => {
                    let (a, c, b) =
                        Self::swap_molecule_one(*from_mol_id, *to_mol_id, context, rng, &claimed)?;
                    ln_bias += b;
                    claimed.extend(c.iter().map(|(gi, _)| *gi));
                    actions.extend(a);
                    for (gi, change) in c {
                        Self::merge_change(&mut group_changes, gi, change)?;
                    }
                }
                ReactionOp::SwapAtom {
                    from_id,
                    to_id,
                    molecule_ids,
                } => {
                    let (a, c, b) = Self::swap_atom_one(
                        *from_id,
                        *to_id,
                        molecule_ids,
                        context,
                        rng,
                        &claimed_atoms,
                    )?;
                    ln_bias += b;
                    if let SpeciationAction::SwapAtomKind {
                        abs_index,
                        new_atom_id,
                        ..
                    } = &a
                    {
                        claimed_atoms.push((*abs_index, *new_atom_id));
                    }
                    actions.push(a);
                    Self::merge_change(&mut group_changes, c.0, c.1)?;
                }
            }
        }

        Some((actions, group_changes, ln_bias))
    }
}

impl<T: ObserveContext> MoveProposal<T> for SpeciationMove {
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

        Some(ProposedMove::speciation(actions, group_changes))
    }

    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> crate::montecarlo::Bias {
        // Dimensionless: the criterion multiplies by the same thermal energy it divides
        // the total by, so the reaction bias is exactly ln K_eff whatever the temperature.
        if let Some(ln_bias) = self.trial_ln_bias {
            crate::montecarlo::Bias::Dimensionless(-ln_bias)
        } else {
            crate::montecarlo::Bias::None
        }
    }

    fn on_trial_outcome(&mut self, _context: &T, accepted: bool) {
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
    use crate::cell::BoundaryConditions;
    use crate::group::GroupCollection;
    use crate::propagate::MoveProposal;
    use crate::Change;
    use crate::WithSimulationCell;
    use float_cmp::assert_approx_eq;

    const TEST_YAML: &str = "tests/files/speciation_test.yaml";
    const THERMAL_ENERGY: f64 = crate::R_IN_KJ_PER_MOL * 298.15;

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

        let thermal_energy = 2.479;
        let config: ReactionConfig = serde_yml::from_str(&format!(
            r#"["= M", !dG {}]"#,
            -thermal_energy * 10.0_f64.ln()
        ))
        .unwrap();
        assert!((config.1.to_k(thermal_energy) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn speciation_move_yaml() {
        let yaml =
            "temperature: 298.15\nreactions:\n  - [\"= M\", !K 10.0]\n  - [\"⚛A = ⚛B\", !pK 0.0]";
        let mv: SpeciationMove = serde_yml::from_str(yaml).unwrap();
        assert_eq!(mv.temperature, Some(298.15));
        assert_eq!(mv.reactions.len(), 2);
    }

    /// The `temperature` key is deprecated and optional; the system value is used.
    #[test]
    fn speciation_move_yaml_without_temperature() {
        let mv: SpeciationMove = serde_yml::from_str("reactions:\n  - [\"= M\", !K 10.0]").unwrap();
        assert_eq!(mv.temperature, None);

        let context = make_context();
        let mut mv = mv;
        mv.finalize(&context, THERMAL_ENERGY).unwrap();
        assert_approx_eq!(f64, mv.thermal_energy, THERMAL_ENERGY, epsilon = 1e-12);
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
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        assert_eq!(mv.resolved.len(), 1);
        let r = &mv.resolved[0];
        // Forward: activate M (product side)
        assert!(
            matches!(r.forward_ops[0], ReactionOp::ActivateMolecule(id) if id == MoleculeId::new(0))
        );
        // Backward: deactivate M
        assert!(
            matches!(r.backward_ops[0], ReactionOp::DeactivateMolecule(id) if id == MoleculeId::new(0))
        );
        assert_approx_eq!(f64, r.effective_ln_k, 10.0_f64.ln());
    }

    #[test]
    fn finalize_resolves_atom_swap() {
        let context = make_context();
        // A and B are atoms in molecule AB (mol_id=1)
        let mut mv = make_move("⚛A = ⚛B", 1.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let r = &mv.resolved[0];
        assert!(matches!(
            r.forward_ops[0],
            ReactionOp::SwapAtom { from_id, to_id, .. }
                if from_id == AtomKindId::new(0) && to_id == AtomKindId::new(1)
        ));
        assert!(matches!(
            r.backward_ops[0],
            ReactionOp::SwapAtom { from_id, to_id, .. }
                if from_id == AtomKindId::new(1) && to_id == AtomKindId::new(0)
        ));
    }

    #[test]
    fn finalize_rejects_zero_temperature() {
        let context = make_context();
        let mut mv = make_move("= M", 1.0);
        mv.temperature = Some(0.0);
        assert!(mv.finalize(&context, THERMAL_ENERGY).is_err());
    }

    /// A `temperature` disagreeing with the system's used to silently raise every
    /// equilibrium constant to the power T_move/T_system. It is now a startup error.
    #[test]
    fn finalize_rejects_temperature_disagreeing_with_system() {
        let context = make_context(); // system temperature is 298.15 K
        let mut mv = make_move("= M", 1.0);
        mv.temperature = Some(310.0);

        let err = mv
            .finalize(&context, THERMAL_ENERGY)
            .expect_err("a mismatched move temperature must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("310"),
            "error should name the move value: {msg}"
        );
        assert!(
            msg.contains("298.15"),
            "error should name the system value: {msg}"
        );
    }

    #[test]
    fn finalize_rejects_negative_k() {
        let context = make_context();
        let mut mv = make_move("= M", -1.0);
        assert!(mv.finalize(&context, THERMAL_ENERGY).is_err());
    }

    #[test]
    fn finalize_computes_thermal_energy() {
        let context = make_context();
        let mut mv = make_move("= M", 1.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();
        assert_approx_eq!(f64, mv.thermal_energy, THERMAL_ENERGY, epsilon = 1e-10);
    }

    // --- Feasibility (try_build_actions) ---

    #[test]
    fn insertion_feasible_with_empty_groups() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        // M has 5 inactive groups -> insertion should be feasible
        let mut rng = rand::thread_rng();
        let result = mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng);
        assert!(result.is_some(), "Insertion should be feasible");
    }

    #[test]
    fn deletion_feasible_with_full_groups() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        // M has 5 active groups -> deletion (backward) should be feasible
        let mut rng = rand::thread_rng();
        let result = mv.try_build_actions(&mv.resolved[0], Direction::Backward, &context, &mut rng);
        assert!(result.is_some(), "Deletion should be feasible");
    }

    #[test]
    fn insertion_infeasible_when_no_empty_groups() {
        let mut context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        // Activate all M groups so none are empty
        let mol_id = MoleculeId::new(0);
        let empty_groups: Vec<usize> = context.find_molecules(mol_id, GroupSize::Empty).to_vec();
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
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        // Deactivate all M groups so none are full
        let mol_id = MoleculeId::new(0);
        let full_groups: Vec<usize> = context.find_molecules(mol_id, GroupSize::Full).to_vec();
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
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

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
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

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
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let mut rng = rand::thread_rng();
        // Try multiple times since direction is random and might fail
        for _ in 0..20 {
            if let Some(proposed) = mv.propose_move(&context, &mut rng) {
                assert!(matches!(
                    proposed.target(),
                    crate::propagate::MoveTarget::System
                ));
                assert!(matches!(
                    proposed.transform(),
                    crate::transform::Transform::Speciation(_)
                ));
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
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let mut rng = rand::thread_rng();
        assert!(mv.propose_move(&context, &mut rng).is_none());
    }

    // --- Bias ---

    #[test]
    fn bias_returns_dimensionless_after_propose() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            if let Some(proposed) = mv.propose_move(&context, &mut rng) {
                let bias =
                    MoveProposal::<Backend>::bias(&mv, proposed.change(), &NewOld::from(0.0, 0.0));
                assert!(matches!(bias, crate::montecarlo::Bias::Dimensionless(_)));
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
        let propagate = Propagate::from_file(TEST_YAML, &context, THERMAL_ENERGY).unwrap();

        let thermal_energy = THERMAL_ENERGY;
        let mut mc = MarkovChain::new(
            context,
            propagate,
            thermal_energy,
            AnalysisCollection::default(),
        )
        .unwrap();

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

    /// Charge-conserving 2:1 ion exchange `2 Na⁺ ⇌ Ca²⁺` with explicit Coulomb energy.
    /// Exercises the coefficient-2 reactant path (insertion/deletion of two distinct Na
    /// groups + one Ca) and checks the incremental energy stays consistent with a full
    /// recompute — the regression that the swap-overlay/duplicate-group bugs would break.
    fn charge_swap_yaml(atomic: bool) -> String {
        let flag = if atomic { "\n    atomic: true" } else { "" };
        format!(
            r#"atoms:
  - {{name: Na, mass: 23.0, charge: 1.0, sigma: 4.0}}
  - {{name: Ca, mass: 40.0, charge: 2.0, sigma: 4.0}}
molecules:
  - name: na
    atoms: [Na]{flag}
  - name: ca
    atoms: [Ca]{flag}
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium:
    permittivity: !Fixed 78.0
    temperature: 298.15
  energy:
    nonbonded:
      default:
        - !Coulomb {{cutoff: 14.0}}
  blocks:
    - molecule: na
      N: 20
      active: 20
      insert: !RandomAtomPos {{}}
    - molecule: ca
      N: 10
      active: 0
      insert: !RandomAtomPos {{}}
propagate:
  seed: !Fixed 42
  criterion: Metropolis
  repeat: 2000
  collections:
    - !Stochastic
      moves:
        - !SpeciationMove
          temperature: 298.15
          reactions:
            - ["na + na = ca", !dG 0.0]
"#
        )
    }

    /// Run the `2 Na = Ca` system and return (energy drift, active Na count, active Ca count).
    /// Molecule ids: 0 = na, 1 = ca (declaration order).
    fn run_charge_swap(atomic: bool) -> (f64, usize, usize) {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let yaml = charge_swap_yaml(atomic);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let path = tmp.path().to_str().unwrap();
        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();
        let initial_energy = mc.system_energy();
        for step in mc.iter() {
            step.unwrap();
        }
        let drift = mc.energy_drift(initial_energy);
        // `count_active_molecules` works for both flavours: a partly-filled atomic
        // mega-group is `Partial`, never `Full`, so counting Full *groups* would report
        // zero for the atomic case however many atoms are active.
        let n_na = mc.context().count_active_molecules(MoleculeId::new(0));
        let n_ca = mc.context().count_active_molecules(MoleculeId::new(1));
        (drift, n_na, n_ca)
    }

    /// `2 Na⁺ ⇌ Ca²⁺` on individual (non-atomic) single-atom groups: the coefficient-2
    /// reactant deletes two DISTINCT groups, so the cross-term between the two removed Na
    /// (and the inserted Ca) is handled by the multi-group energy path. Verifies BOTH charge
    /// conservation (each Ca consumes exactly two Na, so Na + 2·Ca stays at the initial total
    /// charge of 20) and energy consistency (incremental vs. full recompute).
    #[test]
    fn charge_conserving_swap() {
        let (drift, n_na, n_ca) = run_charge_swap(false);
        assert!(n_ca > 0, "reaction never fired (no Ca formed)");
        assert_eq!(
            n_na + 2 * n_ca,
            20,
            "charge not conserved: {n_na} Na + {n_ca} Ca"
        );
        assert!(drift < 1e-6, "energy drift {drift:.6e} for 2 Na = Ca");
    }

    /// The same reaction on an ATOMIC mega-group must give the same physics as on separate
    /// molecular groups: the two deletions land in one mega-group, so they have to be
    /// coalesced into a single change entry (otherwise the energy path pairs the group with
    /// itself) and must pick two *distinct* slots (otherwise one Na is deleted, not two).
    #[test]
    fn charge_conserving_swap_atomic() {
        let (drift, n_na, n_ca) = run_charge_swap(true);
        assert!(n_ca > 0, "reaction never fired (no Ca formed)");
        assert_eq!(
            n_na + 2 * n_ca,
            20,
            "charge not conserved: {n_na} Na + {n_ca} Ca"
        );
        assert!(
            drift < 1e-6,
            "energy drift {drift:.6e} for atomic 2 Na = Ca"
        );
    }

    /// The proposal-level defect behind `charge_conserving_swap_atomic`. Deleting two atoms
    /// from one mega-group must (a) pick two distinct slots and (b) emit a *single* change
    /// entry for that group — `multi_group_change` pairs entries with one another, so a
    /// repeated group index makes it pair the group with itself and corrupt ΔU.
    #[test]
    fn atomic_multi_delete_picks_distinct_slots() {
        let yaml = charge_swap_yaml(true);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();

        let mut rng = rand::thread_rng();
        let context = Backend::new(tmp.path(), None, &mut rng).unwrap();
        let mut mv = make_move("na + na = ca", 1.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let mut proposals = 0;
        for attempt in 0..200 {
            let Some((actions, changes, _)) =
                mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng)
            else {
                continue;
            };
            proposals += 1;

            let groups: Vec<usize> = changes.iter().map(|(gi, _)| *gi).collect();
            let unique: std::collections::BTreeSet<usize> = groups.iter().copied().collect();
            assert_eq!(
                groups.len(),
                unique.len(),
                "attempt {attempt}: duplicate group index in changes {groups:?}"
            );

            let deleted: Vec<usize> = actions
                .iter()
                .filter_map(|a| match a {
                    SpeciationAction::DeactivateAtom { abs_index, .. } => Some(*abs_index),
                    _ => None,
                })
                .collect();
            assert_eq!(deleted.len(), 2, "expected two deletions, got {deleted:?}");
            assert_ne!(
                deleted[0], deleted[1],
                "attempt {attempt}: both deletions hit the same atom {deleted:?}"
            );
        }
        assert!(proposals > 0, "no proposal was ever built");
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
        [0, 1, 2, 3].map(|id| context.count_molecules(MoleculeId::new(id), GroupSize::Full))
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
            let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
            let thermal_energy = THERMAL_ENERGY;
            let mut mc = MarkovChain::new(
                context,
                propagate,
                thermal_energy,
                AnalysisCollection::default(),
            )
            .unwrap();

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

    // --- Input validation: bad YAML must fail at startup, not sample wrong physics ---

    /// An atom name that matches no atom kind was silently dropped, leaving the reaction with
    /// no operations: `propose_move` returned `None` forever and the run finished reporting a
    /// titration that never happened.
    #[test]
    fn unknown_atom_in_reaction_is_error() {
        let context = make_context();
        let mut mv = make_move("⚛Zz = ⚛A", 1.0);
        let err = mv
            .finalize(&context, THERMAL_ENERGY)
            .expect_err("unknown atom kind must be rejected");
        assert!(err.to_string().contains("Zz"), "{err}");
    }

    /// Reactant and product atoms were zipped, so a longer side was silently truncated and the
    /// move performed a different reaction than the one written.
    #[test]
    fn unbalanced_atom_stoichiometry_is_error() {
        let context = make_context();
        let mut mv = make_move("⚛A + ⚛A = ⚛B", 1.0);
        assert!(mv.finalize(&context, THERMAL_ENERGY).is_err());
    }

    /// `to_k` overflows to +∞ for a large `lnK` or a large negative `dG` (a kJ/J units slip),
    /// and `k > 0.0` happily accepts ∞ — after which the Metropolis criterion accepts every
    /// trial and the run degenerates in silence.
    #[test]
    fn non_finite_equilibrium_constant_is_error() {
        let context = make_context();

        for yaml in [
            "reactions:\n  - [\"= M\", !lnK 1000.0]",
            "reactions:\n  - [\"= M\", !dG -1.0e6]",
        ] {
            let mut mv: SpeciationMove = serde_yml::from_str(yaml).unwrap();
            let err = mv
                .finalize(&context, THERMAL_ENERGY)
                .expect_err("a non-finite equilibrium constant must be rejected: {yaml}");
            assert!(err.to_string().contains("finite"), "{err}");
        }
    }

    /// `find_molecules` returns `Some(&[])` for a declared kind with no groups, so the
    /// `is_some()` guards in `validate_reaction_groups` never fired: forgetting a `blocks:`
    /// entry gave a full-length run at 0 % acceptance instead of a startup error.
    #[test]
    fn reaction_needs_allocated_groups() {
        let yaml = r#"atoms:
  - {name: A, mass: 1.0, sigma: 1.0}
molecules:
  - name: M
    atoms: [A]
  - name: Unallocated
    atoms: [A]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: {}
  blocks:
    - molecule: M
      N: 4
      active: 2
      insert: !RandomAtomPos {}
propagate:
  seed: !Fixed 42
  repeat: 1
  collections: []
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let mut rng = rand::thread_rng();
        let context = Backend::new(tmp.path(), None, &mut rng).unwrap();

        let mut mv = make_move("= Unallocated", 1.0);
        let err = mv
            .finalize(&context, THERMAL_ENERGY)
            .expect_err("a reaction naming a molecule with no groups must be rejected");
        assert!(err.to_string().contains("Unallocated"), "{err}");
    }

    /// A one-to-one reaction between two kinds resolves to a *molecular swap*, which moves a
    /// whole group from Full to Empty. An atomic mega-group is neither — it is Partial — so
    /// the swap could never find a group to act on: `propose_move` returned `None` on every
    /// call and the reaction silently never fired.
    #[test]
    fn molecular_swap_between_atomic_kinds_is_error() {
        let yaml = charge_swap_yaml(true);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let mut rng = rand::thread_rng();
        let context = Backend::new(tmp.path(), None, &mut rng).unwrap();

        let mut mv = make_move("na = ca", 1.0);
        let err = mv
            .finalize(&context, THERMAL_ENERGY)
            .expect_err("a molecular swap between atomic kinds must be rejected");
        assert!(err.to_string().contains("atomic"), "{err}");
    }

    // --- Repeated atom swaps within one reaction ---

    /// Four titratable sites per molecule, five molecules: 20 `HA` and no `A` to start.
    fn double_swap_yaml(repeat: usize) -> String {
        format!(
            r#"atoms:
  - {{name: HA, mass: 1.0, charge: 0.0, sigma: 3.0}}
  - {{name: A, mass: 1.0, charge: -1.0, sigma: 3.0}}
molecules:
  - name: site4
    atoms: [HA, HA, HA, HA]
system:
  cell: !Cuboid [40.0, 40.0, 40.0]
  medium:
    permittivity: !Fixed 78.0
    temperature: 298.15
  energy:
    nonbonded:
      default:
        - !Coulomb {{cutoff: 18.0}}
  blocks:
    - molecule: site4
      N: 5
      active: 5
      insert: !RandomAtomPos {{}}
propagate:
  seed: !Fixed 42
  criterion: MetropolisHastings
  repeat: {repeat}
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          reactions:
            - ["⚛HA + ⚛HA = ⚛A + ⚛A", !K 1.0]
"#
        )
    }

    /// A reaction swapping two atoms must pick two *distinct* atoms, and the second swap
    /// must see the counts left by the first. Rescanning the unchanged state for both gives
    /// `N/(N_to+1)` twice instead of `N(N-1)/((N_to+1)(N_to+2))`, and lets one atom be
    /// converted twice — releasing one proton where the bias assumed two.
    #[test]
    fn paired_atom_swap_selects_distinct_atoms_and_correct_bias() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), double_swap_yaml(10).as_bytes()).unwrap();
        let mut rng = rand::thread_rng();
        let context = Backend::new(tmp.path(), None, &mut rng).unwrap();

        let mut mv = make_move("⚛HA + ⚛HA = ⚛A + ⚛A", 1.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        // 20 HA, 0 A: ln K + [ln 20 − ln 1] + [ln 19 − ln 2]
        let expected = 20.0_f64.ln() - 1.0_f64.ln() + 19.0_f64.ln() - 2.0_f64.ln();

        let mut proposals = 0;
        for attempt in 0..200 {
            let Some((actions, changes, ln_bias)) =
                mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng)
            else {
                continue;
            };
            proposals += 1;

            let swapped: Vec<usize> = actions
                .iter()
                .filter_map(|a| match a {
                    SpeciationAction::SwapAtomKind { abs_index, .. } => Some(*abs_index),
                    _ => None,
                })
                .collect();
            assert_eq!(swapped.len(), 2, "expected two swaps, got {swapped:?}");
            assert_ne!(
                swapped[0], swapped[1],
                "attempt {attempt}: both swaps hit the same atom {swapped:?}"
            );

            let groups: Vec<usize> = changes.iter().map(|(gi, _)| *gi).collect();
            let unique: std::collections::BTreeSet<usize> = groups.iter().copied().collect();
            assert_eq!(
                groups.len(),
                unique.len(),
                "attempt {attempt}: duplicate group index in changes {groups:?}"
            );

            assert_approx_eq!(f64, ln_bias, expected, epsilon = 1e-9);
        }
        assert!(proposals > 0, "no proposal was ever built");
    }

    /// Two swaps landing in one group emit two `UpdateIdentity` entries for it unless they
    /// are coalesced, which the energy path then pairs with itself.
    #[test]
    fn repeated_atom_swap_energy_drift() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), double_swap_yaml(2_000).as_bytes()).unwrap();
        let path = tmp.path();

        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();

        let initial_energy = mc.system_energy();
        for step in mc.iter() {
            step.unwrap();
        }
        let drift = mc.energy_drift(initial_energy);
        assert!(
            drift < 1e-6,
            "energy drift {drift:.6e} for paired atom swap"
        );
    }

    // --- A titratable atom shared by several molecule kinds ---

    /// `HA` sites live in two different molecule kinds. Ideal, so the equilibrium is exactly
    /// Henderson–Hasselbalch and every site titrates independently.
    fn shared_site_yaml(ph: f64, repeat: usize) -> String {
        let activity = 10.0_f64.powf(-ph);
        format!(
            r#"atoms:
  - {{name: HA, mass: 1.0, charge: 0.0, sigma: 3.0}}
  - {{name: A, mass: 1.0, charge: -1.0, sigma: 3.0}}
  - {{name: X, mass: 1.0, charge: 0.0, sigma: 3.0}}
  - {{name: H+, mass: 1.0, activity: {activity:.6e}}}
molecules:
  - name: siteA
    atoms: [HA]
  - name: siteB
    atoms: [X, HA]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  energy: {{}}
  blocks:
    - molecule: siteA
      N: 20
      active: 20
      insert: !RandomAtomPos {{}}
    - molecule: siteB
      N: 20
      active: 20
      insert: !RandomAtomPos {{}}
propagate:
  seed: !Fixed 42
  criterion: MetropolisHastings
  repeat: {repeat}
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          repeat: 4
          reactions:
            - ["⚛HA = ⚛A + ~H+", !pK 4.0]
"#
        )
    }

    /// Count deprotonated (`A`) and protonated (`HA`) atoms within each molecule kind.
    fn count_protonation(context: &Backend, molecule_id: MoleculeId) -> (usize, usize) {
        let (mut n_a, mut n_ha) = (0, 0);
        let a = AtomKindId::new(1);
        let ha = AtomKindId::new(0);
        let selection = crate::montecarlo::GroupSelection::ByMoleculeId(molecule_id);
        for gi in context.select(&selection) {
            for i in context.groups()[gi].iter_active() {
                match context.atom_kind(i) {
                    kind if kind == a => n_a += 1,
                    kind if kind == ha => n_ha += 1,
                    _ => {}
                }
            }
        }
        (n_a, n_ha)
    }

    /// When one molecule kind hosts the protonated state and a *different* kind hosts the
    /// deprotonated one, with none hosting both, the two are indistinguishable from a
    /// titratable site sitting beside an unrelated species that reuses the atom kind — a free
    /// `A⁻` ion pool next to a titratable `HA`. Titrating both would protonate free ions into
    /// sites; titrating one would silently freeze the other. Reject it and say so.
    #[test]
    fn atom_swap_split_across_unrelated_kinds_is_error() {
        let yaml = r#"atoms:
  - {name: HA, mass: 1.0, charge: 0.0, sigma: 3.0}
  - {name: A, mass: 1.0, charge: -1.0, sigma: 3.0}
  - {name: H+, mass: 1.0, activity: 1.0e-4}
molecules:
  - name: site
    atoms: [HA]
  - name: freeion
    atoms: [A]
    atomic: true
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: {}
  blocks:
    - {molecule: site, N: 10, active: 10, insert: !RandomAtomPos {}}
    - {molecule: freeion, N: 10, active: 5, insert: !RandomAtomPos {}}
propagate:
  seed: !Fixed 42
  repeat: 1
  collections: []
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let mut rng = rand::thread_rng();
        let context = Backend::new(tmp.path(), None, &mut rng).unwrap();

        let mut mv = make_move("⚛HA = ⚛A + ~H+", 1.0);
        let err = mv
            .finalize(&context, THERMAL_ENERGY)
            .expect_err("a swap split across unrelated kinds must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("Ambiguous"), "{msg}");
        assert!(msg.contains("site") && msg.contains("freeion"), "{msg}");
    }

    /// A titratable atom kind may occur in several molecule kinds. Binding the swap to the
    /// first matching kind freezes every other kind's sites in their initial protonation
    /// state for the whole run — a silently wrong titration curve — and computes the
    /// `N_from/(N_to+1)` factor over the wrong population.
    #[test]
    fn atom_swap_titrates_every_molecule_kind() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        // pH = pKa, so each site is 50 % deprotonated at equilibrium.
        let repeat = 20_000;
        let equilibrate = 4_000;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), shared_site_yaml(4.0, repeat).as_bytes()).unwrap();
        let path = tmp.path();

        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();

        let mut sums = [0.0_f64; 2];
        let mut samples = 0usize;
        for step in 0..repeat {
            mc.propagate
                .propagate(
                    &mut mc.context,
                    mc.thermal_energy,
                    &mut mc.step,
                    &mut mc.analyses,
                )
                .unwrap();
            if step >= equilibrate {
                for (id, sum) in sums.iter_mut().enumerate() {
                    let (n_a, n_ha) = count_protonation(&mc.context, MoleculeId::new(id));
                    *sum += n_a as f64 / (n_a + n_ha) as f64;
                }
                samples += 1;
            }
        }

        for (id, sum) in sums.iter().enumerate() {
            let fraction = sum / samples as f64;
            assert!(
                (fraction - 0.5).abs() < 0.05,
                "molecule kind {id}: deprotonated fraction {fraction:.3}, expected 0.5 at pH = pKa"
            );
        }
    }

    // --- Grand-canonical insertion of a polyatomic molecule ---

    /// A bonded dimer with a 3 Å equilibrium bond, given inline so the fixture is
    /// self-contained. `slots` groups, `active` of them active.
    fn dimer_yaml(k: f64, slots: usize, active: usize, repeat: usize) -> String {
        format!(
            r#"atoms:
  - {{name: D1, mass: 1.0, sigma: 2.0, epsilon: 0.5}}
  - {{name: D2, mass: 1.0, sigma: 2.0, epsilon: 0.5}}
molecules:
  - name: dimer
    from_structure: [{{D1: [0.0, 0.0, 0.0]}}, {{D2: [3.0, 0.0, 0.0]}}]
    bonds:
      - {{index: [0, 1], kind: !Harmonic {{k: 100.0, req: 3.0}}}}
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  energy:
    nonbonded:
      default:
        - !WeeksChandlerAndersen {{mixing: LB}}
  blocks:
    - molecule: dimer
      N: {slots}
      active: {active}
      insert: !RandomCOM {{}}
propagate:
  seed: !Fixed 42
  criterion: MetropolisHastings
  repeat: {repeat}
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          reactions:
            - ["= dimer", !K {k:.10}]
"#
        )
    }

    /// How far a group's coordinates depart from the ones its stored orientation claims (RMSD, Å).
    ///
    /// The stored quaternion claims to carry the molecule's reference conformation onto its
    /// coordinates; this asks whether it does. Deliberately *not* an angle against a fitted
    /// rotation: a linear or symmetric molecule has many orientations its coordinates cannot tell
    /// apart, and comparing quaternions would report those agreements as disagreements.
    fn orientation_error(context: &Backend, group_index: usize) -> Option<f64> {
        use crate::context::WithTopology;
        let group = &context.groups()[group_index];
        let topology = context.topology();
        let reference = topology
            .moleculekind(group.molecule())
            .reference_positions();
        let current: Vec<crate::Point> = group.iter_active().map(|i| context.position(i)).collect();
        let gathered = crate::geometry::gather_molecule(&current, context.cell());
        crate::geometry::orientation_residual(reference, &gathered, group.quaternion())
    }

    fn dimer_context(slots: usize, active: usize) -> (tempfile::NamedTempFile, Backend) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), dimer_yaml(1.0, slots, active, 10).as_bytes()).unwrap();
        let mut rng = rand::thread_rng();
        let context = Backend::new(tmp.path(), None, &mut rng).unwrap();
        (tmp, context)
    }

    /// An inserted polyatomic must arrive with its template geometry intact. Placing every
    /// atom at one point (`vec![pos; capacity]`) gives a bond length of zero: with a
    /// repulsive potential the trial energy is +∞ and the insertion can never be accepted;
    /// with a soft one the molecule is accepted in an impossible collapsed geometry.
    #[test]
    fn inserted_polyatomic_keeps_template_geometry() {
        let (_tmp, context) = dimer_context(10, 2);
        let mut mv = make_move("= dimer", 1.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let mut rng = rand::thread_rng();
        let mut insertions = 0;
        for _ in 0..200 {
            let Some((actions, _, _)) =
                mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng)
            else {
                continue;
            };
            for action in &actions {
                let SpeciationAction::ActivateGroup { positions, .. } = action else {
                    continue;
                };
                insertions += 1;
                assert_eq!(positions.len(), 2, "dimer must get two positions");
                // Minimum image: the template may straddle a periodic boundary.
                let bond = context.cell().distance(&positions[0], &positions[1]).norm();
                assert_approx_eq!(f64, bond, 3.0, epsilon = 1e-9);
            }
        }
        assert!(insertions > 0, "no insertion was ever proposed");
    }

    /// A molecule declared with a plain `atoms:` list has no reference geometry — only
    /// `from_structure` populates it. Building the insertion from `reference_positions()`
    /// alone therefore produced an *empty* coordinate list, and the transform activated the
    /// group without writing any positions: the molecule reappeared at stale coordinates
    /// while the bias assumed a uniform random insertion, breaking detailed balance.
    #[test]
    fn insertion_places_a_molecule_that_has_no_reference_geometry() {
        let yaml = r#"atoms:
  - {name: Na, mass: 23.0, charge: 1.0, sigma: 3.0}
  - {name: Cl, mass: 35.0, charge: -1.0, sigma: 4.0}
molecules:
  - name: NaCl
    atoms: [Na, Cl]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: {}
  blocks:
    - {molecule: NaCl, N: 10, active: 2, insert: !RandomAtomPos {}}
propagate:
  seed: !Fixed 42
  repeat: 1
  collections: []
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let mut rng = rand::thread_rng();
        let context = Backend::new(tmp.path(), None, &mut rng).unwrap();

        let mut mv = make_move("= NaCl", 20.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        for _ in 0..50 {
            let Some((actions, _, _)) =
                mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng)
            else {
                continue;
            };
            let Some(SpeciationAction::ActivateGroup { positions, .. }) = actions
                .iter()
                .find(|a| matches!(a, SpeciationAction::ActivateGroup { .. }))
            else {
                continue;
            };
            assert_eq!(
                positions.len(),
                2,
                "an inserted molecule must get one position per atom, even with no \
                 `from_structure` reference geometry"
            );
            return;
        }
        panic!("no insertion proposed in 50 attempts");
    }

    /// The insertion applies a random orientation, so the group's stored quaternion must be
    /// updated to match its new coordinates. A stale one corrupts orientation-dependent
    /// energies (6D tables) and any analysis reading `Group::quaternion`.
    #[test]
    fn polyatomic_insertion_sets_group_orientation() {
        let (_tmp, mut context) = dimer_context(10, 2);
        let mut mv = make_move("= dimer", 1.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let Some((actions, _, _)) =
                mv.try_build_actions(&mv.resolved[0], Direction::Forward, &context, &mut rng)
            else {
                continue;
            };
            let Some(SpeciationAction::ActivateGroup { group_index, .. }) = actions
                .iter()
                .find(|a| matches!(a, SpeciationAction::ActivateGroup { .. }))
                .cloned()
            else {
                continue;
            };

            crate::transform::Transform::Speciation(actions)
                .on_system(&mut context)
                .unwrap();

            // An insertion places the template at a random orientation; the group must report
            // the one its coordinates actually have.
            let error = orientation_error(&context, group_index)
                .expect("a polyatomic group has a reference conformation");
            assert!(
                error < 1e-9,
                "stored orientation misses the coordinates it was placed with by {error:.2e} Å"
            );
            return;
        }
        panic!("no insertion proposed in 200 attempts");
    }

    /// End to end: the dimer is inserted at its equilibrium bond length, so its
    /// intramolecular energy is zero and the ideal-gas relation ⟨N⟩ = K·c₀·V survives even
    /// with a repulsive nonbonded term at this low density. Collapsed insertions would
    /// instead cost ~450 kJ/mol of bond energy (plus a WCA overlap) and never be accepted,
    /// driving ⟨N⟩ to zero.
    #[test]
    #[ignore = "statistical; run explicitly (cargo test --release -- --ignored)"]
    fn dimer_gcmc_energy_drift_and_count() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let volume = 30.0_f64.powi(3);
        let lambda = 8.0;
        let k = lambda / (crate::MOLAR_TO_INV_ANGSTROM3 * volume);

        let repeat = 40_000;
        let equilibrate = 8_000;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), dimer_yaml(k, 60, 8, repeat).as_bytes()).unwrap();
        let path = tmp.path();

        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();
        let initial_energy = mc.system_energy();

        let (mut sum, mut samples) = (0.0_f64, 0usize);
        for step in 0..repeat {
            mc.propagate
                .propagate(
                    &mut mc.context,
                    mc.thermal_energy,
                    &mut mc.step,
                    &mut mc.analyses,
                )
                .unwrap();
            if step >= equilibrate {
                sum += mc.context.count_active_molecules(MoleculeId::new(0)) as f64;
                samples += 1;
            }
        }

        let drift = mc.energy_drift(initial_energy);
        assert!(drift < 1e-6, "energy drift {drift:.6e} for dimer GCMC");

        let mean = sum / samples as f64;
        assert!(
            (mean - lambda).abs() < 0.1 * lambda,
            "⟨N⟩ = {mean:.3}, expected ≈ K·c₀·V = {lambda:.3}"
        );
    }

    // --- Polyprotic titration by molecular swap: phytate (IP6), 13 states ---

    /// Stepwise dissociation constants of phytic acid (IP6), 12 protons.
    /// De Stefano et al., <https://doi.org/10.1021/je020124m>.
    // The 8th pKa, 6.28, is measured data — not an approximation of τ, as clippy suspects.
    #[allow(clippy::approx_constant)]
    const PHYTATE_PKA: [f64; 12] = [
        1.1, 1.5, 1.7, 2.1, 2.5, 2.9, 5.72, 6.28, 6.81, 7.60, 9.94, 11.84,
    ];

    /// Mole fraction of the state retaining `m` protons, for a polyprotic acid with stepwise
    /// `pk`. Exact solution: de Levie, *General Expressions for Acid–Base Titrations of
    /// Arbitrary Mixtures*, Anal. Chem. 68, 585 (1996),
    /// <https://doi.org/10.1021/ac950430l>.
    ///
    /// χ_m = [H⁺]^m · Π_{j≤n−m} K_j ⁄ ( [H⁺]ⁿ + Σ_{i≤n} [H⁺]^{n−i} · Π_{j≤i} K_j )
    fn polyprotic_fraction(m: usize, ph: f64, pk: &[f64]) -> f64 {
        let n = pk.len();
        let h = 10.0_f64.powf(-ph);
        let k = |i: usize| 10.0_f64.powf(-pk[i]);
        let prod = |upto: usize| (0..upto).map(k).product::<f64>();

        let numerator = h.powi(m as i32) * prod(n - m);
        let denominator = h.powi(n as i32)
            + (0..n)
                .map(|i| h.powi((n - i - 1) as i32) * prod(i + 1))
                .sum::<f64>();
        numerator / denominator
    }

    /// Mean charge ⟨Z⟩ = −Σ_m (n − m)·χ_m.
    fn polyprotic_mean_charge(ph: f64, pk: &[f64]) -> f64 {
        let n = pk.len();
        -(0..=n)
            .map(|m| (n - m) as f64 * polyprotic_fraction(m, ph, pk))
            .sum::<f64>()
    }

    /// No interactions: the equilibrium is then exactly the analytic one.
    const IDEAL_ENERGY: &str = "  energy: {}\n";

    /// Screened electrostatics plus a soft core. The phosphates carry −1 and −2, so a
    /// protonation swap changes the charge distribution and ΔU is genuinely non-zero.
    const INTERACTING_ENERGY: &str = "  energy:\n    nonbonded:\n      default:\n        \
         - !Coulomb {cutoff: 30.0}\n        - !WeeksChandlerAndersen {mixing: LB}\n";

    /// Thirteen protonation states of phytate, each a 7-atom molecule (an inositol centre
    /// bead plus six phosphates) exchanged as a whole molecule kind.
    fn phytate_yaml(ph: f64, n_molecules: usize, repeat: usize, energy: &str) -> String {
        // Sites carrying the k-th extra negative charge, maximising the minimum separation.
        const MAXSEP: [&[usize]; 7] = [
            &[],
            &[0],
            &[0, 3],
            &[0, 2, 4],
            &[0, 1, 3, 5],
            &[0, 1, 2, 4, 5],
            &[0, 1, 2, 3, 4, 5],
        ];
        const SITES: [[f64; 3]; 6] = [
            [4.5, 0.0, 1.5],
            [2.25, 3.897, -0.5],
            [-2.25, 3.897, -0.5],
            [-4.5, 0.0, -0.5],
            [-2.25, -3.897, -0.5],
            [2.25, -3.897, -0.5],
        ];
        let n_states = PHYTATE_PKA.len() + 1;
        // Start from the state closest to neutral at this pH so equilibration is short.
        let start = (0..n_states)
            .max_by(|a, b| {
                polyprotic_fraction(n_states - 1 - a, ph, &PHYTATE_PKA)
                    .partial_cmp(&polyprotic_fraction(n_states - 1 - b, ph, &PHYTATE_PKA))
                    .unwrap()
            })
            .unwrap();

        let site_kind = |state: usize, j: usize| -> &'static str {
            if state <= 6 {
                if MAXSEP[state].contains(&j) {
                    "PH1"
                } else {
                    "PH0"
                }
            } else if MAXSEP[state - 6].contains(&j) {
                "PH2"
            } else {
                "PH1"
            }
        };

        let mut yaml = format!(
            "atoms:\n  - {{name: INO, mass: 180.0, charge: 0.0, sigma: 6.2, epsilon: 0.8368}}\n\
             \x20 - {{name: PH0, mass: 95.0, charge: 0.0, sigma: 5.8, epsilon: 0.8368}}\n\
             \x20 - {{name: PH1, mass: 95.0, charge: -1.0, sigma: 5.8, epsilon: 0.8368}}\n\
             \x20 - {{name: PH2, mass: 95.0, charge: -2.0, sigma: 5.8, epsilon: 0.8368}}\n\
             \x20 - {{name: H+, mass: 1.0, sigma: 1.0, epsilon: 0.8368, activity: {:.6e}}}\nmolecules:\n",
            10.0_f64.powf(-ph)
        );
        let exclusions = (0..7)
            .flat_map(|i| (i + 1..7).map(move |j| format!("[{i},{j}]")))
            .collect::<Vec<_>>()
            .join(", ");
        for state in 0..n_states {
            let sites = (0..6)
                .map(|j| {
                    let [x, y, z] = SITES[j];
                    format!("{{{}: [{x}, {y}, {z}]}}", site_kind(state, j))
                })
                .collect::<Vec<_>>()
                .join(", ");
            // All intramolecular pairs excluded: the electrostatic repulsion between the
            // phosphates is already carried empirically by the pKa ladder, so counting it
            // again would skew the equilibrium. This is also what the molecular swap's
            // `ResizeExcludeIntra` assumes — see `molecular_swap_energy_drift_with_interactions`.
            yaml += &format!(
                "  - name: \"PA{state}\"\n    from_structure: [{{INO: [0.0, 0.0, 0.0]}}, {sites}]\n    \
                 exclusions: [{exclusions}]\n"
            );
        }
        yaml += &format!(
            "system:\n  cell: !Cuboid [100.0, 100.0, 100.0]\n  \
             medium: {{permittivity: !Water, temperature: 298.15}}\n{energy}  blocks:\n"
        );
        for state in 0..n_states {
            let active = if state == start { n_molecules } else { 0 };
            yaml += &format!(
                "    - {{molecule: \"PA{state}\", N: {n_molecules}, active: {active}, \
                 insert: !RandomCOM {{}}}}\n"
            );
        }
        yaml += &format!(
            "propagate:\n  seed: !Fixed 42\n  criterion: MetropolisHastings\n  repeat: {repeat}\n  \
             collections:\n    - !Deterministic\n      moves:\n        - !SpeciationMove\n          \
             repeat: 10\n          reactions:\n"
        );
        for (i, pk) in PHYTATE_PKA.iter().enumerate() {
            yaml += &format!("            - [\"PA{i} = PA{} + ~H+\", !pK {pk}]\n", i + 1);
        }
        yaml
    }

    /// Phytate (IP6) titrates through 13 protonation states, swapped as whole 7-atom
    /// molecules. Ideal, so the sampled mean charge must reproduce the exact polyprotic
    /// solution across the whole titration curve — a 13-state, polyatomic exercise of the
    /// molecular-swap path that no other test covers.
    #[test]
    #[ignore = "statistical; run explicitly (cargo test --release -- --ignored)"]
    fn phytate_mean_charge_matches_exact_polyprotic_solution() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let n_states = PHYTATE_PKA.len() + 1;
        let (n_molecules, repeat, equilibrate) = (20, 20_000, 4_000);

        for ph in [2.0, 5.0, 7.0, 9.0, 11.0] {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(
                tmp.path(),
                phytate_yaml(ph, n_molecules, repeat, IDEAL_ENERGY).as_bytes(),
            )
            .unwrap();
            let path = tmp.path();

            let mut rng = rand::thread_rng();
            let context = Backend::new(path, None, &mut rng).unwrap();
            let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
            let mut mc = MarkovChain::new(
                context,
                propagate,
                THERMAL_ENERGY,
                AnalysisCollection::default(),
            )
            .unwrap();

            let (mut charge_sum, mut samples) = (0.0_f64, 0usize);
            for step in 0..repeat {
                mc.propagate
                    .propagate(
                        &mut mc.context,
                        mc.thermal_energy,
                        &mut mc.step,
                        &mut mc.analyses,
                    )
                    .unwrap();
                if step < equilibrate {
                    continue;
                }
                let mut total = 0usize;
                let mut charge = 0.0;
                for state in 0..n_states {
                    let n = mc
                        .context
                        .count_molecules(MoleculeId::new(state), GroupSize::Full);
                    total += n;
                    charge -= (state * n) as f64;
                }
                assert_eq!(total, n_molecules, "molecules not conserved at pH {ph}");
                charge_sum += charge / total as f64;
                samples += 1;
            }

            let observed = charge_sum / samples as f64;
            let expected = polyprotic_mean_charge(ph, &PHYTATE_PKA);
            assert!(
                (observed - expected).abs() < 0.15,
                "pH {ph}: ⟨Z⟩ = {observed:.3} e, exact solution gives {expected:.3} e"
            );
        }
    }

    /// The analytic titration tests above are all ideal, so ΔU ≡ 0 and no energy bug can
    /// show up in them. The molecular-swap path in particular computes ΔU with
    /// `GroupChange::ResizeExcludeIntra` — intramolecular energy is deliberately left out
    /// and absorbed into K — which only an interacting system exercises.
    ///
    /// Here the phosphates carry −1 and −2, so every protonation swap changes the charge
    /// distribution and ΔU is real. The incremental energy must then agree with a full
    /// recompute; a drift means the swap's ΔU is wrong even though the ideal test passes.
    #[test]
    fn molecular_swap_energy_drift_with_interactions() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            phytate_yaml(7.0, 12, 2_000, INTERACTING_ENERGY).as_bytes(),
        )
        .unwrap();
        let path = tmp.path();

        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();

        let initial_energy = mc.system_energy();
        for step in mc.iter() {
            step.unwrap();
        }

        // The reaction must actually fire, or the drift check is vacuous.
        let start = mc
            .context()
            .count_molecules(MoleculeId::new(9), GroupSize::Full);
        assert_ne!(
            start, 12,
            "no swap was ever accepted; drift check is vacuous"
        );

        let drift = mc.energy_drift(initial_energy);
        assert!(
            drift < 1e-6,
            "energy drift {drift:.6e} for an interacting molecular swap"
        );
    }

    /// A one-atom molecular group has no orientation, and deriving one must not invent it.
    ///
    /// Deriving the orientation from coordinates has to stay a no-op where there are too few
    /// coordinates to define a frame — a single point is the same point however you turn it.
    /// The group keeps the identity it was built with, and nothing panics on the way.
    #[test]
    fn one_atom_molecular_group_keeps_an_identity_orientation() {
        let yaml = r#"
atoms:
  - {name: A, mass: 1.0, sigma: 2.0}
molecules:
  - name: mono
    from_structure: [{A: [0.0, 0.0, 0.0]}]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: {}
  blocks:
    - {molecule: mono, N: 8, active: 2, insert: !RandomCOM {}}
propagate: {seed: !Fixed 3, criterion: MetropolisHastings, repeat: 0, collections: []}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let mut rng = rand::thread_rng();
        let mut context = Backend::new(tmp.path(), None, &mut rng).unwrap();

        // Activate an empty slot: the path that derives an orientation from written coordinates.
        let gi = 5;
        assert!(context.groups()[gi].is_empty());
        crate::transform::Transform::Speciation(vec![SpeciationAction::ActivateGroup {
            group_index: gi,
            positions: vec![crate::Point::new(3.0, -4.0, 5.0)],
        }])
        .on_system(&mut context)
        .unwrap();

        assert_eq!(context.groups()[gi].len(), 1);
        assert_eq!(
            *context.groups()[gi].quaternion(),
            crate::UnitQuaternion::identity(),
            "a single atom has no orientation to derive"
        );
        // A single point sits at zero residual whichever way you turn it — which is the point.
        assert_eq!(orientation_error(&context, gi), Some(0.0));
        // The mass center still follows the coordinates, single atom or not.
        assert_eq!(
            context.groups()[gi].mass_center().copied(),
            Some(crate::Point::new(3.0, -4.0, 5.0))
        );
    }

    /// Swapping molecules of *different* shape still leaves each describing its own conformation.
    ///
    /// The overlay writes the incoming kind's template, so the swapped-in molecule is a rigid
    /// image of *its* reference, not of the one it replaced. Deriving the orientation from the
    /// coordinates is what makes that work: an alignment rotation threaded through from the
    /// overlay would relate two different frames and be wrong the moment the shapes differ.
    #[test]
    fn swap_between_differently_shaped_molecules_keeps_each_orientation_honest() {
        let yaml = r#"
atoms:
  - {name: T1, mass: 1.0, sigma: 2.0}
  - {name: T2, mass: 2.0, sigma: 2.0}
  - {name: T3, mass: 3.0, sigma: 2.0}
molecules:
  # Straight and bent: deliberately different conformations, and different masses per site,
  # so an orientation fitted against the wrong reference cannot pass by coincidence.
  - name: straight
    from_structure: [{T1: [-4.0, 0.0, 0.0]}, {T2: [0.0, 0.0, 0.0]}, {T3: [4.0, 0.0, 0.0]}]
  - name: bent
    from_structure: [{T1: [-3.0, 0.0, 0.0]}, {T2: [0.0, 2.5, 0.0]}, {T3: [3.0, 0.0, 0.0]}]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: {}
  blocks:
    - {molecule: straight, N: 6, active: 6, insert: !RandomCOM {rotate: true}}
    - {molecule: bent, N: 6, active: 0, insert: !RandomCOM {rotate: true}}
propagate:
  seed: !Fixed 7
  criterion: MetropolisHastings
  repeat: 300
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          repeat: 1
          reactions:
            - ["straight = bent", !K 1.0]
"#;
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let path = tmp.path();

        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();
        for step in mc.iter() {
            step.unwrap();
        }

        let context = mc.context();
        let bent_active = context.count_molecules(MoleculeId::new(1), GroupSize::Full);
        assert!(
            bent_active > 0,
            "no swap was accepted; the check is vacuous"
        );

        for gi in 0..context.groups().len() {
            if context.groups()[gi].is_empty() {
                continue;
            }
            let error =
                orientation_error(context, gi).expect("a trimer has a reference conformation");
            assert!(
                error < 1e-6,
                "group {gi} stores an orientation that misses its coordinates by {error:.2e} Å"
            );
        }
    }

    /// A swapped-in molecule's stored orientation must describe the coordinates it was given.
    ///
    /// The swap overlays the incoming template onto the outgoing molecule's pose, which is a
    /// reorientation; a group whose quaternion still says "unrotated" is lying to every consumer
    /// that reads it — 6D tabulated energies, body-frame analyses, the GPU rigid integrator and
    /// the checkpoint. Nothing in this fixture reads it, which is exactly why it went unnoticed.
    #[test]
    fn molecular_swap_sets_group_orientation() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            phytate_yaml(7.0, 12, 500, IDEAL_ENERGY).as_bytes(),
        )
        .unwrap();
        let path = tmp.path();

        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();
        for step in mc.iter() {
            step.unwrap();
        }

        use crate::context::WithTopology;
        let context = mc.context();
        let topology = context.topology();
        let mut checked = 0usize;
        let mut wrong: Vec<String> = Vec::new();

        for gi in 0..context.groups().len() {
            let group = &context.groups()[gi];
            let kind = topology.moleculekind(group.molecule());
            if group.is_empty() || kind.reference_positions().len() < 2 {
                continue;
            }
            checked += 1;
            let error = orientation_error(context, gi)
                .expect("a 7-bead phytate has a reference conformation");
            if error > 1e-6 {
                wrong.push(format!(
                    "group {gi}: stored orientation is off by {error:.1}°"
                ));
            }
        }

        assert!(checked > 0, "no molecular groups were checked");
        assert!(
            wrong.is_empty(),
            "{} of {checked} groups store an orientation that does not match their coordinates:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }

    /// The drift above is worth catching before the run rather than after, so a swap between
    /// kinds with differing atoms and un-excluded intramolecular pairs warns at startup.
    /// A swap between kinds with identical atoms cannot differ in intramolecular energy and
    /// must stay quiet — `molswap_phosphate` is exactly that case.
    #[test]
    fn intramolecular_exclusion_warning_only_when_it_can_drift() {
        fn warns(yaml: &str, reaction: &str) -> bool {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
            let mut rng = rand::thread_rng();
            let context = Backend::new(tmp.path(), None, &mut rng).unwrap();
            let mut mv = make_move(reaction, 1.0);
            mv.finalize(&context, THERMAL_ENERGY).unwrap();
            let topology = crate::context::WithTopology::topology_ref(&context);

            // The warning fires exactly when the resolved swap has differing atom kinds and
            // fewer than all-pairs exclusions; assert on that condition directly.
            mv.resolved[0].forward_ops.iter().any(|op| {
                let ReactionOp::SwapMolecule {
                    from_mol_id,
                    to_mol_id,
                } = op
                else {
                    return false;
                };
                let (from, to) = (
                    topology.moleculekind(*from_mol_id),
                    topology.moleculekind(*to_mol_id),
                );
                if from.atom_indices() == to.atom_indices() {
                    return false;
                }
                let n = from.atom_indices().len();
                let all_pairs = n * n.saturating_sub(1) / 2;
                from.exclusions().len() < all_pairs || to.exclusions().len() < all_pairs
            })
        }

        let charged_no_exclusions = phytate_yaml(7.0, 4, 1, INTERACTING_ENERGY)
            .lines()
            .filter(|line| !line.trim_start().starts_with("exclusions:"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert!(
            warns(&charged_no_exclusions, "PA9 = PA10 + ~H+"),
            "differing atoms with no exclusions must warn: this configuration drifts"
        );

        let charged_with_exclusions = phytate_yaml(7.0, 4, 1, INTERACTING_ENERGY);
        assert!(
            !warns(&charged_with_exclusions, "PA9 = PA10 + ~H+"),
            "all intramolecular pairs excluded: nothing can drift, so do not warn"
        );
    }

    // --- Grand-canonical ideal gas: ⟨N⟩ = K·c₀·V ---

    /// Ideal gas in a 10 Å cube with a single insertion/deletion reaction `= particle`.
    fn ideal_gas_yaml(k: f64, slots: usize, active: usize, repeat: usize) -> String {
        format!(
            r#"atoms:
  - {{name: X, mass: 1.0, sigma: 1.0}}
molecules:
  - name: particle
    atoms: [X]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  energy: {{}}
  blocks:
    - molecule: particle
      N: {slots}
      active: {active}
      insert: !RandomAtomPos {{}}
propagate:
  seed: !Fixed 42
  criterion: MetropolisHastings
  repeat: {repeat}
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          temperature: 298.15
          repeat: 4
          reactions:
            - ["= particle", !K {k:.10}]
"#
        )
    }

    /// The grand-canonical ideal gas is Poisson-distributed with λ = K·c₀·V, where
    /// c₀ = N_A·10⁻²⁷ Å⁻³ is the 1 M standard state. This pins the standard-state
    /// convention of the acceptance criterion: without the c₀ factor the same K would
    /// target ⟨N⟩ = K·V, larger by ~1/c₀ ≈ 1660.
    ///
    /// The variance is the sharper assertion: any rescaling of the bias exponent (e.g. a
    /// move temperature differing from the system temperature) leaves the mode roughly in
    /// place but destroys the Poisson relation Var(N) = ⟨N⟩.
    #[test]
    #[ignore = "statistical; run explicitly (cargo test --release -- --ignored)"]
    fn ideal_gas_gcmc_matches_k_c0_v() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let volume = 10.0_f64.powi(3);
        let lambda = 20.0;
        let k = lambda / (crate::MOLAR_TO_INV_ANGSTROM3 * volume);

        let repeat = 60_000;
        let equilibrate = 10_000;
        let yaml = ideal_gas_yaml(k, 200, 20, repeat);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml.as_bytes()).unwrap();
        let path = tmp.path();

        let mut rng = rand::thread_rng();
        let context = Backend::new(path, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(path, &context, THERMAL_ENERGY).unwrap();
        let mut mc = MarkovChain::new(
            context,
            propagate,
            THERMAL_ENERGY,
            AnalysisCollection::default(),
        )
        .unwrap();

        let (mut sum, mut sum_sq, mut samples) = (0.0_f64, 0.0_f64, 0usize);
        for _ in 0..repeat {
            let running = mc.propagate.propagate(
                &mut mc.context,
                mc.thermal_energy,
                &mut mc.step,
                &mut mc.analyses,
            );
            assert!(running.unwrap(), "simulation ended early");
            samples += 1;
            if samples > equilibrate {
                let n = mc
                    .context
                    .count_molecules(MoleculeId::new(0), GroupSize::Full)
                    as f64;
                sum += n;
                sum_sq += n * n;
            }
        }

        let n = (samples - equilibrate) as f64;
        let mean = sum / n;
        let variance = sum_sq / n - mean * mean;

        assert!(
            (mean - lambda).abs() < 0.5,
            "⟨N⟩ = {mean:.3}, expected K·c₀·V = {lambda:.3}"
        );
        assert!(
            (variance / mean - 1.0).abs() < 0.15,
            "Var(N)/⟨N⟩ = {:.3}, expected 1 (Poisson); ⟨N⟩ = {mean:.3}",
            variance / mean
        );
    }

    // --- Temperature consistency ---

    /// The acceptance criterion forms `exp(-(ΔU + bias)/thermal_energy)` using the *system*
    /// thermal energy. The reaction bias must therefore be dimensionless: whatever scale the
    /// criterion divides by, the applied bias has to come out as exactly `ln K_eff`.
    ///
    /// Evaluating at two different scales is what pins this. The move used to return
    /// `Bias::Energy(-kT_move · ln K_eff)` built from a temperature of its own, so the
    /// applied bias scaled as `kT_move/kT_system` — silently raising every equilibrium
    /// constant to the power `T_move/T_system` whenever the two disagreed.
    #[test]
    fn speciation_bias_is_dimensionless() {
        let context = make_context();
        let mut mv = make_move("= M", 10.0);
        mv.finalize(&context, THERMAL_ENERGY).unwrap();

        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let Some(proposed) = mv.propose_move(&context, &mut rng) else {
                continue;
            };
            let ln_bias = mv.trial_ln_bias.expect("propose sets the trial bias");
            let bias =
                MoveProposal::<Backend>::bias(&mv, proposed.change(), &NewOld::from(0.0, 0.0));

            for scale in [THERMAL_ENERGY, 2.0 * THERMAL_ENERGY] {
                let applied = bias.to_energy(scale).expect("not a ForceAccept") / scale;
                assert_approx_eq!(f64, applied, -ln_bias, epsilon = 1e-12);
            }
            return;
        }
        panic!("no proposal in 20 attempts");
    }
}
