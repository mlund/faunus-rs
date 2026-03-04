//! Reaction ensemble (speciation) Monte Carlo move.
//!
//! Performs molecular insertion/deletion and atom-type swaps according to
//! chemical reactions. The acceptance criterion follows Smith & Triska (1994),
//! using `entropy_bias()` for the combinatorial `V^Δν · ∏[N!/(N+ν)!]` factors.
//! For atom swaps, a `N_from/(N_to+1)` combinatorial factor ensures detailed
//! balance, consistent with ESPResSo's reaction ensemble implementation.

use crate::chemistry::reaction::{Direction, Participant, Reaction};
use crate::energy::ExternalPressure;
use crate::group::GroupSize;
use crate::montecarlo::{entropy_bias, NewOld};
use crate::propagate::{
    default_repeat, default_weight, tagged_yaml, Displacement, MoveProposal, MoveTarget,
    ProposedMove,
};
use crate::transform::{SpeciationAction, Transform};
use crate::{cell::Shape, Change, Context, GroupChange};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// A single reaction parsed from YAML.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReactionConfig {
    /// Reaction string, e.g. "= NaCl" or "⚛A = ⚛B"
    reaction: String,
    /// Equilibrium constant (not ln K)
    #[serde(alias = "K")]
    equilibrium_constant: f64,
}

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

/// Trial state stored between `propose_move` and `bias`.
#[derive(Clone, Debug)]
struct TrialState {
    ln_bias: f64,
}

/// Reaction ensemble Monte Carlo move.
///
/// Supports molecular insertion/deletion and atom-type swaps.
///
/// # YAML example
/// ```yaml
/// - !SpeciationMove
///   temperature: 298.15
///   reactions:
///     - { reaction: "= NaCl", K: 100.0 }
///     - { reaction: "⚛A = ⚛B", K: 1.0 }
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
    /// Trial state for current move (between propose and bias).
    #[serde(skip)]
    trial_state: Option<TrialState>,
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
        // Find the molecule containing both atom kinds
        let molecule_id = topology
            .moleculekinds()
            .iter()
            .position(|m| m.atom_indices().contains(&from.0) && m.atom_indices().contains(&to.0))
            .ok_or_else(|| {
                anyhow::anyhow!("No molecule contains both atom '{}' and '{}'", from.1, to.1)
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

impl SpeciationMove {
    /// Resolve reaction strings to topology IDs.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.temperature > 0.0,
            "SpeciationMove: temperature must be positive"
        );
        self.thermal_energy = ExternalPressure::thermal_energy_from_temperature(self.temperature);

        let topology = context.topology();
        self.resolved.clear();

        for config in &self.reactions {
            anyhow::ensure!(
                config.equilibrium_constant > 0.0,
                "Equilibrium constant must be positive for reaction '{}'",
                config.reaction
            );

            let reaction = Reaction::from_reaction(&config.reaction, config.equilibrium_constant)?;
            let (reactants, products) = reaction.get();

            // Build forward operations: consume reactants, produce products
            let mut forward_ops = Vec::new();
            let mut backward_ops = Vec::new();

            // Molecular participants (atoms and implicits handled separately)
            for p in reactants {
                if let Participant::Molecule(name) = p {
                    let id = find_molecule_index(name, &topology)?;
                    forward_ops.push(ReactionOp::DeactivateMolecule(id));
                    backward_ops.push(ReactionOp::ActivateMolecule(id));
                }
            }
            for p in products {
                if let Participant::Molecule(name) = p {
                    let id = find_molecule_index(name, &topology)?;
                    forward_ops.push(ReactionOp::ActivateMolecule(id));
                    backward_ops.push(ReactionOp::DeactivateMolecule(id));
                }
            }

            // Atom swap participants
            let (swap_fwd, swap_bwd) = resolve_atom_swaps(reactants, products, &topology)?;
            forward_ops.extend(swap_fwd);
            backward_ops.extend(swap_bwd);

            let effective_ln_k = config.equilibrium_constant.ln();

            self.resolved.push(ResolvedReaction {
                effective_ln_k,
                forward_ops,
                backward_ops,
            });
        }

        // Validate that required groups exist
        for resolved in &self.resolved {
            for op in resolved
                .forward_ops
                .iter()
                .chain(resolved.backward_ops.iter())
            {
                match op {
                    ReactionOp::ActivateMolecule(mol_id)
                    | ReactionOp::DeactivateMolecule(mol_id) => {
                        let name = topology.moleculekinds()[*mol_id].name();
                        let has_groups = context
                            .group_lists()
                            .find_molecules(*mol_id, GroupSize::Full)
                            .is_some()
                            || context
                                .group_lists()
                                .find_molecules(*mol_id, GroupSize::Empty)
                                .is_some();
                        anyhow::ensure!(
                            has_groups,
                            "No groups found for molecule '{name}' needed by reaction"
                        );
                    }
                    ReactionOp::SwapAtom { molecule_id, .. } => {
                        let name = topology.moleculekinds()[*molecule_id].name();
                        let has_groups = context
                            .group_lists()
                            .find_molecules(*molecule_id, GroupSize::Full)
                            .is_some();
                        anyhow::ensure!(
                            has_groups,
                            "No active groups for molecule '{name}' needed by atom swap"
                        );
                    }
                }
            }
        }

        log::info!(
            "SpeciationMove: {} reactions, kT = {:.4} kJ/mol",
            self.resolved.len(),
            self.thermal_energy,
        );

        Ok(())
    }

    /// Try to build speciation actions for the current direction of a resolved reaction.
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

        for op in ops {
            match op {
                ReactionOp::DeactivateMolecule(mol_id) => {
                    let full_groups = context
                        .group_lists()
                        .find_molecules(*mol_id, GroupSize::Full)?;
                    if full_groups.is_empty() {
                        return None;
                    }
                    let &group_index = full_groups.iter().choose(rng)?;
                    let n_old = full_groups.len();
                    let n_new = n_old - 1;
                    ln_bias -= entropy_bias(NewOld::from(n_new, n_old), vol);

                    actions.push(SpeciationAction::DeactivateGroup(group_index));
                    group_changes.push((group_index, GroupChange::Resize(GroupSize::Empty)));
                }
                ReactionOp::ActivateMolecule(mol_id) => {
                    let empty_groups = context
                        .group_lists()
                        .find_molecules(*mol_id, GroupSize::Empty)?;
                    if empty_groups.is_empty() {
                        return None;
                    }
                    let &group_index = empty_groups.iter().choose(rng)?;

                    // Count current active groups
                    let n_old = context
                        .group_lists()
                        .find_molecules(*mol_id, GroupSize::Full)
                        .map_or(0, |gs| gs.len());
                    let n_new = n_old + 1;
                    ln_bias -= entropy_bias(NewOld::from(n_new, n_old), vol);

                    // Generate random position inside the simulation cell
                    // Note: get_point_inside requires ThreadRng; ideally the cell API should be generic
                    let mut thread_rng = rand::thread_rng();
                    let random_pos = context.cell().get_point_inside(&mut thread_rng);
                    let group = &context.groups()[group_index];
                    let num_atoms = group.capacity();

                    // Place all atoms of the molecule at the random position
                    // (for multi-atom molecules, a proper conformational sampling would be needed)
                    let positions = vec![random_pos; num_atoms];

                    actions.push(SpeciationAction::ActivateGroup {
                        group_index,
                        positions,
                    });
                    group_changes.push((group_index, GroupChange::Resize(GroupSize::Full)));
                }
                ReactionOp::SwapAtom {
                    from_id,
                    to_id,
                    molecule_id,
                } => {
                    let full_groups = context
                        .group_lists()
                        .find_molecules(*molecule_id, GroupSize::Full)?;
                    if full_groups.is_empty() {
                        return None;
                    }

                    // Count atoms of each type for combinatorial factor
                    let (mut n_from, mut n_to) = (0usize, 0usize);
                    let mut from_atoms: Vec<(usize, usize)> = Vec::new();
                    for &gi in full_groups {
                        for i in context.groups()[gi].iter_active() {
                            let kind = context.get_atomkind(i);
                            if kind == *from_id {
                                n_from += 1;
                                from_atoms.push((gi, i));
                            } else if kind == *to_id {
                                n_to += 1;
                            }
                        }
                    }

                    if n_from == 0 {
                        return None;
                    }

                    // Pick a random atom of from_id type
                    let &(group_index, abs_index) = from_atoms.choose(rng)?;

                    // Detailed balance requires N_from!/(N_from-1)! / [(N_to+1)!/N_to!]
                    // = N_from / (N_to + 1), consistent with ESPResSo's reaction ensemble
                    ln_bias += (n_from as f64).ln() - ((n_to + 1) as f64).ln();

                    actions.push(SpeciationAction::SwapAtomKind {
                        group_index,
                        abs_index,
                        new_atom_id: *to_id,
                    });
                    group_changes.push((group_index, GroupChange::UpdateIdentity(vec![abs_index])));
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
        let resolved = self.resolved.choose(rng)?;
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

        self.trial_state = Some(TrialState { ln_bias });

        Some(ProposedMove {
            change: Change::Groups(group_changes),
            displacement: Displacement::None,
            transform: Transform::Speciation(actions),
            target: MoveTarget::System,
        })
    }

    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> crate::montecarlo::Bias {
        if let Some(trial) = &self.trial_state {
            crate::montecarlo::Bias::Energy(-self.thermal_energy * trial.ln_bias)
        } else {
            crate::montecarlo::Bias::None
        }
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        tagged_yaml("SpeciationMove", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::GroupCollection;
    use crate::platform::aos::AosPlatform;
    use crate::propagate::MoveProposal;
    use crate::WithCell;
    use float_cmp::assert_approx_eq;

    const TEST_YAML: &str = "tests/files/speciation_test.yaml";

    fn make_context() -> AosPlatform {
        let mut rng = rand::thread_rng();
        AosPlatform::new(TEST_YAML, None, &mut rng).unwrap()
    }

    fn make_move(reaction: &str, k: f64) -> SpeciationMove {
        serde_yaml::from_str(&format!(
            "temperature: 298.15\nreactions:\n  - {{ reaction: \"{reaction}\", K: {k} }}"
        ))
        .unwrap()
    }

    // --- YAML deserialization ---

    #[test]
    fn reaction_config_yaml() {
        let config: ReactionConfig =
            serde_yaml::from_str(r#"{ reaction: "= NaCl", K: 100.0 }"#).unwrap();
        assert_eq!(config.reaction, "= NaCl");
        assert_eq!(config.equilibrium_constant, 100.0);
    }

    #[test]
    fn speciation_move_yaml() {
        let yaml = "temperature: 298.15\nreactions:\n  - { reaction: '= M', K: 10.0 }\n  - { reaction: '⚛A = ⚛B', K: 1.0 }";
        let mv: SpeciationMove = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(mv.temperature, 298.15);
        assert_eq!(mv.reactions.len(), 2);
    }

    #[test]
    fn unknown_field_rejected() {
        let yaml = r#"{ temperature: 300.0, reactions: [], bogus: 42 }"#;
        assert!(serde_yaml::from_str::<SpeciationMove>(yaml).is_err());
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
        let expected_kt = ExternalPressure::thermal_energy_from_temperature(298.15);
        assert_approx_eq!(f64, mv.thermal_energy, expected_kt, epsilon = 1e-10);
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
            .group_lists()
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
            .group_lists()
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
                assert!(mv.trial_state.is_some());
                return;
            }
        }
        panic!("propose_move should succeed at least once in 20 tries");
    }

    #[test]
    fn propose_move_returns_none_when_empty() {
        let context = make_context();
        let mut mv: SpeciationMove =
            serde_yaml::from_str("temperature: 298.15\nreactions: []").unwrap();
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
                let bias = MoveProposal::<AosPlatform>::bias(
                    &mv,
                    &proposed.change,
                    &NewOld::from(0.0, 0.0),
                );
                assert!(matches!(bias, crate::montecarlo::Bias::Energy(_)));
                return;
            }
        }
        panic!("Should get at least one proposal");
    }

    #[test]
    fn bias_returns_none_without_propose() {
        let mv = make_move("= M", 10.0);
        let bias =
            MoveProposal::<AosPlatform>::bias(&mv, &Change::Everything, &NewOld::from(0.0, 0.0));
        assert!(matches!(bias, crate::montecarlo::Bias::None));
    }

    // --- Full simulation round-trip ---

    #[test]
    fn speciation_simulation_energy_drift() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let mut rng = rand::thread_rng();
        let context = AosPlatform::new(TEST_YAML, None, &mut rng).unwrap();
        let propagate = Propagate::from_file(TEST_YAML, &context).unwrap();

        let kt = ExternalPressure::thermal_energy_from_temperature(298.15);
        let mut mc =
            MarkovChain::new(context, propagate, kt, AnalysisCollection::default()).unwrap();

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
}
