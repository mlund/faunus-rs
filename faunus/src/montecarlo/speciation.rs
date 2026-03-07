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

/// Conversion from molar (mol/L) to number density (1/Å³).
const MOLAR_TO_INV_ANGSTROM3: f64 = physical_constants::AVOGADRO_CONSTANT * 1e-27;

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
    /// Molar free energy in kJ/mol; `K = exp(-ΔG / kT)`.
    #[serde(rename = "dG")]
    DeltaG(f64),
}

impl EquilibriumConstant {
    /// Convert to K. The `kt` parameter (kJ/mol) is needed for the `dG` variant.
    fn to_k(&self, kt: f64) -> f64 {
        match self {
            Self::K(k) => *k,
            Self::LnK(ln_k) => ln_k.exp(),
            Self::Pk(pk) => 10.0_f64.powf(-pk),
            Self::DeltaG(dg) => (-dg / kt).exp(),
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

/// Generate a random point inside the cell using rejection sampling.
///
/// Uses `bounding_box` + `is_inside` so that any `RngCore` can be used
/// (the `Shape::get_point_inside` API requires `ThreadRng`).
fn random_point_inside(cell: &impl crate::cell::Shape, rng: &mut dyn RngCore) -> crate::Point {
    let bbox = cell
        .bounding_box()
        .expect("Cell must have a bounding box for GCMC insertion");
    loop {
        let point = crate::Point::new(
            rng.gen_range(-bbox.x..bbox.x),
            rng.gen_range(-bbox.y..bbox.y),
            rng.gen_range(-bbox.z..bbox.z),
        );
        if cell.is_inside(&point) {
            return point;
        }
    }
}

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
            let mass = atomkinds[context.get_atomkind(i)].mass();
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
            let k = config.1.to_k(self.thermal_energy);
            anyhow::ensure!(
                k > 0.0,
                "Equilibrium constant must be positive for reaction '{}'",
                config.0
            );

            let reaction = Reaction::from_reaction(&config.0, k)?;
            let (reactants, products) = reaction.get();

            // Build forward operations: consume reactants, produce products
            let mut forward_ops = Vec::new();
            let mut backward_ops = Vec::new();

            // Collect molecular participants
            let reactant_mols: Vec<usize> = reactants
                .iter()
                .filter_map(|p| match p {
                    Participant::Molecule(name) => Some(find_molecule_index(name, &topology)),
                    _ => None,
                })
                .collect::<anyhow::Result<_>>()?;
            let product_mols: Vec<usize> = products
                .iter()
                .filter_map(|p| match p {
                    Participant::Molecule(name) => Some(find_molecule_index(name, &topology)),
                    _ => None,
                })
                .collect::<anyhow::Result<_>>()?;

            // Detect molecular swaps: paired reactant/product molecules with equal atom count
            let mut swap_reactants: Vec<usize> = Vec::new();
            let mut swap_products: Vec<usize> = Vec::new();
            for (ri, &from_id) in reactant_mols.iter().enumerate() {
                for (pi, &to_id) in product_mols.iter().enumerate() {
                    if !swap_products.contains(&pi)
                        && from_id != to_id
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

            // Atom swap participants
            let (swap_fwd, swap_bwd) = resolve_atom_swaps(reactants, products, &topology)?;
            forward_ops.extend(swap_fwd);
            backward_ops.extend(swap_bwd);

            // Fold activities into effective K:
            // - Implicit species: consumed reactants increase K_eff, produced products decrease it
            // - Molecular fugacities (insert/delete only): sign is reversed, compensating for
            //   entropy_bias using N/V instead of N/(V·z). Swap molecules are excluded since
            //   total molecule count is conserved.
            let mut effective_ln_k = k.ln();
            for (participants, sign) in [(reactants, 1.0_f64), (products, -1.0_f64)] {
                for p in participants {
                    if let Participant::Implicit(name) = p {
                        let activity = find_implicit_activity(name, &topology)?;
                        effective_ln_k += sign * activity.ln();
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
                            let z = activity * MOLAR_TO_INV_ANGSTROM3;
                            effective_ln_k -= sign * z.ln();
                        }
                    }
                }
            }

            self.resolved.push(ResolvedReaction {
                effective_ln_k,
                forward_ops,
                backward_ops,
            });
        }

        // Validate that required groups exist
        let has_any_groups = |mol_id: usize| -> bool {
            context
                .group_lists()
                .find_molecules(mol_id, GroupSize::Full)
                .is_some()
                || context
                    .group_lists()
                    .find_molecules(mol_id, GroupSize::Empty)
                    .is_some()
        };
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
                        anyhow::ensure!(
                            has_any_groups(*mol_id),
                            "No groups found for molecule '{name}' needed by reaction"
                        );
                    }
                    ReactionOp::SwapAtom { molecule_id, .. } => {
                        let name = topology.moleculekinds()[*molecule_id].name();
                        let has_full = context
                            .group_lists()
                            .find_molecules(*molecule_id, GroupSize::Full)
                            .is_some();
                        anyhow::ensure!(
                            has_full,
                            "No active groups for molecule '{name}' needed by atom swap"
                        );
                    }
                    ReactionOp::SwapMolecule {
                        from_mol_id,
                        to_mol_id,
                    } => {
                        for &mol_id in [from_mol_id, to_mol_id] {
                            let name = topology.moleculekinds()[mol_id].name();
                            anyhow::ensure!(
                                has_any_groups(mol_id),
                                "No groups found for molecule '{name}' needed by molecular swap"
                            );
                        }
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

                    // Generate random position from bounding box via rejection sampling
                    let random_pos = random_point_inside(context.cell(), rng);
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
                ReactionOp::SwapMolecule {
                    from_mol_id,
                    to_mol_id,
                } => {
                    // Find a full group of the source molecule
                    let full_groups = context
                        .group_lists()
                        .find_molecules(*from_mol_id, GroupSize::Full)?;
                    if full_groups.is_empty() {
                        return None;
                    }
                    let &from_group = full_groups.iter().choose(rng)?;
                    let n_from = full_groups.len();

                    // Find an empty group of the target molecule
                    let empty_groups = context
                        .group_lists()
                        .find_molecules(*to_mol_id, GroupSize::Empty)?;
                    if empty_groups.is_empty() {
                        return None;
                    }
                    let &to_group = empty_groups.iter().choose(rng)?;
                    let n_to = context
                        .group_lists()
                        .find_molecules(*to_mol_id, GroupSize::Full)
                        .map_or(0, |gs| gs.len());

                    // Combinatorial bias: N_from / (N_to + 1)
                    ln_bias += (n_from as f64).ln() - ((n_to + 1) as f64).ln();

                    // Overlay target template onto source orientation
                    let from_active = context.groups()[from_group].iter_active();
                    let to_grp = &context.groups()[to_group];
                    let positions = overlay_swap_positions(from_active, to_grp, context, rng);

                    // Intramolecular energy is excluded from ΔU because it is
                    // absorbed into the equilibrium constant K.
                    actions.push(SpeciationAction::DeactivateGroup(from_group));
                    group_changes.push((
                        from_group,
                        GroupChange::ResizeExcludeIntra(GroupSize::Empty),
                    ));

                    actions.push(SpeciationAction::ActivateGroup {
                        group_index: to_group,
                        positions,
                    });
                    group_changes
                        .push((to_group, GroupChange::ResizeExcludeIntra(GroupSize::Full)));
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
    use crate::platform::soa::SoaPlatform;
    use crate::propagate::MoveProposal;
    use crate::WithCell;
    use float_cmp::assert_approx_eq;

    const TEST_YAML: &str = "tests/files/speciation_test.yaml";

    fn make_context() -> SoaPlatform {
        let mut rng = rand::thread_rng();
        SoaPlatform::new(TEST_YAML, None, &mut rng).unwrap()
    }

    fn make_move(reaction: &str, k: f64) -> SpeciationMove {
        serde_yaml::from_str(&format!(
            "temperature: 298.15\nreactions:\n  - [\"{reaction}\", !K {k}]"
        ))
        .unwrap()
    }

    // --- YAML deserialization ---

    #[test]
    fn reaction_config_yaml_k() {
        let config: ReactionConfig = serde_yaml::from_str(r#"["= NaCl", !K 100.0]"#).unwrap();
        assert_eq!(config.0, "= NaCl");
        assert!((config.1.to_k(1.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn reaction_config_yaml_pk() {
        let config: ReactionConfig =
            serde_yaml::from_str(r#"["⚛HA = ⚛A + ~H+", !pK 4.0]"#).unwrap();
        assert!((config.1.to_k(1.0) - 1e-4).abs() < 1e-14);
    }

    #[test]
    fn reaction_config_yaml_lnk() {
        let config: ReactionConfig = serde_yaml::from_str(r#"["= M", !lnK -2.302585]"#).unwrap();
        assert!((config.1.to_k(1.0) - 0.1).abs() < 1e-5);
    }

    #[test]
    fn reaction_config_yaml_dg() {
        // dG = 0 => K = 1; dG = -kT·ln(10) => K = 10
        let config: ReactionConfig = serde_yaml::from_str(r#"["= Na+ + Cl-", !dG 0.0]"#).unwrap();
        assert!((config.1.to_k(2.479) - 1.0).abs() < 1e-10);

        let kt = 2.479;
        let config: ReactionConfig =
            serde_yaml::from_str(&format!(r#"["= M", !dG {}]"#, -kt * 10.0_f64.ln())).unwrap();
        assert!((config.1.to_k(kt) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn speciation_move_yaml() {
        let yaml =
            "temperature: 298.15\nreactions:\n  - [\"= M\", !K 10.0]\n  - [\"⚛A = ⚛B\", !pK 0.0]";
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
                let bias = MoveProposal::<SoaPlatform>::bias(
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
            MoveProposal::<SoaPlatform>::bias(&mv, &Change::Everything, &NewOld::from(0.0, 0.0));
        assert!(matches!(bias, crate::montecarlo::Bias::None));
    }

    // --- Full simulation round-trip ---

    #[test]
    fn speciation_simulation_energy_drift() {
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;

        let mut rng = rand::thread_rng();
        let context = SoaPlatform::new(TEST_YAML, None, &mut rng).unwrap();
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
    fn count_phosphate_species(context: &SoaPlatform) -> [usize; 4] {
        let gl = context.group_lists();
        [0, 1, 2, 3].map(|id| {
            gl.find_molecules(id, GroupSize::Full)
                .map_or(0, |g| g.len())
        })
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
            let context = SoaPlatform::new(path, None, &mut rng).unwrap();
            let propagate = Propagate::from_file(path, &context).unwrap();
            let kt = ExternalPressure::thermal_energy_from_temperature(298.15);
            let mut mc =
                MarkovChain::new(context, propagate, kt, AnalysisCollection::default()).unwrap();

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
