// Copyright 2023 Mikael Lund
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

//! # Support for Monte Carlo sampling

use crate::analysis::{AnalysisCollection, Analyze};
use crate::energy::EnergyChange;
use crate::group::*;
use crate::propagate::{Displacement, Propagate};
use crate::state::{GroupState, State};
use crate::{time::Timer, Context};
use anyhow::Result;
use average::{Estimate, Mean};
use log;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Div;
use std::{cmp::Ordering, ops::Neg};

mod crankshaft;
mod pivot;
mod rotate;
mod translate;
mod volume;

pub use crankshaft::CrankshaftMove;
pub use pivot::PivotMove;
pub use rotate::RotateMolecule;
pub use translate::*;
pub use volume::VolumeMove;

/// Look up a molecule kind by name and return its id.
fn find_molecule_id(
    context: &impl Context,
    molecule_name: &str,
    move_name: &str,
) -> anyhow::Result<usize> {
    context
        .topology()
        .moleculekinds()
        .iter()
        .position(|x| x.name() == molecule_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Molecule '{}' in '{}' move does not exist.",
                molecule_name,
                move_name
            )
        })
}

/// Pick a random group index of the specified molecule type.
fn random_group(
    context: &impl Context,
    rng: &mut (impl Rng + ?Sized),
    molecule_id: usize,
) -> Option<usize> {
    let select = GroupSelection::ByMoleculeId(molecule_id);
    context.select(&select).iter().copied().choose(rng)
}

/// Pick a random atom from the specified group.
/// Returns an absolute index of the atom.
fn random_atom(
    context: &impl Context,
    rng: &mut (impl Rng + ?Sized),
    group_index: usize,
    atom_id: Option<usize>,
) -> Option<usize> {
    let select = atom_id.map_or(ParticleSelection::Active, ParticleSelection::ById);

    context
        .groups()
        .get(group_index)
        .expect("Group should exist.")
        .select(&select, context)
        .expect("Selection should be successful.")
        .iter()
        .copied()
        .choose(rng)
}

/// Custom bias to be added to the energy after a given move
///
/// Some moves may need to add additional bias not captured by the Hamiltonian.
#[derive(Clone, Copy, Debug)]
pub enum Bias {
    /// Custom bias to be added to the energy
    Energy(f64),
    /// Force acceptance of the move regardless of energy change
    ForceAccept,
    /// No bias
    None,
}

/// Named helper struct to handle `new`, `old` pairs.
///
/// Used e.g. for data before and after a Monte Carlo move
/// and reduces risk mixing up the order or old and new values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NewOld<T> {
    pub new: T,
    pub old: T,
}

impl<T> NewOld<T> {
    pub const fn from(new: T, old: T) -> Self {
        Self { new, old }
    }
}

impl NewOld<usize> {
    /// Difference `new - old` as a signed integer
    pub const fn difference(&self) -> i32 {
        self.new as i32 - self.old as i32
    }
}

impl NewOld<f64> {
    /// Difference `new - old`
    pub fn difference(&self) -> f64 {
        self.new - self.old
    }
}

impl Copy for NewOld<usize> {}
impl Copy for NewOld<f64> {}

/// # Helper class to keep track of accepted and rejected moves
///
/// It is optionally possible to let this class keep track of a single mean square displacement
/// which can be useful for many Monte Carlo moves.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct MoveStatistics {
    /// Number of trial moves
    pub num_trials: usize,
    /// Number of accepted moves
    pub num_accepted: usize,
    /// Mean square displacement of some quantity (optional)
    pub mean_square_displacement: Option<Mean>,
    /// Timer that measures the time spent in the move
    #[serde(skip_deserializing)]
    pub timer: Timer,
    /// Custom statistics and information (only serialized)
    #[serde(skip_deserializing)]
    pub info: HashMap<String, crate::topology::Value>,
    /// Sum of energy changes due to this move
    pub energy_change_sum: f64,
}

impl MoveStatistics {
    /// Register an accepted move and increment counters
    pub fn accept(&mut self, energy_change: f64, displacement: Displacement) {
        self.num_trials += 1;
        self.num_accepted += 1;
        self.energy_change_sum += energy_change;
        self.update_msd(displacement);
    }

    /// Register a rejected move and increment counters
    pub fn reject(&mut self) {
        self.num_trials += 1;
        self.update_msd(Displacement::Zero);
    }

    /// Acceptance ratio
    pub fn acceptance_ratio(&self) -> f64 {
        self.num_accepted as f64 / self.num_trials as f64
    }

    /// Update mean square displacement if possible
    pub(crate) fn update_msd(&mut self, displacement: Displacement) {
        if let Ok(dp) = f64::try_from(displacement) {
            self.mean_square_displacement
                .get_or_insert_with(Mean::new)
                .add(dp * dp);
        }
    }
}

/// All possible acceptance criteria for Monte Carlo moves
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Copy)]
pub enum AcceptanceCriterion {
    #[default]
    #[serde(alias = "Metropolis")]
    /// Metropolis-Hastings acceptance criterion
    /// More information: <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>
    MetropolisHastings,
    /// Energy minimization acceptance criterion
    /// This will always accept a move if the new energy is lower than the old energy.
    Minimize,
}

impl AcceptanceCriterion {
    /// Acceptance criterion based on an old and new energy.
    ///
    /// The energies are normalized by the given thermal energy, _kT_,.
    pub(crate) fn accept(
        &self,
        energy: NewOld<f64>,
        bias: Bias,
        thermal_energy: f64,
        rng: &mut (impl Rng + ?Sized),
    ) -> bool {
        match self {
            Self::MetropolisHastings => {
                // useful for hard-sphere systems where initial configurations may overlap
                if energy.old.is_infinite() && energy.new.is_finite() {
                    log::trace!("Accepting infinite -> finite energy change");
                    return true;
                }
                // always accept if negative infinity
                if energy.new.is_infinite() && energy.new.is_sign_negative() {
                    return true;
                }

                let du = energy.difference()
                    + match bias {
                        Bias::Energy(bias) => bias,
                        Bias::None => 0.0,
                        Bias::ForceAccept => return true,
                    };
                // Reject if du is NaN (e.g. from inf - inf during hard-sphere overlap)
                if du.is_nan() {
                    return false;
                }
                let p = f64::min(1.0, f64::exp(-du / thermal_energy));
                rng.r#gen::<f64>() < p
            }

            Self::Minimize => {
                if energy.old.is_infinite() && energy.new.is_finite() {
                    return true;
                }
                energy.difference()
                    + match bias {
                        Bias::Energy(bias) => bias,
                        Bias::None => 0.0,
                        Bias::ForceAccept => return true,
                    }
                    <= 0.0
            }
        }
    }
}

/// # Monte Carlo simulation instance
///
/// This maintains a single [`Context`] with backup/undo support, and a [`Propagate`] section
/// specifying what moves to perform and how often. The MC loop computes old energy before
/// mutation, applies the transform with backup, then accepts (discard backup) or rejects (undo).
///
/// The MarkovChain can be converted into an `Iterator` where each iteration corresponds to one 'propagate' cycle.
#[derive(Debug)]
pub struct MarkovChain<T: Context> {
    /// Description of moves to perform.
    propagate: Propagate<T>,
    /// Simulation context.
    context: T,
    /// Current step.
    step: usize,
    /// Thermal energy - must be same unit as energy.
    thermal_energy: f64,
    /// Collection of analyses to perform during the simulation.
    analyses: AnalysisCollection<T>,
}

impl<T: Context> MarkovChain<T> {
    pub const fn iter(&mut self) -> MarkovChainIterator<'_, T> {
        MarkovChainIterator { markov: self }
    }
}

/// Iterator over MarkovChain.
/// Necessary if we want to access MarkovChain after the iteration is finished.
#[derive(Debug)]
pub struct MarkovChainIterator<'a, T: Context> {
    markov: &'a mut MarkovChain<T>,
}

impl<T: Context> Iterator for MarkovChainIterator<'_, T> {
    type Item = anyhow::Result<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.markov.propagate.propagate(
            &mut self.markov.context,
            self.markov.thermal_energy,
            &mut self.markov.step,
            &mut self.markov.analyses,
        ) {
            Err(e) => Some(Err(e)),
            Ok(true) => Some(Ok(self.markov.step)),
            Ok(false) => None,
        }
    }
}

impl<T: Context + 'static> MarkovChain<T> {
    pub fn new(
        context: T,
        propagate: Propagate<T>,
        thermal_energy: f64,
        analyses: AnalysisCollection<T>,
    ) -> Result<Self> {
        Ok(Self {
            context,
            thermal_energy,
            step: 0,
            propagate,
            analyses,
        })
    }

    /// Compute total system energy using `Change::Everything`.
    pub fn system_energy(&self) -> f64 {
        self.context
            .hamiltonian()
            .energy(&self.context, &crate::Change::Everything)
    }

    /// Absolute energy drift: |initial + sum(accepted ΔE) - current|.
    pub fn energy_drift(&self, initial_energy: f64) -> f64 {
        let sum_du = self.propagate.energy_change_sum();
        (initial_energy + sum_du - self.system_energy()).abs()
    }

    /// Propagate instance describing moves to perform.
    pub const fn context(&self) -> &T {
        &self.context
    }

    pub const fn propagation(&self) -> &Propagate<T> {
        &self.propagate
    }

    /// Collection of analyses.
    pub fn analyses(&self) -> &AnalysisCollection<T> {
        &self.analyses
    }

    /// Run end-of-simulation analyses (e.g. `Trajectory` with `frequency: End`).
    pub fn finalize_analyses(&mut self) -> Result<()> {
        self.analyses.finalize(&self.context)
    }

    /// Set the thermal energy, _kT_.
    ///
    /// This is used to normalize the energy change when determining the acceptance probability.
    /// Must match the unit of the energy.
    pub const fn set_thermal_energy(&mut self, thermal_energy: f64) {
        self.thermal_energy = thermal_energy;
    }
    /// Append an analysis to the back of the collection.
    pub fn add_analysis(&mut self, analysis: Box<dyn Analyze<T>>) {
        self.analyses.push(analysis)
    }
}

impl<T: Context + crate::WithCell<SimCell = crate::cell::Cell> + 'static> MarkovChain<T> {
    /// Extract the current simulation state for checkpointing.
    pub fn save_state(&self) -> State {
        let context = &self.context;
        State {
            particles: context.get_all_particles(),
            cell: context.cell().clone(),
            groups: context
                .groups()
                .iter()
                .map(|g| GroupState {
                    molecule: g.molecule(),
                    capacity: g.capacity(),
                    size: g.size(),
                })
                .collect(),
            step: self.step,
        }
    }

    /// Restore simulation state from a checkpoint.
    ///
    /// Validates topology compatibility before modifying any state,
    /// so a mismatched state file is rejected cleanly.
    pub fn load_state(&mut self, state: State) -> Result<()> {
        let num_particles = self.context.num_particles();
        let num_groups = self.context.groups().len();

        if state.particles.len() != num_particles {
            anyhow::bail!(
                "Particle count mismatch: state has {}, context has {}",
                state.particles.len(),
                num_particles
            );
        }
        if state.groups.len() != num_groups {
            anyhow::bail!(
                "Group count mismatch: state has {}, context has {}",
                state.groups.len(),
                num_groups
            );
        }

        // Catch topology changes that alter atom types but preserve total count
        for (i, state_p) in state.particles.iter().enumerate() {
            let ctx_id = self.context.particle(i).atom_id;
            if state_p.atom_id != ctx_id {
                anyhow::bail!(
                    "Particle {} atom_id mismatch: state has {}, topology has {}",
                    i,
                    state_p.atom_id,
                    ctx_id
                );
            }
        }

        // Catch molecule reordering or resized molecule definitions
        for (i, (gs, group)) in state
            .groups
            .iter()
            .zip(self.context.groups().iter())
            .enumerate()
        {
            if gs.molecule != group.molecule() {
                anyhow::bail!(
                    "Group {} molecule mismatch: state has {}, topology has {}",
                    i,
                    gs.molecule,
                    group.molecule()
                );
            }
            if gs.capacity != group.capacity() {
                anyhow::bail!(
                    "Group {} capacity mismatch: state has {}, topology has {}",
                    i,
                    gs.capacity,
                    group.capacity()
                );
            }
        }

        self.context
            .set_particles(0..num_particles, state.particles.iter())?;
        *self.context.cell_mut() = state.cell;
        for (i, gs) in state.groups.iter().enumerate() {
            self.context.resize_group(i, gs.size)?;
        }
        for i in 0..num_groups {
            self.context.update_mass_center(i);
        }
        self.context.update(&crate::Change::Everything)?;

        self.step = state.step;
        log::info!("Restored simulation state");
        Ok(())
    }
}

/// Entropy bias due to a change in number of particles
///
/// See:
/// - <https://en.wikipedia.org/wiki/Entropy_(statistical_thermodynamics)#Entropy_of_mixing>
/// - <https://doi.org/10/fqcpg3>
///
/// # Examples
/// ~~~
/// use faunus::montecarlo::*;
/// let vol = NewOld::from(1.0, 1.0);
/// assert_eq!(entropy_bias(NewOld::from(0, 0), vol.clone()), 0.0);
/// assert_eq!(entropy_bias(NewOld::from(2, 1), vol.clone()), f64::ln(2.0));
/// assert_eq!(entropy_bias(NewOld::from(1, 2), vol.clone()), f64::ln(0.5));
/// ~~~
///
/// Note that the volume unit should match so that n/V matches the unit of the chemical potential
pub fn entropy_bias(n: NewOld<usize>, volume: NewOld<f64>) -> f64 {
    let dn = n.difference();
    match dn.cmp(&0) {
        Ordering::Equal => {
            if volume.difference().abs() > f64::EPSILON {
                unimplemented!("Entropy bias currently cannot be used for volume changes")
            }
            0.0
        }
        Ordering::Greater => (0..dn)
            .map(|i| f64::from(n.old as i32 + i + 1).div(volume.new).ln())
            .sum(),
        Ordering::Less => (0..-dn)
            .map(|i| f64::from(n.old as i32 - i).div(volume.old).ln())
            .sum::<f64>()
            .neg(),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::platform::reference::ReferencePlatform;
    use float_cmp::assert_approx_eq;

    #[test]
    fn translate_molecules_simulation() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/translate_molecules_simulation.yaml",
            None,
            &mut rng,
        )
        .unwrap();

        let propagate =
            Propagate::from_file("tests/files/translate_molecules_simulation.yaml", &context)
                .unwrap();

        let mut markov_chain =
            MarkovChain::new(context, propagate, 1.0, AnalysisCollection::default()).unwrap();

        let initial_energy = markov_chain.system_energy();

        for step in markov_chain.iter() {
            step.unwrap();
        }

        let drift = markov_chain.energy_drift(initial_energy);
        assert!(drift < 1e-10, "Energy drift {drift:.6e} exceeds tolerance");

        let move1_stats = markov_chain.propagate.collections()[0].moves()[0].statistics();

        assert_eq!(move1_stats.num_trials, 73);
        assert_eq!(move1_stats.num_accepted, 71);
        assert_approx_eq!(
            f64,
            move1_stats.energy_change_sum,
            -3.3952572353350177,
            epsilon = 1e-14
        );

        let move2_stats = markov_chain.propagate.collections()[0].moves()[1].statistics();

        assert_eq!(move2_stats.num_trials, 81);
        assert_eq!(move2_stats.num_accepted, 79);
        assert_approx_eq!(
            f64,
            move2_stats.energy_change_sum,
            -1.1611869334060376,
            epsilon = 1e-14
        );

        let move3_stats = markov_chain.propagate.collections()[0].moves()[2].statistics();

        assert_eq!(move3_stats.num_trials, 0);
        assert_eq!(move3_stats.num_accepted, 0);
        assert_approx_eq!(f64, move3_stats.energy_change_sum, 0.0, epsilon = 1e-14);

        let move4_stats = markov_chain.propagate.collections()[1].moves()[0].statistics();

        assert_eq!(move4_stats.num_trials, 100);
        assert_eq!(move4_stats.num_accepted, 94);
        assert_approx_eq!(
            f64,
            move4_stats.energy_change_sum,
            -61.739122509342266,
            epsilon = 1e-14
        );

        let move5_stats = markov_chain.propagate.collections()[2].moves()[0].statistics();

        assert_eq!(move5_stats.num_trials, 500);
        assert_eq!(move5_stats.num_accepted, 466);
        assert_approx_eq!(
            f64,
            move5_stats.energy_change_sum,
            -515.1334649717062,
            epsilon = 1e-14
        );

        let context = &markov_chain.context;
        println!("{:?}", context.particles());

        let p1 = &context.particles()[0];
        let p2 = &context.particles()[1];
        let p3 = &context.particles()[2];

        assert_approx_eq!(f64, p1.pos.x, p2.pos.x + 1.0, epsilon = 0.0000001);
        assert_approx_eq!(f64, p1.pos.x, p3.pos.x + 1.0, epsilon = 0.0000001);
        assert_approx_eq!(f64, p2.pos.x, p3.pos.x, epsilon = 0.0000001);

        assert_approx_eq!(f64, p1.pos.y, p2.pos.y, epsilon = 0.0000001);
        assert_approx_eq!(f64, p1.pos.y + 1.0, p3.pos.y, epsilon = 0.0000001);
        assert_approx_eq!(f64, p2.pos.y + 1.0, p3.pos.y, epsilon = 0.0000001);

        assert_approx_eq!(f64, p1.pos.z, p2.pos.z, epsilon = 0.0000001);
        assert_approx_eq!(f64, p1.pos.z, p3.pos.z, epsilon = 0.0000001);
        assert_approx_eq!(f64, p2.pos.z, p3.pos.z, epsilon = 0.0000001);

        let p4 = &context.particles()[3];
        let p5 = &context.particles()[4];
        let p6 = &context.particles()[5];

        assert_approx_eq!(f64, p4.pos.x + 1.0, p5.pos.x, epsilon = 0.0000001);
        assert_approx_eq!(f64, p4.pos.x + 1.0, p6.pos.x, epsilon = 0.0000001);
        assert_approx_eq!(f64, p5.pos.x, p6.pos.x, epsilon = 0.0000001);

        assert_approx_eq!(f64, p4.pos.y, p5.pos.y, epsilon = 0.0000001);
        assert_approx_eq!(f64, p4.pos.y, p6.pos.y, epsilon = 0.0000001);
        assert_approx_eq!(f64, p5.pos.y, p6.pos.y, epsilon = 0.0000001);

        assert_approx_eq!(f64, p4.pos.z, p5.pos.z, epsilon = 0.0000001);
        assert_approx_eq!(f64, p4.pos.z, p6.pos.z + 1.0, epsilon = 0.0000001);
        assert_approx_eq!(f64, p5.pos.z, p6.pos.z + 1.0, epsilon = 0.0000001);

        let p7 = &context.particles()[6];
        let p8 = &context.particles()[7];
        let p9 = &context.particles()[8];

        assert_approx_eq!(f64, p7.pos.x, p8.pos.x, epsilon = 0.0000001);
        assert_approx_eq!(f64, p7.pos.x, p9.pos.x, epsilon = 0.0000001);
        assert_approx_eq!(f64, p8.pos.x, p9.pos.x, epsilon = 0.0000001);

        assert_approx_eq!(f64, p7.pos.y, p8.pos.y + 1.0, epsilon = 0.0000001);
        assert_approx_eq!(f64, p7.pos.y, p9.pos.y + 2.0, epsilon = 0.0000001);
        assert_approx_eq!(f64, p8.pos.y, p9.pos.y + 1.0, epsilon = 0.0000001);

        assert_approx_eq!(f64, p7.pos.z, p8.pos.z, epsilon = 0.0000001);
        assert_approx_eq!(f64, p7.pos.z, p9.pos.z, epsilon = 0.0000001);
        assert_approx_eq!(f64, p8.pos.z, p9.pos.z, epsilon = 0.0000001);
    }
}
