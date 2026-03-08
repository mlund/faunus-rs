//! Gibbs ensemble Monte Carlo: inter-box moves and ensemble coordinator.
//!
//! Two simulation boxes exchange volume and particles while running
//! independent intra-box MC in parallel via `std::thread::scope`.
//! Reference: Panagiotopoulos, Mol. Phys. 61, 813 (1987), doi:10.1080/00268978700101491.

use super::{MarkovChain, MoveStatistics};
use crate::cell::{Shape, VolumeScalePolicy};
use crate::energy::EnergyChange;
use crate::group::GroupSize;
use crate::propagate::tagged_yaml;
use crate::transform::Transform;
use crate::{Change, Context};
use anyhow::Result;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::Deserialize;
use std::fmt::Debug;

/// Inter-box move operating on two simulation contexts simultaneously.
pub trait GibbsMove<T: Context>: Debug + Send {
    fn perform(
        &mut self,
        c0: &mut T,
        c1: &mut T,
        thermal_energy: f64,
        rng: &mut StdRng,
    ) -> Result<()>;

    fn to_yaml(&self) -> Option<serde_yaml::Value>;
}

// ---------------------------------------------------------------------------
// Volume exchange
// ---------------------------------------------------------------------------

/// Exchange volume between two Gibbs ensemble boxes.
///
/// Proposes ΔV uniformly in `[-dV/2, dV/2]`, grows one box and shrinks the
/// other by the same amount.  Acceptance includes the ideal-gas entropy bias
/// `N₁·ln(V₁'/V₁) + N₂·ln(V₂'/V₂)`.
#[derive(Debug)]
pub struct GibbsVolumeExchange {
    dv: f64,
    statistics: MoveStatistics,
}

impl GibbsVolumeExchange {
    pub fn new(dv: f64) -> Self {
        Self {
            dv,
            statistics: MoveStatistics::default(),
        }
    }
}

impl<T: Context> GibbsMove<T> for GibbsVolumeExchange {
    fn perform(
        &mut self,
        c0: &mut T,
        c1: &mut T,
        thermal_energy: f64,
        rng: &mut StdRng,
    ) -> Result<()> {
        self.statistics.timer.start();

        let v0 = c0
            .cell()
            .volume()
            .ok_or_else(|| anyhow::anyhow!("box 0 has no volume"))?;
        let v1 = c1
            .cell()
            .volume()
            .ok_or_else(|| anyhow::anyhow!("box 1 has no volume"))?;
        let dv = (rng.r#gen::<f64>() - 0.5) * self.dv;
        let v0_new = v0 + dv;
        let v1_new = v1 - dv;

        if v0_new < 1.0 || v1_new < 1.0 {
            self.statistics.reject();
            return Ok(());
        }

        let n0 = c0.num_active_particles() as f64;
        let n1 = c1.num_active_particles() as f64;

        let old_energy_0 = c0.hamiltonian().energy(c0, &Change::Everything);
        let old_energy_1 = c1.hamiltonian().energy(c1, &Change::Everything);

        Transform::VolumeScale(VolumeScalePolicy::Isotropic, v0_new).on_system_with_backup(c0)?;
        Transform::VolumeScale(VolumeScalePolicy::Isotropic, v1_new).on_system_with_backup(c1)?;

        c0.update_with_backup(&Change::Everything)?;
        c1.update_with_backup(&Change::Everything)?;

        let new_energy_0 = c0.hamiltonian().energy(c0, &Change::Everything);
        let new_energy_1 = c1.hamiltonian().energy(c1, &Change::Everything);

        let du = (new_energy_0 - old_energy_0) + (new_energy_1 - old_energy_1);
        let bias = n0 * (v0_new / v0).ln() + n1 * (v1_new / v1).ln();
        let exponent = -du / thermal_energy + bias;

        if rng.r#gen::<f64>() < f64::exp(exponent) {
            self.statistics
                .accept(du, crate::propagate::Displacement::Custom(dv));
            c0.discard_backup();
            c1.discard_backup();
        } else {
            self.statistics.reject();
            c0.undo()?;
            c1.undo()?;
        }
        Ok(())
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        let mut map = serde_yaml::Mapping::new();
        map.insert("dV".into(), self.dv.into());
        map.insert(
            "statistics".into(),
            serde_yaml::to_value(&self.statistics).ok()?,
        );
        tagged_yaml("GibbsVolumeExchange", &map)
    }
}

// ---------------------------------------------------------------------------
// Particle transfer
// ---------------------------------------------------------------------------

/// Transfer a molecule between two Gibbs ensemble boxes.
///
/// Picks a random direction, deactivates a molecule in the source box and
/// activates an empty slot in the target box at a random position.
/// Acceptance follows Panagiotopoulos Eq. 8.
#[derive(Debug)]
pub struct GibbsParticleTransfer {
    molecule_id: usize,
    molecule_name: String,
    statistics: MoveStatistics,
}

impl GibbsParticleTransfer {
    pub fn new(molecule_id: usize, molecule_name: String) -> Self {
        Self {
            molecule_id,
            molecule_name,
            statistics: MoveStatistics::default(),
        }
    }

    /// Count active (Full + Partial) groups of the tracked molecule type.
    fn count_active_molecules(context: &impl Context, molecule_id: usize) -> usize {
        let gl = context.group_lists();
        let full = gl
            .find_molecules(molecule_id, GroupSize::Full)
            .map_or(0, |s| s.len());
        let partial = gl
            .find_molecules(molecule_id, GroupSize::Partial(0))
            .map_or(0, |s| s.len());
        full + partial
    }

    /// Transfer one molecule from `src` to `tgt`.
    fn transfer(
        &mut self,
        src: &mut impl Context,
        tgt: &mut impl Context,
        thermal_energy: f64,
        rng: &mut StdRng,
    ) -> Result<()> {
        let src_group = src
            .group_lists()
            .find_molecules(self.molecule_id, GroupSize::Full)
            .and_then(|gs| gs.iter().copied().choose(rng));
        let tgt_group = tgt
            .group_lists()
            .find_molecules(self.molecule_id, GroupSize::Empty)
            .and_then(|gs| gs.iter().copied().choose(rng));

        let (Some(src_group), Some(tgt_group)) = (src_group, tgt_group) else {
            self.statistics.reject();
            return Ok(());
        };

        let n_src = Self::count_active_molecules(src, self.molecule_id) as f64;
        let n_tgt = Self::count_active_molecules(tgt, self.molecule_id) as f64;
        let v_src = src
            .cell()
            .volume()
            .ok_or_else(|| anyhow::anyhow!("source has no volume"))?;
        let v_tgt = tgt
            .cell()
            .volume()
            .ok_or_else(|| anyhow::anyhow!("target has no volume"))?;

        let old_e_src = src.hamiltonian().energy(src, &Change::Everything);
        let old_e_tgt = tgt.hamiltonian().energy(tgt, &Change::Everything);

        src.save_system_backup();
        tgt.save_system_backup();

        // copy source molecule positions and compute COM
        let src_indices: Vec<usize> = src.groups()[src_group].iter_active().collect();
        let com = src.mass_center(&src_indices);

        // shift positions to a random position in target cell
        let shift = tgt.cell().get_point_inside(&mut rand::thread_rng()) - com;
        let positions: Vec<_> = src_indices
            .iter()
            .map(|&i| src.position(i) + shift)
            .collect();
        let tgt_start = tgt.groups()[tgt_group].start();
        let tgt_indices = tgt_start..tgt_start + positions.len();
        // atom kinds are already correct in the target slot
        tgt.set_positions(tgt_indices, positions.iter());

        Transform::Deactivate.on_group(src_group, src)?;
        Transform::Activate.on_group(tgt_group, tgt)?;

        src.update_with_backup(&Change::Everything)?;
        tgt.update_with_backup(&Change::Everything)?;

        let new_e_src = src.hamiltonian().energy(src, &Change::Everything);
        let new_e_tgt = tgt.hamiltonian().energy(tgt, &Change::Everything);

        let du = (new_e_src - old_e_src) + (new_e_tgt - old_e_tgt);
        // Panagiotopoulos Eq. 8
        let bias = (v_tgt * n_src / (v_src * (n_tgt + 1.0))).ln();
        let exponent = -du / thermal_energy + bias;

        if rng.r#gen::<f64>() < f64::exp(exponent) {
            self.statistics
                .accept(du, crate::propagate::Displacement::None);
            src.discard_backup();
            tgt.discard_backup();
        } else {
            self.statistics.reject();
            src.undo()?;
            tgt.undo()?;
        }
        Ok(())
    }
}

impl<T: Context> GibbsMove<T> for GibbsParticleTransfer {
    fn perform(
        &mut self,
        c0: &mut T,
        c1: &mut T,
        thermal_energy: f64,
        rng: &mut StdRng,
    ) -> Result<()> {
        self.statistics.timer.start();

        if rng.r#gen::<bool>() {
            self.transfer(c0, c1, thermal_energy, rng)
        } else {
            self.transfer(c1, c0, thermal_energy, rng)
        }
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        let mut map = serde_yaml::Mapping::new();
        map.insert("molecule".into(), self.molecule_name.clone().into());
        map.insert(
            "statistics".into(),
            serde_yaml::to_value(&self.statistics).ok()?,
        );
        tagged_yaml("GibbsParticleTransfer", &map)
    }
}

// ---------------------------------------------------------------------------
// YAML deserialization
// ---------------------------------------------------------------------------

/// Deserialization helper for Gibbs inter-box moves.
#[derive(Clone, Debug, Deserialize)]
pub enum GibbsMoveBuilder {
    GibbsVolumeExchange {
        #[serde(alias = "dV")]
        dv: f64,
    },
    GibbsParticleTransfer {
        molecule: String,
    },
}

impl GibbsMoveBuilder {
    /// Build a concrete `GibbsMove` from the deserialized config.
    pub fn build<T: Context + 'static>(self, context: &T) -> Result<Box<dyn GibbsMove<T>>> {
        Ok(match self {
            Self::GibbsVolumeExchange { dv } => {
                anyhow::ensure!(dv > 0.0, "GibbsVolumeExchange: dV must be positive");
                Box::new(GibbsVolumeExchange::new(dv))
            }
            Self::GibbsParticleTransfer { molecule } => {
                let molecule_id = crate::montecarlo::find_molecule_id(
                    context,
                    &molecule,
                    "GibbsParticleTransfer",
                )?;
                Box::new(GibbsParticleTransfer::new(molecule_id, molecule))
            }
        })
    }
}

/// Top-level Gibbs ensemble configuration parsed from `propagate.gibbs`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GibbsConfig {
    /// Number of intra-box propagation cycles between inter-box moves.
    pub intra_steps: usize,
    /// Inter-box moves to perform each Gibbs sweep.
    pub moves: Vec<GibbsMoveBuilder>,
}

// ---------------------------------------------------------------------------
// Ensemble coordinator
// ---------------------------------------------------------------------------

/// Gibbs ensemble: two boxes with parallel intra-box MC and sequential inter-box moves.
pub struct GibbsEnsemble<T: Context> {
    boxes: [MarkovChain<T>; 2],
    inter_moves: Vec<Box<dyn GibbsMove<T>>>,
    intra_steps: usize,
    max_sweeps: usize,
    thermal_energy: f64,
    rng: StdRng,
}

impl<T: Context> Debug for GibbsEnsemble<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GibbsEnsemble")
            .field("intra_steps", &self.intra_steps)
            .field("max_sweeps", &self.max_sweeps)
            .field("thermal_energy", &self.thermal_energy)
            .field("inter_moves", &self.inter_moves)
            .finish()
    }
}

impl<T: Context + Send + 'static> GibbsEnsemble<T> {
    pub fn new(
        boxes: [MarkovChain<T>; 2],
        inter_moves: Vec<Box<dyn GibbsMove<T>>>,
        intra_steps: usize,
        max_sweeps: usize,
        thermal_energy: f64,
        rng: StdRng,
    ) -> Self {
        Self {
            boxes,
            inter_moves,
            intra_steps,
            max_sweeps,
            thermal_energy,
            rng,
        }
    }

    /// Number of Gibbs sweeps (outer loop iterations).
    pub fn max_sweeps(&self) -> usize {
        self.max_sweeps
    }

    /// Number of intra-box propagation cycles per sweep.
    pub fn intra_steps(&self) -> usize {
        self.intra_steps
    }

    /// Run sequential inter-box moves (called between intra-box sweeps).
    pub fn perform_inter_moves(&mut self) -> Result<()> {
        let (first, rest) = self.boxes.split_at_mut(1);
        for mv in &mut self.inter_moves {
            mv.perform(
                first[0].context_mut(),
                rest[0].context_mut(),
                self.thermal_energy,
                &mut self.rng,
            )?;
        }
        Ok(())
    }

    /// Access both MarkovChains.
    pub fn boxes(&self) -> &[MarkovChain<T>; 2] {
        &self.boxes
    }

    /// Mutable access to both MarkovChains.
    pub fn boxes_mut(&mut self) -> &mut [MarkovChain<T>; 2] {
        &mut self.boxes
    }

    /// Finalize analyses in both boxes.
    pub fn finalize_analyses(&mut self) -> Result<()> {
        self.boxes[0].finalize_analyses()?;
        self.boxes[1].finalize_analyses()?;
        Ok(())
    }

    /// Serialize inter-box move results to YAML.
    pub fn inter_moves_to_yaml(&self) -> Vec<serde_yaml::Value> {
        self.inter_moves
            .iter()
            .filter_map(|m| m.to_yaml())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::WithCell;

    #[test]
    fn gibbs_volume_exchange_conserves_total_volume() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/translate_molecules_simulation.yaml",
            None,
            &mut rng,
        )
        .unwrap();

        let mut c0 = context.clone();
        let mut c1 = context;

        let total_before = c0.cell().volume().unwrap() + c1.cell().volume().unwrap();

        let mut ve = GibbsVolumeExchange::new(5.0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            ve.perform(&mut c0, &mut c1, 1.0, &mut rng).unwrap();
        }

        let total_after = c0.cell().volume().unwrap() + c1.cell().volume().unwrap();
        assert!(
            (total_after - total_before).abs() < 1e-10,
            "Total volume not conserved: {total_before} -> {total_after}"
        );
        assert_eq!(ve.statistics.num_trials, 100);
    }

    #[test]
    fn gibbs_move_builder_volume_yaml() {
        let yaml = "!GibbsVolumeExchange { dV: 10.0 }";
        let builder: GibbsMoveBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder, GibbsMoveBuilder::GibbsVolumeExchange { dv } if dv == 10.0));
    }

    #[test]
    fn gibbs_move_builder_transfer_yaml() {
        let yaml = "!GibbsParticleTransfer { molecule: LJ }";
        let builder: GibbsMoveBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(
            matches!(builder, GibbsMoveBuilder::GibbsParticleTransfer { molecule } if molecule == "LJ")
        );
    }

    #[test]
    fn gibbs_config_yaml() {
        let yaml = "
intra_steps: 100
moves:
  - !GibbsVolumeExchange { dV: 10 }
  - !GibbsParticleTransfer { molecule: LJ }
";
        let config: GibbsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.intra_steps, 100);
        assert_eq!(config.moves.len(), 2);
    }
}
