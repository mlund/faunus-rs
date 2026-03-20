//! Gibbs ensemble Monte Carlo: inter-box moves and ensemble coordinator.
//!
//! Two simulation boxes exchange volume and particles while running
//! independent intra-box MC in parallel via `std::thread::scope`.
//! Reference: Panagiotopoulos, Mol. Phys. 61, 813 (1987),
//! [doi:10.1080/00268978700101491](https://doi.org/10.1080/00268978700101491).

use super::speciation::random_point_inside;
use super::{MarkovChain, MoveStatistics};
use crate::cell::{Shape, VolumeScalePolicy};
use crate::energy::EnergyChange;
use crate::group::GroupSize;
use crate::propagate::tagged_yaml;
use crate::transform::{SpeciationAction, Transform};
use crate::{Change, Context};
use anyhow::Result;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Metropolis accept/reject for two-context Gibbs moves.
fn accept_or_reject(
    a: &mut impl Context,
    b: &mut impl Context,
    du: f64,
    exponent: f64,
    displacement: crate::propagate::Displacement,
    statistics: &mut MoveStatistics,
    rng: &mut StdRng,
) -> Result<()> {
    if rng.r#gen::<f64>() < exponent.exp() {
        statistics.accept(du, displacement);
        a.discard_backup();
        b.discard_backup();
    } else {
        statistics.reject();
        a.undo()?;
        b.undo()?;
    }
    Ok(())
}

/// Inter-box move operating on two simulation contexts simultaneously.
pub(crate) trait GibbsMove<T: Context>: Debug + Send {
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
// Volume displacement method
// ---------------------------------------------------------------------------

/// How to propose volume changes between two Gibbs ensemble boxes.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub(crate) enum VolumeDisplacementMethod {
    /// ln(V₁/V₂) space — Frenkel & Smit, Sec. 8.3.2.
    /// Bias includes (N+1) from the Jacobian of the ln(V) transformation.
    #[default]
    Logarithmic,
    /// Direct linear volume transfer δV — Allen & Tildesley, Eq. 9.75.
    Linear,
}

// ---------------------------------------------------------------------------
// Volume exchange
// ---------------------------------------------------------------------------

/// Exchange volume between two Gibbs ensemble boxes.
///
/// Default: logarithmic displacement in ln(V₁/V₂) space (Frenkel & Smit, Sec. 8.3.2).
/// Optional: linear displacement δV (Allen & Tildesley, Eq. 9.75).
#[derive(Debug)]
struct GibbsVolumeExchange {
    dv: f64,
    method: VolumeDisplacementMethod,
    statistics: MoveStatistics,
}

impl GibbsVolumeExchange {
    fn new(dv: f64, method: VolumeDisplacementMethod) -> Self {
        Self {
            dv,
            method,
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

        // N = translatable mass centers (not atoms) for the V^N partition function factor
        let n0 = c0.num_active_mass_centers() as f64;
        let n1 = c1.num_active_mass_centers() as f64;

        let (v0_new, v1_new, bias, displacement) = match self.method {
            VolumeDisplacementMethod::Logarithmic => {
                // Displace in ln(V₁/V₂) space to avoid sampling issues near V→0.
                // V_tot is conserved by construction: V₁' + V₂' = V_tot.
                let v_tot = v0 + v1;
                let f = ((v0 / v1).ln() + (rng.r#gen::<f64>() - 0.5) * self.dv).exp();
                let v0_new = v_tot / (1.0 / f + 1.0);
                let v1_new = v_tot - v0_new;
                let ln_ratio = (v0_new / v0).ln();
                // (N+1): the +1 comes from the Jacobian d(lnV)/dV = 1/V
                let bias = (n0 + 1.0) * ln_ratio + (n1 + 1.0) * (v1_new / v1).ln();
                (v0_new, v1_new, bias, ln_ratio)
            }
            VolumeDisplacementMethod::Linear => {
                let dv = (rng.r#gen::<f64>() - 0.5) * self.dv;
                let v0_new = v0 + dv;
                let v1_new = v1 - dv;
                // No Jacobian correction needed for linear displacement
                let bias = n0 * (v0_new / v0).ln() + n1 * (v1_new / v1).ln();
                (v0_new, v1_new, bias, dv)
            }
        };

        // Reject unphysical volumes (floor avoids numerical issues with ln)
        if v0_new < 1.0 || v1_new < 1.0 {
            self.statistics.reject();
            return Ok(());
        }

        let old_energy_0 = c0.hamiltonian().energy(c0, &Change::Everything);
        let old_energy_1 = c1.hamiltonian().energy(c1, &Change::Everything);

        Transform::VolumeScale(VolumeScalePolicy::Isotropic, v0_new).on_system_with_backup(c0)?;
        Transform::VolumeScale(VolumeScalePolicy::Isotropic, v1_new).on_system_with_backup(c1)?;

        c0.update_with_backup(&Change::Everything)?;
        c1.update_with_backup(&Change::Everything)?;

        let new_energy_0 = c0.hamiltonian().energy(c0, &Change::Everything);
        let new_energy_1 = c1.hamiltonian().energy(c1, &Change::Everything);

        let du = (new_energy_0 - old_energy_0) + (new_energy_1 - old_energy_1);
        let exponent = -du / thermal_energy + bias;

        accept_or_reject(
            c0,
            c1,
            du,
            exponent,
            crate::propagate::Displacement::Custom(displacement),
            &mut self.statistics,
            rng,
        )
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        let mut map = serde_yaml::Mapping::new();
        map.insert("dV".into(), self.dv.into());
        map.insert("method".into(), serde_yaml::to_value(self.method).ok()?);
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
/// Acceptance follows [Panagiotopoulos Eq. 8](https://doi.org/10.1080/00268978700101491).
#[derive(Debug)]
struct GibbsParticleTransfer {
    molecule_id: usize,
    molecule_name: String, // kept for YAML output without topology access
    statistics: MoveStatistics,
}

impl GibbsParticleTransfer {
    fn new(molecule_id: usize, molecule_name: String) -> Self {
        Self {
            molecule_id,
            molecule_name,
            statistics: MoveStatistics::default(),
        }
    }

    /// Transfer one molecule/atom from `src` to `tgt`.
    fn transfer(
        &mut self,
        src: &mut impl Context,
        tgt: &mut impl Context,
        thermal_energy: f64,
        rng: &mut StdRng,
    ) -> Result<()> {
        let is_atomic = src.topology_ref().moleculekinds()[self.molecule_id].atomic();
        if is_atomic {
            self.transfer_atomic(src, tgt, thermal_energy, rng)
        } else {
            self.transfer_molecular(src, tgt, thermal_energy, rng)
        }
    }

    /// Transfer a molecular group from `src` to `tgt`.
    fn transfer_molecular(
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

        let n_src = src.count_active_molecules(self.molecule_id) as f64;
        let n_tgt = tgt.count_active_molecules(self.molecule_id) as f64;
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
        // thread_rng required by Shape::get_point_inside signature
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

        accept_or_reject(
            src,
            tgt,
            du,
            exponent,
            crate::propagate::Displacement::None,
            &mut self.statistics,
            rng,
        )
    }

    /// Transfer a single atom between atomic mega-groups in `src` and `tgt`.
    fn transfer_atomic(
        &mut self,
        src: &mut impl Context,
        tgt: &mut impl Context,
        thermal_energy: f64,
        rng: &mut StdRng,
    ) -> Result<()> {
        let src_gi = src.group_lists().find_atomic_group(self.molecule_id);
        let tgt_gi = tgt.group_lists().find_atomic_group(self.molecule_id);

        let (Some(src_gi), Some(tgt_gi)) = (src_gi, tgt_gi) else {
            self.statistics.reject();
            return Ok(());
        };

        let n_src = src.groups()[src_gi].len();
        let tgt_group = &tgt.groups()[tgt_gi];
        let n_tgt = tgt_group.len();

        // Reject if source is empty or target is full
        if n_src == 0 || tgt_group.is_full() {
            self.statistics.reject();
            return Ok(());
        }

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

        // Pick random active atom in source, random position in target
        let rel_idx = rng.gen_range(0..n_src);
        let abs_idx = src.groups()[src_gi].to_absolute_index(rel_idx)?;
        let position = random_point_inside(tgt.cell(), rng);

        Transform::Speciation(vec![SpeciationAction::DeactivateAtom {
            group_index: src_gi,
            abs_index: abs_idx,
        }])
        .on_system(src)?;

        Transform::Speciation(vec![SpeciationAction::ActivateAtom {
            group_index: tgt_gi,
            position,
        }])
        .on_system(tgt)?;

        src.update_with_backup(&Change::Everything)?;
        tgt.update_with_backup(&Change::Everything)?;

        let new_e_src = src.hamiltonian().energy(src, &Change::Everything);
        let new_e_tgt = tgt.hamiltonian().energy(tgt, &Change::Everything);

        let du = (new_e_src - old_e_src) + (new_e_tgt - old_e_tgt);
        // Panagiotopoulos Eq. 8: same formula, N counts atoms for atomic groups
        let bias = (v_tgt * n_src as f64 / (v_src * (n_tgt as f64 + 1.0))).ln();
        let exponent = -du / thermal_energy + bias;

        accept_or_reject(
            src,
            tgt,
            du,
            exponent,
            crate::propagate::Displacement::None,
            &mut self.statistics,
            rng,
        )
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
pub(crate) enum GibbsMoveBuilder {
    GibbsVolumeExchange {
        #[serde(alias = "dV")]
        dv: f64,
        #[serde(default)]
        method: VolumeDisplacementMethod,
    },
    GibbsParticleTransfer {
        molecule: String,
    },
}

impl GibbsMoveBuilder {
    /// Build a concrete `GibbsMove` from the deserialized config.
    pub(crate) fn build<T: Context + 'static>(self, context: &T) -> Result<Box<dyn GibbsMove<T>>> {
        Ok(match self {
            Self::GibbsVolumeExchange { dv, method } => {
                anyhow::ensure!(dv > 0.0, "GibbsVolumeExchange: dV must be positive");
                Box::new(GibbsVolumeExchange::new(dv, method))
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
pub(crate) struct GibbsConfig {
    /// Number of intra-box propagation cycles between inter-box moves.
    pub(crate) intra_steps: usize,
    /// Inter-box moves to perform each Gibbs sweep.
    pub(crate) moves: Vec<GibbsMoveBuilder>,
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
    pub(crate) fn new(
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

    fn make_contexts() -> (Backend, Backend) {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/translate_molecules_simulation.yaml",
            None,
            &mut rng,
        )
        .unwrap();
        (context.clone(), context)
    }

    fn assert_volume_conserved(method: VolumeDisplacementMethod) {
        let (mut c0, mut c1) = make_contexts();
        let total_before = c0.cell().volume().unwrap() + c1.cell().volume().unwrap();

        let mut ve = GibbsVolumeExchange::new(5.0, method);
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
    fn volume_exchange_conserves_total_volume_logarithmic() {
        assert_volume_conserved(VolumeDisplacementMethod::Logarithmic);
    }

    #[test]
    fn volume_exchange_conserves_total_volume_linear() {
        assert_volume_conserved(VolumeDisplacementMethod::Linear);
    }

    #[test]
    fn gibbs_move_builder_volume_yaml_default() {
        let yaml = "!GibbsVolumeExchange { dV: 10.0 }";
        let builder: GibbsMoveBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builder, GibbsMoveBuilder::GibbsVolumeExchange { dv, .. } if dv == 10.0));
    }

    #[test]
    fn gibbs_move_builder_volume_yaml_linear() {
        let yaml = "!GibbsVolumeExchange { dV: 10.0, method: Linear }";
        let builder: GibbsMoveBuilder = serde_yaml::from_str(yaml).unwrap();
        assert!(
            matches!(builder, GibbsMoveBuilder::GibbsVolumeExchange { dv, method: VolumeDisplacementMethod::Linear } if dv == 10.0)
        );
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
