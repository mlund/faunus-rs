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

use crate::cell::{Shape, VolumeScalePolicy};
use crate::montecarlo::NewOld;
use crate::propagate::{tagged_yaml, Displacement, MoveProposal};
use crate::{Change, Context};
use rand::prelude::*;
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Monte Carlo move for proposing volume changes in the NPT ensemble.
///
/// The volume is sampled logarithmically:
/// `V_new = exp(ln(V_old) + (rand - 0.5) * dV)`
///
/// No move-level bias is needed because the `ExternalPressure` energy term
/// already computes `P*V - (N+1)*kT*ln(V)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeMove {
    /// Volume displacement parameter for logarithmic sampling.
    #[serde(alias = "dV")]
    volume_displacement: f64,
    /// Policy for how to scale the volume (default: Isotropic).
    #[serde(default)]
    method: VolumeScalePolicy,
    /// Move selection weight.
    #[serde(skip_serializing)]
    pub(crate) weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
}

impl crate::Info for VolumeMove {
    fn short_name(&self) -> Option<&'static str> {
        Some("volume")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Volume move (NPT)")
    }
}

impl VolumeMove {
    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        if self.volume_displacement <= 0.0 {
            anyhow::bail!(
                "VolumeMove: volume displacement (dV) must be positive, got {}",
                self.volume_displacement
            );
        }
        let volume = context
            .cell()
            .volume()
            .ok_or_else(|| anyhow::anyhow!("VolumeMove: cell has no defined volume"))?;
        if volume.is_infinite() {
            anyhow::bail!("VolumeMove: cannot use volume move with infinite cell");
        }
        if volume <= 0.0 {
            anyhow::bail!("VolumeMove: cell volume must be positive, got {}", volume);
        }
        Ok(())
    }
}

impl<T: Context> MoveProposal<T> for VolumeMove {
    fn propose_move(
        &mut self,
        context: &mut T,
        rng: &mut dyn RngCore,
    ) -> Option<(Change, Displacement)> {
        let old_volume = context.cell().volume()?;
        if old_volume.is_infinite() {
            return None;
        }

        let ln_new_volume = old_volume.ln() + (rng.r#gen::<f64>() - 0.5) * self.volume_displacement;
        let new_volume = ln_new_volume.exp();

        let old_volume = context
            .scale_volume_and_positions(new_volume, self.method)
            .ok()?;

        let change = Change::Volume(self.method, NewOld::from(new_volume, old_volume));
        let displacement = Displacement::Custom(new_volume - old_volume);
        Some((change, displacement))
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        tagged_yaml("VolumeMove", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_move_yaml_defaults() {
        let yaml = "{ dV: 0.03, weight: 1.0 }";
        let vm: VolumeMove = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(vm.volume_displacement, 0.03);
        assert_eq!(vm.weight, 1.0);
        assert_eq!(vm.method, VolumeScalePolicy::Isotropic);
        assert_eq!(vm.repeat, 1);
    }

    #[test]
    fn test_volume_move_yaml_explicit() {
        let yaml = "{ dV: 0.05, method: ScaleZ, weight: 0.5, repeat: 2 }";
        let vm: VolumeMove = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(vm.volume_displacement, 0.05);
        assert_eq!(vm.weight, 0.5);
        assert_eq!(vm.method, VolumeScalePolicy::ScaleZ);
        assert_eq!(vm.repeat, 2);
    }

    #[test]
    fn test_volume_move_yaml_unknown_field() {
        let yaml = "{ dV: 0.03, weight: 1.0, unknown: 42 }";
        assert!(serde_yaml::from_str::<VolumeMove>(yaml).is_err());
    }
}
