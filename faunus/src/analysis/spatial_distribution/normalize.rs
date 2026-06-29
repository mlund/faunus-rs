use crate::MOLAR_TO_INV_ANGSTROM3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum OutputScale {
    RelativeBulk,
    Molar,
}

impl OutputScale {
    pub(super) const fn from_bulk_normalize(bulk_normalize: bool) -> Self {
        if bulk_normalize {
            Self::RelativeBulk
        } else {
            Self::Molar
        }
    }

    pub(super) const fn unit_label(self) -> &'static str {
        match self {
            Self::RelativeBulk => "relative_density",
            Self::Molar => "mol/L",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct Normalization {
    bulk_density_observations: f64,
    reference_observations: f64,
}

impl Normalization {
    pub(super) fn observe_reference(
        &mut self,
        weight: f64,
        eligible_targets: usize,
        volume: Option<f64>,
        scale: OutputScale,
    ) -> anyhow::Result<()> {
        self.reference_observations += weight;
        if scale == OutputScale::RelativeBulk {
            let volume = volume.ok_or_else(|| {
                anyhow::anyhow!("SpatialDistribution: bulk normalization requires cell volume")
            })?;
            anyhow::ensure!(
                volume > 0.0,
                "SpatialDistribution: cell volume must be positive"
            );
            self.bulk_density_observations += weight * eligible_targets as f64 / volume;
        }
        Ok(())
    }

    pub(super) const fn reference_observations(&self) -> f64 {
        self.reference_observations
    }

    pub(super) fn normalize_count(&self, count: f64, voxel_volume: f64, scale: OutputScale) -> f64 {
        match scale {
            OutputScale::RelativeBulk => {
                let ideal = voxel_volume * self.bulk_density_observations;
                if ideal > 0.0 {
                    count / ideal
                } else {
                    0.0
                }
            }
            OutputScale::Molar => {
                let ideal = voxel_volume * self.reference_observations;
                if ideal > 0.0 {
                    count / ideal / MOLAR_TO_INV_ANGSTROM3
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn relative_bulk_uniform_gas_gives_one() {
        let mut norm = Normalization::default();
        norm.observe_reference(1.0, 10, Some(100.0), OutputScale::RelativeBulk)
            .unwrap();
        let voxel_volume = 2.0;
        let ideal_count = voxel_volume * 10.0 / 100.0;
        assert_relative_eq!(
            norm.normalize_count(ideal_count, voxel_volume, OutputScale::RelativeBulk),
            1.0
        );
    }

    #[test]
    fn molar_conversion_from_particles_per_angstrom3() {
        let mut norm = Normalization::default();
        norm.observe_reference(1.0, 0, None, OutputScale::Molar)
            .unwrap();
        let concentration = norm.normalize_count(1.0, 1.0, OutputScale::Molar);
        assert_relative_eq!(concentration, 1.0 / MOLAR_TO_INV_ANGSTROM3);
    }
}
