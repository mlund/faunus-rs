//! GPU-friendly spline data extraction from NonbondedMatrixSplined.

use crate::energy::NonbondedMatrixSplined;
use interatomic::twobody::IsotropicTwobodyEnergy;

/// GPU-aligned spline parameters for a single atom-type pair.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSplineParams {
    pub r_min: f32,
    pub r_max: f32,
    pub n_coeffs: u32,
    /// Offset into the global coefficients buffer
    pub coeff_offset: u32,
    /// Energy at r_min for extrapolation below r_min
    pub u_at_rmin: f32,
    pub _pad: [f32; 3],
}

/// GPU-aligned spline coefficients for one interval.
///
/// Energy coefficients `u` and force coefficients `f` are stored separately
/// so the energy-only shader path can skip loading force data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSplineCoeffs {
    /// Energy polynomial coefficients [u0, u1, u2, u3]
    pub u: [f32; 4],
    /// Force polynomial coefficients [f0, f1, f2, f3]
    pub f: [f32; 4],
}

/// All spline data needed for GPU computation.
pub struct GpuSplineData {
    pub coefficients: Vec<GpuSplineCoeffs>,
    /// Parameters per atom-type pair (n_types × n_types, row-major)
    pub params: Vec<GpuSplineParams>,
    pub n_types: usize,
}

impl GpuSplineData {
    /// Extract GPU spline data from a NonbondedMatrixSplined.
    pub fn from_matrix(matrix: &NonbondedMatrixSplined) -> Self {
        let potentials = matrix.get_potentials();
        let shape = potentials.raw_dim();
        let n_types = shape[0];

        let mut coefficients = Vec::new();
        let mut params = Vec::with_capacity(n_types * n_types);

        for i in 0..n_types {
            for j in 0..n_types {
                let potential = potentials.get((i, j)).expect("Valid index");
                let stats = potential.stats();
                let coeffs = potential.coefficients();

                let coeff_offset = coefficients.len() as u32;
                let n_coeffs = coeffs.len() as u32;

                let u_at_rmin = potential.isotropic_twobody_energy(stats.rsq_min) as f32;

                params.push(GpuSplineParams {
                    r_min: stats.r_min as f32,
                    r_max: stats.r_max as f32,
                    n_coeffs,
                    coeff_offset,
                    u_at_rmin,
                    _pad: [0.0; 3],
                });

                for c in coeffs {
                    coefficients.push(GpuSplineCoeffs {
                        u: [c.u[0] as f32, c.u[1] as f32, c.u[2] as f32, c.u[3] as f32],
                        f: [c.f[0] as f32, c.f[1] as f32, c.f[2] as f32, c.f[3] as f32],
                    });
                }
            }
        }

        Self {
            coefficients,
            params,
            n_types,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_spline_alignment() {
        assert_eq!(std::mem::size_of::<GpuSplineParams>(), 32);
        assert_eq!(std::mem::size_of::<GpuSplineCoeffs>(), 32);
    }
}
