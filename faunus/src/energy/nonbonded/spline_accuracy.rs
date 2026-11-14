//! Setup-time check that spline tables reproduce the exact pair potentials.
//!
//! A table can be badly off with no other symptom; total energies of large
//! molecules then drift by many kT at close contact.

use super::{NonbondedMatrix, NonbondedMatrixSplined};
use interatomic::twobody::{IsotropicTwobodyEnergy, SplineStats};

/// Distances sampled per atom-kind pair.
const NUM_SAMPLES: usize = 1000;

/// Absolute error per pair, in kT, that stays small summed over the atom pairs of two large molecules.
const ABSOLUTE_TOLERANCE: f64 = 0.01;

/// Relative error allowed on top, so a steep wall isn't flagged for interpolation noise.
const RELATIVE_TOLERANCE: f64 = 1e-3;

/// Energies beyond this, in kT, are thermally unreachable, so table errors there don't matter.
const REACHABLE_ENERGY: f64 = 15.0;

/// Worst disagreement between a spline table and its exact potential.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SplineDeviation {
    /// Atom kind indices of the pair.
    pub atom_kinds: (usize, usize),
    /// Distance in Å.
    pub distance: f64,
    /// Exact energy, shifted as the table is, in kJ/mol.
    pub exact: f64,
    /// Tabulated energy in kJ/mol.
    pub splined: f64,
    /// Inner edge of the table in Å; below it the table extrapolates linearly.
    pub table_edge: f64,
}

impl SplineDeviation {
    /// Absolute energy error in kJ/mol.
    pub fn error(&self) -> f64 {
        (self.splined - self.exact).abs()
    }
}

/// The sample furthest beyond tolerance over all atom-kind pairs, or `None` if
/// every table is accurate. Samples run from half the table's inner edge (where
/// it extrapolates linearly) to the cutoff; thermally unreachable energies are
/// skipped, so the check needs no notion of contact distance.
pub(crate) fn worst_deviation(
    exact: &NonbondedMatrix,
    splined: &NonbondedMatrixSplined,
    thermal_energy: f64,
) -> Option<SplineDeviation> {
    let energy_cap = REACHABLE_ENERGY * thermal_energy;
    let excess = |d: &SplineDeviation| {
        d.error() / (thermal_energy * ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * d.exact.abs())
    };
    let exact = exact.get_potentials();
    let splined = splined.get_potentials();
    let mut worst: Option<SplineDeviation> = None;
    let mut worst_excess = 1.0;
    for i in 0..exact.order() {
        for j in i..exact.order() {
            let (Some(u), Some(table)) = (exact.get((i, j)), splined.get((i, j))) else {
                continue;
            };
            let stats = table.stats();
            let r_start = 0.5 * stats.r_min;
            for k in 0..NUM_SAMPLES {
                // Offset keeps samples off grid nodes, where the error vanishes.
                let r = r_start + (k as f64 + 0.37) / NUM_SAMPLES as f64 * (stats.r_max - r_start);
                let reference = shifted_reference(u, &stats, r);
                if reference.is_nan() || reference.abs() > energy_cap {
                    continue;
                }
                let candidate = SplineDeviation {
                    atom_kinds: (i, j),
                    distance: r,
                    exact: reference,
                    splined: table.isotropic_twobody_energy(r * r),
                    table_edge: stats.r_min,
                };
                let candidate_excess = excess(&candidate);
                if candidate_excess > worst_excess {
                    worst_excess = candidate_excess;
                    worst = Some(candidate);
                }
            }
        }
    }
    worst
}

/// The exact potential with the shifts the table applies, so a deliberate
/// shift isn't reported as error.
fn shifted_reference(u: &impl IsotropicTwobodyEnergy, stats: &SplineStats, r: f64) -> f64 {
    u.isotropic_twobody_energy(r * r) - stats.energy_shift - (r - stats.r_max) * stats.force_shift
}
