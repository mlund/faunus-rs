// Copyright 2025 Mikael Lund
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

//! External pressure energy term for the NPT ensemble.
//!
//! Implements the isobaric contribution to the partition function:
//! `E = P·V - (N + 1)·kT·ln(V)` in kJ/mol,
//! where N is the number of independently translatable entities.

use crate::cell::Shape;
use crate::change::GroupChange;
use crate::{Change, Context};
use serde::{Deserialize, Serialize};

use super::EnergyTerm;

/// Molar gas constant in kJ/(mol·K). The factor 1e-3 converts from J to kJ.
const R_KJ_PER_MOL_K: f64 = physical_constants::MOLAR_GAS_CONSTANT * 1e-3;

/// Pa → kJ/(mol·Å³). Derives from PV having units J when P is in Pa and V in m³;
/// multiplying by N_A gives kJ/mol·m³, then 1e-30 m³/ų and 1e-3 kJ/J yield the 1e-33.
const PA_TO_INTERNAL: f64 = physical_constants::AVOGADRO_CONSTANT * 1e-33;

/// mM → number density in ų. 1 mM = 1e-3 mol/L = 1e-3·N_A / (1e27 ų/L) particles/ų.
const MILLIMOLAR_TO_ANGSTROM3: f64 = physical_constants::AVOGADRO_CONSTANT * 1e-30;

/// External pressure contribution to the NPT ensemble.
///
/// Energy: `E = P·V - (N + 1)·kT·ln(V)` in kJ/mol,
/// where N is the number of independently translatable entities
/// (individual atoms for single-atom molecule kinds, 1 per molecule
/// for multi-atom molecule kinds).
#[derive(Debug, Clone)]
pub struct ExternalPressure {
    pressure: f64,       // kJ/(mol·Å³)
    thermal_energy: f64, // kT in kJ/mol
}

/// Pressure with unit, serialized as a YAML tag (e.g. `!atm 1.0`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pressure {
    #[serde(rename = "atm")]
    Atm(f64),
    #[serde(rename = "bar")]
    Bar(f64),
    #[serde(rename = "kT")]
    Kt(f64),
    #[serde(rename = "mM")]
    MilliMolar(f64),
    Pa(f64),
}

impl Pressure {
    /// Convert to internal pressure units: kJ/(mol·Å³).
    pub fn to_internal(&self, thermal_energy: f64) -> f64 {
        match self {
            Self::Pa(p) => p * PA_TO_INTERNAL,
            Self::Atm(p) => p * physical_constants::STANDARD_ATMOSPHERE * PA_TO_INTERNAL,
            Self::Bar(p) => p * 1e5 * PA_TO_INTERNAL, // 1 bar = 10⁵ Pa by definition
            Self::Kt(p) => p * thermal_energy,
            Self::MilliMolar(p) => p * MILLIMOLAR_TO_ANGSTROM3 * thermal_energy,
        }
    }
}

impl ExternalPressure {
    /// Create new external pressure energy term.
    pub fn new(pressure: &Pressure, thermal_energy: f64) -> Self {
        let pressure_internal = pressure.to_internal(thermal_energy);
        log::info!(
            "External pressure: {:.6e} kJ/(mol·Å³), kT = {:.4} kJ/mol",
            pressure_internal,
            thermal_energy
        );
        Self {
            pressure: pressure_internal,
            thermal_energy,
        }
    }

    /// Compute thermal energy kT in kJ/mol from temperature in Kelvin.
    pub fn thermal_energy_from_temperature(temperature: f64) -> f64 {
        R_KJ_PER_MOL_K * temperature
    }

    /// Count independently translatable entities N for the NPT partition function.
    ///
    /// Atomic groups contribute each particle separately because they lack
    /// internal structure; molecular groups contribute 1 because the molecule
    /// translates as a rigid unit.
    fn count_entities(context: &impl Context) -> usize {
        let topology = context.topology_ref();
        let mol_kinds = topology.moleculekinds();
        context
            .groups()
            .iter()
            .filter(|g| !g.is_empty())
            .map(|g| {
                if mol_kinds[g.molecule()].len() == 1 {
                    g.len()
                } else {
                    1
                }
            })
            .sum()
    }

    /// Compute the isobaric energy: `P·V - (N+1)·kT·ln(V)`.
    fn compute(&self, context: &impl Context) -> f64 {
        let volume = match context.cell().volume() {
            Some(v) if v > 0.0 => v,
            _ => return 0.0,
        };
        let n = Self::count_entities(context) as f64;
        self.pressure
            .mul_add(volume, -((n + 1.0) * self.thermal_energy * volume.ln()))
    }

    /// Compute energy for a given change.
    pub fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(..) => self.compute(context),
            // N changes when particles are added/removed (GCMC)
            Change::SingleGroup(_, GroupChange::Resize(_)) => self.compute(context),
            Change::Groups(changes) => {
                if changes
                    .iter()
                    .any(|(_, gc)| matches!(gc, GroupChange::Resize(_)))
                {
                    self.compute(context)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

impl From<ExternalPressure> for EnergyTerm {
    fn from(ep: ExternalPressure) -> Self {
        Self::ExternalPressure(ep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn pressure_pascal_conversion() {
        let kt = 2.4789; // kJ/mol at ~298 K
        let p = Pressure::Pa(101325.0);
        let internal = p.to_internal(kt);
        assert_approx_eq!(f64, internal, 101325.0 * PA_TO_INTERNAL, epsilon = 1e-15);
    }

    #[test]
    fn pressure_atm_conversion() {
        let kt = 2.4789;
        let p_pa = Pressure::Pa(101325.0);
        let p_atm = Pressure::Atm(1.0);
        assert_approx_eq!(
            f64,
            p_atm.to_internal(kt),
            p_pa.to_internal(kt),
            epsilon = 1e-15
        );
    }

    #[test]
    fn pressure_bar_conversion() {
        let kt = 2.4789;
        let p_pa = Pressure::Pa(1e5);
        let p_bar = Pressure::Bar(1.0);
        assert_approx_eq!(
            f64,
            p_bar.to_internal(kt),
            p_pa.to_internal(kt),
            epsilon = 1e-15
        );
    }

    #[test]
    fn pressure_kt_conversion() {
        let kt = 2.4789;
        let p = Pressure::Kt(0.5);
        assert_approx_eq!(f64, p.to_internal(kt), 0.5 * kt, epsilon = 1e-15);
    }

    #[test]
    fn pressure_millimolar_conversion() {
        let kt = 2.4789;
        let p = Pressure::MilliMolar(1.0);
        assert_approx_eq!(
            f64,
            p.to_internal(kt),
            MILLIMOLAR_TO_ANGSTROM3 * kt,
            epsilon = 1e-20
        );
    }

    #[test]
    fn pressure_yaml_roundtrip() {
        let yaml = "!atm 1.0";
        let p: Pressure = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(p, Pressure::Atm(v) if (v - 1.0).abs() < f64::EPSILON));

        let yaml = "!bar 2.5";
        let p: Pressure = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(p, Pressure::Bar(v) if (v - 2.5).abs() < f64::EPSILON));

        let yaml = "!Pa 101325.0";
        let p: Pressure = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(p, Pressure::Pa(v) if (v - 101325.0).abs() < f64::EPSILON));

        let yaml = "!kT 0.1";
        let p: Pressure = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(p, Pressure::Kt(v) if (v - 0.1).abs() < f64::EPSILON));

        let yaml = "!mM 10.0";
        let p: Pressure = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(p, Pressure::MilliMolar(v) if (v - 10.0).abs() < f64::EPSILON));
    }

    #[test]
    fn thermal_energy_from_temperature() {
        let kt = ExternalPressure::thermal_energy_from_temperature(298.15);
        assert_approx_eq!(f64, kt, R_KJ_PER_MOL_K * 298.15, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cell::Shape;
    use crate::group::GroupCollection;
    use crate::platform::reference::ReferencePlatform;
    use crate::{WithCell, WithTopology};
    use float_cmp::assert_approx_eq;
    use std::path::Path;

    #[test]
    fn energy_with_reference_platform() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let kt = ExternalPressure::thermal_energy_from_temperature(298.15);
        let pressure = Pressure::Atm(1.0);
        let ep = ExternalPressure::new(&pressure, kt);

        let volume = context.cell().volume().unwrap();
        assert!(volume > 0.0);

        let topology = context.topology_ref();
        let mol_kinds = topology.moleculekinds();
        let n: usize = context
            .groups()
            .iter()
            .filter(|g| !g.is_empty())
            .map(|g| {
                if mol_kinds[g.molecule()].len() == 1 {
                    g.len()
                } else {
                    1
                }
            })
            .sum();

        let expected = ep.pressure * volume - (n as f64 + 1.0) * kt * volume.ln();
        let energy = ep.energy(&context, &Change::Everything);
        assert_approx_eq!(f64, energy, expected, epsilon = 1e-10);

        // No change → 0
        assert_approx_eq!(
            f64,
            ep.energy(&context, &Change::None),
            0.0,
            epsilon = 1e-15
        );
    }
}
