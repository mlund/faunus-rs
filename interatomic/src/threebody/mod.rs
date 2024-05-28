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

//! ## Threebody interactions
//!
//! Module for describing exactly three particles interacting with each other, such as in a torsion.

use dyn_clone::DynClone;

pub mod cosine;
pub mod harmonic;

pub use cosine::CosineTorsion;
pub use harmonic::HarmonicTorsion;

/// Potential energy between three particles as a function of angle between them.
pub trait ThreebodyAngleEnergy: DynClone {
    /// Interaction energy between three particles.
    fn threebody_angle_energy(&self, angle: f64) -> f64;
}

dyn_clone::clone_trait_object!(ThreebodyAngleEnergy);
