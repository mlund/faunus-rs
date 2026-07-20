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

//! # Implementation of the deserialization of the hamiltonian.

use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    marker::PhantomData,
    path::Path,
};

use crate::topology::AtomKind;
use anyhow::Context as AnyhowContext;
#[cfg(test)]
use interatomic::coulomb::permittivity::VACUUM as VACUUM_PERMITTIVITY;
use interatomic::coulomb::{permittivity::RelativePermittivity, DebyeLength};
#[cfg(test)]
use interatomic::twobody::NoInteraction;
use interatomic::{
    twobody::{
        AshbaughHatch, CustomPotential, HardSphere, IonIon, IsotropicTwobodyEnergy, KimHummer,
        LennardJones, WeeksChandlerAndersen,
    },
    CombinationRule,
};
use serde::{Deserialize, Serialize};
use unordered_pair::UnorderedPair;

use super::constrain::ConstrainBuilder;
use super::contact_tessellation::ContactTessellationEnergyBuilder;
use super::custom_external::CustomExternalBuilder;
use super::custom_pair::CustomPairBuilder;
use super::ewald::EwaldBuilder;
use super::external_pressure::Pressure;
use super::penalty::PenaltyBuilder;
use super::polymer_depletion::PolymerDepletionBuilder;
use super::sasa::SasaEnergyBuilder;
use super::tabulated::{Tabulated3DBuilder, Tabulated6DBuilder};
use interatomic::twobody::{GridType, SplineConfig};

/// Bounds required for a coulomb scheme to be used with `IonIon` and `Box<dyn>`.
trait CoulombScheme:
    interatomic::coulomb::pairwise::MultipoleEnergy
    + Clone
    + Debug
    + PartialEq
    + 'static
    + Sync
    + Display
    + Send
{
}
impl<T> CoulombScheme for T where
    T: interatomic::coulomb::pairwise::MultipoleEnergy
        + Clone
        + Debug
        + PartialEq
        + 'static
        + Sync
        + Display
        + Send
{
}

/// Specifies whether the parameters for the interaction are
/// directly provided or should be calculated using a combination rule.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum DirectOrMixing<T: IsotropicTwobodyEnergy> {
    /// Calculate the parameters using the provided combination rule.
    ///
    /// Held in a named struct so `deny_unknown_fields` can reject typos: serde
    /// forbids that attribute on an untagged enum's inline struct variant, but
    /// allows it on a newtype variant's inner struct.
    Mixing(MixingParams<T>),
    /// The parameters for the interaction are specifically provided.
    Direct(T),
}

/// Combination-rule parameters for [`DirectOrMixing::Mixing`].
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MixingParams<T> {
    /// Combination rule to use for mixing.
    mixing: CombinationRule,
    /// Optional cutoff for the interaction.
    cutoff: Option<f64>,
    #[serde(skip)]
    /// Marker specifying the interaction type.
    _phantom: PhantomData<T>,
}

/// Construct a potential from combined atom parameters, factoring out the
/// per-type mixing logic that was previously duplicated across five
/// `PairInteraction::to_boxed` match arms.
pub trait FromMixing: IsotropicTwobodyEnergy + Clone + 'static {
    fn from_mixing(combined: &AtomKind, cutoff: Option<f64>) -> anyhow::Result<Self>;
}

impl FromMixing for KimHummer {
    fn from_mixing(combined: &AtomKind, _cutoff: Option<f64>) -> anyhow::Result<Self> {
        Ok(Self::new(
            combined.epsilon().context("Epsilons not defined!")?,
            combined.sigma().context("Sigmas not defined!")?,
        ))
    }
}

impl FromMixing for LennardJones {
    fn from_mixing(combined: &AtomKind, _cutoff: Option<f64>) -> anyhow::Result<Self> {
        Ok(Self::new(
            combined.epsilon().context("Epsilons not defined!")?,
            combined.sigma().context("Sigmas not defined!")?,
        ))
    }
}

impl FromMixing for WeeksChandlerAndersen {
    fn from_mixing(combined: &AtomKind, _cutoff: Option<f64>) -> anyhow::Result<Self> {
        Ok(Self::new(
            combined.epsilon().context("Epsilons not defined!")?,
            combined.sigma().context("Sigmas not defined!")?,
        ))
    }
}

impl FromMixing for HardSphere {
    fn from_mixing(combined: &AtomKind, _cutoff: Option<f64>) -> anyhow::Result<Self> {
        Ok(Self::new(combined.sigma().context("Sigmas not defined!")?))
    }
}

impl FromMixing for AshbaughHatch {
    fn from_mixing(combined: &AtomKind, cutoff: Option<f64>) -> anyhow::Result<Self> {
        let lj = LennardJones::new(
            combined.epsilon().context("Epsilons not defined!")?,
            combined.sigma().context("Sigmas not defined!")?,
        );
        Ok(Self::new(
            lj,
            combined.lambda().context("No lambda defined!")?,
            cutoff.context("Cutoff undefined!")?,
        ))
    }
}

impl<T: FromMixing> DirectOrMixing<T> {
    /// Resolve to a concrete instance, applying mixing rules if needed.
    fn to_concrete(&self, atom1: &AtomKind, atom2: &AtomKind) -> anyhow::Result<T> {
        match self {
            Self::Direct(inner) => Ok(inner.clone()),
            Self::Mixing(params) => {
                let combined = AtomKind::combine(params.mixing, atom1, atom2);
                T::from_mixing(&combined, params.cutoff)
            }
        }
    }

    /// Convert to a boxed trait object, applying mixing rules if needed.
    fn to_boxed(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        Ok(Box::new(self.to_concrete(atom1, atom2)?))
    }
}

/// Types of pair interactions
// The attribute below is inert: it only bites on struct variants, and every
// variant here is a newtype. Unknown keys are caught by the wrapped potential,
// so a new variant must carry `deny_unknown_fields` on its own type.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub enum PairInteraction {
    /// Ashbaugh-Hatch potential.
    #[serde(alias = "AH")]
    AshbaughHatch(DirectOrMixing<AshbaughHatch>),
    /// Kim-Hummer coarse-grained protein potential.
    #[serde(alias = "KH")]
    KimHummer(DirectOrMixing<KimHummer>),
    /// Lennard-Jones potential.
    LennardJones(DirectOrMixing<LennardJones>),
    /// Weeks-Chandler-Andersen potential.
    #[serde(alias = "WCA")]
    WeeksChandlerAndersen(DirectOrMixing<WeeksChandlerAndersen>),
    /// Hard sphere potential.
    HardSphere(DirectOrMixing<HardSphere>),
    /// Truncated Ewald potential.
    CoulombEwald(interatomic::coulomb::pairwise::EwaldTruncated),
    /// Real-space Ewald potential.
    #[serde(alias = "Ewald")]
    CoulombRealSpaceEwald(interatomic::coulomb::pairwise::RealSpaceEwald),
    /// Plain coulombic potential.
    #[serde(alias = "Coulomb")]
    CoulombPlain(interatomic::coulomb::pairwise::Plain),
    /// Reaction field.
    #[serde(alias = "ReactionField")]
    CoulombReactionField(interatomic::coulomb::pairwise::ReactionField),
    /// Fanourgakis coulomb scheme.
    #[serde(alias = "Fanourgakis")]
    CoulombFanourgakis(interatomic::coulomb::pairwise::Fanourgakis),
    /// Custom pair potential from math expression.
    CustomPotential(Box<CustomPotential>),
}

impl PairInteraction {
    /// True if this variant is an electrostatic interaction.
    pub fn is_coulomb(&self) -> bool {
        matches!(
            self,
            Self::CoulombEwald(_)
                | Self::CoulombRealSpaceEwald(_)
                | Self::CoulombPlain(_)
                | Self::CoulombReactionField(_)
                | Self::CoulombFanourgakis(_)
        )
    }

    /// Convert to a boxed `IsotropicTwobodyEnergy` trait object for a given pair of atom types.
    ///
    /// ## Notes
    /// - A mixing rule is applied, if needed.
    fn to_boxed(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        let mixed = AtomKind::combine(CombinationRule::Arithmetic, atom1, atom2);
        let charge_product = mixed.charge();

        match self {
            Self::KimHummer(x) => x.to_boxed(atom1, atom2),
            Self::LennardJones(x) => x.to_boxed(atom1, atom2),
            Self::WeeksChandlerAndersen(x) => x.to_boxed(atom1, atom2),
            Self::AshbaughHatch(x) => x.to_boxed(atom1, atom2),
            Self::HardSphere(x) => x.to_boxed(atom1, atom2),
            Self::CoulombPlain(scheme) => {
                Self::make_coulomb(charge_product, medium.unwrap(), scheme.clone())
            }
            Self::CoulombEwald(scheme) => {
                Self::make_coulomb(charge_product, medium.unwrap(), scheme.clone())
            }
            Self::CoulombRealSpaceEwald(scheme) => {
                Self::make_coulomb(charge_product, medium.unwrap(), scheme.clone())
            }
            Self::CoulombReactionField(scheme) => {
                Self::make_coulomb(charge_product, medium.unwrap(), scheme.clone())
            }
            Self::CoulombFanourgakis(scheme) => {
                Self::make_coulomb(charge_product, medium.unwrap(), scheme.clone())
            }
            Self::CustomPotential(custom) => Ok(Box::new(custom.as_ref().clone())),
        }
    }
    /// Create an `IonIon<T>` from a scheme and medium, applying permittivity and Debye length.
    fn make_ionion<T: CoulombScheme>(
        charge_product: f64,
        medium: interatomic::coulomb::Medium,
        scheme: T,
    ) -> IonIon<T> {
        let mut ionion = IonIon::new(charge_product, medium.clone().into(), scheme);
        ionion.set_permittivity(medium.permittivity()).unwrap();
        if let Some(e) = medium
            .debye_length()
            .take_if(|_| ionion.debye_length().is_none())
            .and_then(|d| ionion.set_debye_length(Some(d)).err())
        {
            log::warn!(
                "Couldn't copy global medium::debye_length to ion-ion pair potential: {}",
                e
            )
        };
        log::debug!("{}", &ionion);
        ionion
    }

    /// Helper to create a boxed coulombic interaction with a generic scheme.
    fn make_coulomb<T: CoulombScheme>(
        charge_product: f64,
        medium: interatomic::coulomb::Medium,
        scheme: T,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        Ok(Box::new(Self::make_ionion(charge_product, medium, scheme)))
    }

    /// Classify a non-Coulomb interaction into a [`ShortRange`] variant.
    pub(crate) fn to_short_range(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
    ) -> anyhow::Result<super::pairpot::ShortRange> {
        use super::pairpot::ShortRange;
        #[allow(clippy::wildcard_enum_match_arm)] // Coulomb variants are dispatched by to_coulomb()
        match self {
            Self::LennardJones(x) => Ok(ShortRange::LennardJones(x.to_concrete(atom1, atom2)?)),
            Self::WeeksChandlerAndersen(x) => Ok(ShortRange::Wca(x.to_concrete(atom1, atom2)?)),
            Self::AshbaughHatch(x) => Ok(ShortRange::AshbaughHatch(x.to_concrete(atom1, atom2)?)),
            Self::KimHummer(x) => Ok(ShortRange::KimHummer(x.to_concrete(atom1, atom2)?)),
            Self::HardSphere(x) => Ok(ShortRange::HardSphere(x.to_concrete(atom1, atom2)?)),
            Self::CustomPotential(custom) => Ok(ShortRange::Dynamic(
                interatomic::twobody::ArcPotential::new(custom.as_ref().clone()),
            )),
            _ => unreachable!("Coulomb variants should use to_coulomb()"),
        }
    }

    /// Classify a Coulomb interaction into a [`Coulomb`] variant.
    pub(crate) fn to_coulomb(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
        medium: interatomic::coulomb::Medium,
    ) -> anyhow::Result<super::pairpot::Coulomb> {
        use super::pairpot::Coulomb;
        let mixed = AtomKind::combine(CombinationRule::Arithmetic, atom1, atom2);
        let charge_product = mixed.charge();
        #[allow(clippy::wildcard_enum_match_arm)]
        // short-range variants are dispatched by to_short_range()
        match self {
            Self::CoulombPlain(scheme) => Ok(Coulomb::Plain(Self::make_ionion(
                charge_product,
                medium,
                scheme.clone(),
            ))),
            Self::CoulombRealSpaceEwald(scheme) => Ok(Coulomb::RealSpaceEwald(Self::make_ionion(
                charge_product,
                medium,
                scheme.clone(),
            ))),
            Self::CoulombEwald(scheme) => Ok(Coulomb::Ewald(Self::make_ionion(
                charge_product,
                medium,
                scheme.clone(),
            ))),
            Self::CoulombReactionField(scheme) => Ok(Coulomb::ReactionField(Self::make_ionion(
                charge_product,
                medium,
                scheme.clone(),
            ))),
            Self::CoulombFanourgakis(scheme) => Ok(Coulomb::Fanourgakis(Self::make_ionion(
                charge_product,
                medium,
                scheme.clone(),
            ))),
            _ => unreachable!("Non-Coulomb variants should use to_short_range()"),
        }
    }
}

/// Structure storing information about the nonbonded interactions in the system in serializable format.
///
/// Three sections control how interactions are assigned to atom pairs:
/// - `default`: base interactions applied to all pairs
/// - `replace`: pair-specific entries that completely replace `default`
/// - `append`: pair-specific entries merged with `default` by interaction type
///
/// The `spline`, `cutoff` and `bounding_spheres` keys configure how the assembled
/// pair matrix is evaluated (splining and group-to-group culling); they live here
/// rather than at the top level because they configure only this term.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PairPotentialBuilder {
    #[serde(default)]
    default: Vec<PairInteraction>,

    #[serde(default, with = "::serde_with::rust::maps_duplicate_key_is_error")]
    replace: HashMap<UnorderedPair<String>, Vec<PairInteraction>>,

    #[serde(default, with = "::serde_with::rust::maps_duplicate_key_is_error")]
    append: HashMap<UnorderedPair<String>, Vec<PairInteraction>>,

    /// Optional spline configuration for the nonbonded interactions.
    /// When present, `NonbondedMatrixSplined` is used instead of `NonbondedMatrix`.
    #[serde(default)]
    spline: Option<SplineOptions>,

    /// Optional global cutoff (Å) for group-to-group bounding-sphere culling of
    /// the (non-splined) nonbonded matrix. Must be ≥ the largest per-pair
    /// potential cutoff to remain exact. Ignored when `spline` is set (the
    /// spline carries its own cutoff).
    #[serde(default)]
    cutoff: Option<f64>,

    /// Enable bounding-sphere culling for the non-splined matrix (default: true).
    /// Only has an effect together with `cutoff`.
    #[serde(default = "default_bounding_spheres")]
    bounding_spheres: bool,
}

// Manual `Default` so `bounding_spheres` matches its serde default (`true`);
// a derived impl would give `false` and silently disable culling for callers
// that build a `PairPotentialBuilder` without deserializing.
impl Default for PairPotentialBuilder {
    fn default() -> Self {
        Self {
            default: Vec::new(),
            replace: HashMap::new(),
            append: HashMap::new(),
            spline: None,
            cutoff: None,
            bounding_spheres: default_bounding_spheres(),
        }
    }
}

impl PairPotentialBuilder {
    /// Merge pairs from an included file. Default lists are concatenated
    /// (skip duplicate types); `replace`/`append` entries and the evaluation
    /// settings (`spline`/`cutoff`/`bounding_spheres`) from the input take
    /// precedence over includes.
    fn merge_from(&mut self, other: Self) {
        for interaction in other.default {
            let disc = std::mem::discriminant(&interaction);
            if self
                .default
                .iter()
                .any(|d| std::mem::discriminant(d) == disc)
            {
                log::warn!(
                    "Duplicate default nonbonded interaction '{interaction:?}' from include file — skipping"
                );
            } else {
                self.default.push(interaction);
            }
        }
        let merge = |dst: &mut HashMap<_, _>, src: HashMap<_, _>| {
            for (key, value) in src {
                dst.entry(key).or_insert(value);
            }
        };
        merge(&mut self.replace, other.replace);
        merge(&mut self.append, other.append);

        // Adopt the include's evaluation settings only where the input left them
        // unset, so nonbonded config factored into an include is honoured rather
        // than silently dropped. `bounding_spheres` follows the culling `cutoff`
        // it qualifies (a plain bool can't distinguish "unset" from "false").
        if self.spline.is_none() {
            self.spline = other.spline;
        }
        if self.cutoff.is_none() {
            self.cutoff = other.cutoff;
            if self.cutoff.is_some() {
                self.bounding_spheres = other.bounding_spheres;
            }
        }
    }

    /// Append a pair interaction to the `default` list.
    pub(crate) fn push_default(&mut self, interaction: PairInteraction) {
        self.default.push(interaction);
    }

    /// Resolve applicable interactions for an atom pair.
    ///
    /// - If the pair is in `replace`, returns those interactions only.
    /// - If the pair is in `append`, merges with `default` by interaction type
    ///   (same type in append replaces that type from default).
    /// - Otherwise returns `default`.
    fn resolve_interactions(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
        filter: impl Fn(&PairInteraction) -> bool,
    ) -> Vec<&PairInteraction> {
        let key = UnorderedPair(atom1.name().to_owned(), atom2.name().to_owned());

        if let Some(interactions) = self.replace.get(&key) {
            return interactions.iter().filter(|i| filter(i)).collect();
        }

        if let Some(pair_interactions) = self.append.get(&key) {
            let pair_discs: std::collections::HashSet<_> = pair_interactions
                .iter()
                .map(std::mem::discriminant)
                .collect();
            return self
                .default
                .iter()
                .filter(|i| !pair_discs.contains(&std::mem::discriminant(i)))
                .chain(pair_interactions.iter())
                .filter(|i| filter(i))
                .collect();
        }

        self.default.iter().filter(|i| filter(i)).collect()
    }

    /// Collect matching interactions into a summed trait object.
    fn collect_interactions(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
        medium: Option<interatomic::coulomb::Medium>,
        filter: impl Fn(&PairInteraction) -> bool,
    ) -> anyhow::Result<Option<Box<dyn IsotropicTwobodyEnergy>>> {
        let interactions = self.resolve_interactions(atom1, atom2, filter);
        if interactions.is_empty() {
            return Ok(None);
        }
        let total: Box<dyn IsotropicTwobodyEnergy> = interactions
            .into_iter()
            .map(|interact| interact.to_boxed(atom1, atom2, medium.clone()))
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .sum();
        Ok(Some(total))
    }

    /// Get interactions for a specific pair of atoms and collect them into a single `IsotropicTwobodyEnergy` trait object.
    /// If this pair of atoms has no explicitly defined interactions, get interactions for Default.
    /// If Default is not defined or no interactions have been found, return `NoInteraction` structure and log a warning.
    #[cfg(test)]
    pub(crate) fn get_interaction(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        self.collect_interactions(atom1, atom2, medium, |_| true)
            .map(|opt| {
                opt.unwrap_or_else(|| {
                    log::warn!(
                        "No nonbonded interaction defined for '{} <-> {}'.",
                        atom1.name(),
                        atom2.name()
                    );
                    Box::from(NoInteraction)
                })
            })
    }

    /// Get only the Coulomb part of the interaction for a given atom pair.
    /// Needed separately from `get_interaction()` because excluded-pair
    /// corrections must evaluate Coulomb without the short-range component.
    ///
    /// Returns `None` if no Coulomb interaction is configured.
    pub(crate) fn get_coulomb_interaction(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Option<Box<dyn IsotropicTwobodyEnergy>>> {
        self.collect_interactions(atom1, atom2, medium, PairInteraction::is_coulomb)
    }

    /// Build a [`PairPot`] for a given atom pair, classifying short-range and
    /// Coulomb components into enum variants for inline dispatch.
    pub(crate) fn get_pair_pot(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<super::pairpot::PairPot> {
        use super::pairpot::{Coulomb, PairPot, ShortRange};

        let sr_list = self.resolve_interactions(atom1, atom2, |i| !i.is_coulomb());
        let coul_list = self.resolve_interactions(atom1, atom2, PairInteraction::is_coulomb);

        if sr_list.is_empty() && coul_list.is_empty() {
            log::warn!(
                "No nonbonded interaction defined for '{} <-> {}'.",
                atom1.name(),
                atom2.name()
            );
            return Ok(PairPot::default());
        }

        // Classify short-range: single known type → typed variant; else Dynamic
        let short_range = match sr_list.as_slice() {
            [] => ShortRange::None,
            [single] => single.to_short_range(atom1, atom2)?,
            _ => {
                let total: Box<dyn IsotropicTwobodyEnergy> = sr_list
                    .into_iter()
                    .map(|i| i.to_boxed(atom1, atom2, medium.clone()))
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .into_iter()
                    .sum();
                ShortRange::Dynamic(interatomic::twobody::ArcPotential(total.into()))
            }
        };

        // Classify Coulomb: single known type → typed variant; else Dynamic
        let coulomb = match coul_list.as_slice() {
            [] => Coulomb::None,
            [single] => single.to_coulomb(
                atom1,
                atom2,
                medium
                    .clone()
                    .expect("Medium required for Coulomb interactions"),
            )?,
            _ => {
                let total: Box<dyn IsotropicTwobodyEnergy> = coul_list
                    .into_iter()
                    .map(|i| i.to_boxed(atom1, atom2, medium.clone()))
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .into_iter()
                    .sum();
                Coulomb::Dynamic(interatomic::twobody::ArcPotential(total.into()))
            }
        };

        Ok(PairPot::from_parts(short_range, coulomb))
    }

    /// True if any configured interaction is a Coulomb variant.
    pub(crate) fn has_coulomb(&self) -> bool {
        self.default
            .iter()
            .chain(self.replace.values().flatten())
            .chain(self.append.values().flatten())
            .any(|i| i.is_coulomb())
    }

    /// Spline configuration for this nonbonded term, if any.
    pub(crate) fn spline(&self) -> Option<&SplineOptions> {
        self.spline.as_ref()
    }

    /// Group-to-group culling cutoff (Å) for the non-splined matrix, if set.
    pub(crate) fn cutoff(&self) -> Option<f64> {
        self.cutoff
    }

    /// Whether bounding-sphere culling is enabled for the non-splined matrix.
    pub(crate) fn bounding_spheres(&self) -> bool {
        self.bounding_spheres
    }

    /// Override the culling cutoff (test-only; inputs set it via YAML).
    #[cfg(test)]
    pub(crate) fn set_cutoff(&mut self, cutoff: Option<f64>) {
        self.cutoff = cutoff;
    }
}

const fn default_spline_table_points() -> usize {
    2000
}

/// Configuration for splined nonbonded potentials.
///
/// When present in the YAML input, nonbonded interactions will be
/// tabulated using cubic Hermite splines for faster evaluation.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SplineOptions {
    /// Cutoff distance for splined potentials (Ångström).
    pub cutoff: f64,
    /// Number of grid points for the spline table.
    #[serde(default = "default_spline_table_points")]
    pub table_points: usize,
    /// Grid spacing strategy for spline construction.
    #[serde(default)]
    pub grid_type: GridType,
    /// Shift energy to zero at cutoff (default: true).
    #[serde(default = "default_shift_energy")]
    pub shift_energy: bool,
    /// Shift force to zero at cutoff (default: false).
    #[serde(default = "default_shift_force")]
    pub shift_force: bool,
    /// Build a cell list for spatial acceleration (default: true).
    #[serde(default = "default_cell_list")]
    pub cell_list: bool,
    /// Use bounding-sphere culling of distant group pairs (default: true).
    #[serde(default = "default_bounding_spheres")]
    pub bounding_spheres: bool,
}

const fn default_cell_list() -> bool {
    true
}

const fn default_bounding_spheres() -> bool {
    true
}

const fn default_shift_energy() -> bool {
    true
}

const fn default_shift_force() -> bool {
    false
}

impl SplineOptions {
    /// Convert to interatomic's SplineConfig.
    pub fn to_spline_config(&self) -> SplineConfig {
        SplineConfig {
            n_points: self.table_points,
            grid_type: self.grid_type,
            shift_energy: self.shift_energy,
            shift_force: self.shift_force,
            ..Default::default()
        }
    }
}

/// A single entry in the `energy:` input list.
///
/// The `energy:` section is a sequence of externally-tagged entries (`!Nonbonded`,
/// `!Sasa`, …), mirroring `analysis:` and `moves:`. Order is irrelevant — the total
/// energy is a commutative sum — so [`Hamiltonian::new`](super::Hamiltonian) and
/// `finalize` route entries by variant kind (via the accessors on
/// [`HamiltonianBuilder`]), not by position.
///
/// The tag is inert for strictness: an unknown *key inside* an entry is rejected by
/// the wrapped builder's own `deny_unknown_fields`; an unknown *tag* is an
/// unknown-variant error named by
/// [`from_tagged_list`](crate::auxiliary::from_tagged_list).
#[derive(Debug, Clone, Deserialize)]
pub enum EnergyTermBuilder {
    /// Nonbonded pair interactions plus their evaluation config (spline/cutoff/…).
    Nonbonded(PairPotentialBuilder),
    /// Solvent-accessible-surface-area energy. May repeat (e.g. different probes).
    Sasa(SasaEnergyBuilder),
    /// Contact tessellation energy between rigid bodies.
    ContactTessellation(ContactTessellationEnergyBuilder),
    /// Collective-variable constraint (hard or harmonic). May repeat.
    Constrain(ConstrainBuilder),
    /// External pressure for the NPT ensemble. The unit is an inner map key,
    /// e.g. `!Pressure {atm: 1}` — a node cannot carry both `!Pressure` and `!atm`,
    /// so `singleton_map` reads the otherwise tag-valued `Pressure` unit as a map key.
    #[serde(alias = "Isobaric", with = "yaml_serde::with::singleton_map")]
    Pressure(Pressure),
    /// Custom external potential from a math expression. May repeat.
    CustomExternal(CustomExternalBuilder),
    /// User-defined energy/force between two rigid-body centers of mass. May repeat.
    CustomPair(CustomPairBuilder),
    /// Ewald reciprocal-space energy configuration.
    Ewald(EwaldBuilder),
    /// Polymer depletion many-body interaction.
    PolymerDepletion(PolymerDepletionBuilder),
    /// Tabulated 6D rigid molecule-molecule energy tables.
    Tabulated6d(Tabulated6DBuilder),
    /// Tabulated 3D rigid molecule-atom energy tables.
    Tabulated3d(Tabulated3DBuilder),
    /// Static flat-histogram bias loaded from a Wang-Landau checkpoint.
    Penalty(PenaltyBuilder),
}

/// Generate `EnergyTermBuilder::kind` and the by-kind accessors on
/// [`HamiltonianBuilder`] from one table, so each term's uniqueness (`unique` vs
/// `iter`) is stated once and drives both the duplicate policy and the read path.
/// Adding an enum variant without a row here makes `kind`'s `match` non-exhaustive —
/// a compile error — keeping the enum and this table in step. `unique` yields
/// `Option<&T>` (the first match); `iter` yields every match.
macro_rules! energy_dispatch {
    ( $( $variant:ident => $flavor:ident $accessor:ident : $ty:ty ),* $(,)? ) => {
        impl EnergyTermBuilder {
            /// The entry's tag name and whether it may appear more than once. The
            /// name labels duplicate errors; the flag drives `validate_uniqueness`.
            const fn kind(&self) -> (&'static str, bool) {
                match self {
                    $(
                        Self::$variant(_) =>
                            (stringify!($variant), energy_dispatch!(@repeatable $flavor)),
                    )*
                }
            }
        }
        impl HamiltonianBuilder {
            $( energy_dispatch!(@accessor $flavor $accessor $variant $ty); )*
        }
    };
    (@repeatable unique) => { false };
    (@repeatable iter) => { true };
    (@accessor unique $name:ident $variant:ident $ty:ty) => {
        #[allow(clippy::wildcard_enum_match_arm)]
        pub(crate) fn $name(&self) -> Option<&$ty> {
            self.terms.iter().find_map(|t| match t {
                EnergyTermBuilder::$variant(b) => Some(b),
                _ => None,
            })
        }
    };
    (@accessor iter $name:ident $variant:ident $ty:ty) => {
        #[allow(clippy::wildcard_enum_match_arm)]
        pub(crate) fn $name(&self) -> impl Iterator<Item = &$ty> {
            self.terms.iter().filter_map(|t| match t {
                EnergyTermBuilder::$variant(b) => Some(b),
                _ => None,
            })
        }
    };
}

/// Deserialized `energy:` section: an ordered list of energy-term builders.
///
/// A thin newtype over the list; the type name stays stable for the many call sites
/// that thread a `&HamiltonianBuilder`, while [`Hamiltonian`](super::Hamiltonian)
/// reads terms through the by-kind accessors rather than named struct fields.
#[derive(Debug, Clone)]
pub struct HamiltonianBuilder {
    terms: Vec<EnergyTermBuilder>,
}

// One row per term: variant => (unique | iter) accessor : builder type.
energy_dispatch! {
    Nonbonded           => unique nonbonded            : PairPotentialBuilder,
    Sasa                => iter   sasas                : SasaEnergyBuilder,
    ContactTessellation => unique contact_tessellation : ContactTessellationEnergyBuilder,
    Constrain           => iter   constrains           : ConstrainBuilder,
    Pressure            => unique pressure             : Pressure,
    CustomExternal      => iter   custom_externals     : CustomExternalBuilder,
    CustomPair          => iter   custom_pairs         : CustomPairBuilder,
    Ewald               => unique ewald                : EwaldBuilder,
    PolymerDepletion    => unique polymer_depletion    : PolymerDepletionBuilder,
    Tabulated6d         => unique tabulated6d          : Tabulated6DBuilder,
    Tabulated3d         => unique tabulated3d          : Tabulated3DBuilder,
    Penalty             => unique penalty              : PenaltyBuilder,
}

impl HamiltonianBuilder {
    /// Mutable access to the single nonbonded builder, for include-merging.
    #[allow(clippy::wildcard_enum_match_arm)]
    fn nonbonded_mut(&mut self) -> Option<&mut PairPotentialBuilder> {
        self.terms.iter_mut().find_map(|t| match t {
            EnergyTermBuilder::Nonbonded(b) => Some(b),
            _ => None,
        })
    }

    /// Override the nonbonded culling cutoff (test-only; inputs set it via YAML).
    #[cfg(test)]
    pub(crate) fn set_nonbonded_cutoff(&mut self, cutoff: Option<f64>) {
        self.nonbonded_mut()
            .expect("nonbonded term must be present")
            .set_cutoff(cutoff);
    }

    /// Get hamiltonian from faunus input file.
    ///
    /// This assumes this YAML layout:
    /// ```yaml
    /// system:
    ///   energy:
    ///     - !Nonbonded
    ///         default: [...]
    /// ```
    ///
    /// The `!Nonbonded` term from `include` files is merged in; the input file takes precedence.
    pub(crate) fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let yaml = crate::auxiliary::read_yaml(&path)?;
        let full: yaml_serde::Value = yaml_serde::from_str(&yaml)?;

        let energy = full
            .get("system")
            .and_then(|system| system.get("energy"))
            .ok_or_else(|| anyhow::anyhow!("Could not find `system.energy` in the YAML file."))?;

        let mut builder = Self {
            terms: crate::auxiliary::from_tagged_list("system/energy", energy)?,
        };

        // Merge the nonbonded term from included files (input overrides include).
        if let Some(includes) = full.get("include").and_then(|v| v.as_sequence()) {
            let parent_dir = path.as_ref().parent().unwrap_or(Path::new("."));
            for entry in includes {
                let Some(rel) = entry.as_str() else { continue };
                let inc_path = parent_dir.join(rel);
                let inc_yaml = crate::auxiliary::read_yaml(&inc_path)?;
                let inc_full: yaml_serde::Value = yaml_serde::from_str(&inc_yaml)?;
                let Some(inc_energy) = inc_full.get("energy") else {
                    continue;
                };
                // include files carry `energy:` at top level, so the bare label is correct here
                let inc_terms: Vec<EnergyTermBuilder> =
                    crate::auxiliary::from_tagged_list("energy", inc_energy)?;
                #[allow(clippy::wildcard_enum_match_arm)]
                if let Some(inc_nb) = inc_terms.into_iter().find_map(|t| match t {
                    EnergyTermBuilder::Nonbonded(pb) => Some(pb),
                    _ => None,
                }) {
                    match builder.nonbonded_mut() {
                        Some(nb) => nb.merge_from(inc_nb),
                        None => builder.terms.push(EnergyTermBuilder::Nonbonded(inc_nb)),
                    }
                }
            }
        }

        builder.validate_uniqueness()?;
        Ok(builder)
    }

    /// Parse a Hamiltonian from a YAML string with `system.energy` or `energy` structure.
    ///
    /// Does not support `include` file merging.
    pub fn from_str(yaml: &str) -> anyhow::Result<Self> {
        let full: yaml_serde::Value = yaml_serde::from_str(yaml)?;
        // Try navigating system.energy first, then fall back to energy
        let (label, energy) = if let Some(system) = full.get("system") {
            let energy = system
                .get("energy")
                .ok_or_else(|| anyhow::anyhow!("Could not find `energy` in `system`"))?;
            ("system/energy", energy)
        } else if let Some(energy) = full.get("energy") {
            ("energy", energy)
        } else {
            anyhow::bail!("Could not find `system.energy` or `energy` in the YAML string")
        };
        let builder = Self {
            terms: crate::auxiliary::from_tagged_list(label, energy)?,
        };
        builder.validate_uniqueness()?;
        Ok(builder)
    }

    /// Check that every atom kind named in the nonbonded `replace`/`append`
    /// sections exists in the topology.
    ///
    /// The per-term duplicate policy is enforced earlier, at parse time
    /// ([`validate_uniqueness`](Self::validate_uniqueness)), so every construction
    /// path is covered rather than only those that reach this atom-kind check.
    pub(crate) fn validate(&self, atom_kinds: &[AtomKind]) -> anyhow::Result<()> {
        if let Some(pb) = self.nonbonded() {
            for key @ UnorderedPair(x, y) in pb.replace.keys().chain(pb.append.keys()) {
                for name in [x, y] {
                    anyhow::ensure!(
                        atom_kinds.iter().any(|atom| atom.name() == name),
                        "Atom kind '{name}' specified in `nonbonded` does not exist."
                    );
                }
                anyhow::ensure!(
                    !(pb.replace.contains_key(key) && pb.append.contains_key(key)),
                    "Pair [{x}, {y}] cannot appear in both `replace` and `append`."
                );
            }
        }

        Ok(())
    }

    /// Reject a second copy of any energy term that must be unique.
    ///
    /// Called from [`from_file`](Self::from_file)/[`from_str`](Self::from_str) so the
    /// policy holds by construction — a downstream caller (e.g. `Hamiltonian::from_file`)
    /// cannot skip it the way it can skip the atom-kind [`validate`](Self::validate).
    fn validate_uniqueness(&self) -> anyhow::Result<()> {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for term in &self.terms {
            let (name, repeatable) = term.kind();
            if repeatable {
                continue;
            }
            anyhow::ensure!(
                seen.insert(name),
                "energy term `{name}` may appear only once"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::topology::AtomKindBuilder;
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn hamiltonian_deserialization_pass() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        let pb = builder.nonbonded().unwrap();

        assert_eq!(
            pb.default,
            vec![
                PairInteraction::LennardJones(DirectOrMixing::Direct(LennardJones::new(1.5, 6.0))),
                PairInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
                    WeeksChandlerAndersen::new(1.3, 8.0)
                )),
                PairInteraction::CoulombPlain(interatomic::coulomb::pairwise::Plain::new(
                    11.0,
                    Some(1.0),
                ))
            ]
        );

        assert_eq!(pb.replace.len(), 2);

        let ow_ow = pb
            .replace
            .get(&UnorderedPair("OW".into(), "OW".into()))
            .unwrap();
        assert_eq!(
            ow_ow,
            &[
                PairInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
                    WeeksChandlerAndersen::new(1.5, 3.0)
                )),
                PairInteraction::HardSphere(DirectOrMixing::Mixing(MixingParams {
                    mixing: CombinationRule::Geometric,
                    cutoff: None,
                    _phantom: Default::default()
                })),
                PairInteraction::CoulombReactionField(
                    interatomic::coulomb::pairwise::ReactionField::new(11.0, 100.0, 1.5, true)
                ),
            ]
        );

        let ow_hw = pb
            .replace
            .get(&UnorderedPair("OW".into(), "HW".into()))
            .unwrap();
        assert_eq!(
            ow_hw,
            &[
                PairInteraction::HardSphere(DirectOrMixing::Mixing(MixingParams {
                    mixing: CombinationRule::LorentzBerthelot,
                    cutoff: None,
                    _phantom: Default::default()
                })),
                PairInteraction::CoulombEwald(interatomic::coulomb::pairwise::EwaldTruncated::new(
                    11.0, 0.1
                )),
            ]
        );
    }

    #[test]
    fn hamiltonian_deserialization_fail_duplicate() {
        let error =
            HamiltonianBuilder::from_file("tests/files/nonbonded_duplicate.yaml").unwrap_err();
        assert_eq!(
            &error.to_string(),
            "in `system/energy` entry 1 (!Nonbonded): invalid entry: found duplicate key"
        );
    }

    #[test]
    fn hamiltonian_deserialization_fail_duplicate_default() {
        let error = HamiltonianBuilder::from_file("tests/files/nonbonded_duplicate_default.yaml")
            .unwrap_err();
        // `read_yaml` names the offending file; `{:#}` flattens the chain onto the serde reason.
        let message = format!("{error:#}");
        assert!(
            message.contains("nonbonded_duplicate_default.yaml"),
            "{message}"
        );
        assert!(message.contains("duplicate entry with key"), "{message}");
    }

    #[test]
    fn hamiltonian_builder_validate() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        let atom_ow = AtomKindBuilder::default()
            .name("OW")
            .id(0)
            .mass(16.0)
            .charge(1.0)
            .build()
            .unwrap();

        let atom_hw = AtomKindBuilder::default()
            .name("HW")
            .id(1)
            .mass(1.0)
            .charge(0.0)
            .build()
            .unwrap();

        let atoms = [atom_ow.clone(), atom_hw.clone()];
        builder.validate(&atoms).unwrap();

        let atoms = [atom_ow.clone()];
        let error = builder.validate(&atoms).unwrap_err();
        assert_eq!(
            &error.to_string(),
            "Atom kind 'HW' specified in `nonbonded` does not exist."
        );

        let atoms = [atom_hw.clone()];
        let error = builder.validate(&atoms).unwrap_err();
        assert_eq!(
            &error.to_string(),
            "Atom kind 'OW' specified in `nonbonded` does not exist."
        );
    }

    // we can not (easily) test equality of the trait objects so we test the equality of their behavior
    fn assert_behavior(
        obj1: Box<dyn IsotropicTwobodyEnergy>,
        obj2: Box<dyn IsotropicTwobodyEnergy>,
    ) {
        let testing_distances = [0.00201, 0.7, 12.3, 12457.6];

        for &dist in testing_distances.iter() {
            assert_approx_eq!(
                f64,
                obj1.isotropic_twobody_energy(dist),
                obj2.isotropic_twobody_energy(dist)
            );
        }
    }

    // TODO: These tests are commented out as they test a private interface that was
    // subsequently refactored. They should be re-enabled using the public interface
    // once it is stable.

    // #[test]
    // fn test_convert_nonbonded() {
    //     // Lennard Jones -- direct
    //     let expected = LennardJones::new(1.5, 3.2);
    //     let nonbonded =
    //         NonbondedInteraction::LennardJones(DirectOrMixing::Direct(expected.clone()));

    //     let converted = nonbonded.convert(None, None, None, None).unwrap().unwrap();

    //     assert_behavior(converted, Box::new(expected));

    //     // Lennard Jones -- mixing
    //     let expected = LennardJones::new(1.5, 4.5);
    //     let nonbonded = NonbondedInteraction::LennardJones(DirectOrMixing::Mixing {
    //         mixing: CombinationRule::Arithmetic,
    //         _phantom: PhantomData,
    //     });

    //     let converted = nonbonded
    //         .convert(None, Some((2.0, 1.0)), Some((8.2, 0.8)), None)
    //         .unwrap()
    //         .unwrap();

    //     assert_behavior(converted, Box::new(expected));

    //     // Hard Sphere -- mixing
    //     let expected = HardSphere::new(3.0);
    //     let nonbonded = NonbondedInteraction::HardSphere(DirectOrMixing::Mixing {
    //         mixing: CombinationRule::Geometric,
    //         _phantom: PhantomData,
    //     });

    //     let converted = nonbonded
    //         .convert(None, None, Some((4.5, 2.0)), None)
    //         .unwrap()
    //         .unwrap();

    //     assert_behavior(converted, Box::new(expected));

    //     // Coulomb Reaction Field -- charged atoms
    //     let expected = interatomic::coulomb::pairwise::ReactionField::new(11.0, 15.0, 1.5, false);
    //     let nonbonded = NonbondedInteraction::CoulombReactionField(expected.clone());
    //     let charge = (1.0, -1.0);

    //     let converted = nonbonded
    //         .convert(Some(charge), None, None, None)
    //         .unwrap()
    //         .unwrap();

    //     assert_behavior(
    //         converted,
    //         Box::new(IonIon::new(charge.0 * charge.1, expected)),
    //     );

    //     // Coulomb Reaction Field -- uncharged atom => should result in None
    //     let coulomb = interatomic::coulomb::pairwise::ReactionField::new(11.0, 15.0, 1.5, false);
    //     let nonbonded = NonbondedInteraction::CoulombReactionField(coulomb.clone());
    //     let charge = (0.0, -1.0);

    //     assert!(nonbonded
    //         .convert(Some(charge), None, None, None)
    //         .unwrap()
    //         .is_none());
    // }

    #[test]
    fn test_get_interaction() {
        let medium = interatomic::coulomb::Medium::new(
            298.15,
            interatomic::coulomb::permittivity::Permittivity::Vacuum,
            None,
        );

        let interaction1 = PairInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
            WeeksChandlerAndersen::new(1.5, 3.2),
        ));
        let interaction2 =
            PairInteraction::CoulombPlain(interatomic::coulomb::pairwise::Plain::new(11.0, None));

        let interaction3 = PairInteraction::HardSphere(DirectOrMixing::Mixing(MixingParams {
            mixing: CombinationRule::Arithmetic,
            cutoff: None,
            _phantom: PhantomData,
        }));

        let atom1 = AtomKindBuilder::default()
            .name("NA")
            .id(0)
            .mass(12.0)
            .charge(1.0)
            .sigma(1.0)
            .build()
            .unwrap();

        let atom2 = AtomKindBuilder::default()
            .name("CL")
            .id(1)
            .mass(16.0)
            .charge(-1.0)
            .sigma(3.0)
            .build()
            .unwrap();

        let atom3 = AtomKindBuilder::default()
            .name("K")
            .id(2)
            .mass(32.0)
            .charge(0.0)
            .sigma(2.0)
            .build()
            .unwrap();

        let mut nonbonded = PairPotentialBuilder {
            default: vec![interaction1.clone(), interaction2.clone()],
            replace: HashMap::from([(
                UnorderedPair("NA".into(), "CL".into()),
                vec![
                    interaction1.clone(),
                    interaction2.clone(),
                    interaction3.clone(),
                ],
            )]),
            append: HashMap::new(),
            ..Default::default()
        };

        let expected = interaction1.to_boxed(&atom1, &atom2, None).unwrap()
            + interaction2
                .to_boxed(&atom1, &atom2, Some(medium.clone()))
                .unwrap()
            + interaction3
                .to_boxed(&atom1, &atom2, Some(medium.clone()))
                .unwrap();

        let interaction = nonbonded
            .get_interaction(&atom1, &atom2, Some(medium.clone()))
            .unwrap();
        assert_behavior(interaction, expected.clone());

        // changed order of atoms = same result
        let interaction = nonbonded
            .get_interaction(&atom2, &atom1, Some(medium.clone()))
            .unwrap();
        assert_behavior(interaction, expected);

        // default
        let expected = interaction1.to_boxed(&atom2, &atom1, None).unwrap();
        let interaction = nonbonded
            .get_interaction(&atom1, &atom3, Some(medium.clone()))
            .unwrap();
        assert_behavior(interaction, expected);

        // no interaction
        nonbonded.default.clear();
        let expected = Box::<NoInteraction>::default();
        let interaction = nonbonded
            .get_interaction(&atom1, &atom3, Some(medium.clone()))
            .unwrap();
        assert_behavior(interaction, expected);
    }

    #[test]
    fn test_get_interaction_empty() {
        let medium = interatomic::coulomb::Medium::new(
            298.15,
            interatomic::coulomb::permittivity::Permittivity::Vacuum,
            None,
        );

        let plain_coulomb = interatomic::coulomb::pairwise::Plain::new(11.0, None);
        let truncated_ewald = interatomic::coulomb::pairwise::EwaldTruncated::new(11.0, 0.2);
        let hardsphere = HardSphere::from_combination_rule(CombinationRule::Arithmetic, (1.0, 3.0));

        let atom1 = AtomKindBuilder::default()
            .name("NA")
            .id(0)
            .mass(12.0)
            .charge(1.0)
            .sigma(1.0)
            .build()
            .unwrap();

        let atom2 = AtomKindBuilder::default()
            .name("BB")
            .id(1)
            .mass(16.0)
            .charge(0.0)
            .sigma(3.0)
            .build()
            .unwrap();

        // first two interactions evaluate to 0
        let mut nonbonded = PairPotentialBuilder {
            default: Vec::new(),
            replace: HashMap::from([(
                UnorderedPair("NA".into(), "BB".into()),
                vec![
                    PairInteraction::CoulombPlain(plain_coulomb.clone()),
                    PairInteraction::CoulombEwald(truncated_ewald.clone()),
                    PairInteraction::HardSphere(DirectOrMixing::Direct(hardsphere.clone())),
                ],
            )]),
            append: HashMap::new(),
            ..Default::default()
        };

        let expected = Box::new(IonIon::new(0.0, VACUUM_PERMITTIVITY, plain_coulomb.clone()))
            as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(IonIon::new(
                0.0,
                VACUUM_PERMITTIVITY,
                truncated_ewald.clone(),
            )) as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(hardsphere) as Box<dyn IsotropicTwobodyEnergy>;

        let interaction = nonbonded
            .get_interaction(&atom1, &atom2, Some(medium.clone()))
            .unwrap();
        assert_behavior(interaction, expected);

        // all interactions evaluate to 0
        nonbonded.replace.insert(
            UnorderedPair("NA".into(), "BB".into()),
            vec![
                PairInteraction::CoulombPlain(plain_coulomb.clone()),
                PairInteraction::CoulombEwald(truncated_ewald.clone()),
            ],
        );

        let expected = Box::new(IonIon::new(0.0, VACUUM_PERMITTIVITY, plain_coulomb))
            as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(IonIon::new(0.0, VACUUM_PERMITTIVITY, truncated_ewald))
                as Box<dyn IsotropicTwobodyEnergy>;

        let interaction = nonbonded
            .get_interaction(&atom1, &atom2, Some(medium.clone()))
            .unwrap();
        assert_behavior(interaction, expected);
    }

    #[test]
    fn test_kimhummer_deserialization() {
        let builder =
            HamiltonianBuilder::from_file("tests/files/nonbonded_kimhummer.yaml").unwrap();

        let pb = builder.nonbonded().unwrap();

        assert_eq!(
            pb.default,
            vec![PairInteraction::KimHummer(DirectOrMixing::Mixing(
                MixingParams {
                    mixing: CombinationRule::LorentzBerthelot,
                    cutoff: None,
                    _phantom: Default::default()
                }
            ))]
        );

        assert_eq!(pb.replace.len(), 2);

        assert_eq!(
            pb.replace[&UnorderedPair("A".into(), "A".into())],
            vec![PairInteraction::KimHummer(DirectOrMixing::Direct(
                KimHummer::new(-0.5, 6.0)
            ))]
        );

        assert_eq!(
            pb.replace[&UnorderedPair("B".into(), "B".into())],
            vec![PairInteraction::KimHummer(DirectOrMixing::Direct(
                KimHummer::new(0.3, 8.0)
            ))]
        );
    }

    #[test]
    fn test_custom_potential_deserialization() {
        let builder = HamiltonianBuilder::from_file("tests/files/nonbonded_custom.yaml").unwrap();
        let pb = builder.nonbonded().unwrap();

        assert!(!pb.default.is_empty());

        let atom_a = AtomKindBuilder::default()
            .name("A")
            .id(0)
            .mass(1.0)
            .charge(0.0)
            .build()
            .unwrap();

        let boxed = pb.default[0]
            .to_boxed(&atom_a, &atom_a, None)
            .expect("to_boxed should succeed for CustomPotential");
        let energy = boxed.isotropic_twobody_energy(3.4 * 3.4);
        assert_approx_eq!(f64, energy, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_spline_options_deserialization() {
        let builder =
            HamiltonianBuilder::from_file("tests/files/nonbonded_interactions_splined.yaml")
                .unwrap();

        // Check that spline options are present and correctly parsed
        let pb = builder
            .nonbonded()
            .expect("Nonbonded interactions should be present");
        let spline = pb.spline().expect("Spline options should be present");
        assert_approx_eq!(f64, spline.cutoff, 15.0);
        assert_eq!(spline.table_points, 2000);
        assert_eq!(spline.grid_type, GridType::PowerLaw2);

        // Verify conversion to SplineConfig works
        let config = spline.to_spline_config();
        assert_eq!(config.n_points, 2000);
        assert_eq!(config.grid_type, GridType::PowerLaw2);
    }

    #[test]
    fn test_pairpot_merge_from() {
        let pair_aa = UnorderedPair("A".into(), "A".into());
        let pair_ab = UnorderedPair("A".into(), "B".into());

        let interaction1 = vec![PairInteraction::KimHummer(DirectOrMixing::Direct(
            KimHummer::new(-0.5, 6.0),
        ))];
        let interaction2 = vec![PairInteraction::KimHummer(DirectOrMixing::Direct(
            KimHummer::new(0.3, 8.0),
        ))];
        let interaction3 = vec![PairInteraction::KimHummer(DirectOrMixing::Direct(
            KimHummer::new(0.1, 5.0),
        ))];

        let mut base = PairPotentialBuilder {
            replace: HashMap::from([(pair_aa.clone(), interaction1.clone())]),
            ..Default::default()
        };
        let other = PairPotentialBuilder {
            replace: HashMap::from([
                (pair_aa.clone(), interaction2.clone()),
                (pair_ab.clone(), interaction3.clone()),
            ]),
            ..Default::default()
        };

        base.merge_from(other);

        // existing key kept (input overrides include)
        assert_eq!(base.replace[&pair_aa], interaction1);
        // new key inserted from include
        assert_eq!(base.replace[&pair_ab], interaction3);
        assert_eq!(base.replace.len(), 2);
    }

    #[test]
    fn test_pairpot_merge_from_default() {
        let pair_aa = UnorderedPair("A".into(), "A".into());

        let coulomb =
            PairInteraction::CoulombPlain(interatomic::coulomb::pairwise::Plain::new(40.0, None));
        let lj = PairInteraction::LennardJones(DirectOrMixing::Direct(LennardJones::new(1.0, 3.0)));
        let kh = PairInteraction::KimHummer(DirectOrMixing::Direct(KimHummer::new(0.1, 5.0)));

        let mut base = PairPotentialBuilder {
            default: vec![coulomb.clone()],
            replace: HashMap::from([(pair_aa.clone(), vec![kh.clone()])]),
            ..Default::default()
        };
        // Include has LJ as default — different variant, should be merged
        let other = PairPotentialBuilder {
            default: vec![lj.clone()],
            ..Default::default()
        };

        base.merge_from(other);

        // Different variants are concatenated
        assert_eq!(base.default, vec![coulomb, lj]);
        // Pair-specific unchanged
        assert_eq!(base.replace[&pair_aa], vec![kh]);
    }

    #[test]
    fn test_pairpot_merge_from_default_duplicate_skipped() {
        let kh1 = PairInteraction::KimHummer(DirectOrMixing::Direct(KimHummer::new(-0.5, 6.0)));
        let kh2 = PairInteraction::KimHummer(DirectOrMixing::Direct(KimHummer::new(0.3, 8.0)));

        let mut base = PairPotentialBuilder {
            default: vec![kh1.clone()],
            ..Default::default()
        };
        let other = PairPotentialBuilder {
            default: vec![kh2],
            ..Default::default()
        };

        base.merge_from(other);

        // Same variant from include is skipped
        assert_eq!(base.default, vec![kh1]);
    }

    #[test]
    fn test_pairpot_merge_from_adopts_eval_settings() {
        let spline = SplineOptions {
            cutoff: 12.0,
            table_points: 500,
            grid_type: GridType::default(),
            shift_energy: true,
            shift_force: false,
            cell_list: true,
            bounding_spheres: true,
        };

        // Input left spline/culling unset → adopt the include's, including its
        // `bounding_spheres: false` which rides along with the culling cutoff.
        let mut base = PairPotentialBuilder::default();
        base.merge_from(PairPotentialBuilder {
            spline: Some(spline),
            cutoff: Some(30.0),
            bounding_spheres: false,
            ..Default::default()
        });
        assert_eq!(base.spline().map(|s| s.cutoff), Some(12.0));
        assert_eq!(base.cutoff(), Some(30.0));
        assert!(!base.bounding_spheres());

        // Input set its own cutoff → include's cutoff and bounding_spheres ignored.
        let mut base = PairPotentialBuilder {
            cutoff: Some(50.0),
            ..Default::default()
        };
        base.merge_from(PairPotentialBuilder {
            cutoff: Some(30.0),
            bounding_spheres: false,
            ..Default::default()
        });
        assert_eq!(base.cutoff(), Some(50.0));
        assert!(base.bounding_spheres());
    }

    #[test]
    fn from_str_error_names_the_section() {
        // `energy` is a scalar, not a mapping, so deserialization fails; the
        // section label must survive so the user knows where to look.
        let err = HamiltonianBuilder::from_str("system:\n  energy: 42\n").unwrap_err();
        assert!(
            format!("{err:#}").contains("system/energy"),
            "error should name the section: {err:#}"
        );
    }

    #[test]
    fn repeatable_terms_are_allowed_but_a_second_unique_term_is_rejected() {
        // Two `!Sasa` are independent runtime terms (e.g. different probes) and stand.
        let two_sasa = HamiltonianBuilder::from_str(
            "energy:\n  - !Sasa {probe_radius: 1.4}\n  - !Sasa {probe_radius: 2.0}\n",
        )
        .unwrap();
        assert_eq!(two_sasa.sasas().count(), 2);

        // A second `!Pressure` would double-count; rejected at parse time (so no
        // caller can bypass it), naming the term.
        let err = HamiltonianBuilder::from_str(
            "energy:\n  - !Pressure {atm: 1}\n  - !Pressure {bar: 2}\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("Pressure"), "{err}");
    }

    #[test]
    fn pressure_reads_its_unit_as_a_map_key_under_both_tags() {
        // The unit cannot be a second YAML tag next to `!Pressure`, so it is a map
        // key; `singleton_map` turns `{atm: 1}` back into the tagged `Pressure` enum.
        for yaml in [
            "energy:\n  - !Pressure {atm: 1}\n",
            "energy:\n  - !Isobaric {atm: 1}\n",
        ] {
            let builder = HamiltonianBuilder::from_str(yaml).unwrap();
            assert!(matches!(builder.pressure(), Some(Pressure::Atm(p)) if *p == 1.0));
        }
    }

    #[test]
    fn terms_route_by_kind_regardless_of_list_order() {
        // Order never affects the total energy, so the accessors must find a term
        // wherever it sits in the list.
        for yaml in [
            "energy:\n  - !Nonbonded {default: []}\n  - !Sasa {probe_radius: 1.4}\n",
            "energy:\n  - !Sasa {probe_radius: 1.4}\n  - !Nonbonded {default: []}\n",
        ] {
            let builder = HamiltonianBuilder::from_str(yaml).unwrap();
            assert!(builder.nonbonded().is_some());
            assert_eq!(builder.sasas().count(), 1);
        }
    }

    #[test]
    fn nonbonded_is_merged_from_included_files_with_input_winning() {
        let dir = tempfile::tempdir().unwrap();
        // Include contributes a KimHummer default (a different variant than the parent's).
        std::fs::write(
            dir.path().join("lib.yaml"),
            "energy:\n  - !Nonbonded\n      default:\n        - !KimHummer {mixing: LB}\n",
        )
        .unwrap();
        let input = dir.path().join("input.yaml");
        std::fs::write(
            &input,
            "include: [lib.yaml]\nsystem:\n  energy:\n    - !Nonbonded\n        \
             default:\n          - !Coulomb {cutoff: 10.0}\n",
        )
        .unwrap();

        let builder = HamiltonianBuilder::from_file(&input).unwrap();
        let pb = builder.nonbonded().unwrap();
        // Parent's Coulomb and the include's KimHummer are different variants, so both survive.
        assert_eq!(pb.default.len(), 2);
    }
}
