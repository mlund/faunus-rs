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

/// Structure used for (de)serializing the Hamiltonian of the system.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct HamiltonianBuilder {
    /// Nonbonded interactions defined for the system.
    #[serde(rename = "nonbonded")]
    pub pairpot_builder: Option<PairPotentialBuilder>,

    /// Solvent Accessible Surface Area (SASA) energy term.
    pub sasa: Option<SasaEnergyBuilder>,

    /// Contact tessellation energy between rigid bodies.
    pub contact_tessellation: Option<ContactTessellationEnergyBuilder>,

    /// Collective variable constraints (hard or harmonic).
    pub constrain: Option<Vec<ConstrainBuilder>>,

    /// External pressure for the NPT ensemble.
    #[serde(alias = "isobaric")]
    pub pressure: Option<Pressure>,

    /// Custom external potentials from math expressions.
    pub custom_external: Option<Vec<CustomExternalBuilder>>,

    /// User-defined energy/force between two rigid-body centers of mass.
    pub custom_pair: Option<Vec<CustomPairBuilder>>,

    /// Ewald reciprocal-space energy configuration.
    pub ewald: Option<EwaldBuilder>,

    /// Polymer depletion many-body interaction.
    pub polymer_depletion: Option<PolymerDepletionBuilder>,

    /// Tabulated 6D rigid molecule-molecule energy tables.
    pub tabulated6d: Option<Tabulated6DBuilder>,

    /// Tabulated 3D rigid molecule-atom energy tables.
    pub tabulated3d: Option<Tabulated3DBuilder>,

    /// Static flat-histogram bias loaded from a Wang-Landau checkpoint.
    pub penalty: Option<PenaltyBuilder>,
}

impl HamiltonianBuilder {
    /// Get hamiltonian from faunus input file.
    ///
    /// This assumes this YAML layout:
    /// ```yaml
    /// system:
    ///   energy:
    ///     nonbonded:
    ///       ...
    /// ```
    ///
    /// Nonbonded pairs from `include` files are merged in; the input file takes precedence.
    pub(crate) fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let yaml = crate::auxiliary::read_yaml(&path)?;
        let full: serde_yml::Value = serde_yml::from_str(&yaml)?;

        let mut current = &full;
        for key in ["system", "energy"] {
            current = match current.get(key) {
                Some(x) => x,
                None => anyhow::bail!("Could not find `{}` in the YAML file.", key),
            }
        }

        let mut builder: Self = crate::auxiliary::from_section_value("system/energy", current)?;

        // Merge nonbonded from included files (input overrides)
        if let Some(includes) = full.get("include").and_then(|v| v.as_sequence()) {
            let parent_dir = path.as_ref().parent().unwrap_or(Path::new("."));
            for entry in includes {
                if let Some(rel) = entry.as_str() {
                    let inc_path = parent_dir.join(rel);
                    let inc_yaml = crate::auxiliary::read_yaml(&inc_path)?;
                    let inc_full: serde_yml::Value = serde_yml::from_str(&inc_yaml)?;
                    if let Some(energy_val) = inc_full.get("energy") {
                        let inc_builder: Self =
                            crate::auxiliary::from_section_value("energy", energy_val)?;
                        // include files carry `energy:` at top level, so the bare label is correct here
                        if let Some(inc_pairpot) = inc_builder.pairpot_builder {
                            builder
                                .pairpot_builder
                                .get_or_insert_default()
                                .merge_from(inc_pairpot);
                        }
                    }
                }
            }
        }

        Ok(builder)
    }

    /// Parse a Hamiltonian from a YAML string with `system.energy` or `energy` structure.
    ///
    /// Does not support `include` file merging.
    pub fn from_str(yaml: &str) -> anyhow::Result<Self> {
        let full: serde_yml::Value = serde_yml::from_str(yaml)?;
        // Try navigating system.energy first, then fall back to energy
        let (label, current) = if let Some(system) = full.get("system") {
            let energy = system
                .get("energy")
                .ok_or_else(|| anyhow::anyhow!("Could not find `energy` in `system`"))?;
            ("system/energy", energy)
        } else if let Some(energy) = full.get("energy") {
            ("energy", energy)
        } else {
            anyhow::bail!("Could not find `system.energy` or `energy` in the YAML string")
        };
        crate::auxiliary::from_section_value(label, current)
    }

    /// Check that all atom kinds referred to in the pair potentials exist.
    pub(crate) fn validate(&self, atom_kinds: &[AtomKind]) -> anyhow::Result<()> {
        if let Some(pb) = &self.pairpot_builder {
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
}

#[cfg(test)]
mod tests {
    use crate::topology::AtomKindBuilder;
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn hamiltonian_deserialization_pass() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        let pb = builder.pairpot_builder.unwrap();

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
            "in `system/energy` section: invalid entry: found duplicate key"
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

        let pb = builder.pairpot_builder.unwrap();

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
        let pb = builder.pairpot_builder.unwrap();

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
            .pairpot_builder
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
}
