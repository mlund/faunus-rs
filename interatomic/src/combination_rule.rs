use num::{Float, NumCast};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Combination rules for mixing epsilon and sigma values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CombinationRule {
    /// Arithmetic mean on both epsilon and sigma.
    #[cfg_attr(feature = "serde", serde(alias = "arithmetic"))]
    Arithmetic,
    /// Geometric mean on both epsilon and sigma.
    #[cfg_attr(feature = "serde", serde(alias = "geometric"))]
    Geometric,
    /// The Lotentz-Berthelot combination rule (geometric mean on epsilon, arithmetic mean on sigma).
    #[cfg_attr(
        feature = "serde",
        serde(alias = "LB", alias = "lorentzberthelot", alias = "lorentz-berthelot")
    )]
    LorentzBerthelot,
    /// The Fender-Halsey combination rule (harmonic mean on epsilon, arithmetic mean on sigma).
    #[cfg_attr(
        feature = "serde",
        serde(alias = "FH", alias = "fenderhalsey", alias = "fender-halsey")
    )]
    FenderHalsey,
}

impl CombinationRule {
    /// Combines epsilon and sigma pairs using the selected combination rule
    pub fn mix(&self, epsilons: (f64, f64), sigmas: (f64, f64)) -> (f64, f64) {
        let epsilon = self.mix_epsilons(epsilons);
        let sigma = self.mix_sigmas(sigmas);
        (epsilon, sigma)
    }

    /// Combine epsilon values using the selected combination rule
    pub fn mix_epsilons(&self, epsilons: (f64, f64)) -> f64 {
        match self {
            Self::LorentzBerthelot | Self::Geometric => geometric_mean(epsilons),
            Self::FenderHalsey => harmonic_mean(epsilons),
            Self::Arithmetic => arithmetic_mean(epsilons),
        }
    }

    /// Combine sigma values using the selected combination rule
    pub fn mix_sigmas(&self, sigmas: (f64, f64)) -> f64 {
        match self {
            Self::Geometric => geometric_mean(sigmas),
            _ => arithmetic_mean(sigmas),
        }
    }
}

/// See Pythagorean means on [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_means)
fn geometric_mean<T: Float>(values: (T, T)) -> T {
    T::sqrt(values.0 * values.1)
}

/// See Pythagorean means on [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_means)
pub(crate) fn arithmetic_mean<T: Float>(values: (T, T)) -> T {
    (values.0 + values.1) * NumCast::from(0.5).unwrap()
}

/// See Pythagorean means on [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_means)
fn harmonic_mean<T: Float>(values: (T, T)) -> T {
    values.0 * values.1 / (values.0 + values.1) * NumCast::from(2.0).unwrap()
}

/// Transform x^2 --> x when serializing
#[cfg(feature = "serde")]
pub fn sqrt_serialize<S>(x: &f64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_f64(x.sqrt())
}

/// Transform x --> x^2 when deserializing
#[cfg(feature = "serde")]
pub fn square_deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(f64::deserialize(deserializer)?.powi(2))
}

/// Transform x --> x/4 when serializing
#[cfg(feature = "serde")]
pub fn divide4_serialize<S>(x: &f64, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_f64(x / 4.0)
}

/// Transform x --> 4x when deserializing
#[cfg(feature = "serde")]
pub fn multiply4_deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(f64::deserialize(deserializer)? * 4.0)
}
