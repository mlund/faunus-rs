// Copyright 2026 Mikael Lund
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

//! Errors returned by the public simulation interface.

use std::path::PathBuf;

/// Anything that can go wrong while loading, running or saving a simulation.
///
/// The variants separate the failures a caller can act on — a missing file, a malformed
/// input, an unsupported request — from the rest, which arrive as [`Error::Other`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A file could not be read or written.
    #[error("cannot access {path}")]
    Io {
        /// The file that could not be accessed.
        path: PathBuf,
        /// The underlying operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// The input is malformed or describes an inconsistent system.
    #[error("invalid input: {0}")]
    Input(String),
    /// The input is well-formed but asks for something this entry point cannot do.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The simulation started but could not run to completion.
    ///
    /// The cause is kept as a source rather than flattened into a string, so a caller can walk
    /// the chain or downcast to the failure underneath — a full disk, say, versus a bad move.
    #[error("simulation failed")]
    Run(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A failure that has not yet been given a variant of its own.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Result alias for the public simulation interface.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Build an [`Error::Io`] naming the file that failed.
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

// `anyhow` is an internal dependency: boxing keeps the source chain without naming it
// in the public API, so the error type stays stable as internals migrate to `Error`.
impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self::Other(error.into())
    }
}
