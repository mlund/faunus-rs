// Copyright 2023 Mikael Lund
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

//! Information such as name, citation, url etc. for concepts

/// Defines information about a concept, like a short name, citation, url etc.
pub trait Info {
    /// Returns a short name for the concept. Use `_` for spaces and avoid weird characters.
    /// This is typically used as keywords in user input and output, e.g. in JSON files.
    fn short_name(&self) -> Option<&'static str> {
        None
    }
    /// Returns a long name for the concept. Spaces are allowed.
    fn long_name(&self) -> Option<&'static str> {
        None
    }

    /// Returns a citation string which should be a
    /// 1. Digital Object Identifier (DOI) in the format `doi:...` (preferred)
    /// 2. URL in the format `https://...`
    fn citation(&self) -> Option<&'static str> {
        None
    }
    /// Tries to extract a URL from the citation string
    fn url(&self) -> Option<String> {
        if self.citation()?.starts_with("doi:") {
            Some(format!(
                "https://doi.org/{}",
                &self.citation().unwrap()[4..]
            ))
        } else if self.citation()?.starts_with("https://")
            || self.citation()?.starts_with("http://")
        {
            Some(self.citation().unwrap().to_string())
        } else {
            None
        }
    }
}
