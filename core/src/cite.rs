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

/// Defines a citation which can be used to reference the source of a model
pub trait Citation {
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
