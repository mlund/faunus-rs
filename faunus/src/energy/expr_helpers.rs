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

//! String helpers shared by the math-expression-based energy terms
//! (`custom_external`, `custom_pair`). Private to the `energy` module.

use std::collections::HashMap;

/// Substitute named constants into an expression string using word-boundary matching.
///
/// Only replaces occurrences where the constant name is not part of a longer
/// identifier (e.g. constant `c` won't clobber `cos` or `rc`).
pub(super) fn substitute_constants(expression: &str, constants: &HashMap<String, f64>) -> String {
    let mut sorted: Vec<_> = constants.iter().collect();
    sorted.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));

    let mut result = expression.to_string();
    for (name, value) in sorted {
        result = replace_whole_word(&result, name, &format!("({value:.17})"));
    }
    result
}

/// Replace all whole-word occurrences of `word` in `text` with `replacement`.
///
/// A match is "whole word" when the characters immediately before and after
/// are not alphanumeric or underscore (i.e. not part of an identifier).
pub(super) fn replace_whole_word(text: &str, word: &str, replacement: &str) -> String {
    fn is_ident_char(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(word) {
        let before_ok = pos == 0 || !rest[..pos].ends_with(is_ident_char);
        let end = pos + word.len();
        let after_ok = end >= rest.len() || !rest[end..].starts_with(is_ident_char);
        result.push_str(&rest[..pos]);
        if before_ok && after_ok {
            result.push_str(replacement);
        } else {
            result.push_str(word);
        }
        rest = &rest[end..];
    }
    result.push_str(rest);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_substitution() {
        let mut constants = HashMap::new();
        constants.insert("sigma".to_string(), 2.0);
        constants.insert("sig".to_string(), 999.0);
        let result = substitute_constants("sigma + sig", &constants);
        assert!(!result.contains("sigma"));
        assert!(result.contains("999"));
    }

    #[test]
    fn constant_substitution_word_boundary() {
        // Single-letter constant `c` must not clobber `cos` or `rc`
        let mut constants = HashMap::new();
        constants.insert("c".to_string(), 3.0);
        let result = substitute_constants("c * cos(x) + c", &constants);
        assert!(result.contains("cos"), "cos was clobbered: {result}");
        assert!(
            !result.contains(" c "),
            "standalone c not replaced: {result}"
        );
    }
}
