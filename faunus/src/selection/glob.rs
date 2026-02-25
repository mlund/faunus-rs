// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! Glob pattern matching for selection expressions.

/// A glob pattern for matching strings (supports `*`, `?`, `[abc]`).
#[derive(Debug, Clone)]
pub struct GlobPattern(String);

impl GlobPattern {
    /// Create a new glob pattern.
    #[must_use]
    pub fn new(pattern: &str) -> Self {
        Self(pattern.to_string())
    }

    /// Test if text matches the pattern.
    #[must_use]
    pub fn matches(&self, text: &str) -> bool {
        glob_match(&self.0, text)
    }
}

/// Match text against a glob pattern (supports `*`, `?`, `[abc]`).
#[allow(clippy::similar_names)]
fn glob_match(pattern: &str, text: &str) -> bool {
    let mut star_pattern_pos: Option<usize> = None;
    let mut star_text_pos: Option<usize> = None;
    let mut pat_idx = 0;
    let mut txt_idx = 0;

    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();

    while txt_idx < text.len() {
        if pat_idx < pattern.len() {
            match pattern[pat_idx] {
                '?' => {
                    pat_idx += 1;
                    txt_idx += 1;
                    continue;
                }
                '*' => {
                    star_pattern_pos = Some(pat_idx);
                    star_text_pos = Some(txt_idx);
                    pat_idx += 1;
                    continue;
                }
                '[' => {
                    if let Some((true, end_idx)) =
                        match_char_class(&pattern, pat_idx, text[txt_idx])
                    {
                        pat_idx = end_idx + 1;
                        txt_idx += 1;
                        continue;
                    }
                }
                c if c == text[txt_idx] => {
                    pat_idx += 1;
                    txt_idx += 1;
                    continue;
                }
                _ => {}
            }
        }

        if let (Some(sp), Some(st)) = (star_pattern_pos, star_text_pos) {
            pat_idx = sp + 1;
            star_text_pos = Some(st + 1);
            txt_idx = st + 1;
            if txt_idx > text.len() {
                return false;
            }
        } else {
            return false;
        }
    }

    while pat_idx < pattern.len() && pattern[pat_idx] == '*' {
        pat_idx += 1;
    }

    pat_idx == pattern.len()
}

/// Match a character class like `[abc]` or `[a-z]`.
/// Returns `(matched, end_index)` where `end_index` is position of `]`.
fn match_char_class(pattern: &[char], start: usize, c: char) -> Option<(bool, usize)> {
    if pattern.get(start) != Some(&'[') {
        return None;
    }

    let mut i = start + 1;
    let mut matched = false;

    while i < pattern.len() && pattern[i] != ']' {
        if i + 2 < pattern.len() && pattern[i + 1] == '-' && pattern[i + 2] != ']' {
            let range_start = pattern[i];
            let range_end = pattern[i + 2];
            if c >= range_start && c <= range_end {
                matched = true;
            }
            i += 3;
        } else {
            if pattern[i] == c {
                matched = true;
            }
            i += 1;
        }
    }

    if i < pattern.len() && pattern[i] == ']' {
        Some((matched, i))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let p = GlobPattern::new("ALA");
        assert!(p.matches("ALA"));
        assert!(!p.matches("ALAB"));
        assert!(!p.matches("ALA "));
        assert!(!p.matches("ala"));
    }

    #[test]
    fn star_wildcard() {
        let p = GlobPattern::new("C*");
        assert!(p.matches("C"));
        assert!(p.matches("CA"));
        assert!(p.matches("CG1"));
        assert!(!p.matches("NC"));
    }

    #[test]
    fn question_wildcard() {
        let p = GlobPattern::new("?A");
        assert!(p.matches("CA"));
        assert!(p.matches("NA"));
        assert!(!p.matches("A"));
        assert!(!p.matches("CAA"));
    }

    #[test]
    fn char_class() {
        let p = GlobPattern::new("[CNO]");
        assert!(p.matches("C"));
        assert!(p.matches("N"));
        assert!(p.matches("O"));
        assert!(!p.matches("S"));
    }

    #[test]
    fn complex_pattern() {
        let p = GlobPattern::new("C[AG]*");
        assert!(p.matches("CA"));
        assert!(p.matches("CG"));
        assert!(p.matches("CG1"));
        assert!(!p.matches("CB"));
    }

    #[test]
    fn star_at_beginning() {
        let p = GlobPattern::new("*A");
        assert!(p.matches("A"));
        assert!(p.matches("CA"));
        assert!(p.matches("CGA"));
        assert!(!p.matches("AB"));
    }

    #[test]
    fn star_in_middle() {
        let p = GlobPattern::new("C*A");
        assert!(p.matches("CA"));
        assert!(p.matches("CGA"));
        assert!(!p.matches("CG"));
    }

    #[test]
    fn char_range() {
        let p = GlobPattern::new("[a-z]");
        assert!(p.matches("a"));
        assert!(p.matches("m"));
        assert!(p.matches("z"));
        assert!(!p.matches("A"));
    }
}
