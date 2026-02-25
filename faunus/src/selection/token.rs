// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer for the VMD-like selection language.

use super::SelectionError;

/// Token type for lexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    // Keywords
    Chain,
    Resname,
    Resid,
    Name,
    Element,
    Atomtype,
    Atomid,
    Molecule,
    Protein,
    Backbone,
    Sidechain,
    Nucleic,
    Hydrophobic,
    Aromatic,
    Acidic,
    Basic,
    Polar,
    Charged,
    All,
    None,
    // Boolean ops
    And,
    Or,
    Not,
    // Grouping
    LParen,
    RParen,
    // Range separator
    To,
    // Values
    Ident(String),
    Number(i32),
    Colon,
}

/// Convert identifier string to keyword token or leave as identifier.
fn ident_to_token(ident: String) -> Token {
    match ident.to_lowercase().as_str() {
        "chain" | "segid" => Token::Chain,
        "resname" | "resn" => Token::Resname,
        "resid" | "resi" | "resseq" | "resnum" => Token::Resid,
        "name" | "atomname" => Token::Name,
        "element" | "elem" => Token::Element,
        "atomtype" | "type" => Token::Atomtype,
        "atomid" => Token::Atomid,
        "molecule" => Token::Molecule,
        "protein" => Token::Protein,
        "backbone" => Token::Backbone,
        "sidechain" => Token::Sidechain,
        "nucleic" | "nucleicacid" => Token::Nucleic,
        "hydrophobic" => Token::Hydrophobic,
        "aromatic" => Token::Aromatic,
        "acidic" => Token::Acidic,
        "basic" => Token::Basic,
        "polar" => Token::Polar,
        "charged" => Token::Charged,
        "all" | "everything" => Token::All,
        "none" | "nothing" => Token::None,
        "and" | "&&" => Token::And,
        "or" | "||" => Token::Or,
        "not" | "!" => Token::Not,
        "to" => Token::To,
        _ => Token::Ident(ident),
    }
}

/// Check if character can start an identifier.
const fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '*' | '?' | '[' | '\'')
}

/// Check if character can continue an identifier.
const fn is_ident_char(c: char, in_bracket: bool) -> bool {
    in_bracket || c.is_ascii_alphanumeric() || matches!(c, '_' | '*' | '?' | '[' | ']' | '-' | '\'')
}

/// Tokenize input string.
pub fn tokenize(input: &str) -> Result<Vec<(Token, usize)>, SelectionError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();

    while let Some(&(pos, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        let single = match c {
            '(' => Some(Token::LParen),
            ')' => Some(Token::RParen),
            ':' => Some(Token::Colon),
            _ => None,
        };
        if let Some(token) = single {
            tokens.push((token, pos));
            chars.next();
            continue;
        }

        // Quoted strings
        if c == '"' || c == '\'' {
            let (token, start) = tokenize_quoted(&mut chars, c, pos);
            tokens.push((token, start));
            continue;
        }

        // Numbers (including negative)
        if c.is_ascii_digit()
            || (c == '-'
                && chars
                    .clone()
                    .nth(1)
                    .is_some_and(|(_, ch)| ch.is_ascii_digit()))
        {
            let (token, start) = tokenize_number(&mut chars, pos)?;
            tokens.push((token, start));
            continue;
        }

        // Identifiers and keywords
        if is_ident_start(c) {
            let (token, start) = tokenize_ident(&mut chars, pos);
            tokens.push((token, start));
            continue;
        }

        return Err(SelectionError {
            message: format!("Unexpected character: {c}"),
            position: pos,
        });
    }

    Ok(tokens)
}

/// Tokenize a quoted string.
fn tokenize_quoted(
    chars: &mut std::iter::Peekable<std::str::CharIndices>,
    quote: char,
    pos: usize,
) -> (Token, usize) {
    chars.next(); // consume opening quote
    let start = pos + 1;
    let mut value = String::new();
    while let Some(&(_, ch)) = chars.peek() {
        if ch == quote {
            chars.next();
            break;
        }
        value.push(ch);
        chars.next();
    }
    (Token::Ident(value), start)
}

/// Tokenize a number (including negative).
fn tokenize_number(
    chars: &mut std::iter::Peekable<std::str::CharIndices>,
    pos: usize,
) -> Result<(Token, usize), SelectionError> {
    let start = pos;
    let mut num_str = String::new();

    if chars.peek().is_some_and(|(_, c)| *c == '-') {
        num_str.push('-');
        chars.next();
    }
    while let Some(&(_, ch)) = chars.peek() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            chars.next();
        } else {
            break;
        }
    }
    let num: i32 = num_str.parse().map_err(|_| SelectionError {
        message: format!("Invalid number: {num_str}"),
        position: start,
    })?;
    Ok((Token::Number(num), start))
}

/// Tokenize an identifier or keyword.
fn tokenize_ident(
    chars: &mut std::iter::Peekable<std::str::CharIndices>,
    pos: usize,
) -> (Token, usize) {
    let start = pos;
    let mut ident = String::new();
    let mut in_bracket = false;

    while let Some(&(_, ch)) = chars.peek() {
        if ch == '[' {
            in_bracket = true;
        } else if ch == ']' {
            in_bracket = false;
        }
        if is_ident_char(ch, in_bracket) {
            ident.push(ch);
            chars.next();
        } else {
            break;
        }
    }

    (ident_to_token(ident), start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple() {
        let tokens = tokenize("chain A").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, Token::Chain);
        assert!(matches!(tokens[1].0, Token::Ident(ref s) if s == "A"));
    }

    #[test]
    fn tokenize_boolean() {
        let tokens = tokenize("protein and backbone").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].0, Token::Protein);
        assert_eq!(tokens[1].0, Token::And);
        assert_eq!(tokens[2].0, Token::Backbone);
    }

    #[test]
    fn tokenize_range() {
        let tokens = tokenize("resid 10 to 20").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, Token::Resid);
        assert_eq!(tokens[1].0, Token::Number(10));
        assert_eq!(tokens[2].0, Token::To);
        assert_eq!(tokens[3].0, Token::Number(20));
    }

    #[test]
    fn tokenize_parens() {
        let tokens = tokenize("(chain A or chain B)").unwrap();
        assert_eq!(tokens[0].0, Token::LParen);
        assert_eq!(tokens[tokens.len() - 1].0, Token::RParen);
    }

    #[test]
    fn tokenize_faunus_keywords() {
        let tokens = tokenize("molecule water").unwrap();
        assert_eq!(tokens[0].0, Token::Molecule);

        let tokens = tokenize("atomtype CA").unwrap();
        assert_eq!(tokens[0].0, Token::Atomtype);

        let tokens = tokenize("element C*").unwrap();
        assert_eq!(tokens[0].0, Token::Element);

        let tokens = tokenize("atomid 0 to 5").unwrap();
        assert_eq!(tokens[0].0, Token::Atomid);
    }

    #[test]
    fn tokenize_negative_number() {
        let tokens = tokenize("resid -5 to 5").unwrap();
        assert_eq!(tokens[1].0, Token::Number(-5));
        assert_eq!(tokens[3].0, Token::Number(5));
    }

    #[test]
    fn tokenize_quoted() {
        let tokens = tokenize("name \"CA\"").unwrap();
        assert!(matches!(tokens[1].0, Token::Ident(ref s) if s == "CA"));
    }
}
