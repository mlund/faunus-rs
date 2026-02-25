// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! Recursive descent parser for the selection language.

use super::expr::Expr;
use super::glob::GlobPattern;
use super::token::Token;
use super::SelectionError;

/// Parser state.
pub struct Parser<'a> {
    tokens: &'a [(Token, usize)],
    pos: usize,
}

impl<'a> Parser<'a> {
    pub const fn new(tokens: &'a [(Token, usize)]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn current_pos(&self) -> usize {
        self.tokens.get(self.pos).map_or(0, |(_, p)| *p)
    }

    fn advance(&mut self) -> Option<&Token> {
        if self.pos < self.tokens.len() {
            let t = &self.tokens[self.pos].0;
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    /// Parse the full token stream into an expression.
    pub fn parse(&mut self) -> Result<Expr, SelectionError> {
        if self.tokens.is_empty() {
            return Err(SelectionError {
                message: "Empty selection".to_string(),
                position: 0,
            });
        }
        let expr = self.parse_or()?;
        if self.pos < self.tokens.len() {
            return Err(SelectionError {
                message: "Unexpected token after expression".to_string(),
                position: self.current_pos(),
            });
        }
        Ok(expr)
    }

    fn parse_or(&mut self) -> Result<Expr, SelectionError> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, SelectionError> {
        let mut left = self.parse_not()?;
        while self.peek() == Some(&Token::And) {
            self.advance();
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, SelectionError> {
        if self.peek() == Some(&Token::Not) {
            self.advance();
            let inner = self.parse_not()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, SelectionError> {
        let pos = self.current_pos();
        match self.peek() {
            Some(Token::LParen) => self.parse_parenthesized(),
            Some(Token::Chain) => self.parse_pattern_keyword("chain", Expr::Chain),
            Some(Token::Resname) => self.parse_pattern_keyword("resname", Expr::Resname),
            Some(Token::Name) => self.parse_pattern_keyword("name", Expr::Name),
            Some(Token::Element) => self.parse_pattern_keyword("element", Expr::Element),
            Some(Token::Atomtype) => self.parse_pattern_keyword("atomtype", Expr::Atomtype),
            Some(Token::Molecule) => self.parse_pattern_keyword("molecule", Expr::Molecule),
            Some(Token::Resid) => self.parse_range_keyword("resid", Expr::Resid),
            Some(Token::Atomid) => self.parse_range_keyword("atomid", Expr::Atomid),
            Some(Token::Protein) => self.advance_and_ok(Expr::Protein),
            Some(Token::Backbone) => self.advance_and_ok(Expr::Backbone),
            Some(Token::Sidechain) => self.advance_and_ok(Expr::Sidechain),
            Some(Token::Nucleic) => self.advance_and_ok(Expr::Nucleic),
            Some(Token::Hydrophobic) => self.advance_and_ok(Expr::Hydrophobic),
            Some(Token::Aromatic) => self.advance_and_ok(Expr::Aromatic),
            Some(Token::Acidic) => self.advance_and_ok(Expr::Acidic),
            Some(Token::Basic) => self.advance_and_ok(Expr::Basic),
            Some(Token::Polar) => self.advance_and_ok(Expr::Polar),
            Some(Token::Charged) => self.advance_and_ok(Expr::Charged),
            Some(Token::All) => self.advance_and_ok(Expr::All),
            Some(Token::None) => self.advance_and_ok(Expr::None),
            Some(Token::And | Token::Or) => Err(SelectionError {
                message: "Unexpected boolean operator".to_string(),
                position: pos,
            }),
            Some(Token::Ident(s)) => Err(SelectionError {
                message: format!("Unknown keyword: {s}"),
                position: pos,
            }),
            _ => Err(SelectionError {
                message: "Expected selection expression".to_string(),
                position: pos,
            }),
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn advance_and_ok(&mut self, expr: Expr) -> Result<Expr, SelectionError> {
        self.advance();
        Ok(expr)
    }

    fn parse_parenthesized(&mut self) -> Result<Expr, SelectionError> {
        self.advance(); // consume '('
        let inner = self.parse_or()?;
        if self.peek() != Some(&Token::RParen) {
            return Err(SelectionError {
                message: "Missing closing parenthesis".to_string(),
                position: self.current_pos(),
            });
        }
        self.advance();
        Ok(inner)
    }

    fn parse_pattern_keyword(
        &mut self,
        name: &str,
        constructor: fn(Vec<GlobPattern>) -> Expr,
    ) -> Result<Expr, SelectionError> {
        let pos = self.current_pos();
        self.advance();
        let patterns = self.parse_patterns();
        if patterns.is_empty() {
            return Err(SelectionError {
                message: format!("{name} requires at least one argument"),
                position: pos,
            });
        }
        Ok(constructor(patterns))
    }

    fn parse_range_keyword(
        &mut self,
        name: &str,
        constructor: fn(Vec<(i32, i32)>) -> Expr,
    ) -> Result<Expr, SelectionError> {
        let pos = self.current_pos();
        self.advance();
        let ranges = self.parse_ranges()?;
        if ranges.is_empty() {
            return Err(SelectionError {
                message: format!("{name} requires at least one argument"),
                position: pos,
            });
        }
        Ok(constructor(ranges))
    }

    fn parse_patterns(&mut self) -> Vec<GlobPattern> {
        let mut patterns = Vec::new();
        while let Some(Token::Ident(s)) = self.peek() {
            patterns.push(GlobPattern::new(s));
            self.advance();
        }
        patterns
    }

    fn parse_ranges(&mut self) -> Result<Vec<(i32, i32)>, SelectionError> {
        let mut ranges = Vec::new();

        while let Some(token) = self.peek() {
            match token {
                Token::Number(n) => {
                    let start = *n;
                    self.advance();

                    match self.peek() {
                        Some(Token::To | Token::Colon) => {
                            let separator = if self.peek() == Some(&Token::To) {
                                "'to'"
                            } else {
                                "':'"
                            };
                            self.advance();
                            if let Some(Token::Number(end)) = self.peek() {
                                ranges.push((start, *end));
                                self.advance();
                            } else {
                                return Err(SelectionError {
                                    message: format!("Expected number after {separator}"),
                                    position: self.current_pos(),
                                });
                            }
                        }
                        _ => {
                            ranges.push((start, start));
                        }
                    }
                }
                Token::Ident(s) => {
                    if let Some((start, end)) = parse_colon_range(s) {
                        ranges.push((start, end));
                        self.advance();
                    } else if let Ok(n) = s.parse::<i32>() {
                        ranges.push((n, n));
                        self.advance();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        Ok(ranges)
    }
}

/// Parse "N:M" format from a string.
fn parse_colon_range(s: &str) -> Option<(i32, i32)> {
    let (left, right) = s.split_once(':')?;
    let start: i32 = left.parse().ok()?;
    let end: i32 = right.parse().ok()?;
    Some((start, end))
}

#[cfg(test)]
mod tests {
    use super::super::token::tokenize;
    use super::*;

    fn parse(input: &str) -> Result<Expr, SelectionError> {
        let tokens = tokenize(input)?;
        Parser::new(&tokens).parse()
    }

    #[test]
    fn parse_chain() {
        let expr = parse("chain A").unwrap();
        assert!(matches!(expr, Expr::Chain(_)));
    }

    #[test]
    fn parse_resid_range() {
        let expr = parse("resid 10 to 20").unwrap();
        assert!(matches!(expr, Expr::Resid(ref ranges) if ranges == &[(10, 20)]));
    }

    #[test]
    fn parse_resid_colon() {
        let expr = parse("resid 10:20").unwrap();
        assert!(matches!(expr, Expr::Resid(ref ranges) if ranges == &[(10, 20)]));
    }

    #[test]
    fn parse_and_or() {
        let expr = parse("chain A and resname ALA or chain B").unwrap();
        assert!(matches!(expr, Expr::Or(_, _)));
    }

    #[test]
    fn parse_not() {
        let expr = parse("not protein").unwrap();
        assert!(matches!(expr, Expr::Not(_)));
    }

    #[test]
    fn parse_parens() {
        let expr = parse("(chain A or chain B) and protein").unwrap();
        assert!(matches!(expr, Expr::And(_, _)));
    }

    #[test]
    fn parse_molecule() {
        let expr = parse("molecule water").unwrap();
        assert!(matches!(expr, Expr::Molecule(_)));
    }

    #[test]
    fn parse_element() {
        let expr = parse("element C*").unwrap();
        assert!(matches!(expr, Expr::Element(_)));
    }

    #[test]
    fn parse_atomtype() {
        let expr = parse("atomtype CA").unwrap();
        assert!(matches!(expr, Expr::Atomtype(_)));
    }

    #[test]
    fn parse_atomid_range() {
        let expr = parse("atomid 0 to 5").unwrap();
        assert!(matches!(expr, Expr::Atomid(ref ranges) if ranges == &[(0, 5)]));
    }

    #[test]
    fn parse_empty_fails() {
        assert!(parse("").is_err());
    }

    #[test]
    fn parse_missing_arg() {
        assert!(parse("chain").is_err());
        assert!(parse("resname").is_err());
        assert!(parse("resid").is_err());
    }

    #[test]
    fn parse_missing_paren() {
        assert!(parse("(chain A").is_err());
    }

    #[test]
    fn parse_standalone_keywords() {
        assert!(matches!(parse("protein"), Ok(Expr::Protein)));
        assert!(matches!(parse("backbone"), Ok(Expr::Backbone)));
        assert!(matches!(parse("sidechain"), Ok(Expr::Sidechain)));
        assert!(matches!(parse("nucleic"), Ok(Expr::Nucleic)));
        assert!(matches!(parse("all"), Ok(Expr::All)));
        assert!(matches!(parse("none"), Ok(Expr::None)));
        assert!(matches!(parse("hydrophobic"), Ok(Expr::Hydrophobic)));
        assert!(matches!(parse("aromatic"), Ok(Expr::Aromatic)));
        assert!(matches!(parse("acidic"), Ok(Expr::Acidic)));
        assert!(matches!(parse("basic"), Ok(Expr::Basic)));
        assert!(matches!(parse("polar"), Ok(Expr::Polar)));
        assert!(matches!(parse("charged"), Ok(Expr::Charged)));
    }
}
