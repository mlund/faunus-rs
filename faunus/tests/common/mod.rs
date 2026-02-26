//! Shared utilities for integration tests.
//!
//! Provides a recursive YAML comparator that walks two `serde_yaml::Value`
//! trees and reports all mismatches with their full dotted path.

use serde_yaml::Value;

/// Compare two YAML values recursively.
///
/// Returns a list of human-readable difference descriptions.
/// Keys listed in `ignored_keys` are skipped entirely (e.g. `"timer"`).
pub fn compare_yaml(
    expected: &Value,
    actual: &Value,
    float_tolerance: f64,
    ignored_keys: &[&str],
) -> Vec<String> {
    let mut diffs = Vec::new();
    compare_recursive(expected, actual, "", float_tolerance, ignored_keys, &mut diffs);
    diffs
}

/// Assert that two YAML values match, panicking with all diffs on failure.
pub fn assert_yaml_eq(
    expected: &Value,
    actual: &Value,
    float_tolerance: f64,
    ignored_keys: &[&str],
) {
    let diffs = compare_yaml(expected, actual, float_tolerance, ignored_keys);
    if !diffs.is_empty() {
        let report = diffs
            .iter()
            .enumerate()
            .map(|(i, d)| format!("  {}. {}", i + 1, d))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "YAML comparison found {} difference(s):\n{}",
            diffs.len(),
            report
        );
    }
}

fn compare_recursive(
    expected: &Value,
    actual: &Value,
    path: &str,
    tol: f64,
    ignored_keys: &[&str],
    diffs: &mut Vec<String>,
) {
    match (expected, actual) {
        (Value::Null, Value::Null) => {}
        (Value::Bool(a), Value::Bool(b)) => {
            if a != b {
                diffs.push(format!("{path}: bool mismatch: expected {a}, got {b}"));
            }
        }
        (Value::String(a), Value::String(b)) => {
            if a != b {
                diffs.push(format!("{path}: string mismatch: expected \"{a}\", got \"{b}\""));
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            compare_numbers(a, b, path, tol, diffs);
        }
        (Value::Sequence(a), Value::Sequence(b)) => {
            if a.len() != b.len() {
                diffs.push(format!(
                    "{path}: sequence length mismatch: expected {}, got {}",
                    a.len(),
                    b.len()
                ));
            } else {
                for (i, (ea, eb)) in a.iter().zip(b.iter()).enumerate() {
                    let child = format!("{path}[{i}]");
                    compare_recursive(ea, eb, &child, tol, ignored_keys, diffs);
                }
            }
        }
        (Value::Mapping(a), Value::Mapping(b)) => {
            compare_mappings(a, b, path, tol, ignored_keys, diffs);
        }
        (Value::Tagged(a), Value::Tagged(b)) => {
            if a.tag != b.tag {
                diffs.push(format!(
                    "{path}: tag mismatch: expected !{}, got !{}",
                    a.tag, b.tag
                ));
            }
            compare_recursive(&a.value, &b.value, path, tol, ignored_keys, diffs);
        }
        _ => {
            diffs.push(format!(
                "{path}: type mismatch: expected {}, got {}",
                type_name(expected),
                type_name(actual)
            ));
        }
    }
}

fn compare_numbers(
    a: &serde_yaml::Number,
    b: &serde_yaml::Number,
    path: &str,
    tol: f64,
    diffs: &mut Vec<String>,
) {
    match (a.as_i64(), b.as_i64()) {
        // Both are integers — exact match
        (Some(ai), Some(bi)) => {
            if ai != bi {
                diffs.push(format!(
                    "{path}: integer mismatch: expected {ai}, got {bi}"
                ));
            }
        }
        _ => {
            // At least one is a float — compare with tolerance
            let af = a.as_f64().unwrap_or(f64::NAN);
            let bf = b.as_f64().unwrap_or(f64::NAN);
            if (af - bf).abs() > tol {
                diffs.push(format!(
                    "{path}: float mismatch: expected {af}, got {bf} (diff = {}, tol = {tol})",
                    (af - bf).abs()
                ));
            }
        }
    }
}

fn child_path(parent: &str, key: &Value) -> String {
    let key = match key {
        Value::String(s) => s.clone(),
        other => format!("{other:?}"),
    };
    if parent.is_empty() {
        key
    } else {
        format!("{parent}.{key}")
    }
}

fn is_ignored(key: &Value, ignored_keys: &[&str]) -> bool {
    matches!(key, Value::String(s) if ignored_keys.contains(&s.as_str()))
}

fn compare_mappings(
    a: &serde_yaml::Mapping,
    b: &serde_yaml::Mapping,
    path: &str,
    tol: f64,
    ignored_keys: &[&str],
    diffs: &mut Vec<String>,
) {
    // Check keys in expected that are missing in actual
    for (k, v) in a {
        if is_ignored(k, ignored_keys) {
            continue;
        }
        let child = child_path(path, k);
        match b.get(k) {
            Some(bv) => compare_recursive(v, bv, &child, tol, ignored_keys, diffs),
            None => diffs.push(format!("{child}: key missing in actual")),
        }
    }

    // Check keys in actual that are missing in expected
    for k in b.keys() {
        if is_ignored(k, ignored_keys) {
            continue;
        }
        if !a.contains_key(k) {
            diffs.push(format!("{}: unexpected key in actual", child_path(path, k)));
        }
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Sequence(_) => "sequence",
        Value::Mapping(_) => "mapping",
        Value::Tagged(_) => "tagged",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_values_produce_no_diffs() {
        let yaml = "a: 1\nb: 2.5\nc: hello";
        let v: Value = serde_yaml::from_str(yaml).unwrap();
        assert!(compare_yaml(&v, &v, 1e-10, &[]).is_empty());
    }

    #[test]
    fn integer_mismatch_reported() {
        let a: Value = serde_yaml::from_str("x: 1").unwrap();
        let b: Value = serde_yaml::from_str("x: 2").unwrap();
        let diffs = compare_yaml(&a, &b, 1e-10, &[]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("integer mismatch"));
    }

    #[test]
    fn float_within_tolerance() {
        let a: Value = serde_yaml::from_str("x: 1.0000000001").unwrap();
        let b: Value = serde_yaml::from_str("x: 1.0000000002").unwrap();
        assert!(compare_yaml(&a, &b, 1e-9, &[]).is_empty());
    }

    #[test]
    fn float_outside_tolerance() {
        let a: Value = serde_yaml::from_str("x: 1.0").unwrap();
        let b: Value = serde_yaml::from_str("x: 2.0").unwrap();
        let diffs = compare_yaml(&a, &b, 1e-10, &[]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("float mismatch"));
    }

    #[test]
    fn ignored_keys_are_skipped() {
        let a: Value = serde_yaml::from_str("x: 1\ntimer: 99").unwrap();
        let b: Value = serde_yaml::from_str("x: 1\ntimer: 0").unwrap();
        assert!(compare_yaml(&a, &b, 1e-10, &["timer"]).is_empty());
    }

    #[test]
    fn missing_key_reported() {
        let a: Value = serde_yaml::from_str("x: 1\ny: 2").unwrap();
        let b: Value = serde_yaml::from_str("x: 1").unwrap();
        let diffs = compare_yaml(&a, &b, 1e-10, &[]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("missing in actual"));
    }

    #[test]
    fn extra_key_reported() {
        let a: Value = serde_yaml::from_str("x: 1").unwrap();
        let b: Value = serde_yaml::from_str("x: 1\ny: 2").unwrap();
        let diffs = compare_yaml(&a, &b, 1e-10, &[]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("unexpected key"));
    }

    #[test]
    fn sequence_length_mismatch() {
        let a: Value = serde_yaml::from_str("x: [1, 2]").unwrap();
        let b: Value = serde_yaml::from_str("x: [1]").unwrap();
        let diffs = compare_yaml(&a, &b, 1e-10, &[]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("sequence length"));
    }

    #[test]
    fn nested_path_reported() {
        let a: Value = serde_yaml::from_str("a:\n  b:\n    c: 1").unwrap();
        let b: Value = serde_yaml::from_str("a:\n  b:\n    c: 2").unwrap();
        let diffs = compare_yaml(&a, &b, 1e-10, &[]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("a.b.c"));
    }

    #[test]
    fn tagged_values_compared() {
        let a: Value = serde_yaml::from_str("!Foo\nx: 1").unwrap();
        let b: Value = serde_yaml::from_str("!Foo\nx: 1").unwrap();
        assert!(compare_yaml(&a, &b, 1e-10, &[]).is_empty());
    }

    #[test]
    fn tagged_mismatch_reported() {
        let a: Value = serde_yaml::from_str("!Foo\nx: 1").unwrap();
        let b: Value = serde_yaml::from_str("!Bar\nx: 1").unwrap();
        let diffs = compare_yaml(&a, &b, 1e-10, &[]);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("tag mismatch"));
    }
}
