//! Regression tests for deterministic MC simulations.

mod common;

use std::path::{Path, PathBuf};

macro_rules! regression_test {
    ($name:ident) => {
        mod $name {
            use super::*;

            fn dir() -> PathBuf {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(concat!("tests/files/", stringify!($name)))
            }

            #[test]
            #[ignore]
            fn fixtures() {
                common::generate_fixtures(&dir());
            }

            #[test]
            #[ignore]
            fn regression() {
                common::run_regression(&dir());
            }
        }
    };
}

regression_test!(npt_polymers);
regression_test!(npt_water);
