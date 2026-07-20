// Copyright 2024 Mikael Lund
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

//! Small crate-internal declarative macros shared by the analysis, energy, and
//! move builders. Each removes a boilerplate pattern that recurred across dozens
//! of sites; keeping them here avoids a per-subsystem copy of the same idea.

/// Build a `yaml_serde::Value::Mapping` from `"key" => value` pairs.
///
/// Values are `.into()`-converted, mirroring the hand-rolled
/// `Mapping::new()` + `insert` + `Value::Mapping` scaffold it replaces.
/// Insertion order is preserved, so serialized output is unchanged.
macro_rules! yaml_map {
    ($($key:expr => $value:expr),* $(,)?) => {{
        let mut map = yaml_serde::Mapping::new();
        $( map.insert($key.into(), $value.into()); )*
        yaml_serde::Value::Mapping(map)
    }};
}

/// Implement [`Info`](crate::Info) with static short/long names and an optional citation.
macro_rules! impl_info {
    ($ty:ty, $short:expr, $long:expr $(,)?) => {
        impl $crate::Info for $ty {
            fn short_name(&self) -> Option<&'static str> {
                Some($short)
            }
            fn long_name(&self) -> Option<&'static str> {
                Some($long)
            }
        }
    };
    ($ty:ty, $short:expr, $long:expr, $citation:expr $(,)?) => {
        impl $crate::Info for $ty {
            fn short_name(&self) -> Option<&'static str> {
                Some($short)
            }
            fn long_name(&self) -> Option<&'static str> {
                Some($long)
            }
            fn citation(&self) -> Option<&'static str> {
                Some($citation)
            }
        }
    };
}

/// Emit the two `Analyze::sampling`/`sampling_mut` accessors for a type whose
/// bookkeeping lives in a field named `sampling`. Invoke inside the `impl
/// Analyze` block; the remaining methods (`perform_sample`, `results`) stay
/// hand-written since they differ per analysis.
macro_rules! impl_sampling_accessors {
    () => {
        fn sampling(&self) -> &$crate::analysis::Sampling {
            &self.sampling
        }
        fn sampling_mut(&mut self) -> &mut $crate::analysis::Sampling {
            &mut self.sampling
        }
    };
}
