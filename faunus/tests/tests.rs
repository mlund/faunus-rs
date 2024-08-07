// Copyright 2023-2024 Mikael Lund and Ladislav Bartos
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

//! Integration tests for Faunus-rs.

use faunus::{
    montecarlo::MarkovChain, platform::reference::ReferencePlatform, propagate::Propagate,
};

#[test]
fn translate_molecules_simulation() {
    let mut rng = rand::thread_rng();
    let context = ReferencePlatform::new(
        "tests/files/translate_molecules_simulation.yaml",
        None::<String>,
        &mut rng,
    )
    .unwrap();

    let propagate =
        Propagate::from_file("tests/files/translate_molecules_simulation.yaml", &context).unwrap();

    let markov_chain = MarkovChain::new(context, propagate, 1.0);

    for step in markov_chain {
        step.unwrap();
    }

    todo!("Finish the test.")
}
