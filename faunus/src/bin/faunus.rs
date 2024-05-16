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

use faunus::{energy::Hamiltonian, *};
use std::rc::Rc;

fn main() {
    let mut _hamiltonian: Hamiltonian = energy::Hamiltonian::new(vec![]);

    let _top = Rc::new(topology::Topology::default());
    //let a = Particle::default();
    //let context = platform::reference::ReferencePlatform::new(cell::Cuboid::cubic(90.0), top);
    //println!("Hello, world! {:?} ", context);
}
