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

//! # Inteface to the [`chemfiles`] crate
//! 
//! This module implements the [`topology::Atom`] trait for [`chemfiles::Atom`].

use crate::topology;
use chemfiles::{Atom};

impl topology::Identity for Atom {
    fn name(&self) -> String {
        self.name()
    }
    fn id(&self) -> usize {
        unimplemented!("chemfiles::Atom does not have an id")
    }

    fn index(&self) -> usize {
        unimplemented!("chemfiles::Atom does not have an index")
    }

    fn set_index(&mut self, _index: usize) {
        unimplemented!("chemfiles::Atom does not have an index")
    }
}

impl topology::CustomProperty for chemfiles::Atom {
    fn set_property(&mut self, key: &str, value: topology::Value) -> anyhow::Result<()> {
        let property = match value {
            topology::Value::Bool(b) => chemfiles::Property::Bool(b),
            topology::Value::Float(f) => chemfiles::Property::Double(f),
            _ => anyhow::bail!("chemfiles::Atom does not support this property type")
        };
        self.set(key, property);
        Ok(())
    }

    fn get_property(&self, key: &str) -> Option<topology::Value> {
        let property = self.get(key).unwrap();
        match property {
            chemfiles::Property::Bool(b) => Some(topology::Value::Bool(b)),
            chemfiles::Property::Double(f) => Some(topology::Value::Float(f)),
            _ => None
        }
    }
}

impl topology::Atom for Atom {
    fn charge(&self) -> f64 {
        self.charge()
    }
    fn mass(&self) -> f64 {
        self.mass()
    }
    fn epsilon(&self) -> f64 {
        0.0
    }
    fn element(&self) -> Option<String> {
        Some(self.atomic_type())
    }
    fn pos(&self) -> Option<&crate::Point> {
        None
    }
    fn sigma(&self) -> f64 {
        0.0
    }
}
