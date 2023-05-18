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

use serde::{Deserialize, Serialize, __private::de};
use serde_xml_rs::{from_str, to_string};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
//#[serde(rename = "Info")]
pub struct Info {
    #[serde(rename = "DateGenerated")]
    date_generated: String,
    #[serde(rename = "Reference")]
    reference: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename = "Type")]
pub struct AtomType {
    element: String,
    name: String,
    class: String,
    mass: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ForceField {
    #[serde(rename = "Info")]
    pub info: Info,
    //#[serde(rename = "atom_types")]
    #[serde(rename = "$value")]
    pub atom_types: Vec<AtomType>,
}

#[test]
fn test_info() {
    let src = r#"<Info DateGenerated="2019-11-20" Reference="J. Chem. Theory Comput. 8 (2012)"/>"#;
    let should_be = Info {
        date_generated: "2019-11-20".to_string(),
        reference: "J. Chem. Theory Comput. 8 (2012)".to_string(),
    };
    let item: Info = from_str(src).unwrap();
    assert_eq!(item, should_be);
}

#[test]
fn test_type() {
    let src = r#"<Type element="C" name="opls_138" class="C138" mass="12.01100"/>"#;
    let should_be = AtomType {
        element: "C".to_string(),
        name: "opls_138".to_string(),
        class: "C138".to_string(),
        mass: "12.01100".to_string(),
    };
    let item: AtomType = from_str(src).unwrap();
    assert_eq!(item, should_be);

    let src = r#"<Info>
    <DateGenerated>2019-11-20"</DateGenerated><Reference>J. Chem. Theory Comput. 8 (2012)</Reference></Info>
    <AtomTypes>
    <Type element="C" name="opls_138" class="C138" mass="12.01100"/>
    <Type element="C" name="opls_135" class="C135" mass="12.01100"/>
    </AtomTypes>"#;
    let should_be: Vec<AtomType> = vec![
        AtomType {
            element: "C".to_string(),
            name: "opls_138".to_string(),
            class: "C138".to_string(),
            mass: "12.01100".to_string(),
        },
        AtomType {
            element: "C".to_string(),
            name: "opls_135".to_string(),
            class: "C135".to_string(),
            mass: "12.01100".to_string(),
        },
    ];
    let item: ForceField = from_str(src).unwrap();
    println!("{:?}", item);
    //    assert_eq!(item.atom_types, should_be);
}
