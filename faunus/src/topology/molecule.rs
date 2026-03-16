// Copyright 2023-2024 Mikael Lund
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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use derive_builder::Builder;
use derive_getters::Getters;
use serde::{Deserialize, Serialize};
use unordered_pair::UnorderedPair;

use crate::topology::{Chain, DegreesOfFreedom, Residue, Value};
use crate::Point;
use validator::{Validate, ValidationError};

use super::bond::BondKind;
use super::{Bond, BondGraph, CustomProperty, Dihedral, IndexRange, Indexed, Torsion};

/// Source of molecular structure: atom names, reference positions, and/or bonds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
enum StructureSource {
    /// Path to a structure file (xyz, pdb, etc.)
    File(PathBuf),
    /// Inline list of named positions, e.g. `[{OW: [0,0,0]}, {HW: [1,0,0]}]`
    Inline(Vec<HashMap<String, Point>>),
    /// FASTA sequence with harmonic bond parameters
    Fasta(FastaStructure),
}

/// FASTA sequence with harmonic bond parameters for building linear, flexible peptides.
///
/// Each letter maps to a three-letter atom/residue name (e.g. `A` → `ALA`).
/// Harmonic bonds are automatically generated between consecutive residues.
/// Lowercase `n`/`c` map to `NTR`/`CTR` termini.
///
/// See [IUPAC-IUB 1983 nomenclature](https://doi.org/10.1111/j.1432-1033.1984.tb07877.x).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FastaStructure {
    /// FASTA sequence string, e.g. `"nAGGKc"`
    pub sequence: String,
    /// Harmonic bond force constant (kJ/mol/Å²)
    pub k: f64,
    /// Equilibrium bond distance (Å)
    pub req: f64,
}

/// Description of molecule properties.
///
/// Molecule is a collection of atoms that can (but not do not have to be) connected by bonds.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate, Getters, Builder)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_molecule"))]
#[builder(default)]
pub struct MoleculeKind {
    /// Unique name.
    #[builder(setter(into))]
    name: String,
    /// Unique identifier.
    /// Only defined if the MoleculeKind is inside of Topology.
    #[serde(skip)]
    #[getter(skip)]
    id: usize,
    /// Names of atom kinds forming the molecule.
    #[serde(default)]
    atoms: Vec<String>,
    /// Indices / atom ids of atom kinds forming the molecule.
    /// Populated once the molecule is added to a topology.
    #[serde(skip)]
    atom_indices: Vec<usize>,
    /// Intramolecular bonds between the atoms.
    #[serde(default)]
    #[validate(nested)]
    bonds: Vec<Bond>,
    /// Intramolecular dihedrals.
    #[serde(default)]
    #[validate(nested)]
    dihedrals: Vec<Dihedral>,
    /// Intramolecular torsions.
    #[serde(default)]
    #[validate(nested)]
    torsions: Vec<Torsion>,
    /// Generate an exclusions list from bonds.
    /// Add all atom pairs which are excluded_neighbours or less bonds apart.
    #[serde(default)] // default value is 0
    excluded_neighbours: usize,
    /// List of atom pairs which nonbonded interactions are excluded.
    #[serde(default)]
    exclusions: HashSet<UnorderedPair<usize>>,
    /// Internal degrees of freedom.
    #[serde(default)]
    #[getter(skip)]
    degrees_of_freedom: DegreesOfFreedom,
    /// Names of atoms forming the molecule.
    #[serde(default)]
    atom_names: Vec<Option<String>>,
    /// Residues forming the molecule.
    #[validate(custom(function = "super::Residue::validate"))]
    #[serde(default)]
    residues: Vec<Residue>,
    /// Chains forming the molecule.
    #[validate(custom(function = "super::Chain::validate"))]
    #[serde(default)]
    chains: Vec<Chain>,
    /// Persistent connectivity for bond-walking algorithms (COM, pivot).
    #[serde(skip)]
    #[builder(default)]
    #[getter(skip)]
    bond_graph: BondGraph,
    /// Does it make sense to calculate center of mass for the molecule?
    #[serde(default = "default_true")]
    #[getter(skip)]
    has_com: bool,
    /// GCMC fugacity for insertion/deletion reactions
    #[serde(default)]
    #[getter(skip)]
    activity: Option<f64>,
    /// Single-atom molecules pooled in one expandable group
    #[serde(default)]
    #[getter(skip)]
    atomic: bool,
    /// Opt in to Coulomb correction for excluded pairs.
    /// Needed because the splined nonbonded potential bundles SR + Coulomb,
    /// so exclusions skip both — but charge titration and alchemical moves
    /// require Coulomb between bonded neighbors whose charges can change.
    #[serde(default)]
    #[getter(skip)]
    keep_excluded_coulomb: bool,
    /// Map of custom properties.
    #[serde(default)]
    custom: HashMap<String, Value>,
    /// Source of molecular structure: file path, inline positions, or FASTA sequence.
    #[serde(default)]
    #[builder(setter(strip_option, custom), default)]
    from_structure: Option<StructureSource>,
    /// Reference positions extracted from `from_structure` during finalization.
    #[serde(skip)]
    #[builder(default)]
    #[getter(skip)]
    reference_positions: Vec<Point>,
}

impl MoleculeKindBuilder {
    /// Populate `atoms` and `reference_positions` from structure file.
    ///
    /// # Panics
    /// Panics if the file doesn't exist or is of unknown format.
    pub fn from_structure(&mut self, filename: impl AsRef<Path>) -> &mut Self {
        let data = super::io::read_structure(&filename).unwrap();
        log::debug!("Loaded {} atoms from structure file", data.names.len());
        self.reference_positions = Some(data.positions);
        self.atoms(data.names)
    }
}

const fn default_true() -> bool {
    true
}

impl MoleculeKind {
    pub const fn id(&self) -> usize {
        self.id
    }

    #[allow(dead_code)] // used in tests
    pub(crate) const fn degrees_of_freedom(&self) -> DegreesOfFreedom {
        self.degrees_of_freedom
    }

    pub(crate) fn has_bonded_potentials(&self) -> bool {
        !self.bonds.is_empty() || !self.torsions.is_empty() || !self.dihedrals.is_empty()
    }

    /// Resolve the effective name for atom at relative index `i`.
    ///
    /// Returns the molecule-level override if set, otherwise falls back to the atom kind name.
    pub fn resolved_atom_name<'a>(&'a self, i: usize, atomkinds: &'a [super::AtomKind]) -> &'a str {
        self.atom_names[i]
            .as_deref()
            .unwrap_or_else(|| atomkinds[self.atom_indices[i]].name())
    }

    /// Map a group-relative index to the topology-level index.
    /// Atomic mega-groups have one topology entry shared by all particles.
    pub const fn topology_index(&self, relative_index: usize) -> usize {
        if self.atomic {
            0
        } else {
            relative_index
        }
    }

    pub const fn has_com(&self) -> bool {
        self.has_com
    }

    pub(crate) fn set_has_com(&mut self, has_com: bool) {
        self.has_com = has_com;
    }

    pub const fn activity(&self) -> Option<f64> {
        self.activity
    }

    pub const fn atomic(&self) -> bool {
        self.atomic
    }

    pub const fn keep_excluded_coulomb(&self) -> bool {
        self.keep_excluded_coulomb
    }

    pub const fn bond_graph(&self) -> &BondGraph {
        &self.bond_graph
    }

    /// Reference positions from `from_structure` (file or inline), COM-unshifted.
    pub fn reference_positions(&self) -> &[Point] {
        &self.reference_positions
    }

    /// Expand `from_structure` into atom names, reference positions, and/or bonds.
    ///
    /// - `File`: reads atom names and positions from a structure file.
    /// - `Inline`: extracts atom names and positions from inline entries.
    /// - `Fasta`: expands a FASTA sequence into atom names and harmonic bonds.
    pub(crate) fn expand_structure(&mut self) -> anyhow::Result<()> {
        let source = match self.from_structure.take() {
            Some(s) => s,
            None => return Ok(()),
        };
        match source {
            StructureSource::File(path) => {
                let data = super::io::read_structure(&path)?;
                self.atoms = data.names;
                self.reference_positions = data.positions;
                log::debug!("Set {} atoms from {}", self.atoms.len(), path.display());
            }
            StructureSource::Inline(entries) => {
                for mut entry in entries {
                    anyhow::ensure!(
                        entry.len() == 1,
                        "molecule '{}': each inline position entry must have exactly one atom",
                        self.name
                    );
                    let (name, pos) = entry.drain().next().expect("validated len == 1");
                    self.atoms.push(name);
                    self.reference_positions.push(pos);
                }
                log::debug!("Set {} atoms from inline positions", self.atoms.len());
            }
            StructureSource::Fasta(fasta) => {
                let path = std::path::Path::new(&fasta.sequence);
                let names = if path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("fasta"))
                {
                    log::info!("Reading FASTA sequence from '{}'", path.display());
                    let file_sequence = super::residue::read_fasta_file(path)?;
                    super::residue::fasta_to_residue_names(&file_sequence)?
                } else {
                    super::residue::fasta_to_residue_names(&fasta.sequence)?
                };
                if names.is_empty() {
                    anyhow::bail!("molecule '{}': FASTA sequence is empty", self.name);
                }
                let harmonic = interatomic::twobody::Harmonic::new(fasta.k, fasta.req);
                let bond_kind = BondKind::Harmonic(harmonic);
                for i in 1..names.len() {
                    self.bonds
                        .push(Bond::new([i - 1, i], bond_kind.clone(), Default::default()));
                }
                self.atoms = names.into_iter().map(String::from).collect();
                log::debug!(
                    "FASTA expanded '{}' into {} atoms and {} bonds",
                    self.name,
                    self.atoms.len(),
                    self.bonds.len()
                );
            }
        }
        anyhow::ensure!(
            self.reference_positions.is_empty()
                || self.reference_positions.len() == self.atoms.len(),
            "molecule '{}': {} reference positions but {} atoms",
            self.name,
            self.reference_positions.len(),
            self.atoms.len()
        );
        Ok(())
    }

    /// Set indices of atom types.
    pub(super) fn set_atom_indices(&mut self, indices: Vec<usize>) {
        self.atom_indices = indices;
    }

    /// Set names of all atoms of the molecule to None.
    pub(super) fn empty_atom_names(&mut self) {
        self.atom_names = vec![None; self.atoms.len()]
    }

    /// Set molecule id
    pub(super) const fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    /// Build the bond graph and generate nonbonded exclusions from it.
    ///
    /// Rigid (and RigidAlchemical) molecules exclude all internal pairs since
    /// intramolecular distances are constant and the SR energy offset is
    /// meaningless. For RigidAlchemical, the `ExcludedCoulomb` term adds
    /// back the charge-dependent part when `keep_excluded_coulomb` is set.
    pub(super) fn finalize_bonds(&mut self) {
        self.bond_graph = BondGraph::from_bonds(&self.bonds, self.atoms.len());
        if self.degrees_of_freedom.is_rigid() {
            let n = self.atoms.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    self.exclusions.insert(UnorderedPair(i, j));
                }
            }
        } else {
            self.exclusions
                .extend(self.bond_graph.pairs_within(self.excluded_neighbours));
        }
    }

    /// Get number of atoms in the molecule.
    pub fn len(&self) -> usize {
        self.atom_indices().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atom_indices().is_empty()
    }
}

fn validate_molecule(molecule: &MoleculeKind) -> Result<(), ValidationError> {
    let n_atoms = molecule.atoms.len();

    // bonds must only exist between defined atoms
    if !molecule.bonds.iter().all(|x| x.lower(n_atoms)) {
        return Err(ValidationError::new("").with_message("bond between undefined atoms".into()));
    }

    // torsions must only exist between defined atoms
    if !molecule.torsions.iter().all(|x| x.lower(n_atoms)) {
        return Err(ValidationError::new("").with_message("torsion between undefined atoms".into()));
    }

    // dihedrals must only exist between defined atoms
    if !molecule.dihedrals.iter().all(|x| x.lower(n_atoms)) {
        return Err(
            ValidationError::new("").with_message("dihedral between undefined atoms".into())
        );
    }

    // residues can't contain undefined atoms (empty residues can contain any indices)
    if molecule
        .residues
        .iter()
        .any(|r| !r.is_empty() && r.range().end > n_atoms)
    {
        return Err(
            ValidationError::new("").with_message("residue contains undefined atoms".into())
        );
    }

    // chains can't contain undefined atoms
    if molecule
        .chains
        .iter()
        .any(|c| !c.is_empty() && c.range().end > n_atoms)
    {
        return Err(ValidationError::new("").with_message("chain contains undefined atoms".into()));
    }

    // exclusions can't contain undefined atoms or the same atom twice
    if molecule
        .exclusions
        .iter()
        .map(|e| e.into_ordered_tuple())
        .any(|(i, j)| i == j)
    {
        return Err(ValidationError::new("").with_message("exclusion between the same atom".into()));
    }

    if molecule
        .exclusions
        .iter()
        .map(|e| e.into_ordered_tuple())
        .any(|(i, j)| i >= n_atoms || j >= n_atoms)
    {
        return Err(
            ValidationError::new("").with_message("exclusion between undefined atoms".into())
        );
    }

    // vector of atom names must correspond to the number of atoms (or be empty)
    if molecule.atom_names.len() != n_atoms {
        return Err(ValidationError::new("").with_message(
            "the number of atom names does not match the number of atoms in a molecule".into(),
        ));
    }

    if molecule.atomic {
        if n_atoms != 1 {
            return Err(ValidationError::new("")
                .with_message("atomic molecule must have exactly one atom type".into()));
        }
        if molecule.has_bonded_potentials() {
            return Err(ValidationError::new("")
                .with_message("atomic molecule must not have bonded potentials".into()));
        }
    }

    Ok(())
}

impl CustomProperty for MoleculeKind {
    fn get_property(&self, key: &str) -> Option<Value> {
        self.custom.get(key).cloned()
    }

    fn set_property(&mut self, key: &str, value: Value) -> anyhow::Result<()> {
        self.custom.insert(key.to_string(), value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::bond::{BondKind, BondOrder};

    #[test]
    fn generate_exclusions_n1() {
        let mut molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .atoms("ABCDEFG".chars().map(|c| c.to_string()).collect())
            .atom_indices((0..8).collect())
            .bonds(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [5, 6],
                    [3, 6],
                    [4, 6],
                ]
                .iter()
                .map(|&ij| Bond::new(ij, BondKind::Unspecified, BondOrder::Unspecified))
                .collect(),
            )
            .degrees_of_freedom(DegreesOfFreedom::Free)
            .excluded_neighbours(1)
            .build()
            .unwrap();

        molecule.finalize_bonds();

        assert_eq!(molecule.exclusions.len(), 8);
        assert!(molecule.exclusions.contains(&UnorderedPair(0, 1)));
        assert!(molecule.exclusions.contains(&UnorderedPair(1, 2)));
        assert!(molecule.exclusions.contains(&UnorderedPair(2, 3)));
        assert!(molecule.exclusions.contains(&UnorderedPair(3, 4)));
        assert!(molecule.exclusions.contains(&UnorderedPair(3, 5)));
        assert!(molecule.exclusions.contains(&UnorderedPair(5, 6)));
        assert!(molecule.exclusions.contains(&UnorderedPair(4, 6)));
        assert!(molecule.exclusions.contains(&UnorderedPair(3, 6)));
    }

    #[test]
    fn generate_exclusions_n2() {
        let mut molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .atoms("ABCDEFG".chars().map(|c| c.to_string()).collect())
            .atom_indices((0..8).collect())
            .bonds(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [5, 6],
                    [3, 6],
                    [4, 6],
                ]
                .iter()
                .map(|&ij| Bond::new(ij, BondKind::Unspecified, BondOrder::Unspecified))
                .collect(),
            )
            .degrees_of_freedom(DegreesOfFreedom::Free)
            .excluded_neighbours(2)
            .build()
            .unwrap();

        molecule.finalize_bonds();

        assert_eq!(molecule.exclusions.len(), 14);
        let expected = [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 5],
            [4, 6],
            [5, 6],
        ];

        for pair in expected {
            assert!(molecule
                .exclusions
                .contains(&UnorderedPair(pair[0], pair[1])));
        }
    }

    #[test]
    fn generate_exclusions_n3() {
        let mut molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .atoms("ABCDEFG".chars().map(|c| c.to_string()).collect())
            .atom_indices((0..8).collect())
            .bonds(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [5, 6],
                    [3, 6],
                    [4, 6],
                ]
                .iter()
                .map(|&ij| Bond::new(ij, BondKind::Unspecified, BondOrder::Unspecified))
                .collect(),
            )
            .degrees_of_freedom(DegreesOfFreedom::Free)
            .excluded_neighbours(3)
            .build()
            .unwrap();

        molecule.finalize_bonds();

        assert_eq!(molecule.exclusions.len(), 18);
        let expected = [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 5],
            [4, 6],
            [5, 6],
        ];

        for pair in expected {
            assert!(molecule
                .exclusions
                .contains(&UnorderedPair(pair[0], pair[1])));
        }
    }

    #[test]
    fn rigid_molecule_excludes_all_internal_pairs() {
        let mut molecule = MoleculeKindBuilder::default()
            .name("RIG")
            .atoms(vec!["A".into(), "B".into(), "C".into(), "D".into()])
            .atom_indices(vec![0, 1, 2, 3])
            .degrees_of_freedom(DegreesOfFreedom::Rigid)
            .build()
            .unwrap();

        molecule.finalize_bonds();

        // 4 atoms → C(4,2) = 6 exclusion pairs
        assert_eq!(molecule.exclusions.len(), 6);
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(molecule.exclusions.contains(&UnorderedPair(i, j)));
            }
        }
    }

    #[test]
    fn fasta_expands_atoms_and_bonds() {
        let yaml = r#"
            name: peptide
            from_structure: {sequence: "nAGKc", k: 80.33, req: 3.8}
        "#;
        let mut molecule: MoleculeKind = serde_yaml::from_str(yaml).unwrap();
        molecule.expand_structure().unwrap();

        assert_eq!(molecule.atoms, ["NTR", "ALA", "GLY", "LYS", "CTR"]);
        assert_eq!(molecule.bonds.len(), 4);
        for (i, bond) in molecule.bonds.iter().enumerate() {
            assert_eq!(*bond.index(), [i, i + 1]);
        }
    }

    #[test]
    fn atomic_rejects_multi_atom() {
        let molecule = MoleculeKindBuilder::default()
            .name("bad")
            .atoms(vec!["A".into(), "B".into()])
            .atom_indices(vec![0, 1])
            .atom_names(vec![None, None])
            .atomic(true)
            .build()
            .unwrap();
        assert!(molecule.validate().is_err());
    }

    #[test]
    fn atomic_rejects_bonds() {
        let molecule = MoleculeKindBuilder::default()
            .name("bad")
            .atoms(vec!["A".into()])
            .atom_indices(vec![0])
            .atom_names(vec![None])
            .atomic(true)
            .bonds(vec![Bond::new(
                [0, 0],
                BondKind::Unspecified,
                BondOrder::Unspecified,
            )])
            .build()
            .unwrap();
        assert!(molecule.validate().is_err());
    }

    #[test]
    fn atomic_accepts_single_atom() {
        let molecule = MoleculeKindBuilder::default()
            .name("ok")
            .atoms(vec!["X".into()])
            .atom_indices(vec![0])
            .atom_names(vec![None])
            .atomic(true)
            .build()
            .unwrap();
        assert!(molecule.validate().is_ok());
    }

    #[test]
    fn test_load_structure() {
        let molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .from_structure("tests/files/mol2.xyz")
            .build()
            .unwrap();

        assert_eq!(molecule.atoms.len(), 3);
        assert_eq!(
            molecule.atoms.as_slice(),
            ["OW".to_owned(), "OW".to_owned(), "X".to_owned()]
        );
        assert_eq!(molecule.reference_positions.len(), 3);
    }

    #[test]
    fn inline_structure_expands() {
        let yaml = r#"
            name: water
            from_structure:
              - OW: [0.0, 0.0, 0.0]
              - HW: [0.58, 0.76, 0.0]
              - HW: [-0.58, 0.76, 0.0]
        "#;
        let mut molecule: MoleculeKind = serde_yaml::from_str(yaml).unwrap();
        molecule.expand_structure().unwrap();

        assert_eq!(molecule.atoms, ["OW", "HW", "HW"]);
        assert_eq!(molecule.reference_positions.len(), 3);
    }
}
