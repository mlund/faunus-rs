import parmed as pmd # type: ignore
from ruamel.yaml import YAML # type: ignore
parm = pmd.load_file("solvated.top")
bonds = (dict(index=[b.atom1.idx, b.atom2.idx], req=b.type.req, k=b.type.k, type='harmonic') for b in parm.bonds)
atoms = (dict(sigma=a.sigma, epsilon=a.epsilon, name=a.name, charge=a.charge, type=a.type,
                            element=a.element_name, index=a.idx) for a in parm.atoms)
residues = (dict(name=r.name, index=r.idx, chain=r.chain, number=r.number, ter=r.ter,
                                  atom_names=list(map(lambda x: x.name, r.atoms)),
                                  atom_index=list(map(lambda x: x.idx, r.atoms))
                                 ) for r in parm.residues)
angles = (dict(index=[a.atom1.idx, a.atom2.idx, a.atom3.idx], k=a.type.k, theteq=a.type.theteq) for a in parm.angles)
top = dict(atomtypes=list(atoms),
                      residues=list(residues),
                      bonds=list(bonds),
                      angles=list(angles))

yaml=YAML()
#yaml.indent(mapping=4, sequence=6, offset=3)
yaml.compact()
#yaml.dump(top, sys.stdout)
#top
parm.molecules
