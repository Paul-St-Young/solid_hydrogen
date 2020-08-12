# ref 1: https://lammps.sandia.gov/doc/99/data_format.html
# ref 2: ase/io/lammpsdata.py
import numpy as np

units = 'metal'
line_formats = {
  #         atype mass
  'mass': '\n%3d %.6f',
  #        id group atype    q     x     y     z
  'atoms': '\n%5d %5d %3d %.6f %.16e %.16e %.16e',
  #        id   btype   i   j
  'bonds': '\n%5d %3d %5d %5d',
}

def get_prism(cell):
  from ase.calculators.lammpsrun import Prism
  p = Prism(cell)
  return p

def get_species(atoms):
  symbols = atoms.get_chemical_symbols()
  species = sorted(set(symbols))
  return species

def text_cell(cell):
  from ase.calculators.lammpsrun import convert
  p = get_prism(cell)
  xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
    "ASE", units)
  text = ''
  text += "0.0 {0:23.17g} xlo xhi\n".format(xhi)
  text += "0.0 {0:23.17g} ylo yhi\n".format(yhi)
  text += "0.0 {0:23.17g} zlo zhi\n".format(zhi)
  return text

def text_mass(atoms):
  mass_dict = {'H': 1.008}
  species = get_species(atoms)
  text = '\n\nMasses\n'
  line_fmt = line_formats['mass']
  for ispec, spec in enumerate(species):
    mass = mass_dict[spec]
    text += line_fmt % (ispec+1, mass)
  return text

def text_atoms(atoms, mols=None, atom_style='full'):
  from ase.calculators.lammpsrun import convert
  p = get_prism(atoms.get_cell())
  species = get_species(atoms)
  # get Atoms info
  symbols = atoms.get_chemical_symbols()
  charges = atoms.get_initial_charges()
  pos = p.vector_to_lammps(atoms.get_positions(), wrap=True)
  natom = len(symbols)
  if mols is None:  # assign all atoms to the same molecule
    mols = np.ones(natom, dtype=int)
  line_fmt = line_formats['atoms']
  text = '\n\nAtoms\n'
  for iatom, tup in enumerate(zip(mols, symbols, charges, pos)):
    imol, sym, q, r = tup
    s = species.index(sym) + 1
    q = convert(q, "charge", "ASE", units)
    r = convert(r, "distance", "ASE", units)
    text += line_fmt % (iatom+1, imol, s, q, *r)
  return text

def text_bonds(pairs_dict):
  line_fmt = line_formats['bonds']
  text = '\n\nBonds\n'
  ibond = 1
  for btype, pairs in pairs_dict.items():
    for myp in pairs:
      text += line_fmt % (ibond, btype, *(myp+1))
      ibond += 1
  return text

# =========================== composition ===========================
def text_lammps_data_dimers(atoms, pairs_dict):
  species = get_species(atoms)
  na_type = len(species)
  nb_type = len(pairs_dict)
  nbond = sum([len(pairs) for i, pairs in pairs_dict.items()])
  mols = np.ones(len(atoms), dtype=int)
  imol = 1
  for ipair, pairs in pairs_dict.items():
    for pair in pairs:
      i, j = pair
      mols[i] = imol
      mols[j] = imol
      imol += 1

  natom = len(atoms)

  text = 'LAMMPS data\n\n'
  text += '  %5d atoms\n' % natom
  text += '  %5d bonds\n' % nbond
  text += '\n'
  text += '  %5d atom types\n' % na_type
  text += '  %5d bond types\n' % nb_type
  text += '\n'
  cellt = text_cell(atoms.cell)
  text += cellt
  masst = text_mass(atoms)
  text += masst
  atomt = text_atoms(atoms, mols=mols)
  text += atomt
  bondt = text_bonds(pairs_dict)
  text += bondt
  return text
