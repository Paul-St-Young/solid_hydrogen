# ref 1: https://lammps.sandia.gov/doc/99/data_format.html
# ref 2: ase/io/lammpsdata.py
import numpy as np

units = 'metal'
line_formats = {
  #         atype mass
  'mass': '\n%3d %.6f',
  #        id group atype      q    x     y     z
  'atoms_full': '\n%5d %5d %3d %.6f %.16e %.16e %.16e',
  #        id atype          q    x     y     z     mux   muy   muz
  'atoms_dipole': '\n%5d %3d %.6f %.16e %.16e %.16e %.16e %.16e %.16e',
  #        id atype                 x     y     z     q    mux   muy   muz diameter density
  'atoms_dipole_sphere': '\n%5d %3d %.16e %.16e %.16e %.6f %.16e %.16e %.16e %.16e %.16e',
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
  natom = len(atoms)
  symbols = atoms.get_chemical_symbols()
  pos = p.vector_to_lammps(atoms.get_positions(), wrap=True)
  # additional info for atom_style
  if atom_style in ['charge', 'full', 'dipole', 'dipole_sphere']:
    charges = atoms.get_initial_charges()
  if atom_style in ['dipole', 'dipole_sphere']:
    dipoles = atoms.get_dipole_moment()
  if mols is None:  # assign all atoms to the same molecule
    mols = np.ones(natom, dtype=int)
  # write text
  line_fmt = line_formats['atoms_%s' % atom_style]
  text = '\n\nAtoms\n'
  for iatom, tup in enumerate(zip(symbols, pos)):
    sym, r = tup
    sym = symbols[iatom]
    s = species.index(sym) + 1
    if atom_style in ['charge', 'full', 'dipole', 'dipole_sphere']:
      q = charges[iatom]
      q = convert(q, "charge", "ASE", units)
    r = convert(r, "distance", "ASE", units)
    if atom_style in ['molecule', 'full']:
      imol = mols[iatom]
      text += line_fmt % (iatom+1, imol, s, q, *r)
    elif atom_style in ['dipole']:
      d = dipoles[iatom]
      text += line_fmt % (iatom+1, s, q, *r, *d)
    elif atom_style in ['dipole_sphere']:
      d = dipoles[iatom]
      line = line_fmt % (iatom+1, s, *r, q, *d, 0., 1.0)
      text += line
    else:
      msg = 'atom_style %s not yet available' % atom_style
      raise NotImplementedError(msg)
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
