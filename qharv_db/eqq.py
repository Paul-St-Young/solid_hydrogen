import numpy as np

def create_h4(case, rsep, rbs=None, orig=None):
  import numpy as np
  nmol = 2
  ndim = 3
  if rbs is None:
    rbs = [1.4, 1.4]  # bond lengths
  case_map = {  # 0: x, 1: y, 2: z
    'A': (2, 2),  # linear
    'C': (0, 2),  # planar
    'F': (0, 0),  # rectangular
    'H': (0, 1),  # staggered
  }
  orient = case_map[case]
  # put down center of mass
  com = np.zeros([nmol, ndim])
  com[0, 2] = -rsep/2.
  com[1, 2] =  rsep/2.
  # setup orientations
  bond = np.zeros([nmol, ndim])
  for imol, idim in enumerate(orient):
    bond[imol, idim] = rbs[imol]
  # create protons for each molecule
  posl = []
  for imol in range(nmol):
    posl.append(com[imol] + bond[imol]/2.)
    posl.append(com[imol] - bond[imol]/2.)
  pos = np.array(posl)
  if orig is None:
    orig = np.zeros(3)
  return com+orig, pos+orig

def make_one_component(atoms, pos, charges):
  from ase import Atoms
  elem = ['H']*len(pos)
  pbc = atoms.get_pbc()
  cell = None
  if np.any(pbc):
    cell = atoms.get_cell()
  atoms1 = Atoms(
    ''.join(elem), charges=charges, positions=pos,
    pbc=pbc, cell=cell
  )
  atoms1.wrap()
  return atoms1

def index_molecules(ncom):
  ncharge = 4  # 4 chages per molecule
  idxl = []
  for imol in range(ncom):
    idx1 = np.arange(4)*ncom + imol
    idxl.append(idx1)
  idx = np.concatenate(idxl)
  return idx

def fixeps_quadrupole(atoms, params, rmax=1.5):
  from qharv.inspect import axes_pos
  # get parameters
  e = params['e']
  eps = params['eps']
  # create charges
  com, avecs = axes_pos.dimer_rep(atoms, rmax)
  p0 = com - avecs
  p1 = com + avecs
  m0 = com - eps*avecs
  m1 = com + eps*avecs
  pos = np.array(p0.tolist() + p1.tolist() + m0.tolist() + m1.tolist())
  ncharge = len(pos)
  nplus = int(round(ncharge/2))
  charges = np.array([e]*nplus + [-e]*nplus)
  # reorder to + + - -, one molecule at a time
  idx = index_molecules(len(com))
  # make new Atoms
  atoms1 = make_one_component(atoms, pos[idx], charges[idx])
  return atoms1

def fixa_quadrupole(atoms, params, rb_max=1.5):
  from qharv.inspect import axes_pos
  # get parameters
  a1 = params['a1']
  a2 = params['a2']
  r0 = params['r0']
  e = params['e']
  # create charges
  com, avecs = axes_pos.dimer_rep(atoms, rb_max)
  amags = np.linalg.norm(avecs, axis=-1)
  ahats = avecs/amags[:, None]
  rb = a1 - a2*(2*amags-r0)
  p0 = com - r0/2.*ahats
  m0 = com - rb[:, None]/2.*ahats
  p1 = com + r0/2.*ahats
  m1 = com + rb[:, None]/2.*ahats
  pos = np.array(p0.tolist() + p1.tolist() + m0.tolist() + m1.tolist())
  ncom = len(com)
  ncharge = len(pos)
  nplus = int(round(ncharge/2))
  charges = np.array([e]*nplus + [-e]*nplus)
  # reorder to + + - -, one molecule at a time
  idx = index_molecules(len(com))
  # make new Atoms
  atoms1 = make_one_component(atoms, pos[idx], charges[idx])
  return atoms1
