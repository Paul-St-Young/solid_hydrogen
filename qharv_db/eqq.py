import numpy as np

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
  idx0 = list(range(0, ncharge, 2))
  idx1 = list(range(1, ncharge, 2))
  idx = idx0+idx1
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
  ncharge = len(pos)
  nplus = int(round(ncharge/2))
  charges = np.array([e]*nplus + [-e]*nplus)
  # reorder to + + - -, one molecule at a time
  idx0 = list(range(0, ncharge, 2))
  idx1 = list(range(1, ncharge, 2))
  idx = idx0+idx1
  # make new Atoms
  atoms1 = make_one_component(atoms, pos[idx], charges[idx])
  return atoms1
