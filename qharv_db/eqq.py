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

def fixeps_quadrupole(atoms, params, rb_max=1.5):
  from qharv.inspect import axes_pos
  # get parameters
  e = params['e']
  eps = params['eps']
  # create charges
  com, avecs = axes_pos.dimer_rep(atoms, rb_max)
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

def mean_sig(fa):
  nconf, ndim = fa.shape
  fm = fa.mean(axis=0)
  fs = np.sqrt(np.sum((fa-fm)**2)/ndim/(nconf-1))
  return fm, fs

def bootstrap(fa, nsample):
  nconf = len(fa)
  confs = np.arange(nconf)
  fsl = []
  for isample in range(nsample):
    iconfs = np.random.choice(confs, size=nconf, replace=True)
    fm1, fs1 = mean_sig(fa[iconfs])
    fsl.append(fs1)
  fsm = np.mean(fsl)
  fse = np.std(fsl, ddof=1)
  return fsm, fse

def create_h2(avec=None, com=None, rb=None):
  from ase import Atoms
  if avec is None:  # point along z
    avec = np.array([0, 0, 1])
  else:
    amag = np.linalg.norm(avec)
    if not np.isclose(amag, 1):
      raise RuntimeError('avec must be a unit vector')
  if com is None:  # center around origin
    com = np.zeros(3)
  if rb is None:  # equilibrium bond length
    bohr = 0.529177210903  # CODATA 2018
    rb = 1.4*bohr
  pos = rb/2*np.array([avec, -avec])
  atoms = Atoms('H2', positions=com+pos)
  return atoms

def rotate_h2(atoms, rot, center='COM'):
  phi, theta, psi = rot.as_euler('zxz', degrees=True)
  atoms.euler_rotate(phi, theta, psi, center=center)

def h2_random_rotations(mols, seed):
  from ase import Atoms
  from scipy.spatial.transform import Rotation
  # get molecule info
  nmol = len(mols)
  axes = mols.get_cell()
  pbc = mols.get_pbc()
  # create random rotations
  np.random.seed(seed)
  rots = Rotation.random(nmol)
  # assign rotations
  posl = []
  for com, rot in zip(mols.get_positions(), rots):
    h2 = create_h2(com=com)
    rotate_h2(h2, rot)
    posl += h2.get_positions().tolist()
  pos = np.array(posl)
  atoms = Atoms('H%d' % len(pos), cell=axes, positions=pos, pbc=pbc)
  return atoms
