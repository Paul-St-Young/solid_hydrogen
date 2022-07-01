import numpy as np

def npt_prefix(dt, pgpa, temp):
  prefix = 'ts%04d' % (dt*10)
  prefix += '-p%d-t%04d' % (pgpa, temp)
  return prefix

def parse_npt_prefix(prefix):
  tst, pt, tt = prefix.split('-')
  dt = float(tst.replace('ts', ''))/10
  pgpa = int(pt.replace('p', ''))
  temp = int(tt.replace('t', ''))
  meta = {'dt': dt, 'pgpa': pgpa, 'temp': temp}
  return meta

def read_eos(ftraj):
  from ase import io
  import pandas as pd
  traj = io.read(ftraj, ':')
  entryl = []
  for atoms in traj:
    v0 = atoms.get_volume()
    e0 = atoms.get_potential_energy()
    seva = atoms.get_stress()
    assert len(seva) == 6
    p0 = -np.mean(seva[:3])
    natom = len(atoms)
    entry = {'volume': v0/natom, 'energy': e0/natom, 'pressure': p0}
    entryl.append(entry)
  df = pd.DataFrame(entryl)
  return df

# ====================== level 1: structure =======================

def tri_prim():
  axes0 = np.array([
    [1, 0],
    [-1./2, 3**0.5/2],
  ])
  return axes0

def prim_cell_in_plane(a1, a2=None):
  if a2 is None:
    a2 = a1
  axes = np.array([
    [3**0.5/2, -1./2],
    [3**0.5/2,  1./2],
  ])
  axes[0] *= a1
  axes[1] *= a2
  fracs = np.array([
    [0, 0],
    [1./3, 1./3],
    [2./3, 2./3],
  ])
  com = np.dot(fracs, axes)
  return axes, com

def add_orientations_in_plane(com, thetas_in_degree, rb=0.74):
  thetas = np.array(thetas_in_degree)/180*np.pi
  norient = len(thetas)
  nmol = len(com)
  if nmol != norient:
    msg = 'got %d orientations for %d molecules' % (norient, nmol)
    raise RuntimeError(msg)
  x = np.cos(thetas)
  y = np.sin(thetas)
  xy = rb*np.c_[x, y]
  p1 = com - xy/2
  p2 = com + xy/2
  posl = [(r1, r2) for (r1, r2) in zip(p1, p2)]
  pos = np.concatenate(posl, axis=0)
  return pos

def c2c_layer(a, thetas, rb):
  a1 = a*3**0.5
  axes, com = prim_cell_in_plane(a1)
  pos = add_orientations_in_plane(com, thetas, rb=rb)
  return axes, pos, com

def c2c_layer_shift_and_thetas(ilayer):
  if ilayer == 0:
    sfrac = [0, 0]
    thetas = [90, -30, 30]
  elif ilayer == 1:
    sfrac = [1./3, 2./3]
    thetas = [90, 30, -30]
  elif ilayer == 2:
    sfrac = [1, 1]
    thetas = [90, -30, 30]
  elif ilayer == 3:
    sfrac = [1./3, -1./3]
    thetas = [90, 30, -30]
  else:
    msg = 'no layer %d' % ilayer
    raise RuntimeError(msg)
  return sfrac, thetas

def c2c_com(a, c):
  nlayer = 4
  a1 = a*3**0.5
  axes0 = a*tri_prim()
  coml = []
  for ilayer in range(nlayer):
    s1, t1 = c2c_layer_shift_and_thetas(ilayer)
    axes, com1 = prim_cell_in_plane(a1)
    shift1 = np.dot(s1, axes0)
    coml.append(com1+shift1)
  com = stack_layers(coml, c/2)
  return com

def fmmm_com(a, c):
  nlayer = 2
  axes0 = a*tri_prim()
  shifts = np.array([
    [0, 0],
    [0.5, 0.5],
  ])
  coml = []
  for s1 in shifts:
    shift1 = np.dot(s1, axes0)
    com = np.array([0, 0])+shift1
    coml.append(com)
  com = stack_layers(coml, c/2)
  return com

def make_atoms(axes, pos, elem=None):
  from ase import Atoms
  natom = len(pos)
  if elem is None:
    elem = 'H%d' % natom
  atoms = Atoms(elem, cell=axes, positions=pos, pbc=True)
  return atoms

def make_c2c(a, c, rb=0.74):
  """Create C2/c-24 structure

  Args:
    a (float): lattice a parameter, in angstrom
    c (float): lattice c parameter, layer separation z=c/2
    rb (float, optional): H2 bond length, default 0.74
  Return:
    ase.Atoms: C2/c-24 unit cell
  Example:
    >>> atoms = make_c2c(1.79, 2.7)
    >>> io.write('c2c-p200.cif', atoms)
  """
  axes0 = a*tri_prim()
  z = c/2
  nlayer = 4
  posl = []
  for ilayer in range(nlayer):
    s1, t1 = c2c_layer_shift_and_thetas(ilayer)
    axes1, pos1, com1 = c2c_layer(a, t1, rb)
    shift1 = np.dot(s1, axes0)
    posl.append(pos1+shift1)

  axes = np.zeros([3, 3])
  axes[:2, :2] = axes1
  axes[2, 2] = nlayer*z

  pos = stack_layers(posl, z)

  atoms = make_atoms(axes, pos)
  return atoms

def stack_layers(posl, z):
  nlayer = len(posl)
  n_per_layer = len(posl[0])
  pos = np.zeros([nlayer*n_per_layer, 3])
  for ilayer in range(nlayer):
    istart = ilayer*n_per_layer
    iend = istart + n_per_layer
    pos[istart:iend, :2] = posl[ilayer]
    pos[istart:iend, 2] = ilayer*z
  return pos

def make_fmmm(a, c, rb=0.74, t0=30, shift=None):
  z = c/2
  axes0 = a*tri_prim()
  # first layer
  thetas = [t0]
  com = np.array([
    [0.0, 0.0],
  ])
  if shift is None:
    shift = [1./3, 0]
  com += np.dot(shift, axes0)
  p0 = add_orientations_in_plane(com, thetas)
  # second layer
  sfrac = [0.5, 0.5]
  shift = np.dot(sfrac, axes0)
  p1 = p0 + shift
  # combine
  posl = [p0, p1]
  pos = stack_layers(posl, z)
  # expand cell
  axes = np.zeros([3, 3])
  axes[:2, :2] = axes0
  axes[2, 2] = len(posl)*z

  atoms = make_atoms(axes, pos)
  return atoms

def hcp_prim_cell(a, ca=None):
  if ca is None:  # use ideal ratio
    ca = (8./3)**0.5
  c = a*ca
  axes = np.array([
    [a, 0, 0],
    [a/2, 3**0.5*a/2, 0],
    [0, 0, c]
  ])
  pos = np.dot(
    np.array([
      [0, 0, 0],
      [1./3, 1./3, 1./2]
    ]), axes)

  atoms = make_atoms(axes, pos)
  return atoms

def make_mhcpc(s1, rb=0.74, iax=2, check=False):
  # copied from f16b1/prim.py
  axes = s1.get_cell()
  pos = s1.get_positions()

  box = np.diag(axes)
  # make sure cell is diagonal
  assert np.allclose(np.diag(box), axes)

  posl = []
  for center in pos:
    p1 = center.copy()
    p1[iax] += rb/2.
    posl.append(p1.copy())
    p1[iax] -= rb
    posl.append(p1)
  pos1 = np.array(posl)
  if check:  # check pos1
    from qharv.inspect import box_pos
    from qharv.inspect import crystal, volumetric
    pos1 = box_pos.pos_in_box(pos1, box)
    fig, ax = volumetric.figax3d()
    crystal.draw_cell(ax, axes)
    crystal.draw_atoms(ax, pos, c='k', ms=2)
    crystal.draw_atoms(ax, pos1)
    plt.show()

  s2 = make_atoms(axes, pos1)
  s2.wrap()
  return s2

def make_mhcpo(atoms, rb=0.74, theta=np.pi/3):
  posl = []
  a = rb/2  # half bond length
  pos = atoms.get_positions()
  zl = np.unique(pos[:, 2])
  for iz, z1 in enumerate(zl):
    pm = 2*(iz%2)-1
    zsel = pos[:, 2] == z1
    avec = np.array([0, pm*a*np.sin(theta), a*np.cos(theta)])
    for r1 in pos[zsel]:
      posl.append(r1+avec)
      posl.append(r1-avec)
  pos1 = np.array(posl)
  atoms1 = make_atoms(atoms.get_cell(), pos1)
  atoms1.wrap()
  return atoms1

def load_traj(ftraj, traj_fmt=None, istart=0, iend=-1, nevery=1, check=True):
  from ase import io
  if traj_fmt is None:
    if ftraj.endswith('.nc'):
      traj_fmt = 'netcdftrajectory'
  traj = io.read(ftraj, slice(istart, iend, nevery), format=traj_fmt)
  if (len(traj) < 1) and check:
    ntot = len(io.read(ftraj, ':', format=traj_fmt))
    msg = '%d/%d snapshots read using:\n' % (len(traj), ntot)
    msg += ' slice(%d, %d, %d)' % (istart, iend, nevery)
    raise RuntimeError(msg)
  return traj

# ====================== level 1: dynamics =======================

def init_velocities(atoms, temp):
  """Initialize atom velocities according to Maxwell-Bolzmann,
   then zero total linear and angular momenta.

  Args:
    atoms (ase.Atoms): initial configuration
    temp (float): temperature in Kelvin
  """
  from ase import units
  from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
  from ase.md.velocitydistribution import Stationary, ZeroRotation
  MaxwellBoltzmannDistribution(atoms, temp*units.kB)
  Stationary(atoms)
  ZeroRotation(atoms)

# ====================== level 2: param. choice =======================
def hcp_ac(pgpa, force=False):
  # fit from f23a/hcp-prim/fit.py
  pmin = 20   # GPa
  pmax = 200  # GPa
  if ((pgpa < pmin) or (pgpa > pmax)) and (not force):
    msg = 'fit fails below %d GPa or above %d GPa' % (pmin, pmax)
    raise RuntimeError(msg)
  pa = np.array([
    1.814710e-14, -1.871584e-11, 7.877205e-09,
    -1.748124e-06, 2.230823e-04, -1.779103e-02,
    2.586276e+00
  ])
  fa = np.poly1d(pa)
  # pc = np.array([
  #   -6.522742e-12, 6.137292e-09, -2.076912e-06,
  #   3.370382e-04, -3.041165e-02, 4.238903e+00
  # ])
  # fc = np.poly1d(pc)
  pca = np.array([
    3.549756e-15, -3.372556e-12, 1.294774e-09,
    -2.582915e-07, 2.894332e-05, -2.008867e-03,
    1.662028e+00
  ])
  fca = np.poly1d(pca)
  a = fa(pgpa)
  c = a*fca(pgpa)
  return a, c

def get_tilematrix(natom):
  if natom == 16:
    tmat = np.array([
      [2, 0, 0],
      [-1, 2, 0],
      [0, 0, 1]
    ])
  elif natom == 96:
    tmat = np.array([
      [ 3, 0, 0],
      [-2, 4, 0],
      [ 0, 0, 2]
    ])
  elif natom == 128:
    tmat = np.array([
      [ 4, 0, 0],
      [-2, 4, 0],
      [ 0, 0, 2]
    ])
  elif natom == 192:
    tmat = np.array([
      [ 4, 0, 0],
      [-2, 4, 0],
      [ 0, 0, 3]
    ])
  elif natom == 360:
    tmat = np.array([
      [ 5, 0, 0],
      [-3, 6, 0],
      [ 0, 0, 3]
    ])
  elif natom == 576:
    tmat = np.array([
      [ 6, 0, 0],
      [-3, 6, 0],
      [ 0, 0, 4]
    ])
  else:
    msg = 'no tilematrix for N=%d' % natom
    raise RuntimeError(msg)
  return tmat

def hcp_supercell(pgpa, natom, tmat=None, force=False):
  from ase.build.supercells import make_supercell
  a, c = hcp_ac(pgpa, force=force)
  atoms0 = hcp_prim_cell(a, c/a)
  if tmat is None:
    tmat = get_tilematrix(natom)
  atoms1 = make_supercell(atoms0, tmat)
  return atoms1

def mhcpc_supercell(pgpa, natom, force=False, **kwargs):
  atoms1 = hcp_supercell(pgpa, natom, force=force)
  atoms2 = make_mhcpc(atoms1, **kwargs)
  return atoms2

def drum_eos(vb):
  """Static-lattice DMC energy of C2/c-24 (KZK) fit over 60 ~ 270 GPa.

  Args:
    vb (np.array): volume per proton in bohr^3
  Return:
    np.array: energy per proton in ha
  """
  # fit from mat/drum_data/scripts/lowp_eos.py
  popt = np.array([
    2.7230701840642966, 0.5237100076747062, -0.6042128881368943
  ])
  eha = np.poly1d(popt)(1./vb)
  return eha

def drum_peos(rsmin=1.31, rsmax=1.8, drs=0.005, norder=5, return_popt=False):
  rss = np.arange(rsmax, rsmin, -drs)
  vol = 4*np.pi/3*rss**3  # Bohr^3
  eha = drum_eos(vol)
  # fit EOS
  popt = np.polyfit(vol, eha, norder)
  # find volume at target pressure
  p1 = np.polyder(popt, 1)

  def peos(v):
    return -np.poly1d(p1)(v)

  ret = peos
  if return_popt:
    ret = (peos, popt)
  return ret

def drum_bgpa(pgpa, rsmin=1.31, rsmax=1.8, drs=0.005, norder=5):
  """Compressibility of solid hydrogen

  Args:
    pgpa (float): pressure in GPa
    rsmin (float, optional): highest-density to refit, default 1.31
    rsmax (float, optional): lowest-density to refit, default 1.8
    drs (float, optional): density grid, default 5e-3
    norder (float, optional): polynomial order for EOS refit
  Return:
    float: bgpa, compressibility in GPa
  """
  peos, popt = drum_peos(rsmin=rsmin, rsmax=rsmax, drs=drs, norder=norder,
    return_popt=True)
  gpa = 29421  # ha/B^3 -> GPa
  from scipy.optimize import minimize_scalar
  sol = minimize_scalar(lambda v: (peos(v)*gpa-pgpa)**2)
  myv = sol.x
  assert np.isclose(peos(myv)*gpa, pgpa)
  # check rs range
  myrs = (3*myv/(4*np.pi))**(1./3)
  assert (rsmin < myrs) & (myrs < rsmax)
  # calculate bulk modulus at this pressure
  p2 = np.polyder(popt, 2)
  bgpa = myv*np.poly1d(p2)(myv)*gpa
  return bgpa

# ===================== level 2: gather output ======================
def parse_ase_log(flog):
  from qharv.reel import scalar_dat
  with open(flog, 'r') as f:
    lines = f.readlines()
  header = lines[0]
  body = '\n'.join(lines[1:])

  def parse_cols(header):
    cols = []
    tokens = header.split()
    i = 0
    for j in range(len(tokens)):
      if i >= len(tokens):
        break
      token = tokens[i]
      col = token
      if token == 'stress':
        col += tokens[i+1]
        i += 1
      if not token.startswith('--'):
        cols.append(col)
      i += 1
    return cols
  cols = parse_cols(header)

  pcols = ['Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz']

  def expand_stress(cols, pcols):
    cols1 = []
    for col in cols:
      if not col.startswith('stress'):
        cols1.append(col)
      else:
        unit = '[' + col.split('[')[-1]
        for name in pcols:
          cols1.append(name+unit)
    return cols1, unit
  cols1, unit = expand_stress(cols, pcols)
  df = scalar_dat.parse(body)
  df.columns = cols1
  df[[c+unit for c in pcols]] *= -1
  return df

def read_qe_cell(mm, ndim=3):
  alat = ascii_out.name_sep_val(mm, 'lattice parameter (alat)')
  rowl = []
  for idim in range(ndim):
    idx = mm.find(b'a(%d)' % (idim+1))
    mm.seek(idx)
    line = mm.readline().decode()
    rowt = line.split('=')[1]
    row = list(map(float, rowt.split('(')[1].split(')')[0].split()))
    rowl.append(row)
  axes = alat*np.array(rowl)
  return axes

def read_qe_pos(mm):
  alat = ascii_out.name_sep_val(mm, 'lattice parameter (alat)')
  natom = ascii_out.name_sep_val(mm, 'number of atoms/cell', dtype=int)
  idx = mm.find(b'positions (alat units)')
  mm.seek(idx)
  mm.readline()
  rowl = []
  for iatom in range(natom):
    line = mm.readline().decode()
    rowt = line.split('=')[1]
    row = list(map(float, rowt.split('(')[1].split(')')[0].split()))
    rowl.append(row)
  pos = alat*np.array(rowl)
  return pos
