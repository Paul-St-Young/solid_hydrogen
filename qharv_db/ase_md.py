import numpy as np

def npt_prefix(dt, pgpa, temp):
  prefix = 'ts%04d' % (dt*10)
  prefix += '-p%d-t%04d' % (pgpa, temp)
  return prefix

# ====================== level 1: structure =======================

def hcp_prim_cell(a, ca):
  from ase import Atoms
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

  atoms = Atoms('H%d' % len(pos), cell=axes, positions=pos, pbc=1)
  return atoms

def make_mhcpc(s1, rb=0.74, iax=2, check=False):
  # copied from f16b1/prim.py
  from ase import Atoms
  from qharv.inspect import box_pos
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
  pos1 = box_pos.pos_in_box(np.array(posl), box)
  if check:  # check pos1
    from qharv.inspect import crystal, volumetric
    fig, ax = volumetric.figax3d()
    crystal.draw_cell(ax, axes)
    crystal.draw_atoms(ax, pos, c='k', ms=2)
    crystal.draw_atoms(ax, pos1)
    plt.show()

  s2 = Atoms('H%d' % len(pos1), cell=axes, positions=pos1, pbc=1)
  return s2

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
def hcp_ac(pgpa):
  # fit from f23a/hcp-prim/fit.py
  pmin = 20   # GPa
  pmax = 200  # GPa
  if (pgpa < pmin) or (pgpa > pmax):
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
  else:
    msg = 'no tilematrix for N=%d' % natom
    raise RuntimeError(msg)
  return tmat

def hcp_supercell(pgpa, natom):
  from ase.build.supercells import make_supercell
  a, c = hcp_ac(pgpa)
  atoms0 = hcp_prim_cell(a, c/a)
  tmat = get_tilematrix(natom)
  atoms1 = make_supercell(atoms0, tmat)
  return atoms1

def mhcpc_supercell(pgpa, natom):
  atoms1 = hcp_supercell(pgpa, natom)
  atoms2 = make_mhcpc(atoms1)
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
  rss = np.arange(rsmax, rsmin, -drs)
  vol = 4*np.pi/3*rss**3  # Bohr^3
  eha = drum_eos(vol)
  # fit EOS
  popt = np.polyfit(vol, eha, norder)
  # find volume at target pressure
  p1 = np.polyder(popt, 1)
  gpa = 29421  # ha/B^3 -> GPa
  def peos(v):
    return -np.poly1d(p1)(v)*gpa
  from scipy.optimize import minimize_scalar
  sol = minimize_scalar(lambda v: (peos(v)-pgpa)**2)
  myv = sol.x
  assert np.isclose(peos(myv), pgpa)
  # check rs range
  myrs = (3*myv/(4*np.pi))**(1./3)
  assert (rsmin < myrs) & (myrs < rsmax)
  # calculate bulk modulus at this pressure
  p2 = np.polyder(popt, 2)
  bgpa = myv*np.poly1d(p2)(myv)*gpa
  return bgpa
