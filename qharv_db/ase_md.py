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

  #iax = 2  # align along z
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
  from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
  from ase.md.velocitydistribution import Stationary, ZeroRotation
  MaxwellBoltzmannDistribution(atoms, temp*units.kB)
  Stationary(atoms)
  ZeroRotation(atoms)
