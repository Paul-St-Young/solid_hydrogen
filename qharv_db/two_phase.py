import numpy as np

def average_cell(traj):
  axesl = [atoms.get_cell() for atoms in traj]
  axes = np.mean(axesl, axis=0)
  return axes

def average_frames(traj):
  """Average atomic positions from a segment of MD trajectory.
  Assume atomic displacement between frames < L/2.

  Args:
    list: list of ASE Atoms
  Return:
    Atoms: ASE Atoms of averaged frame
  """
  from ase import Atoms
  axes0 = traj[0].get_cell()
  box0 = np.diag(axes0)
  if not np.allclose(axes0, np.diag(box0)):
    msg = 'assuming orthorhombic box'
    raise RuntimeError(msg)
  pos0 = traj[0].get_positions()
  nframe = len(traj)
  ndim = len(box0)
  adr = np.zeros([len(pos0), ndim])
  boxl = [box0]
  for atoms in traj[1:]:
    axes = atoms.get_cell()
    box = np.diag(axes)
    if not np.allclose(axes, np.diag(box)):
      msg = 'assuming orthorhombic box'
      raise RuntimeError(msg)
    boxl.append(box)
    pos = atoms.get_positions()
    dr = pos-pos0
    for idim in range(3):
      lbox = box[idim]
      xsel = dr[:, idim] > lbox/2
      pos[xsel, idim] -= lbox
      xsel = dr[:, idim] < -lbox/2
      pos[xsel, idim] += lbox
    adr += pos-pos0
  adr /= nframe
  abox = np.mean(boxl, axis=0)
  apos = pos0+adr
  atoms = Atoms('H%d' % len(apos), cell=np.diag(abox), positions=apos, pbc=1)
  atoms.wrap()
  return atoms

def layers_from_z(z, bins=128):
  from scipy.signal import find_peaks
  counts, bin_edges = np.histogram(z, bins=bins)
  idx, props = find_peaks(-counts)
  return bin_edges[idx]

def find_layers(atoms, bins=128):
  from scipy.signal import find_peaks
  axes = atoms.get_cell()
  lxlylz = np.diag(axes)
  assert np.allclose(axes, np.diag(lxlylz))
  lz = lxlylz[-1]
  assert lz == max(lxlylz)
  pos = atoms.get_positions()
  # histogram z coordinates
  z = pos[:, 2]
  counts, bin_edges = np.histogram(z, bins=bins)
  # find troughs
  idx, props = find_peaks(-counts)
  # return trough edges
  layers = []
  for i0, i in enumerate(idx):
    j = idx[i0+1] if i0 < len(idx)-1 else -1
    bottom = bin_edges[i]
    top = bin_edges[j]
    layer = (bottom, top)
    layers.append(layer)
  return layers

def extract_layers(layers, traj, wrap=True):
  grouped_pos = []
  for layer in layers:
    grouped_pos.append([])
  for atoms in traj:
    if wrap:
      atoms.wrap()
    pos = atoms.get_positions()
    z = pos[:, 2]
    for igrp, layer in enumerate(layers):
      zmin, zmax = layer
      zsel = (zmin < z) & (z < zmax)
      grouped_pos[igrp].append(pos[zsel])
  #groups = np.array([np.concatenate(group, axis=0) for group in grouped_pos])
  groups = [np.array(group) for group in grouped_pos]
  return groups

def find_clusters(group, **kwargs):
  from sklearn.cluster import DBSCAN
  cluster = DBSCAN(**kwargs).fit(group)
  centers = []
  for label in np.unique(cluster.labels_):
    if label < 0: continue
    sel = cluster.labels_ == label
    centers.append(group[sel].mean(axis=0))
  return np.array(centers)

def ml_mhcp(a1, a2, c, fracx=0.5):
  from ase import Atoms
  a2x = a1/2
  a2y = (a2**2-a2x**2)**0.5
  axes = np.array([
    [a1, 0, 0],
    [a2x, a2y, 0],
    [0, 0, c]
  ])
  fracs = np.array([
    [0, 0, 0],
    [fracx, fracx, 0.5],
  ])
  pos = np.dot(fracs, axes)
  atoms = Atoms('H%d' % len(pos), cell=axes, positions=pos, pbc=1)
  return atoms

def detect_p21c_layers(mols, kfracs=None):
  from qharv.inspect import axes_pos
  # setup S(k) calculation
  from qharv_db import grsk
  if kfracs is None:
    kfracs = np.array([
      [0, -8, 16],
      [0,  8, 16],
    ])
  axes = mols.get_cell()
  raxes = axes_pos.raxes(axes)
  kvecs = np.dot(kfracs, raxes)

  # detect layers
  layers = find_layers(mols)
  groups = extract_layers(layers, [mols])
  grps = [g[0] for g in groups]

  # loop through 3 layers at a time
  nlayer = len(grps)
  layer_desc = []
  for ilayer in range(1, nlayer-1):
    pos0 = grps[ilayer-1]
    pos1 = grps[ilayer]
    pos2 = grps[ilayer+1]
    pos = np.concatenate([pos0, pos1, pos2], axis=0)
    sk1 = grsk.Sk(kvecs, pos)
    layer_desc.append(sk1.sum())
  return layer_desc

