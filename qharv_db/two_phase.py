import numpy as np

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
  grouped_pos = [[]]*len(layers)
  for atoms in traj:
    if wrap:
      atoms.wrap()
    pos = atoms.get_positions()
    z = pos[:, 2]
    for igrp, layer in enumerate(layers):
      zmin, zmax = layer
      zsel = (zmin < z) & (z < zmax)
      grouped_pos[igrp].append(pos[zsel])
  groups = np.array([np.concatenate(group, axis=0) for group in grouped_pos])
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
