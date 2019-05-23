from ase.dft.kpoints import get_bandpath
from qharv.inspect import axes_pos, volumetric, crystal

def cmca4_kpath(axes, npt=16):
  G = [0.0,  0.0, 0.0]
  Y = [0.5, -0.5, 0.0]
  S = [0.5,  0.0, 0.0]
  Z = [0.0,  0.0, 0.5]
  kpts_reduced, kpath, sp_points = get_bandpath([G, Y, S, G, Z],
    axes, npoints=npt)
  return kpts_reduced, kpath, sp_points
