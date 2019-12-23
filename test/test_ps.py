import numpy as np

def test_rho2rs():
  from plot_structures import rho2rs
  # D2
  nprot = 2
  rhol = [0.17089, 0.75]
  rsl = [3.16, 1.93]
  for rho, rs in zip(rhol, rsl):
    assert np.isclose(rho2rs(rho, nprot=nprot), rs, atol=1e-4)

def test_rs2rho():
  from plot_structures import rs2rho
  # D2
  nprot = 2
  rhol = [0.17089, 0.75]
  rsl = [3.16, 1.93]
  for rho, rs in zip(rhol, rsl):
    assert np.isclose(rho, rs2rho(rs, nprot=nprot), atol=1e-4)
