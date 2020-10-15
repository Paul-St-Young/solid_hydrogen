import numpy as np

def test_drum_pgpa():
  from qharv_db.ase_md import drum_bgpa
  p0 = 150.
  b0 = 375.8818237253769
  b1 = drum_bgpa(p0)
  assert np.isclose(b0, b1)
