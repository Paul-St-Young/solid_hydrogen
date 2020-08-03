import numpy as np
from qharv_db import eqq

def test_lebedev_h2():
  import quadpy
  scheme = quadpy.u3.lebedev_005()
  pts = scheme.points
  wts = scheme.weights
  orients, weights = eqq.lebedev_h2()
  iunique = [0, 2, 4, 6, 7, 8, 9]
  for i, j in enumerate(iunique):
    assert np.allclose(pts[j], orients[i]/np.linalg.norm(orients[i]))
    assert np.allclose(2*wts[j], weights[i])

if __name__ == '__main__':
  test_lebedev_h2()
# end __main__
