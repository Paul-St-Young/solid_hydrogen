import numpy as np

def cubic_pos(nx, ndim=3):
  nxnynz = [np.arange(nx)]*ndim
  pos = np.stack(
    np.meshgrid(*nxnynz, indexing='ij'),
    axis=-1
  ).reshape(-1, ndim)
  return pos

def test_cart2sph():
  from qharv_db.grsk import cart2sph
  # get a spherical shell of vectors
  nx = 7
  vecs = cubic_pos(2*nx)-nx
  mags = np.linalg.norm(vecs, axis=-1)
  sel = (5 < mags) & (mags < 6)
  kvecs = vecs[sel]
  # convert to spherical coordinates
  kx, ky, kz = kvecs.T
  kmags, theta, phi = cart2sph(kvecs)
  # convert back and check
  z1 = kmags*np.cos(theta)
  rho = np.sqrt(kx**2+ky**2)
  x1 = rho*np.cos(phi)
  y1 = rho*np.sin(phi)
  assert np.allclose(kz, z1)
  assert np.allclose(ky, y1)
  assert np.allclose(kz, z1)

if __name__ == '__main__':
  test_cart2sph()
# end __main__
