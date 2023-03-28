import numpy as np
import chiesa_correction as chc

def test_coulomb_interaction():
  k = np.linspace(1e-3, 1e3)
  vk2d = chc.coulomb_interaction(k, ndim=2)
  assert np.allclose(vk2d, 2*np.pi/k)
  vk3d = chc.coulomb_interaction(k, ndim=3)
  assert np.allclose(vk3d, 4*np.pi/k**2)

def test_charge_density():
  rs = np.linspace(1e-3, 1e3)
  rho2d = chc.charge_density(rs, ndim=2)
  assert np.allclose(rho2d, 1./(np.pi*rs**2))
  rho3d = chc.charge_density(rs, ndim=3)
  assert np.allclose(rho3d, 1./(4*np.pi/3*rs**3))

def test_fermi():
  rs = np.linspace(1e-3, 1e3)
  kf2d = chc.heg_kfermi(rs, ndim=2)
  assert np.allclose(kf2d, 2**0.5/rs)
  kf3d = chc.heg_kfermi(rs, ndim=3)
  assert np.allclose(kf3d, (9*np.pi/4)**(1./3)/rs)
