#!/usr/bin/env python
import numpy as np
import chiesa_correction as chc

def gauss1d(x):
  return np.exp(-x**2/2.)
def check_int1d():
  lbox = 10.
  xmin = -lbox/2.
  xmax =  lbox/2.
  print('integrate 1D standard Gaussian')

  for nx in [4, 16, 32, 64]:
    myx = np.linspace(xmin, xmax, nx)
    myy = gauss1d(myx)

    print(myy.sum()*lbox/nx)
  print((2*np.pi)**0.5)


def gauss3d(x):
  expo = -np.linalg.norm(x, axis=1)**2/2.
  return np.exp(expo)
def check_int3d():
  lbox = 10.
  xmin = -lbox/2.
  xmax =  lbox/2.
  print('integrate 3D standard Gaussian')

  for nx in [4, 16, 32]:
    myx = lbox*chc.shifted_mp_grid(nx)
    myy = gauss3d(myx)

    print(myy.sum()*(lbox/nx)**3)
  print((2*np.pi)**1.5)


if __name__ == '__main__':

  check_int1d()
  check_int3d()

# end __main__
