from qharv_db import bspline
import numpy as np

def coeff_for_jastrow(param, cusp, delta):
  coeffs = np.zeros(len(param)+4)
  coeffs[0] = param[1] - 2.0*delta*cusp
  coeffs[1] = param[0]
  coeffs[2] = param[1]
  coeffs[3:len(param)+1] = param[2:]

  return coeffs

def create_jastrow_from_param(param, cusp, rcut):
  bc = bspline.BCInfo()
  bc.lBC= bspline.FLAT
  bc.rBC = bspline.FLAT

  grid = bspline.Grid1D(len(param)+2, 0.0, rcut)

  sp = bspline.UBSpline1D(bc, grid)
  
  coeff = coeff_for_jastrow(param, cusp, grid.delta)
  #print 'len coeff after',len(coeff)
  sp.set_coefs(coeff)

  return sp


if __name__ == '__main__':
    param = np.array([0.3112479574, 0.2389691818, 0.1641628936, 0.1008687295, 0.05540550108, 0.02718706524, 0.01190787928,0.004651462998])
    rcut = 4.1040393
    b = create_jastrow_from_param(param, -0.25, rcut) 

    delta = rcut/100.0
    for i in range(100):
        r = delta*i
        #u = b.evaluate_v(r)
        #u,du = b.evaluate_vg(r)
        u,du,ddu = b.evaluate_vgl(r)
        
