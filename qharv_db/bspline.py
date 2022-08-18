# Author: Mark Dewing
# B-splines
# Python implementation of einspline routines

import numpy as np


# Boundary condition types

PERIODIC = 0
DERIV1 = 1
DERIV2 = 2
FLAT = 3
NATURAL = 4
ANTIPERIODIC = 5


class BCInfo:
  def __init__(self):
    self.lBC = PERIODIC
    self.rBC = PERIODIC
    self.lval = 0.0  # function value on the left
    self.rval = 0.0  # function value on the right


class Grid1D:
  def __init__(self, num, start, stop):
    self.num = num
    self.start = start
    self.stop = stop

    self.delta = (stop-start)/(num-1)
    self.delta_inv = 1.0/self.delta


# Parameters for spline basis functions.
Ad = np.array(
  [ -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0
  ])

# Parameters for derivative of spline basis functions.
dAd = np.array(
  (  0.0, -0.5,  1.0, -0.5,
     0.0,  1.5, -2.0,  0.0,
     0.0, -1.5,  1.0,  0.5,
     0.0,  0.5,  0.0,  0.0
  ))

# Parameters for 2nd derivative of spline basis functions.
d2Ad = np.array(
  (  0.0, 0.0, -1.0,  1.0,
     0.0, 0.0,  3.0, -2.0,
     0.0, 0.0, -3.0,  1.0,
     0.0, 0.0,  1.0,  0.0
  ))


def find_coefs_1D_periodic(data):
  N = len(data)
  M = N+3
  a = np.zeros ( (N,N) )

  for i in range(N):
    a[i,i] = 4
    if i < N-1:
      a[i,i+1] = 1
    if i > 0:
      a[i,i-1] = 1

  a[0,N-1] = 1
  a[N-1, 0] = 1

  a = a/6

  # The matrix is sparse, could use more efficient methods
  # for solution (and einspline does)
  x = np.linalg.solve(a, data)
  coefs = np.zeros(M)

  coefs[1:N+1] = x
  coefs[0] = coefs[N]
  coefs[N+1] = coefs[1]
  coefs[N+2] = coefs[2]

  return coefs



class UBSpline1D:
  def __init__(self, bcinfo, grid):
    self.bcinfo = bcinfo
    self.grid = grid

    #if bcinfo.lBC != PERIODIC:
    #  print 'Only Periodic boundaries supported'
    #  assert(False)

    M = grid.num
    if bcinfo.lBC == PERIODIC or bcinfo.lBC == ANTIPERIODIC:
      grid.delta = (grid.stop - grid.start)/grid.num
      grid.delta_inv = 1.0/grid.delta
      ncoefs = M+3
    else:
      grid.delta = (grid.stop - grid.start)/(grid.num - 1)
      grid.delta_inv = 1.0/grid.delta
      ncoefs = M+2

    self.coefs = np.zeros(ncoefs)

    self.tp = np.zeros(4)

  def set_data(self, data):
    coefs = find_coefs_1D_periodic(data)
    self.set_coefs(coefs)

  def set_coefs(self, coefs):
    self.coefs[:] = coefs[:]

  def evaluate_v(self, x):
    x = x - self.grid.start
    u = x * self.grid.delta_inv

    t = u%1
    i = int(u)


    self.tp[0] = t*t*t
    self.tp[1] = t*t
    self.tp[2] = t
    self.tp[3] = 1.0

    # This expression can be further compressed using nested dot products
    val = self.coefs[i+0]*(np.dot(Ad[0:4], self.tp)) + \
          self.coefs[i+1]*(np.dot(Ad[4:8], self.tp)) +  \
          self.coefs[i+2]*(np.dot(Ad[8:12], self.tp)) +  \
          self.coefs[i+3]*(np.dot(Ad[12:16], self.tp))

    return val

  def evaluate_vg(self, x):
    x = x - self.grid.start
    u = x * self.grid.delta_inv

    t = u%1
    i = int(u)

    self.tp[0] = t*t*t
    self.tp[1] = t*t
    self.tp[2] = t
    self.tp[3] = 1.0

    val = self.coefs[i+0]*(np.dot(Ad[0:4], self.tp)) + \
          self.coefs[i+1]*(np.dot(Ad[4:8], self.tp)) + \
          self.coefs[i+2]*(np.dot(Ad[8:12], self.tp)) + \
          self.coefs[i+3]*(np.dot(Ad[12:16], self.tp))

    grad = self.coefs[i+0]*(np.dot(dAd[0:4], self.tp)) +\
           self.coefs[i+1]*(np.dot(dAd[4:8], self.tp)) + \
           self.coefs[i+2]*(np.dot(dAd[8:12], self.tp)) + \
           self.coefs[i+3]*(np.dot(dAd[12:16], self.tp))

    grad = grad * self.grid.delta_inv

    return val, grad

  def evaluate_vgl(self, x):
    x = x - self.grid.start
    u = x * self.grid.delta_inv

    t = u%1
    i = int(u)

    self.tp[0] = t*t*t
    self.tp[1] = t*t
    self.tp[2] = t
    self.tp[3] = 1.0

    val = self.coefs[i+0]*(np.dot(Ad[0:4], self.tp)) + \
          self.coefs[i+1]*(np.dot(Ad[4:8], self.tp)) + \
          self.coefs[i+2]*(np.dot(Ad[8:12], self.tp)) + \
          self.coefs[i+3]*(np.dot(Ad[12:16], self.tp))

    grad = self.coefs[i+0]*(np.dot(dAd[0:4], self.tp)) + \
           self.coefs[i+1]*(np.dot(dAd[4:8], self.tp)) + \
           self.coefs[i+2]*(np.dot(dAd[8:12], self.tp)) + \
           self.coefs[i+3]*(np.dot(dAd[12:16], self.tp))
    grad = grad * self.grid.delta_inv

    lapl = self.coefs[i+0]*(np.dot(d2Ad[0:4], self.tp)) + \
           self.coefs[i+1]*(np.dot(d2Ad[4:8], self.tp)) + \
           self.coefs[i+2]*(np.dot(d2Ad[8:12], self.tp)) + \
           self.coefs[i+3]*(np.dot(d2Ad[12:16], self.tp))
    lapl = lapl*self.grid.delta_inv * self.grid.delta_inv

    return val, grad, lapl

def get_spline_1D(data, a, b):
  '''Create 1D spline. Currently only for periodic boundaries
      data - array of input values
      a,b - end points of interval
  '''

  bc = BCInfo()
  grid = Grid1D(len(data), a, b)
  sp = UBSpline1D(bc, grid)
  sp.set_data(data)
  return sp

def find_coefs_3D_periodic(data):
  NN = data.shape
  MM = [NN[i]+3 for i in range(3)]

  coefs = np.zeros(MM)
  # solve in x direction
  for j in range(data.shape[1]):
    for k in range(data.shape[2]):
      N = data.shape[0]
      M = N+3
      a = np.zeros ( (N,N) )

      for i in range(N):
        a[i,i] = 4
        if i < N-1:
          a[i,i+1] = 1
        if i > 0:
          a[i,i-1] = 1

      a[0,N-1] = 1
      a[N-1, 0] = 1

      a = a/6

      # The matrix is sparse, could use more efficient methods
      # for solution (and einspline does)
      x = np.linalg.solve(a, data[:,j,k])

      coefs[1:N+1,j,k] = x
      coefs[0,j,k] = coefs[N,j,k]
      coefs[N+1,j,k] = coefs[1,j,k]
      coefs[N+2,j,k] = coefs[2,j,k]

  # solve in y direction
  for i in range(coefs.shape[0]):
    for k in range(coefs.shape[2]):
      N = data.shape[0]
      M = N+3
      a = np.zeros ( (N,N) )

      for ix in range(N):
        a[ix,ix] = 4
        if ix < N-1:
          a[ix,ix+1] = 1
        if ix > 0:
          a[ix,ix-1] = 1

      a[0,N-1] = 1
      a[N-1, 0] = 1

      a = a/6

      #print 'a = ',a
      #print ' rhs = ',coefs[i,0:N,k]
      x = np.linalg.solve(a, coefs[i,0:N,k])
      #print 'x  =',x

      coefs[i,1:N+1,k] = x
      coefs[i,0,k] = coefs[i,N,k]
      coefs[i,N+1,k] = coefs[i,1,k]
      coefs[i,N+2,k] = coefs[i,2,k]

  # solve in z direction
  for i in range(coefs.shape[0]):
    for j in range(coefs.shape[1]):
      N = data.shape[0]
      a = np.zeros ( (N,N) )

      for ix in range(N):
        a[ix,ix] = 4
        if ix < N-1:
          a[ix,ix+1] = 1
        if ix > 0:
          a[ix,ix-1] = 1

      a[0,N-1] = 1
      a[N-1, 0] = 1

      a = a/6

      x = np.linalg.solve(a, coefs[i,j,0:N])

      coefs[i,j,1:N+1] = x
      coefs[i,j,0] = coefs[i,j,N]
      coefs[i,j,N+1] = coefs[i,j,1]
      coefs[i,j,N+2] = coefs[i,j,2]

  return coefs




class UBSpline3D:
  def __init__(self, bcinfo, grid):
    # 'grid' is an array of Grid1D of length 3
    self.bcinfo = bcinfo
    self.grid = grid

    if bcinfo.lBC != PERIODIC:
      msg = 'Only Periodic boundaries supported'
      raise RuntimeError(msg)

    ncoefs = [0, 0, 0]
    for idx, grid in enumerate(self.grid):
      M = grid.num
      if bcinfo.lBC == PERIODIC or bcinfo.lBC == ANTIPERIODIC:
        grid.delta = (grid.stop - grid.start)/grid.num
        grid.delta_inv = 1.0/grid.delta
        ncoefs[idx] = M+3
      else:
        grid.delta = (grid.stop - grid.start)/(grid.num - 1)
        grid.delta_inv = 1.0/grid.delta
        ncoefs[idx] = M+2

    self.coefs = np.zeros( (ncoefs[0],ncoefs[1],ncoefs[2]) )

    self.tp = np.zeros(4)

  def set_data(self, data):
    coefs = find_coefs_3D_periodic(data)
    self.set_coefs(coefs)

  def set_coefs(self, coefs):
    self.coefs[:,:,:] = coefs[:,:,:]

  def evaluate_v(self, x):
    tp = self.tp
    aa = np.zeros( (3,4) )
    ii = [0, 0, 0]
    for i in range(3):
      xx = x[i] - self.grid[i].start
      u = xx * self.grid[i].delta_inv
      t = u%1
      ii[i] = int(u)

      tp[0] = t*t*t
      tp[1] = t*t
      tp[2] = t
      tp[3] = 1.0
      aa[i,0] = np.dot(Ad[0:4], tp)
      aa[i,1] = np.dot(Ad[4:8], tp)
      aa[i,2] = np.dot(Ad[8:12], tp)
      aa[i,3] = np.dot(Ad[12:16], tp)


    coefs = self.coefs
    a = aa[0,:]
    b = aa[1,:]
    c = aa[2,:]
    i = ii[0]
    j = ii[1]
    k = ii[2]

    val = 0
    # more compact than einspline, but still not very enlightening
    #bb = np.zeros(4)
    #for idx in range(4):
    #  bb[0] = np.dot(c[:], coefs[i+idx, j, k:k+4])
    #  bb[1] = np.dot(c[:], coefs[i+idx, j+1, k:k+4])
    #  bb[2] = np.dot(c[:], coefs[i+idx, j+2, k:k+4])
    #  bb[3] = np.dot(c[:], coefs[i+idx, j+3, k:k+4])
    #  val += a[idx]*np.dot(b, bb)

    # even more compact, borrowed from multi_UBspline_3d_d code
    for idx in range(4):
      for jdx in range(4):
        val += a[idx]*b[jdx]*np.dot(c[:], coefs[i+idx, j+jdx, k:k+4])

    return val

  def evaluate_vg(self, x):
    tp = self.tp
    aa = np.zeros( (3,4) )
    daa = np.zeros( (3,4) )
    ii = [0, 0, 0]
    for i in range(3):
      xx = x[i] - self.grid[i].start
      u = xx * self.grid[i].delta_inv
      t = u%1
      ii[i] = int(u)

      tp[0] = t*t*t
      tp[1] = t*t
      tp[2] = t
      tp[3] = 1.0
      aa[i,0] = np.dot(Ad[0:4], tp)
      aa[i,1] = np.dot(Ad[4:8], tp)
      aa[i,2] = np.dot(Ad[8:12], tp)
      aa[i,3] = np.dot(Ad[12:16], tp)

      daa[i,0] = np.dot(dAd[0:4], tp)
      daa[i,1] = np.dot(dAd[4:8], tp)
      daa[i,2] = np.dot(dAd[8:12], tp)
      daa[i,3] = np.dot(dAd[12:16], tp)


    coefs = self.coefs
    a = aa[0,:]
    b = aa[1,:]
    c = aa[2,:]
    i = ii[0]
    j = ii[1]
    k = ii[2]
    da = daa[0,:]
    db = daa[1,:]
    dc = daa[2,:]

    # more compact than einspline, but still not very enlightening
    bb = np.zeros(4)
    dbb = np.zeros(4)
    val = 0.0
    gradz = 0.0
    bcP = np.zeros(4)
    dbcP = np.zeros(4)
    for idx in range(4):
      bb[0] =np.dot(c[:], coefs[i+idx, j, k:k+4])
      bb[1] =np.dot(c[:], coefs[i+idx, j+1, k:k+4])
      bb[2] =np.dot(c[:], coefs[i+idx, j+2, k:k+4])
      bb[3] =np.dot(c[:], coefs[i+idx, j+3, k:k+4])
      dbb[0] =np.dot(dc[:], coefs[i+idx, j, k:k+4])
      dbb[1] =np.dot(dc[:], coefs[i+idx, j+1, k:k+4])
      dbb[2] =np.dot(dc[:], coefs[i+idx, j+2, k:k+4])
      dbb[3] =np.dot(dc[:], coefs[i+idx, j+3, k:k+4])
      val += a[idx]*np.dot(b, bb)
      bcP[idx] = np.dot(b, bb)
      dbcP[idx] = np.dot(db, bb)
      gradz += a[idx]*np.dot(b,dbb)

    grad = np.zeros(3)
    grad[0] = self.grid[0].delta_inv * np.dot(da, bcP)
    grad[1] = self.grid[1].delta_inv * np.dot(a, dbcP)
    grad[2] = self.grid[2].delta_inv * gradz

    return val,grad

  def evaluate_vgl(self, x):
    tp = self.tp
    aa = np.zeros( (3,4) )
    daa = np.zeros( (3,4) )
    d2aa = np.zeros( (3,4) )
    ii = [0, 0, 0]
    for i in range(3):
      xx = x[i] - self.grid[i].start
      u = xx * self.grid[i].delta_inv
      t = u%1
      ii[i] = int(u)

      tp[0] = t*t*t
      tp[1] = t*t
      tp[2] = t
      tp[3] = 1.0
      aa[i,0] = np.dot(Ad[0:4], tp)
      aa[i,1] = np.dot(Ad[4:8], tp)
      aa[i,2] = np.dot(Ad[8:12], tp)
      aa[i,3] = np.dot(Ad[12:16], tp)

      daa[i,0] = np.dot(dAd[0:4], tp)
      daa[i,1] = np.dot(dAd[4:8], tp)
      daa[i,2] = np.dot(dAd[8:12], tp)
      daa[i,3] = np.dot(dAd[12:16], tp)

      d2aa[i,0] = np.dot(d2Ad[0:4], tp)
      d2aa[i,1] = np.dot(d2Ad[4:8], tp)
      d2aa[i,2] = np.dot(d2Ad[8:12], tp)
      d2aa[i,3] = np.dot(d2Ad[12:16], tp)


    coefs = self.coefs
    a = aa[0,:]
    b = aa[1,:]
    c = aa[2,:]
    i = ii[0]
    j = ii[1]
    k = ii[2]
    da = daa[0,:]
    db = daa[1,:]
    dc = daa[2,:]
    d2a = d2aa[0,:]
    d2b = d2aa[1,:]
    d2c = d2aa[2,:]

    # more compact than einspline, but still not very enlightening
    bb = np.zeros(4)
    dbb = np.zeros(4)
    d2bb = np.zeros(4)
    val = 0.0
    gradz = 0.0
    lapz = 0.0
    bcP = np.zeros(4)
    dbcP = np.zeros(4)
    d2bcP = np.zeros(4)
    for idx in range(4):
      bb[0] =np.dot(c[:], coefs[i+idx, j, k:k+4])
      bb[1] =np.dot(c[:], coefs[i+idx, j+1, k:k+4])
      bb[2] =np.dot(c[:], coefs[i+idx, j+2, k:k+4])
      bb[3] =np.dot(c[:], coefs[i+idx, j+3, k:k+4])
      dbb[0] =np.dot(dc[:], coefs[i+idx, j, k:k+4])
      dbb[1] =np.dot(dc[:], coefs[i+idx, j+1, k:k+4])
      dbb[2] =np.dot(dc[:], coefs[i+idx, j+2, k:k+4])
      dbb[3] =np.dot(dc[:], coefs[i+idx, j+3, k:k+4])
      d2bb[0] =np.dot(d2c[:], coefs[i+idx, j, k:k+4])
      d2bb[1] =np.dot(d2c[:], coefs[i+idx, j+1, k:k+4])
      d2bb[2] =np.dot(d2c[:], coefs[i+idx, j+2, k:k+4])
      d2bb[3] =np.dot(d2c[:], coefs[i+idx, j+3, k:k+4])
      val += a[idx]*np.dot(b, bb)
      bcP[idx] = np.dot(b, bb)
      dbcP[idx] = np.dot(db, bb)
      d2bcP[idx] = np.dot(d2b, bb)
      gradz += a[idx]*np.dot(b,dbb)
      lapz += a[idx]*np.dot(b,d2bb)

    grad = np.zeros(3)
    grad[0] = self.grid[0].delta_inv * np.dot(da, bcP)
    grad[1] = self.grid[1].delta_inv * np.dot(a, dbcP)
    grad[2] = self.grid[2].delta_inv * gradz


    dx2 = self.grid[0].delta_inv * self.grid[0].delta_inv
    dy2 = self.grid[1].delta_inv * self.grid[1].delta_inv
    dz2 = self.grid[2].delta_inv * self.grid[2].delta_inv

    lapl = dx2*np.dot(d2a, bcP) + \
           dy2*np.dot(a, d2bcP) +  \
           dz2*lapz

    return val,grad,lapl

def get_spline_3D(data, a, b):
  '''Create 3D spline. Currently only for periodic boundaries
      Assumes cubic box.
      data - 3D array of input values
      a,b - box boundaries (assumed same in all dimensions)
  '''
  bc = BCInfo()
  grid = Grid1D(len(data), a, b)
  sp = UBSpline3D(bc, [grid, grid, grid])
  sp.set_data(data)
  return sp


if __name__ == '__main__':
  data = [ 0, 0.951057, 0.587785, -0.587785, -0.951057 ]

  sp = get_spline_1D(data, 0.0, 1.0)

  y = sp.evaluate_v(0.2)
