import numpy as np

def gofv(myr, drij, vec):
  """ grid vector quantity over pair separation
  calculate drij[i, j] dot vec[i] and count for each bin

  Args:
    myr (np.array): bin edges for pair separation (nbin,)
    drij (np.array): displacement table (natom, natom, ndim)
    vec (np.array): vector quantity to grid (natom, ndim)
  Return:
    (np.array, np.array): (ysum, counts)
  Example:
    >>> myr = np.linspace(0.5, 3.0, 32)
    >>> vec = np.random.rand(len(drij), 3)
    >>> ysum, counts = gofv(myr, drij, vec)
  """
  # !!!! assume linear grid
  nr = len(myr)
  rmin = myr[0]
  dr = myr[1]-myr[0]
  if not np.isclose(myr[2]-myr[1], dr):
    raise RuntimeError('not linear grid')
  # histogram
  ysum = np.zeros(nr)
  counts = np.zeros(nr, dtype=int)
  rij = np.linalg.norm(drij, axis=-1)
  nmol = rij.shape[0]
  for i in range(nmol):
    for j in range(i+1, nmol):
      deno = rij[i, j]
      ir = int((deno-rmin)//dr)  # bin index
      if (0 <= ir) and (ir < nr):
        nume = np.dot(drij[i, j], vec[i])
        val = nume/deno
        ysum[ir] += val
        counts[ir] += 1
  return ysum, counts

def gofr_norm(myr, natom, volume):
  """ calculate norm needed to turn counts from gofv to
   pair correlation g(r)

  Args:
    myr (np.array): bin edges for pair separation (nbin,)
    natom (int): number of atoms in the box
    volume (float): volume of the box
  Return:
    np.array: nvec, normalization vector g(r) = nvec*[gofv counts]
  Example:
    >>> ysum, counts = gofv(myr, drij, vec)
    >>> nvec = gofr_norm(myr, len(drij), volume)
    >>> gr = nvec*counts
  """
  # !!!! assume 3 dimensions
  ndim = 3
  # !!!! assume linear grid
  nr = len(myr)
  rmin = myr[0]
  dr = myr[1]-myr[0]
  if not np.isclose(myr[2]-myr[1], dr):
    raise RuntimeError('not linear grid')
  # extend grid by 1
  myr1 = np.concatenate([myr, [myr[-1]+dr]])
  # calculate volume differences
  vnorm = np.diff(4*np.pi/3*myr1**ndim)
  # calculate density normalization
  npair = natom*(natom-1)/2
  rho = npair/volume
  # assemble norm vector
  nvec = 1./(rho*vnorm)
  return nvec
