import numpy as np

def disp_in_box(drij, box):
  """ Enforce minimum image convention (MIC) on displacement vectors.

  Args:
    drij (np.array): ri - rj for all pairs of particles rij
    box (np.array): side lengths of orthombic box
  Return:
    np.array: displacement vectors under MIC.
  """
  ndim = drij.shape[-1]
  try:
    ndim1 = len(box)  # except if box is float
  except TypeError as err:
    #nint = np.around(drij/box)
    #return drij-box*nint
    raise err
  assert ndim1 == ndim
  for idim in range(ndim):
    nint = np.around(drij[:, :, idim]/box[idim])
    drij[:, :, idim] -= box[idim]*nint
  return drij

def displacement_table(pos, box, mic=True):
  """ Calculate displacements ri-rj between all pairs of particles.

  Args:
    pos (np.array): particle positions
    box (float): side lengths of box
    mic (bool, optional): enforce minimum image convention (MIC)
  Return:
    np.array: drij, a table of displacement vectors
     shape (natom, natom, ndim)
  """
  drij = pos[:, np.newaxis] - pos[np.newaxis]
  if mic:
    disp_in_box(drij, box)
  return drij

def gofv(myr, drij, vec):
  """ grid vector quantity over pair separation
  calculate drij[i, j] dot vec[i] and count for each bin

  see usage in grid_force_difference

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
        rhat = -drij[i, j]  # drij is dr_i - dr_j NOT ri->rj
        nume = np.dot(rhat, vec[i])
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

def ysum_ysq_count(ysum, ysq, counts):
  """ calculate mean and error given accumulated sum of val and sq

  Args:
    ysum (np.array): accumulated sum of values, (nentry,)
    ysq (np.array): accumulated sum of squares, (nentry,)
    counts (np.array): number of hits for each entry, (nentry,)
  Return:
    (np.array, np.array, np.array): (sel, ym, ye),
      (valid entries, mean, error)
  Example:
    >>> nr = len(myr)
    >>> ysum = np.zeros(nr)
    >>> ysq = np.zeros(nr)
    >>> csum = np.zeros(nr, dtype=int)
    >>> csq = np.zeros(nr, dtype=int)
    >>> for box, pos, vec in zip(boxl, posl, vecl):
    >>>   drij = displacement_table(pos, box)
    >>>   y1, c1 = gofv(myr, drij, vec)
    >>>   ysum += y1
    >>>   ysq += y1**2
    >>>   csum += c1
    >>>   csq += c1**2
    >>> sel, ym, ye = ysum_ysq_count(ysum, ysq, csum)
    >>> myx = myr[sel]
    >>> # g(r)
    >>> grcnt = len(boxl)*np.ones(len(myy))
    >>> sel1, grm, gre = ysum_ysq_count(csum, csq, grcnt)
    >>> nvec = gofr_norm(myr, len(drij), volume)
    >>> grm *= nvec; gre *= nvec
  """
  sel = counts > 1  # need to divide by counts-1
  ym = ysum[sel]/counts[sel]
  y2 = ysq[sel]/counts[sel]
  ye2 = (y2-ym**2)/(counts[sel]-1)
  ye = ye2**0.5
  return sel, ym, ye

def grid_force_difference(myr, boxl, posl, vecl):
  nr = len(myr)
  ysum = np.zeros(nr)
  ysq = np.zeros(nr)
  counts = np.zeros(nr, dtype=int)
  csq = np.zeros(nr, dtype=int)
  for box, pos, vec in zip(boxl, posl, vecl):
    drij = displacement_table(pos, box)
    y1, c1 = gofv(myr, drij, vec)
    ysum += y1
    ysq += y1**2
    counts += c1
    csq += c1**2
  nconf = len(boxl)
  natom = len(posl[0])
  sel, ym, ye = ysum_ysq_count(ysum, ysq, counts)
  myx = myr[sel]
  sel, cm, ce = ysum_ysq_count(counts, csq, nconf*np.ones(len(myr)))
  nvec = gofr_norm(myr, natom, np.prod(box))
  grm = nvec*cm
  gre = nvec*ce
  return myx, ym, ye, grm, gre