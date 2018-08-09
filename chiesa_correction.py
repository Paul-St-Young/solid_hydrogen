import numpy as np
# =========================== get long-range U(k) =========================== 


def isotropic_jk2(jk2_dat, **kwargs):
  """ extract ee long-range Jastrow from Jk2.dat 
  Jk2.dat should have kx,ky,kz, coeff_real columns

  This function currently assumes an extra coeff_imag columns and checks to 
  make sure it is uniformly 0. Simply comment out the assert to remove.

  Args:
    jk2_dat (str): filename of Jk2.dat
  Returns:
    np.array, np.array: unique k-vector magnitudes, shell-averaged Jk2
  """
  import static_correlation as sc                                               
  data  = np.loadtxt(jk2_dat)                                                   
  kvecs = data[:,:3]                                                            
  jk2m  = -data[:,3]  # conform to finite temperature DM sign convention        
  assert np.allclose(data[:,4],0)  # imaginary part of Jk2 is zero              
  kmags = np.linalg.norm(kvecs,axis=1)                                          
                                                                                
  unique_k, unique_jk2 = sc.shell_avg_sofk(kvecs, jk2m, **kwargs)
  return unique_k, unique_jk2                                                   
# end def isotropic_jk2


# ============================== get full U(k) ============================== 
# The way FT[Usr] is loaded can be improved. Currently I read the sampled 
# values of FT[Usr] on a 100-long 1D kgrid, then Bspline interpolate.
# It would be better to read input file for Bspline knots directly, construct
# Usr(r) as a function, then get FT at desired k vectors.
# TODO: edit isotropic_ftur to do the above.


def isotropic_ftur(ftur_dat,smear=100):                                         
  import static_correlation as sc                                               
  myk, fturm, fture = np.loadtxt(ftur_dat).T                                    
  kint    = map(int,map(round,myk*smear))                                       
  kshells = np.unique(kint)                                                     
                                                                                
  # shell average                                                               
  sels  = [kint == kshell for kshell in kshells]                                
  kvals = np.array([myk[sel].mean() for sel in sels])                           
  ukvals= np.array([fturm[sel].mean() for sel in sels])                         
  return kvals,ukvals                                                           
# end def
                                                                                
                                                                                
def isotropic_uk(kmags,jk2m,myk,fturm):                                         
  import scipy.interpolate as interp                                            
  tck = interp.splrep(myk,fturm)                                                
                                                                                
  dk = (kmags[1:] - kmags[:-1]).max()                                           
  finek = np.concatenate([kmags, np.arange(kmags.max()+dk,myk.max(),dk)])       
  finey = interp.splev(finek,tck)                                               
  sel = finek <= kmags.max()                                                    
  finey[sel] += jk2m                                                            
  return finek, finey                                                           
# end def
                                                                                

def get_uk(jk2_dat,ftur_dat):                                                   
  # load isotropic long-range Jastrow                                           
  kmags, jk2m = isotropic_jk2(jk2_dat)                                          
                                                                                
  # load FT[short-range Jastrow]                                                
  myk, fturm  = isotropic_ftur(ftur_dat)                                        
                                                                                
  # construct full U(k)                                                         
  totk, uk = isotropic_uk(kmags,jk2m,myk,fturm)                                 
  return totk, uk                                                               
# end def


chiesa_rpa_uk = lambda k,a,b:4*np.pi*a*(k**(-2)-(k**2+b**(-1))**(-1))
drum_rpa_uk   = lambda k,A,B:4*np.pi*(A/k**2+B/k)

def fit_rpa_uk(model, totk, uk, kmax):
  import scipy.optimize as op
  sel = (totk <= kmax)
  popt,pcov = op.curve_fit(model,totk[sel],uk[sel])

  fuk = lambda k:model(k,*popt)
  return fuk,popt
# end def


def drum_uk(k, wp, kf):
  """ swap parameters A,B out for effective plasmon freq. wp and Fermi kf """
  A = 4*np.pi/wp
  B = -2*np.pi**2/kf**2
  return A/k**2+B/k

# ================== basic routines for klist  ==================
def cubic_pos(nx):
  from itertools import product
  pos  = np.array([
    spos for spos in product(xrange(nx), repeat=3)
    ], dtype=int)
  return pos


def shifted_mp_grid(nx):
  pos = cubic_pos(nx)/float(nx) + 0.5/nx
  pos = (pos+0.5)%1-0.5
  return pos


def remove_com(pos):
  """ remove the center of mass (com)
  assume equally weighted particles

  Args:
    pos (np.array): position array
  Return:
    np.array: new position array with zero com
  """
  com = pos.mean(axis=0)
  return pos - com[np.newaxis, :]


def mirror_xyz(pos):
  from itertools import product
  natom, ndim = pos.shape
  new_pos = np.zeros([natom*2**ndim, ndim], dtype=pos.dtype)

  iflip = 0
  for dim_mult in product([-1, 1], repeat=ndim):
    pos1 = pos.copy()
    for idim, mult in zip(xrange(ndim), dim_mult):
      pos1[:, idim] *= mult
    new_pos[natom*iflip:natom*(iflip+1)] = pos1
    iflip += 1
  # end for

  return new_pos


def get_kshells(nk, raxes, atol = 1e-8):
  ukvecs = mirror_xyz( cubic_pos(nk) )

  # remove non-unique
  ukvecs = np.unique(ukvecs, axis=0)

  # throw out k vector at zero
  sel    = np.linalg.norm(ukvecs, axis=1) < atol
  ukvecs = ukvecs[~sel]

  kvecs  = np.dot(ukvecs, raxes)
  return kvecs


def get_iso_jk_kmags(raxes, kshell_max, kc, smear_fac=1000.):
  # get k vectors
  kvecs = get_kshells(kshell_max, raxes)

  # get unique k vectors given cubic symmetry
  # step 1: throw out half sphere due to inversion symmetry
  sel   = kvecs[:, 0] >= 0
  kvecs = kvecs[sel]
  # step 2: throw out k vectors > kc
  kmags = np.linalg.norm(kvecs, axis=1)
  kcsel = kmags < kc
  kvecs = kvecs[kcsel]
  # step 3: use isotropic symmetry
  kmags = np.linalg.norm(kvecs, axis=1)
  kshs  = (kmags*smear_fac).astype(int)
  uk, uidx = np.unique(kshs, return_index=True)

  return kmags[uidx]


def get_jk_kvecs(fout):
  """ parse QMCPACK output for kSpace Jastrow kvecs """
  from qharv.reel import ascii_out

  header  = 'kSpace coefficent groups'
  trailer = 'QMCHamiltonian::addOperator'

  mm   = ascii_out.read(fout)
  text = ascii_out.block_text(mm, header, trailer)

  data = []
  for line in text.split('\n'):
    tokens = line.split()
    if len(tokens) != 4: continue
    data.append( map(float, tokens) )

  data  = np.array(data)
  kvecs = data[:,:3]
  kmags = data[:,3]
  return kvecs

# ================ routines for structure factor S(k)  ================


def heg_kfermi(rs):
  """ magnitude of the fermi k vector for the homogeneous electron gas (HEG)

  Args:
    rs (float): Wigner-Seitz radius
  Return:
    float: kf
  """
  density = (4*np.pi*rs**3/3)**(-1)
  kf = (3*np.pi**2*density)**(1./3)
  return kf


def hfsk(karr, kf):
  """ static structure factor of non-interacting Fermions
  Args:
    karr (np.array): k vector magnitudes
    kf (float): Fermi k vector magnitude
  Return:
    float: S0(k, kf)
  """
  skm = 3*karr/(4*kf)-karr**3/(16*kf**3)
  sel = np.where(karr>=2*kf)
  skm[sel] = 1.0
  return skm


def effective_fsk_from_fuk(fuk, rs):
  """ construct S(k) from U(k) assuming HEG kf and RPA """
  density = (4*np.pi*rs**3/3)**(-1)
  kf = heg_kfermi(rs)
  return lambda k:(hfsk(k, kf)**(-1) + 2*density*fuk(k))**(-1)


def effective_fuk_from_fsk(fsk, rs):
  """ construct U(k) from S(k) assuming HEG kf and RPA """
  density = (4*np.pi*rs**3/3)**(-1)
  kf = heg_kfermi(rs)
  return lambda k:( fsk(k)**(-1) - hfsk(k, kf)**(-1) )/(2*density)


def load_dsk(fjson, obs='dsk'):
  import pandas as pd
  df = pd.read_json(fjson)
  kvecs = np.array(df.loc[0,'kvecs'])
  skm   = np.array(df.loc[0,'%s_mean'%obs])
  ske   = np.array(df.loc[0,'%s_error'%obs])
  return kvecs, skm, ske


# ================ routines for jastrow potential U(k)  ================


def gaskell_rpa_uk(k, rs, kf):

  # build pieces
  hfsk_val  = hfsk(k, kf)      # determinant contribution S0(k)
  prefactor = 2*np.pi/3*rs**3  # 1/(2*density)
  arg2sqrt  = hfsk_val**(-2) + 12./(rs**3*k**4)

  # put pieces together
  uk = prefactor * ( -hfsk_val**(-1) + arg2sqrt**0.5 )
  return uk


def gaskell_rpa_sk(k, rs, kf):

  rho = 3./(4*np.pi*rs**3)
  hfsk_val  = hfsk(k, kf)      # determinant contribution S0(k)
  sk = hfsk_val*(1+16.*np.pi*rho*hfsk_val**2/k**4)**(-0.5)
  return sk


def mixed_rpa_sk(k, rs, sk0, ktf):
  rho = 3./(4*np.pi*rs**3)
  vk = 4*np.pi/k**2
  vk1 = 4*np.pi/(k**2+ktf**2)
  ek = 0.5*k**2

  ak = 1.+2*rho*sk0**2*vk/ek
  ak1 = 1.+2*rho*sk0**2*vk1/ek
  return sk0*( 2./(ak**0.5+ak1**0.5) )


def ceperley_rpa_uk(k, rs):

  # build pieces
  vol = (4*np.pi/3*rs**3)
  ak = 12/rs**3/k**4  # k in units of rs

  # put pieces together
  uk = vol* 0.5*(-1+(1+ak)**0.5)
  return uk


def fusr(rcut, uu_coeff, ud_coeff, uu_cusp=-0.25, ud_cusp=-0.50):
  """ construct short-range Jastrow potential function
  using Bspline knots in QMCPACK

  Args:
    rcut (float): real-space cutoff
    uu_coeff (list): a list of floats for up-up Jastrow Bspline knots
    ud_coeff (list): a list of floats for up-down Jastrow Bspline knots
    uu_cusp (float, optional): up-up cusp, default -0.25 
    ud_cusp (float, optional): up-down cusp, default -0.50
  Return:
    callable: short-range Jastrow potential
  """
  from jastrow import create_jastrow_from_param
  juu = create_jastrow_from_param(uu_coeff, uu_cusp, rcut)
  jud = create_jastrow_from_param(ud_coeff, ud_cusp, rcut)
  fusr = lambda r: 0.5*( juu.evaluate_v(r) + jud.evaluate_v(r) )
  return fusr


def evaluate_ft_usr(myk, node, rcut):
  """ return FT[Usr] at given k magnitudes
  assume Bspline Jastrow for 'uu' and 'ud' are in QMCPACK xml input format
  contained in 'node'

  Args:
    myk (list): a list of k vector magnitudes to evaluate FT[Usr] on
    node (lxml.etree.Element): xml node, probably <jastrow>
    rcut (float): coordinate-space cutoff radius
  """
  from scipy.integrate import quad

  # read spline knots
  cuu_name = 'uu'
  cud_name = 'ud'

  uu_node = node.find('.//coefficients[@id="%s"]'%cuu_name)
  uu_coeff = np.array(uu_node.text.split(), dtype=float)

  ud_node = node.find('.//coefficients[@id="%s"]'%cud_name)
  ud_coeff = np.array(ud_node.text.split(), dtype=float)

  # construct e-e Usr(r)
  j2sr = fusr(rcut, uu_coeff, ud_coeff)


  # perform spherical FT
  integrand = lambda r, k: r*np.sin(k*r)/k* j2sr(r)
  intval = lambda k: quad(lambda r:integrand(r, k), 0, rcut)[0]
  ft = 4*np.pi* np.array([intval(k) for k in myk])
  return ft


# =========================== density quantities =========================== 


def rs_kf_wp(rho):
  rs = (3./(4*np.pi*rho))**(1./3)
  kf = (3*np.pi**2*rho)**(1./3)
  wp = (4*np.pi*rho)**0.5
  return rs, kf, wp


# =========================== pure RPA FSC ===========================


def mp_grid(raxes, nx):
  """ shifted uniform grid
  Args:
    raxes (np.array): reciprocal space lattice
    nx (int): number of grid points in each dimension
  """
  ugrid = 1./nx* cubic_pos(nx)
  ucom = ugrid.mean(axis=0)
  ugrid -= ucom[np.newaxis, :]

  qgrid = np.dot(ugrid, raxes)
  return qgrid


def rpa_dv(axes, rs, nx=32):
  """ calculate potential finite size error assuming RPA S(k)

  Args:
    raxes (np.array): reciprocal space lattice
    rs (float): Wigner-Seitz radius i.e. density parameter
  Return:
    float: potential finite size correction
  """
  from qharv.inspect import axes_pos

  # get RPA S(k)
  kf = heg_kfermi(rs)

  fuk = lambda k:gaskell_rpa_uk(k, rs, kf)
  fsk = effective_fsk_from_fuk(fuk, rs)

  # Coulomb v(k)
  fvk = lambda k:4*np.pi/k**2

  # setup integration grid
  raxes = axes_pos.raxes(axes)
  qgrid = mp_grid(raxes, nx)

  # perform quadrature
  #  weights
  rvol = axes_pos.volume(raxes)
  intnorm = 1./(2*np.pi)**3* rvol/nx**3
  #  sum
  qmags = np.linalg.norm(qgrid, axis=1)
  integrand = lambda k:0.5*fvk(k)*fsk(k)
  dvlr = intnorm* integrand(qmags).sum()
  return dvlr


def rpa_dt(axes, rs, nx=32):
  """ calculate kinetic finite size error assuming RPA U(k) and S(k)

  Args:
    raxes (np.array): reciprocal space lattice
    rs (float): Wigner-Seitz radius i.e. density parameter
  Return:
    float: kinetic finite size correction
  """
  from qharv.inspect import axes_pos

  density = (4*np.pi/3*rs**3.)**(-1)

  # get RPA U(k) and S(k)
  kf = heg_kfermi(rs)

  fuk = lambda k:gaskell_rpa_uk(k, rs, kf)
  fsk = effective_fsk_from_fuk(fuk, rs)

  # setup integration grid
  raxes = axes_pos.raxes(axes)
  qgrid = mp_grid(raxes, nx)

  # perform quadrature
  #  weights
  rvol = axes_pos.volume(raxes)
  intnorm = 1./(2*np.pi)**3* rvol/nx**3* density
  #  sum
  qmags = np.linalg.norm(qgrid, axis=1)
  integrand = lambda k:0.5*k**2*fuk(k)**2*fsk(k)
  dtlr = intnorm* integrand(qmags).sum()
  return dtlr

# ================ routines for spherical average  ================


def get_regular_grid_dimensions(gvecs):
  gmin = gvecs.min(axis=0)
  gmax = gvecs.max(axis=0)
  ng = np.around(gmax-gmin+1).astype(int)
  return gmin, gmax, ng


def get_index3d(gvec, gmin, dg):
  idx3d = np.around( (gvec - gmin)/dg )
  return idx3d.astype(int)


def fill_regular_grid(gvecs, skm, fill_value=np.nan):
  """ Fill a regular grid with given scalar field (gvecs, skm).
  gvecs should fill (at least parts of) a structured grid consiting of
  congruent rectangular parallelepipeds. By default, missing points are
  filled with np.nan. User may construct a boolean selector for the
  missing points, then go through and fill them.

  Example:
    kvecs, skm, ske = read_sofk('sofk.dat')  # read S(k) in Cartesian units
    gvecs = np.dot(kvecs, np.linalg.inv(raxes))  # convert to lattice units
    rgvecs, rskm = fill_regular_grid(gvecs, skm)

    # set S(0) to 0
    zsel = (np.linalg.norm(rgvecs, axis=1) == 0).reshape(rskm.shape)
    rskm[zsel] = 0

    # fill k>kc with max( S(k) )
    msel = np.isnan(skm)
    rskm[msel] = skm.max()

  Args:
    gvecs (np.array): reciprocal space points in lattice units
    skm (np.array): S(k) at gvecs
    fill_value (float, optiona): default is np.nan
  Return:
    tuple: (gvecs, rgrid), regular grid basis (gvecs) and values (rgrid).
  """
  from itertools import product
  gdtype = gvecs.dtype
  sdtype = skm.dtype

  ndim = gvecs.shape[1]
  if ndim != 3:
    raise RuntimeError('need to generalize to %d dimensions' % ndim)
  gmin, gmax, ng = get_regular_grid_dimensions(gvecs)
  dg = (gmax-gmin)/(ng-1)

  # construct grid points
  grid_gvecs_iter = product(
    np.linspace(gmin[0], gmax[0], ng[0]),
    np.linspace(gmin[1], gmax[1], ng[1]),
    np.linspace(gmin[2], gmax[2], ng[2]),
  )
  rgvecs = np.array([spos for spos in grid_gvecs_iter], dtype=gdtype)

  # initialize regular grid
  rgrid = np.empty(ng, dtype=sdtype)
  filled = np.zeros(ng, dtype=bool)  # keep track of filled grid points
  rgrid.fill(fill_value)

  # fill regular grid
  for gvec, sk in zip(gvecs, skm):
    idx3d = get_index3d(gvec, gmin, dg)
    rgrid[tuple(idx3d)] = sk
    filled[tuple(idx3d)] = True

  # check that data points did not overlap
  nfill = len(filled[filled])
  if nfill != len(skm):
    raise RuntimeError('%d/%d input data retained' % (nfill, len(skm)))
  return rgvecs, rgrid


def get_skgrid(kvecs, dskm, raxes):
  """ example usage of fill_regular_grid for S(k)

  Args:
    kvecs (np.array): reciprocal space vectors with S(k) data
    dskm (np.array): S(k) data
    raxes (np.array): reciprocal space lattice
  Return:
    tuple: (rkvecs, rdskm), kvectors and S(k) on regular grid
  """

  # get a regular grid in lattice units
  gvecs = np.dot(kvecs, np.linalg.inv(raxes))
  rgvecs, rdskm = fill_regular_grid(gvecs, dskm)
  rdskm = rdskm.flatten()

  # fill S(0)
  zsel = (np.linalg.norm(rgvecs, axis=1)==0)
  rdskm[zsel] = 0

  # fill S(k>kc)
  msel = np.isnan(rdskm)
  rdskm[msel] = dskm.max()

  # convert back to Cartesian coordinate
  rkvecs = np.dot(rgvecs, raxes)
  return rkvecs, rdskm


# ================ routines to analyze spherical average  ================


def get_hess_mat(hess):
  """ construct hessian matrix stored in upper-triangular form

  Args:
    hess (array-like): 6-element array storing unique hessian matrix elements
  Return:
    np.array: hmat, the hessian matrix
  """
  hxx, hyy, hzz, hxy, hxz, hyz = hess
  hmat = np.zeros([3, 3])
  # diagnoal
  hmat[0][0] = hxx
  hmat[1][1] = hyy
  hmat[2][2] = hzz
  # off-diagnoal
  hmat[0][1] = hxy
  hmat[0][2] = hxz
  hmat[1][2] = hyz
  #hmat = 0.5*(hmat+hmat.T)
  hmat[1][0] = hxy
  hmat[2][0] = hxz
  hmat[2][1] = hyz
  return hmat

# ================ routines for free fermion determinant  ================


def freec(iorb, midx):
  """ PW coefficients for the ith free fermion orbital
  e.g. freec(0, np.argsort(gmags))
  see usage in get_free_cmat

  Args:
    iorb (int): orbital index
    midx (np.array): the list of indices that sort the gvectors
  Return:
    np.array: PW coefficients
  """
  ci = np.zeros(len(midx))
  ci[midx[iorb]] = 1.0
  return ci


def get_free_cmat(norb, gvecs):
  """ get coefficient matrix for ground state of norb fermions in gvecs PWs

  Args:
    norb (int): number of occupied orbitals (= # of fermions)
    gvecs (np.array): PW basis as integer vectors (i.e. in rec. lat. basis)
  Return:
    np.array: cmat, coefficient matrix (norb, npw)
  """
  npw = len(gvecs)
  gmags = np.linalg.norm(gvecs, axis=1)
  midx = np.argsort(gmags)
  cmat = np.zeros([norb, npw])
  for iorb in range(norb):
    cmat[iorb, :] = freec(iorb, midx)
  return cmat


def get_gvecs(nsh):
  """ get nsh shells of gvectors

  Args:
    nsh (int): number of shells to get
  Return:
    np.array: gvectors in units of reciprocal lattice
  """
  gvecs = cubic_pos(2*nsh+1)
  com = gvecs.mean(axis=0)
  gvecs -= com.astype(int)
  return gvecs


# ================ routines for any determinant in PW basis  ================


def select_from_rgrid(gvecs, rgvecs):
  """ select a subset of integer vectors from a regular grid
  Example:
    idx = select_from_rgrid(gvecs0, rgvecs)
    assert np.allclose(gvecs0, rgvecs[idx])

  Args:
    gvecs  (np.array): subset of integer vectors
    rgvecs (np.array): regular grid of integer vectors
  Return:
    np.array: a list of indices
  """
  gmin, gmax, ng = get_regular_grid_dimensions(rgvecs)
  idx3d = (gvecs-gmin).T
  idx3d = np.around(idx3d).astype(int)
  idx = np.ravel_multi_index(idx3d, ng)
  return idx


def select(gvecs0, gvecs):
  """ select a subset of gvectors (gvecs0) from the full set gvecs
  Example:
    sel = select(gvecs0, gvecs)
    assert np.allclose(gvecs0, gvecs[sel])

  Args:
    gvecs0 (np.array): subset of integer vectors
    gvecs (np.array): full set of integer vectors
  Return:
    np.array: a list of indices
  """
  gmin, gmax, ng = get_regular_grid_dimensions(gvecs)
  idx0 = select_from_rgrid(gvecs0, gvecs)

  idx = select_from_rgrid(gvecs, gvecs)
  myidx = np.zeros(len(idx0), dtype=int)
  for label, i in enumerate(idx0):
    j = np.where(idx==i)[0][0]
    myidx[label] = j
  return myidx


def get_mijq(gvecs, cmat, gvecs0, gvecs_regular=False):
  """ calculate 1RDMs needed to construct S(k)

  Args:
    gvecs (np.array): integer vectors specifying the PW basis
    cmat (np.array): orbital coefficients as rows (norb, npw)
    gvecs0 (np.array): requested momentum vectors
    gvecs_regular (bool, optional): gvecs fill a regular grid, default False
  Return:
    np.array: mijq (nq, norb*norb)
  """
  dtype = cmat.dtype
  norb, npw = cmat.shape
  if npw != len(gvecs):
    raise RuntimeError('wrong basis for coefficient matrix')

  sel_func = select
  if gvecs_regular:
    sel_func = select_from_rgrid

  # select unshifted grid
  idx0 = sel_func(gvecs0, gvecs)

  mijq = np.zeros([len(gvecs0), norb*norb], dtype=dtype)
  # for each qvec, select a shifted grid
  for iq, qvec in enumerate(gvecs0):
    idx = sel_func(gvecs0+qvec, gvecs)
    mij = np.zeros([norb, norb], dtype=dtype)
    for iorb in range(norb):
      for jorb in range(norb):
        mij[iorb][jorb] = np.dot(cmat[iorb, idx0].conj(), cmat[jorb, idx])
    mijq[iq, :] = mij.ravel()
  return mijq


def get_sk(mijq):
  """ calculate structure factor from a list of 1RDMs in PW basis
  must reshape mijq into a list of matrices instead of a list of vectors
  Example:
    mijq = get_mijq(gvecs, cmat, gvecs0)
    norb, npw = cmat.shape
    ng = len(gvecs0)
    skm0 = get_sk(mijq.reshape(ng, norb, norb))

    ax.plot(np.linalg.norm(gvecs0, axis=1), skm0, 'x')

  Args:
    mijq (np.array): 1-body reduced density matrices, having shape (nq, no, no)
  Return:
    np.array: static structure factor, one at each qvec
  """
  nq, norb1, norb2 = mijq.shape
  if norb1 != norb2:
    raise RuntimeError('mij is not a square matrix')
  norb = norb1
  skm0 = []
  for iq in range(len(mijq)):
    mij = mijq[iq].reshape(norb, norb)
    term1 = np.abs(np.diag(mij).sum())
    term2 = (mij.conj()*mij).sum().real
    skval = 1.+(term1-term2)/norb
    skm0.append(skval)
  return np.array(skm0)
