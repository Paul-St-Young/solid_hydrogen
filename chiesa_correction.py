import numpy as np
# =========================== get long-range U(k) =========================== 


def isotropic_jk2(jk2_dat):                                                     
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
                                                                                
  unique_k, unique_jk2 = sc.shell_avg_sofk(kvecs,jk2m)                          
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

# ================== basic routines for klist  ==================
def cubic_pos(nx):
  from itertools import product
  pos  = np.array([spos for spos in product(xrange(nx),repeat=3)],dtype=float)
  return pos


def mirror_xyz(pos):
  from itertools import product
  natom, ndim = pos.shape
  new_pos = np.zeros([natom*2**ndim,ndim])

  iflip = 0
  for dim_mult in product([-1,1],repeat=ndim):
    pos1 = pos.copy()
    for idim, mult in zip(xrange(ndim),dim_mult):
      pos1[:,idim] *= mult
    new_pos[natom*iflip:natom*(iflip+1)] = pos1
    iflip += 1
  # end for

  return new_pos


def get_kshells(nk, raxes, atol = 1e-8):
  ukvecs = mirror_xyz( cubic_pos(nk) )

  # throw out k vector at zero
  sel    = np.linalg.norm(ukvecs, axis=1) < atol
  ukvecs = ukvecs[~sel]

  kvecs  = np.dot(ukvecs, raxes)
  return kvecs


def get_iso_jk_kmags(raxes, kshell_max, kc, smear_fac=1000.):
  import chiesa_correction as chc
  # get k vectors
  kvecs = chc.get_kshells(kshell_max, raxes)

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

heg_kfermi = lambda rs:((9*np.pi)/(4.*rs**3.))**(1./3)

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
  hfsk_val  = hfsk(k, kf)      # non-interacting S0(k)
  prefactor = 2*np.pi/3*rs**3  # 1/(2*density)
  arg2sqrt  = hfsk_val**(-2) + 12./(rs**3*k**4)

  # put pieces together
  uk = prefactor * ( -hfsk_val**(-1) + arg2sqrt**0.5 )
  return uk


def ceperley_rpa_uk(k, rs):

  # build pieces
  vol = (4*np.pi/3*rs**3)
  ak = 12/rs**3/k**4  # k in units of rs

  # put pieces together
  uk = vol* 0.5*(-1+(1+ak)**0.5)
  return uk


def evaluate_ft_usr(myk, node, rcut):
  """ return FT[Usr] at given k magnitudes
  assume Bspline Jastrow for 'uu' and 'ud' are in QMCPACK xml input format
  contained in 'node'

  Args:
    myk (list): a list of k vector magnitudes to evaluate FT[Usr] on
    node (lxml.etree.Element): xml node, probably <jastrow>
    rcut (float): coordinate-space cutoff radius
  """
  from jastrow import create_jastrow_from_param
  from scipy.integrate import quad

  # hard-code cusp condition
  uu_cusp = -0.25
  ud_cusp = -0.50

  # read spline knots
  cuu_name = 'uu'
  cud_name = 'ud'

  uu_node = node.find('.//coefficients[@id="%s"]'%cuu_name)
  uu_coeff = np.array(uu_node.text.split(), dtype=float)

  ud_node = node.find('.//coefficients[@id="%s"]'%cud_name)
  ud_coeff = np.array(ud_node.text.split(), dtype=float)

  # construct e-e Usr(r)
  juu = create_jastrow_from_param(uu_coeff, uu_cusp, rcut)
  jud = create_jastrow_from_param(ud_coeff, ud_cusp, rcut)
  j2sr = lambda r: 0.5*( juu.evaluate_v(r) + jud.evaluate_v(r) )

  # perform spherical FT
  integrand = lambda r, k: r*np.sin(k*r)/k* j2sr(r)
  intval = lambda k: quad(lambda r:integrand(r, k), 0, rcut)[0]
  ft = 4*np.pi* np.array([intval(k) for k in myk])
  return ft

