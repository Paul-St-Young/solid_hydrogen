#!/usr/bin/env python
import numpy as np
import scipy.interpolate as interp

from qharv.reel import ascii_out
def get_dsk_amat(floc):                                                         
  """ extract A matrix from qmcfinitesize output
  k->0 behavior of 3D structure factor S(k) is fitted to a Gaussian 
  S(k) = k^T A k

  Args:
    floc (str): location of qmcfinitesize output
  Returns:
    np.array: A matrix (3x3)
  """
  mm = ascii_out.read(floc)                                                     
                                                                                
  amat = np.zeros([3,3])                                                        
                                                                                
  # step 1: fill upper triangular part of amat                                  
  xyzm = {'x':0,'y':1,'z':2}  # map x,y,z to index                              
  keyl = ['a_xx','a_yy','a_zz','a_xy','a_xz','a_yz']                            
  for key in keyl:  # order of key matters!                                     
    val = ascii_out.name_sep_val(mm,key)                                        
    xyz_xyz = key.split('_')[-1]                                                
    idx = tuple([xyzm[xyz] for xyz in xyz_xyz])                                 
    amat[idx] = val                                                             
  # end for                                                                     
                                                                                
  # step 2: symmetrize amat                                                     
  amat[(1,0)] = amat[(0,1)]                                                     
  amat[(2,1)] = amat[(1,2)]                                                     
  amat[(2,0)] = amat[(0,2)]                                                     
                                                                                
  return amat                                                                   
# end def get_dsk_amat


def get_volume(fout):
  mm = ascii_out.read(fout)
  omega = ascii_out.name_sep_val(mm, 'Vol', pos=1)
  return omega


def get_data_block(floc,name):
  start_tag = '#'+name + '_START#'
  stop_tag  = '#'+name + '_STOP#'

  mm = ascii_out.read(floc)                                                     
  text = ascii_out.block_text(mm,start_tag,stop_tag)
  lines= text.split('\n')
  header = lines[0]
  data   = np.array(
    [map(float,line.split()) for line in lines[1:-1]]
  ,dtype=float)
  return data
# end def get_data_block


def add_mixed_vint(df2):
  """ add mixed vint (\int vk Sk) column to extrapolated entries
   df2 must have columns ['timestep','vint'], there must be a timestep=0
   entry, and a timestep > 0 entry.
  Args:
    df2 (pd.DataFrame): DMC database
  Returns:
    None
  """
  df2['vmixed'] = np.nan                                                      
  for subdir in df2.subdir.unique():                                          
    sel = (df2.subdir==subdir)                                                
    ts0_sel = (df2.timestep==0)                                               
    # !!!! assume smallest non-zero timestep is best DMC                      
    min_ts  = df2.loc[sel&(~ts0_sel),'timestep'].min()                        
    ts1_sel = (df2.timestep==min_ts)                                          
                                                                              
    # get mixed vint entry                                                    
    entry = df2.loc[sel&(ts1_sel),'vint']                                     
    assert len(entry) == 1                                                    
    vmixed = entry.values[0]                                                  
                                                                              
    # transfer to pure entry                                                  
    df2.loc[ts0_sel,'vmixed'] = vmixed                                        
  # end for        
# end def add_mixed_vint


# ================= reproduce QMCPACK implementation ================= #
# step 1: get long-range Coulomb pair potential vk
def get_vk(fout):
  """ long-range coulomb pair potential """
  data = get_data_block(fout, 'VK')
  vkx, vky = data.T

  # QMCPACK vk is divided by volume, undo!
  omega = get_volume(fout)
  vky *= omega

  return vkx, vky

def get_fvk(fout):
  """ interpolated long-range coulomb pair potential """
  vkx, vky = get_vk(fout)
  tck = interp.splrep(vkx, vky)
  fvk = lambda k:interp.splev(k, tck)
  return fvk


# step 2: get raw static structure factor S(k)
def get_dsk(fjson, obs='dsk'):
  """ raw structure factor """
  import pandas as pd
  df = pd.read_json(fjson)
  kvecs = np.array(df.loc[0,'kvecs'])
  skm   = np.array(df.loc[0,'%s_mean'%obs])
  ske   = np.array(df.loc[0,'%s_error'%obs])
  return kvecs, skm, ske


# step 3: get sum
def get_vsum(vk, skm, omega):
  """
  skm should contain S(k) values at ALL supercell reciprocal vectors used
  vk should be the same length as skm and NOT divided by volume omega
  """
  summand = 0.5*vk*skm
  vsum = 1/omega* summand.sum()
  return vsum


def get_qmcpack_vsum(fjson, fout):
  kvecs, skm, ske = get_dsk(fjson)
  kmags = np.linalg.norm(kvecs, axis=1)

  fvk = get_fvk(fout)
  vk = fvk(kmags)
  omega = get_volume(fout)

  vsum = get_vsum(vk, skm, omega)
  return vsum


# step 4: get sphericall averaged Savg(k) spline
def get_fsk(fout):
  """ interpolated spherically-averaged structure factor """
  data = get_data_block(fout, 'SK_SPLINE')
  skx, sky = data.T
  tck = interp.splrep(skx, sky)
  fsk = lambda k:interp.splev(k, tck)
  return fsk


# step 4: get 1D integrand
def get_intx_inty(fout):
  fsk = get_fsk(fout)
  vkx, vky = get_vk(fout)
  myinty = 0.5*vkx**2*vky*fsk(vkx)
  return vkx, myinty


# step 5: interpolate 1D integrand
def get_fint(fout):
  intx, inty = get_intx_inty(fout)
  padx = np.array([0.0])
  pady = np.array([0.0]*len(padx))
  myx  = np.concatenate([padx, intx])
  myy  = np.concatenate([pady, inty])
  tck  = interp.splrep(myx, myy)
  fint = lambda k:interp.splev(k, tck)
  return fint

# step 6: get integral
def get_vint(fout):
  from scipy.integrate import quad
  vkx, vky = get_vk(fout)
  fint = get_fint(fout)
  intnorm = 1./(2*np.pi**2)
  intval  = quad(fint,0,max(vkx))[0]
  vint = intnorm * intval
  return vint
