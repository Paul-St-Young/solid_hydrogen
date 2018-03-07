#!/usr/bin/env python
import numpy as np

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
