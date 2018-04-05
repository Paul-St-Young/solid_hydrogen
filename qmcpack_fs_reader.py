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
