from nexus import generate_structure

def opt_structure_sfp(sname,func,press,struct_df):
  """ request structure from struct_df using labels (sname,func,press)
  Args:
    sname (str): one of ['cmca4','cmca12','c2c','i41amd']
    func (str): one of ['pbe','blyp','vdw-df']
    press (int): pressure target in kbar
    struct_df (pd.DataFrame): database of structures, must have index (sname,func,press) and columns (axes,pos).
  Returns:
    nexus
  """
  entry = struct_df.loc[sname,func,press]
  axes = entry['axes']
  pos  = entry['pos']
  elem = ['H']*len(pos)

  struct = generate_structure(
    axes = axes,
    pos  = pos,
    elem = elem,
    units= 'B'
  )

  return struct
# end def opt_structure_sfp

def supercell(struct,Tmat,folded=False):
  super_struct = struct.tile(Tmat)
  if not folded:
    super_struct.remove_folded_structure()
  # end if
  return super_struct
# end def

def rs_ca_from_axes_pos(tmat,axes,pos,cidx,aidx=0):
  natom  = len(pos)
  volume = np.dot(axes[0],np.cross(axes[1],axes[2]))
  vol_pp = volume/natom # volume per particle
  rs     = (3.*vol_pp/(4*np.pi))**(1./3)
  axes0  = np.dot(np.linalg.inv(tmat),axes)
  abc = [np.linalg.norm(axes0[i]) for i in range(3)]
  cmag= abc[cidx]
  amag= abc[aidx]
  return rs, cmag/amag
# end def rs_ca_from_axes_pos
