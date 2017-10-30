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
