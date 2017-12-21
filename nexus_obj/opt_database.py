import numpy as np
from nexus import generate_structure
from qharv.inspect import axes_pos

tmat72 = {
'c2c':np.array([[2,1,0],[1,2,0],[0,0,1]]),
'cmca4':np.array([[3,3,0],[-1,2,1],[2,-1,1]]),
'cmca12':np.array([[2,1,-1],[-1,1,0],[2,1,1]]),
'i41amd':np.array([[2,-2,1],[2,3,0],[-2,1,1]])
}

rs2ca_pmap = {
'c2c':[0.59540928,0.99294395],
'cmca4':[0.40630975,1.03284052],
'cmca12':[-0.2654712,1.41515302]
}

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

def i4_structure(rs,ca): # ca is c/a ratio
  from nexus import Structure, generate_structure
  bravais_basis = [[0,0,0.5],[0.5,0.5,0.0],[0.5,0.0,0.25],[0.0,0.5,0.75]] # i41amd

  def rs2a(rs,ca,atoms_per_unit_cell):
    space_per_atom = 4.*np.pi/3*rs**3.
    cell_volume    = space_per_atom*atoms_per_unit_cell
    alat = (cell_volume/ca)**(1./3)
    return alat
  # end def rs2a
  alat = rs2a(rs,ca,len(bravais_basis))

  axes = alat*np.array([
    [1,0,0],
    [0,1,0],
    [0,0,ca]
  ])

  pos  = np.dot(bravais_basis,axes)
  elem = ['H']*len(pos)

  structure = generate_structure(
    axes = axes,
    elem = elem,
    pos  = pos,
    units= 'B',
    #tiling    = tiling #  use folded structure
    # to actually make super cell: use tile(), then remove_folded_structure()
  )
  return structure
# end def i4_strucrture

def rs_ca_from_id(myid):
  # myid e.g. c2c-rs1.27-ca1.75
  st,rst,cat = myid.split('-')
  rs = float(rst.replace('rs',''))
  ca = float(cat.replace('ca',''))
  return rs,ca
# end def rs_ca_from_id

def stretch_dimers(axes,pos,frac,rmax=1.5):
  """ multiply all dimer separations by given fraction
  Args:
    axes (np.array): cell
    pos  (np.array): positions
    frac (float): new bond length = frac * current bond length
    rmax (float,optional): atoms separated by r<rmax are considered dimers, default=1.5 for hydrogen
  Returns:
    np.array: pos1, stretch atomic configuration
    np.array: coml, center of mass of the molecules
  """
  import static_correlation as sc
  # modify h2 bond length
  dimer_dict = sc.get_dimers(rmax,axes,pos)
  upair = dimer_dict['upair']
  udist = dimer_dict['udist']
  nmol  = len(upair)
  if nmol != len(pos)/2:
    raise RuntimeError('wrong number of molecules %d'%nmol)
  # end if
  pos1  = pos.copy()
  coml = []
  for imol in range(nmol):
    # get dimer positions
    pair_ij = upair[imol]
    mol_pos = pos[ pair_ij ]

    # get orientation
    bond_vec = axes_pos.displacement(axes,mol_pos[0],mol_pos[1])
    rbond = np.linalg.norm(bond_vec) # pos0-pos1
    if not np.isclose(rbond,udist[imol]):
      raise RuntimeError('molecule mismatch')
    # end if
    ovec = bond_vec/rbond

    # find COM
    com = 0.5*rbond*ovec + mol_pos[1]
    com = axes_pos.pos_in_axes(axes,com)
    coml.append(com)

    # stretch/shrink molecule
    rbond1 = frac*rbond
    mol_pos1 = mol_pos.copy()
    mol_pos1[0] = com - rbond1/2.*ovec
    mol_pos1[1] = com + rbond1/2.*ovec

    pos1[ pair_ij ] = mol_pos1
  # end for imol
  return axes_pos.pos_in_axes(axes,pos1),np.array(coml)
# end def stretch_dimers
