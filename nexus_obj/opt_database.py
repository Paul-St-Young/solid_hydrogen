import numpy as np
from nexus import generate_structure
from qharv.inspect import axes_pos

tmat72 = {
'c2c':np.array([[2,1,0],[1,2,0],[0,0,1]]),
'cmca4':np.array([[3,3,0],[-1,2,1],[2,-1,1]]),
'cmca12':np.array([[2,1,-1],[-1,1,0],[2,1,1]]),
'i41amd':np.array([[2,-2,1],[2,3,0],[-2,1,1]])
}

rwsc_tmat72 = {
'i41amd':np.array([[1,4,3],[4,1,3],[2,2,0]]) # 94a
}

tmat96 = {
'c2c':np.array([[2,0,2],[1,1,0],[0,0,2]]) # 90a
}

rwsc_tmat96 = {
'i41amd':np.array([[0, 4, 4], [2, 1, 5], [2, 2, 0]]) # 94a
}

rwsc_tmat128 = {
'i41amd':np.array([[0, 5, 3], [5, 1, 4], [2, 2, 0]]) # 94a
}

rs2ca_pmap = {
'c2c':[0.59540928,0.99294395], # 36
'cmca4':[0.40630975,1.03284052], # 42
'cmca12':[-0.2654712,1.41515302], # 43a
'i41amd':[-1.73310421,4.53315281] # 44
}

rs2sig_pmap = { # 45
'c2c':[0.10162707,0.09047979],
'cmca4':[0.18923581,-0.0053223],
'cmca12':[0.17921345,0.00393995],
'i41amd':[0.19753849,-0.00268172]
}

rs2rb_pmap = { # 50/init/bond
'c2c':[0.18850855,1.11832306],
'cmca4':[0.12621216,1.25741813],
'cmca12':[ 0.20065344,1.1204539]
}

rs2carb_pmap = { # 55/sname-rs_ca-rb_grid/grid.py
'c2c':{'ca':[0.25205619,1.43017007],'rb':[-1.33414778,3.53568222,-0.97115101]},
'cmca4':{'ca':[0.28167668,1.19844864],'rb':[-3.41019865,8.77500764,-4.21965519]},
'cmca12':{'ca':[-0.16499329,1.28828894],'rb':[-2.43595819,6.40101745,-2.81177197]}
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

def i4_prim(rs,ca):
  volpp   = 4./3*np.pi*rs**3.
  upos = np.array([
    [0.50,0.50,1.00],
    [0.25,0.75,0.50]
  ])

  # construct conventional cell
  natom   = 4
  volconv = volpp*natom
  alat = (volconv/ca)**(1./3)
  clat = alat*ca
  axes = np.array([[alat,0,0],[0,alat,0],[0,0,clat]])

  # get primitive cell
  fcc0 = np.ones(3)-np.eye(3)
  axes0= np.dot(np.linalg.inv(fcc0),axes)
  pos0 = np.dot(upos,axes0)
  return axes0,pos0
# end def i4_prim

def i4_conv(rs, ca):
  axes0, pos0 = i4_prim(rs, ca)
  tmat = np.ones(3) - np.eye(3)
  from ase import Atoms
  from ase.build import make_supercell
  s0 = Atoms('H%d'%len(pos0), cell=axes0, positions=pos0, pbc=[1,1,1])
  s1 = make_supercell(s0, tmat)
  axes1 = s1.get_cell()
  pos1  = s1.get_positions()
  return axes1, pos1
# end def i4_conv

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
  # modify h2 bond length
  upair,udist = axes_pos.dimer_pairs_and_dists(axes,pos,rmax)
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


def nexus_tile(axes0, pos0, tmat):
  """ tile a single species of particles from primitive to supercell

  for multiple species, tile one species at a time, then concatenate.

  Args:
    axes0 (np.array): primitive lattice vectors
    pos0 (np.array): primitive atomic positions
    tmat (np.array): tile matrix
  Return:
    (np.array, np.array): axes, pos in supercell
  """
  from nexus import Structure
  s0 = Structure(axes=axes0, pos=pos0, elem=['H']*len(pos0), units='B')
  s1 = s0.tile(tmat)
  return s1.axes, s1.pos


def ase_tile(axes0, pos0, tmat):
  """ tile a single species of particles from primitive to supercell

  for multiple species, tile one species at a time, then concatenate.

  Args:
    axes0 (np.array): primitive lattice vectors
    pos0 (np.array): primitive atomic positions
    tmat (np.array): tile matrix
  Return:
    (np.array, np.array): axes, pos in supercell
  """
  from ase import Atoms
  from ase.build import make_supercell
  s0 = Atoms('H%d' % len(pos0), cell=axes0, positions=pos0, pbc=[1, 1, 1])
  s1 = make_supercell(s0, tmat)
  return s1.get_cell(), s1.get_positions()
  s1 = s0.tile(tmat)
  return s1.axes, s1.pos
