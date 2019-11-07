import numpy as np

def get_path(ref_dir, temp=1500, rs=1.51, natom=96):
  path = '%s/T%d/RS%3.2fN%d' % (ref_dir, temp, rs, natom)
  return path

def get_confs(path, confl=None, ndim=3, verbose=True):
  from qharv.reel import mole
  if verbose:
    msg = 'reading conf.s from %s' % path
    print(msg)
  fbox = mole.find('*_box.raw', path)
  boxes = np.loadtxt(fbox)
  fene = mole.find('*_energy.raw', path)
  eall = np.loadtxt(fene)
  fvir = mole.find('*_virial.raw', path)
  vall = np.loadtxt(fvir)
  fpos = mole.find('*_coord.raw', path)
  posa = np.loadtxt(fpos)
  ffor = mole.find('*_force.raw', path)
  fora = np.loadtxt(ffor)
  ha = 1./27.211386245988
  bohr = 0.5291772
  mnconf = len(eall)
  if confl is None:
    confl = range(mnconf)
    nread = mnconf
  else:
    nread = len(confl)
  if verbose:
    msg = 'parsing %d/%d conf.s' % (nread, mnconf)
    print(msg)
  data = []
  for iconf in confl:
    # read cubic box
    box = boxes[iconf]
    lbox = box[0]
    ell = np.diag(box.reshape(ndim, ndim))
    # read energy
    em = eall[iconf]*ha
    # read virial
    vmat = vall[iconf]
    # read particle positions
    pos1 = posa[iconf]
    # read forces
    for1 = fora[iconf]
    # convert to bohr
    ell = [l/bohr for l in ell]
    lbox /= bohr
    pos1 = np.array(pos1)/bohr
    for1 = np.array(for1)*bohr*ha
    vmat = np.array(vmat).reshape(ndim, ndim)*bohr**3*ha
    # calculate rs
    natom = len(pos1)/ndim
    assert len(for1) == natom*ndim
    pos = pos1.reshape(natom, ndim)
    forces = for1.reshape(natom, ndim)
    vol = np.prod(ell)/natom
    rs = ((3*vol)/(4*np.pi))**(1./3)
    entry = {'rs': float(rs), 'natom': int(natom), 'iconf': int(iconf),
             'lbox': float(lbox), 'energy': float(em),
             'box': ell,
             'virial': vmat.tolist(),
             'positions': pos.tolist(),
             'forces': forces.tolist()}
    data.append(entry)
  return data

def get_prefix(temp=1500, rs=1.51, natom=96):
  prefix = 'rs%3.2fT%dN%d' % (rs, temp, natom)
  return prefix
