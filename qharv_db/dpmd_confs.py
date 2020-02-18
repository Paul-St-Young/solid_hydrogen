import numpy as np

def get_path(ref_dir, temp=1500, rs=1.51, natom=96):
  path = '%s/T%d/RS%3.2fN%d' % (ref_dir, temp, rs, natom)
  return path

def get_confs(path, confl=None, ndim=3, verbose=True, au=True):
  from qharv.reel import mole
  if verbose:
    msg = 'reading conf.s from %s' % path
    print(msg)
  fbox = mole.find('*_box.raw', path)
  boxes = np.loadtxt(fbox)
  fene = mole.find('*_energy.raw', path)
  eall = np.loadtxt(fene)
  virial = False
  try:
    fvir = mole.find('*_virial.raw', path)
    vall = np.loadtxt(fvir)
    virial = True
  except RuntimeError as err:
    if 'expect' in str(err):
      pass
    else:
      raise err
  fpos = mole.find('*_coord.raw', path)
  posa = np.loadtxt(fpos)
  ffor = mole.find('*_force.raw', path)
  fora = np.loadtxt(ffor)
  ha = 1./27.211386245988
  bohr = 0.5291772
  if not au:
    ha = 1.
    bohr = 1.
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
    # read particle positions
    pos1 = posa[iconf]
    # read forces
    for1 = fora[iconf]
    # convert to bohr
    ell = [l/bohr for l in ell]
    lbox /= bohr
    pos1 = np.array(pos1)/bohr
    for1 = np.array(for1)*bohr*ha
    # calculate rs
    natom = int(len(pos1)/ndim)
    assert len(for1) == natom*ndim
    pos = pos1.reshape(natom, ndim)
    forces = for1.reshape(natom, ndim)
    vol = np.prod(ell)/natom
    rs = ((3*vol)/(4*np.pi))**(1./3)
    entry = {'rs': float(rs), 'natom': int(natom), 'iconf': int(iconf),
             'lbox': float(lbox), 'energy': float(em),
             'box': ell,
             'positions': pos.tolist(),
             'forces': forces.tolist()}
    if virial:
      # read virial
      vmat = vall[iconf]
      vmat = np.array(vmat).reshape(ndim, ndim)*bohr**3*ha
      entry['virial'] = vmat.tolist()
    data.append(entry)
  return data

def get_prefix(temp=1500, rs=1.51, natom=96):
  prefix = 'rs%3.2fT%dN%d' % (rs, temp, natom)
  return prefix

def text_row(row, per_atom=True):
  from qharv.inspect import box_pos
  ntype = 1  # !!!! hard-code for now
  itype = 0
  box = np.array(row['box'])
  pos = np.array(row['positions'])
  natom = len(pos)
  # write header
  header = '#N %d %d\n' % (natom, ntype)
  ndim = len(box)
  vec = np.zeros(ndim)
  for idim, lbox in enumerate(box):
    name = {0: 'X', 1: 'Y', 2: 'Z'}[idim]
    vec[idim] = lbox
    header += '#%s %.16e %.16e %.16e\n' % (name, vec[0], vec[1], vec[2])
  # add energy
  energy = row['energy']
  if per_atom:
    energy /= natom
  header += '#E %.16e\n' % energy
  header += '#F\n'
  # put positions in box for PBC
  pos = box_pos.pos_in_box(pos, box)
  forces = np.array(row['forces'])
  # write position and forces
  body = ''
  line_fmt = '%d ' % itype + '%.16e %.16e %.16e %.16e %.16e %.16e\n'
  vec = np.zeros(2*ndim)
  for p, f in zip(pos, forces):
    vec[:ndim] = p
    vec[ndim:] = f
    line = line_fmt % tuple(vec)
    body += line
  return header + body

def xyz_snapshot(row, has_energy=False, has_forces=False, has_virial=False):
  """Write one snapshot in extended xyz format

  Args:
    row (dict): snapshot information, must have ['axes', 'positions']
    has_energy (bool): row has 'energy' column, default False
    has_forces (bool): row has 'forces' column, default False
    has_virial (bool): row has 'virial' column, default False
  Return:
    str: one frame in extended xyz format
  """
  ndim = 3
  fmt_map = {
    'energy': '%24.14f',
    'positions': '%12.6f',
    'virial': '%20.6f'
  }
  # get data
  axes = np.array(row['axes'])
  pos = np.array(row['positions'])
  natom = len(pos)
  if 'energy' in row:
    has_energy = True
  if 'forces' in row:
    has_forces = True
  if 'virial' in row:
    has_virial = True
  if has_forces:
    fvs = np.array(row['forces'])
  # write header
  text = '%d\n' % natom
  afmt = '%12.8f '
  aline_fmt = 'Lattice="' + afmt*ndim**2 + '"'
  text += aline_fmt % tuple(axes.ravel())  # !!!! C or F order?
  text += ' Properties=species:S:1:pos:R:3'
  if has_forces:
    text += ':forces:R:3'
  if has_energy:
    energy = row['energy']
    eline = ' energy="' + fmt_map['energy'] % energy + '"'
    text += eline
  if has_virial:  # !!!! C or F order?
    vir = np.array(row['virial'])
    vfmt = fmt_map['virial']
    vline = vfmt*ndim**2 % tuple(vir.ravel())
    text += ' virial="'+vline+'"'
  text += ' pbc="T T T"\n'  # termination of header
  # write body
  ncol = ndim  # positions, forces
  if has_forces:
    ncol += ndim
  vec = np.zeros(ncol)
  line_fmt = fmt_map['positions']*ncol + '\n'
  for iatom in range(natom):
    vec[:ndim] = pos[iatom]
    if has_forces:
      vec[ndim:ndim+ndim] = fvs[iatom]
    line = 'H ' + line_fmt % tuple(vec)
    text += line
  return text

def get_ovconfs(pl, confl=None):
  """ Extract configurations from dump file

  Example:
    >>> pl = import_file('dump.1')  # read LAMMPS dump file
    >>> boxl, posl = get_ovconfs(pl)
  """
  if confl is None:
    nframe = pl.source.num_frames
    confl = np.arange(nframe)
  boxl = []
  posl = []
  for iconf in confl:
    ovdata = pl.compute(iconf)
    axes = np.array(ovdata.cell.matrix)
    pos = np.array(ovdata.particles.positions.array)
    box = np.diag(axes)
    boxl.append(box)
    posl.append(pos)
  return boxl, posl
