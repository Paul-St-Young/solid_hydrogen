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
    #em = np.sum(eall[iconf])*ha
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

def get_extxyz_confs(fxyz, confl=None):
  """ Extract configurations from extended xyz file

  Args:
    fxyz (str): concatenated xyz snapshots
    confl (list, optional): 0-based indexing for configurations
  Return:
    list: traj, a list of ase.Atoms objects

  Example:
    >>> traj = get_extxyz_confs('my.xyz', [0, -1])  # first and last
  """
  from ase.io import read
  # add "virial" to calculators
  from ase.calculators import calculator
  calculator.all_properties.append('virial')

  traj0 = read(fxyz, ':')
  nframe = len(traj0)
  if confl is None:
    confl = range(nframe)
  traj = [traj0[iconf] for iconf in confl]
  return traj

def cache_traj_to_set(set_dir, traj, virial=True):
  import os
  from qharv.plantation.sugar import mkdir
  mkdir(set_dir)
  boxl = [a.get_cell().ravel() for a in traj]
  posl = [a.get_positions().ravel() for a in traj]
  el = [a.get_potential_energy() for a in traj]
  forl = [a.get_forces().ravel() for a in traj]
  names = ['box', 'coord', 'energy', 'force']
  vals = [boxl, posl, el, forl]
  if virial:
    virl = [-a.get_volume()*a.get_stress(voigt=False).ravel()
            for a in traj]
    names.append('virial')
    vals.append(virl)
  for name, val in zip(names, vals):
    np.save(os.path.join(set_dir, '%s.npy' % name), val)

def read_set(set_dir, virial=True):
  import os
  names = ['box', 'coord', 'energy', 'force']
  if virial:
    names.append('virial')
  vals = []
  for name in names:
    fname = os.path.join(set_dir, '%s.npy' % name)
    arr = np.load(fname)
    vals.append(arr)
  return names, vals

def write_lammps_dump(fout, fxyz, charge=False, columns=None):
  import os
  if os.path.isfile(fout):
    raise RuntimeError('%s exists' % fout)
  from ovito.io import import_file, export_file
  pl = import_file(fxyz)
  if columns is None:
    columns = ["Particle Identifier", "Particle Type",
      "Position.X", "Position.Y", "Position.Z"]
  if charge:
    cname = 'Charge'
    columns += [cname]
    def add_charge(frame, data):
      data.particles_.create_property(cname, data=data.particles['initial_charges'])
    pl.modifiers.append(add_charge)
  export_file(pl, fout, 'lammps/dump', columns=columns, multiple_frames=True)

def get_particle_results(dc):
  results = {}
  props = dc.particle_properties
  for p in props.properties:
    results[p.name] = props[p.name].array
  return results

def read_lammps_dump(fdump, nequil=None, iframes=None):
  from ovito.io import import_file
  pl = import_file(fdump) # pipe line
  nframe = pl.source.num_frames
  traj = []
  if iframes is None:
    iframes = range(nframe)
  if nequil is not None:
    iframes = iframes[nequil:]
  for iframe in iframes:
    dc = pl.compute(iframe) # data collection
    atoms = dc.to_ase_atoms()
    atoms.info['Timestep'] = dc.attributes['Timestep']
    keys_to_delete = [key for key in atoms.arrays.keys()
      if key not in ['numbers', 'positions']]
    for key in keys_to_delete:
      del atoms.arrays[key]
    # any results to add?
    results = get_particle_results(dc)
    if 'Charge' in results:  # add charges
      atoms.set_initial_charges(results['Charge'])
    if 'Force' in results:  # add forces
      from ase.calculators.singlepoint import SinglePointCalculator
      calc = SinglePointCalculator(atoms, forces=results['Force'])
      atoms.set_calculator(calc)
    traj.append(atoms)
  return traj

def read_lammps_log(flog, header='Per MPI rank memory', trailer='Loop time'):
  import pandas as pd
  from qharv.reel import ascii_out, scalar_dat
  mm = ascii_out.read(flog)
  blocks = ascii_out.all_lines_with_tag(mm, header)
  dfl = []
  for iblock, idx in enumerate(blocks):
    mm.seek(idx)
    pre = '# '  # !!!! assume first line is header
    try:  # clean run from start to finish
      text1 = ascii_out.block_text(mm, header, trailer)
      df1 = scalar_dat.parse(pre+text1).astype(float)
    except Exception as err:
      mm.seek(idx)  # !!!! decorator "stay" does not finish if Exception
      if type(err) is ValueError:  # unfinished restart run
        text1 = ascii_out.block_text(mm, header, trailer='restart')
        df1 = scalar_dat.parse(pre+text1).astype(float)
      elif type(err) is RuntimeError:  # unfinished final run
        text1 = ascii_out.block_text(mm, header, trailer, force_tail=True)
        df1 = scalar_dat.parse(pre+text1).astype(float)
      else:  # give up
        raise err
    dfl.append(df1)
  mm.close()
  df = pd.concat(dfl, sort=False).reset_index(drop=True)
  return df

def pbar_to_seva(pbar):
  """Convert pressure in bar to stress in eV/A^3"""
  # CODATA 2018
  bohr = 0.529177210903   # A
  ha = 27.211386245988    # ev
  joule = 6.241509126e18  # ev
  gpa = ha/joule/(bohr*1e-10)**3./1e9
  # first bar -> gpa, then gpa -> a.u.
  stress_au = -pbar/1e4/gpa
  # finally, a.u. to eV/A^3
  stress = stress_au*ha/bohr**3
  return stress

def write_lammps_data(ftxt, atoms, **kwargs):
  # FAIL: use ase.io.write(ftxt, atoms, format='lammps-data', atom_style='charge')
  from ovito.io import export_file
  from ovito.pipeline import StaticSource, Pipeline
  from ovito.io.ase import ase_to_ovito
  atoms.set_initial_charges(atoms.get_initial_charges())
  dc = ase_to_ovito(atoms)
  #dc.particles.masses = np.ones(len(atoms))
  #dc.particles_.create_property('masses', data=atoms.get_masses())
  pl = Pipeline(source=StaticSource(data=dc))
  export_file(pl, ftxt, 'lammps/data', atom_style='charge')#**kwargs)


# ====================== LAMMPS analysis ======================

def lammps_scalars(flog, nequil, plot=False):
  from qharv.sieve import scalar_df, mean_df
  df = read_lammps_log(flog)
  if plot:
    names = ['Press', 'Temp', 'TotEng', 'PotEng']
    nrow = len(names)
    fig, axl = plt.subplots(nrow, 1, sharex=True)
    for ax, name in zip(axl, names):
      myy = df[name]
      ax.plot(myy)
      ax.axvline(nequil, c='gray', lw=2, ls='--')
    plt.show()
  mdf = scalar_df.mean_error_scalar_df(df.reset_index(), nequil)
  return mdf

def calc_sofk_nvt(traj, sk_params):
  from qharv_db.grsk import legal_kvecs, Sk, yl_ysql
  from static_correlation import shavg
  # !!!! use initial cell for kvectors
  axes = traj[0].get_cell()
  kvecs = legal_kvecs(axes, sk_params['nsh'])

  skl = []
  for atoms in traj:
    pos = atoms.get_positions()
    sk1 = Sk(kvecs, pos)
    skl.append(sk1)
  skm, ske = yl_ysql(skl, [sk**2 for sk in skl])
  uk, uskm, uske = shavg(kvecs, skm, ske)
  return uk, uskm, uske

def calc_sofk_npt(traj, sk_params):
  from qharv_db.grsk import legal_kvecs, Sk, yl_ysql
  from static_correlation import shavg
  nsh = sk_params['nsh']
  dk = sk_params['dk']
  bin_edges = np.arange(sk_params['kmin'], sk_params['kmax'], dk)
  myk = 0.5*(bin_edges[:-1]+bin_edges[1:])
  nk = len(myk)
  kmin = bin_edges.min()

  skl = []
  for atoms in traj:
    axes = atoms.get_cell()
    kvecs = legal_kvecs(axes, nsh)
    pos = atoms.get_positions()
    sk1 = Sk(kvecs, pos)
    # histogram 1D S(k)
    uk1, usk1, junk = shavg(kvecs, sk1, sk1)
    idx = ((uk1-kmin)//dk).astype(int)
    sel = idx < nk
    msk1 = np.zeros(nk)
    msk1[idx[sel]] += usk1[sel]
    skl.append(msk1)
  uskm, uske = yl_ysql(skl, [sk**2 for sk in skl])
  return myk, uskm, uske
