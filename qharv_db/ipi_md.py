import numpy as np
from qharv.seed import xml

# ======================= level 0: basic input ======================
def barostat(taup, taut):
  baro = xml.make_node('barostat', {'mode': 'isotropic'})
  cell_thermo = xml.make_node('thermostat', {'mode': 'langevin'})
  tnode = xml.make_node('tau', {'units': 'femtosecond'}, str(taut))
  cell_thermo.append(tnode)
  pnode = xml.make_node('tau', {'units': 'femtosecond'}, str(taup))
  baro.append(pnode)
  baro.append(cell_thermo)
  return baro

def pileg(taut, plam=0.5):
  therm = xml.make_node('thermostat', {'mode': 'pile_g'})
  lam_node = xml.make_node('pile_lambda', text=str(plam))
  tnode = xml.make_node('tau', {'units': 'femtosecond'}, str(taut))
  therm.append(lam_node)
  therm.append(tnode)
  return therm

def ensemble(temp, pgpa=None):
  ens = xml.make_node('ensemble')
  tnode = xml.make_node('temperature', {'units': 'kelvin'}, str(temp))
  ens.append(tnode)
  if pgpa is not None:
    p = pgpa*1e3
    pnode = xml.make_node('pressure', {'units': 'megapascal'}, str(p))
    ens.append(pnode)
  return ens

def properties(props=None, **kwargs):
  if props is None:
    quantum = kwargs.pop('quantum', False)
    props = default_properties(quantum=quantum)
  text = text_arr(np.array(props))
  # defaults
  attrib = {
    'stride': str(kwargs.pop('stride', '1')),
    'flush': str(kwargs.pop('flush', '1')),
    'filename': str(kwargs.pop('filename', 'out')),
  }
  prop = xml.make_node('properties', attrib, text)
  return prop

def parse_arr_text(text, dtype=float):
  nums = text.split('[')[1].split(']')[0].split(',')
  arr = np.array(nums, dtype=dtype)
  return arr

def text_arr(arr, **kwargs):
  return np.array2string(arr, separator=',', **kwargs)

# ====================== level 1: custom input ======================
def default_properties(quantum=False):
  e = '{electronvolt}'
  p = '{megapascal}'
  # stability
  props = ['step', 'time{picosecond}', 'conserved'+e, 'potential'+e]
  # ensemble control
  props += ['temperature{kelvin}', 'pressure_md'+p]
  # other
  props += ['virial_cv'+p]
  if quantum:
    props += ['kinetic_tdsc'+e, 'kstress_tdsc'+p]
  return props

def classical_qe(temp, pgpa=None, **kwargs):
  ens_name = 'nvt'
  if pgpa is not None:
    ens_name = 'npt'
  # defaults
  prefix = kwargs.pop('prefix', 'qemd')
  ntherm = kwargs.pop('ntherm', 2)  # property output
  nconf = kwargs.pop('nconf', 20)  # conf. output
  dtfs = kwargs.pop('dtfs', 0.5)  # timestep in fs
  taut = kwargs.pop('taut', 100)  # fs
  taup = kwargs.pop('taup', 100)  # fs
  # make input pieces
  sim = xml.make_node('simulation', {'verbosity': 'high'})
  # <system>: init, forces, ensemble, dynamics
  system = xml.make_node('system')
  init = xml.make_node('initialize', {'nbeads': '1'})
  fnode = xml.make_node('file', {'mode': 'xyz'}, 'init.xyz')
  vnode = xml.make_node('velocities', {
    'mode': 'thermal', 'units': 'kelvin'}, str(temp))
  xml.append(init, [fnode, vnode])
  forces = xml.make_node('forces')
  forces.append(xml.make_node('force', {'forcefield': 'qe'}))
  mot = xml.make_node('motion', {'mode': 'dynamics'})
  dyn = xml.make_node('dynamics', {'mode': ens_name})
  ts = xml.make_node('timestep', {'units': 'femtosecond'}, str(dtfs))
  tnode = pileg(taut)
  dnodes = [ts, tnode]
  if pgpa is not None:
    baro = barostat(taup, taut)
    dnodes.append(baro)
  xml.append(dyn, dnodes)
  mot.append(dyn)
  ens = ensemble(temp, pgpa=pgpa)
  xml.append(system, [init, forces, mot, ens])
  # <output>: properties, trajectory, checkpoint
  output = xml.make_node('output', {'prefix': prefix})
  props = properties(quantum=False, stride=ntherm)
  ptraj = xml.make_node('trajectory', {
    'filename': 'pos',
    'stride': str(nconf),
    'flush': '1',
    'cell_units': 'angstrom',
  }, text='positions{angstrom}')
  ftraj = xml.make_node('trajectory', {
    'filename': 'frc',
    'stride': str(nconf),
    'flush': '1',
    'cell_units': 'angstrom',
  }, text='forces{ev/ang}')
  check = xml.make_node('checkpoint', {'stride': str(nconf)})
  xml.append(output, [props, ptraj, ftraj, check])
  # assemble
  xml.append(sim, [system, output])
  doc = xml.etree.ElementTree(sim)
  # ??? <ffsocket>
  # ??? <prng>
  # ??? <total_steps>
  return doc

# ======================== level 1: structure =======================
def text_ipi_xyz(atoms, fxyz='/var/tmp/ipi.xyz'):
  from ase import io
  io.write(fxyz, atoms)
  # keep only positions
  keys = atoms.arrays.keys()
  keys2del = []
  for key in keys:
    if key not in ['numbers', 'positions']:
      keys2del.append(key)
  for key in keys2del:
    del atoms.arrays[key]
  # write info line
  cell = atoms.get_cell()
  abc = cell.lengths()
  angles = cell.angles()
  line1 = '# CELL(abcABC): %f %f %f %f %f %f' % (*abc, *angles)
  line1 += ' cell{angstrom} Traj: positions{angstrom} Step: 0 Bead:0'
  line1 += '\n'
  # replace info line
  text1 = ''
  with open(fxyz, 'r') as f:
    text1 += f.readline()
    line = f.readline()
    text1 += line1
    for line in f:
      text1 += line
  return text1

# ======================= level 0: basic output ======================
def read_ipi_log(flog):
  from qharv.reel import scalar_dat
  # interpret header lines
  header = '# '
  text = ''
  with open(flog, 'r') as f:
    for line in f:
      if line.startswith('#'):  # header
        if 'col' not in line:
          msg = 'cannot parse: ' + line
          raise RuntimeError(msg)
        col_index, col_label = line.split('-->')
        name = col_label.split()[0]
        if 'cols' in line:  # multi-column
          ij = col_index.split()[2].split('-')
          i, j = list(map(int, ij))
          ncol = j-i+1
          for icol in range(ncol):
            header += '%s_%d ' % (name, icol)
        else:
          header += name + ' '
      else:  # data
        text += line
  # make Scalar TABle format
  header += '\n'
  text1 = header + text
  # parse known format
  df = scalar_dat.parse(text1)
  return df

def read_ipi_xyz(fxyz):
  from ase import io
  from ase.cell import Cell
  from qharv.reel import ascii_out
  mm = ascii_out.read(fxyz)
  idxl = ascii_out.all_lines_with_tag(mm, '# ')
  traj = io.read(fxyz, ':')
  nhead = len(idxl)
  nbody = len(traj)
  if nhead != nbody:
    msg = 'found %d headers for %d bodies' % (nhead, nbody)
    raise RuntimeError(msg)
  headers = []
  for idx in idxl:
    mm.seek(idx)
    header = mm.readline().decode()
    headers.append(header)
  mm.close()

  for header, atoms in zip(headers, traj):
    tokens = header.strip('#').split()
    # read cell
    i0 = tokens.index('CELL(abcABC):')
    cellpart = tokens[i0+1:i0+7]
    cellpar = np.array(cellpart, dtype=float)
    cell = Cell.fromcellpar(cellpar)
    atoms.set_cell(cell)
    atoms.set_pbc(True)
    # read info
    atoms.info = {}
    for i, tok in enumerate(tokens):
      if (i >= i0) and (i < i0+7):
        continue
      if tok.endswith(':'):  # key-val pair
        atoms.info[tok[:-1]] = tokens[i+1]
      if ('{' in tok) and ('}' in tok):  # unit
        key, val = tok.split('{')
        atoms.info[key+'_unit'] = val[:-1]
  return traj

def read_ipi_bead(fpos, ffrc=None):
  from ase import units
  traj0 = read_ipi_xyz(fpos)
  if ffrc is not None:
    traj1 = read_ipi_xyz(ffrc)
  else:
    traj1 = traj0
  for atoms0, atoms1 in zip(traj0, traj1):
    info0 = atoms0.info
    if ffrc is not None:
      info1 = atoms1.info
      assert info0['Step'] == info1['Step']
      assert info0['Bead'] == info1['Bead']
      assert np.allclose(atoms0.get_cell(), atoms1.get_cell())
      assert 'positions_unit' in info0
      assert 'forces_unit' in info1
      info0['forces_unit'] = info1['forces_unit']
      atoms0.arrays['forces'] = atoms1.get_positions()
    # change units to angstrom, ev/ang
    pu = info0['positions_unit']
    if pu == 'angstrom':
      pass  # no conversion
    elif pu == 'atomic_unit':
      pos = atoms0.get_positions()
      atoms0.set_positions(pos*units.Bohr)
    else:
      msg = 'unknown position unit %s' % pu
      raise RuntimeError(msg)
    if ffrc is not None:
      fu = info0['forces_unit']
      if fu == 'ev/ang':
        pass  # no conversion
      elif fu == 'atomic_unit':
        forces = atoms0.arrays['forces']
        atoms0.arrays['forces'] = forces*units.eV/units.Bohr
      else:
        msg = 'unknown force unit %s' % fu
        raise RuntimeError(msg)
  return traj0

# ========================= level 1: restart ========================
def read_restart_atoms(frs):
  from ase import Atoms
  bohr = 0.529177210903  # CODATA 2018
  from qharv_db import ipi_md
  axes = read_restart_cell(frs)
  elem, posl = read_restart_beads(frs)
  traj = [Atoms(elem, cell=axes*bohr, pbc=1, positions=pos*bohr) for
    pos in posl]
  return traj

def read_restart_cell(fres):
  doc = xml.read(fres)
  cnode = doc.find('.//cell')
  text = cnode.text
  cvec = parse_arr_text(text)
  ndim = int(round(len(cvec)**0.5))
  assert ndim**2 == len(cvec)
  cell = cvec.reshape(ndim, ndim)
  return cell

def read_restart_beads(fres):
  doc = xml.read(fres)
  bnode = doc.find('.//beads')
  natom = int(bnode.get('natoms'))
  nbead = int(bnode.get('nbeads'))
  ql = bnode.findall('.//q')
  assert len(ql) == nbead
  elem = parse_arr_text(bnode.find('.//names').text, dtype=str)
  elem = [e.replace('\n', '').strip() for e in elem]
  assert len(elem) == natom
  posl = []
  for qnode in ql:
    pvec = parse_arr_text(qnode.text)
    pos = pvec.reshape(natom, -1)
    posl.append(pos)
  return elem, posl
