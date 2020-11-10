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

def parse_arr_text(text):
  nums = text.split('[')[1].split(']')[0].split(',')
  arr = np.array(nums, dtype=float)
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
  }, text='positions')
  ftraj = xml.make_node('trajectory', {
    'filename': 'frc',
    'stride': str(nconf),
    'flush': '1',
    'cell_units': 'angstrom',
  }, text='forces')
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
