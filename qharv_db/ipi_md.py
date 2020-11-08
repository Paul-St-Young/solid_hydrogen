from qharv.seed import xml

# ========================== level 0: input =========================
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

def parse_arr_text(text):
  import numpy as np
  nums = text.split('[')[1].split(']')[0].split(',')
  arr = np.array(nums, dtype=float)
  return arr

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
