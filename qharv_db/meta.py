import os
import numpy as np
import pandas as pd
from qharv.reel import mole

# ======================== level 0: paths and inputs =========================


def get_pcdirs(
    tmp_dir='inp_text',
    pd_name='proj_dir.dat',
    fdl_name='folder_list.dat'
  ):
  """ read project dir and calc dirs from inputs
  default to the following directory structure:
  $ls -R
  ./inp_text:
  folder_list.dat  proj_dir.dat

  proj_dir.dat should hold the absolute path to project folder
  folder_list.dat should hold a list of QMC calculation directories

  Args:
    tmp_dir (str, optional): temporary folder to hold inputs and meta data
    pd_name (str, optional): file holding project directory
    fdl_name (str, optional): file holding calculation folder list
  Return:
    tuple: (str, list) i.e. (proj_dir, folder_list)
  """

  # locate project directory
  pd_dat  = os.path.join(tmp_dir, pd_name)
  with open(pd_dat, 'r') as f:
    proj_dir = f.read().strip('\n')

  # locate folders to analyze
  fdlist_dat = os.path.join(tmp_dir, fdl_name)
  with open(fdlist_dat,'r') as f:
    folder_list = f.read().split('\n')[:-1]
  return proj_dir, folder_list

def sra_label(subdir, sep='-'):
  tokens = subdir.split(sep)
  sname  = tokens[0]
  rst    = tokens[1]
  cat    = tokens[2]
  rs = float(rst.replace('rs', ''))
  ca = float(cat.replace('ca', ''))
  return {'sname':sname, 'rs':rs, 'ca':ca}

def get_task_dir(subdir):
  sra = sra_label(subdir)
  sname = sra['sname']
  rs = sra['rs']
  task_dir_map = {
    'c2c': '56-c2c-dft-geo/ecut50-k8',
    'cmca4': '57-cmca4-dft-geo/ecut50-k8',
    'cmca12': '58-cmca12-dft-geo/ecut50-k8',
    'i41amd': '44-i4-twist/ecut50-k8'
  }
  if rs < 1.21:
    task_dir_map = {
      'c2c': '79-c2c-dft-geo/ecut50-k8',
      'cmca4': '78-cmca4-dft-geo/ecut50-k8',
      'i41amd': '77-i4-twist/ecut50-k8',
      'cmca12': None
    }
  if (sname == 'cmca4') & (rs < 1.19):
    task_dir_map['cmca4'] = '86-cmca4-kgrid12-geo/ecut50-k8'
  task_dir = task_dir_map[sname]
  return task_dir

def collect_first_input(folderl):
  """ collect input text into a database

  example:
    proj_dir, fdl = get_pcdirs()
    folderl = [os.path.join(proj_dir, folder) for folder in fdl]
    mdf = collect_first_input(folderl)

  Args:
    folderl (list): a list of QMC calculation folders
  Return:
    pd.DataFrame: mdf contains ['path', 'inp_text'] columns. mdf collects the
    first input xml mole finds in each folder.
  """
  data = []
  for folder in folderl:
    fin = mole.files_with_regex('*.in.xml',folder)[0]
    with open(fin,'r') as f:
      inp_text = f.read()
    # end with
    entry = {'path':folder,'inp_text':inp_text}
    data.append(entry)
  # end for
  mdf = pd.DataFrame(data)
  return mdf


def find_all_groups_and_series(folderl):
  gsdata = []
  for folder in folderl:
    flist = mole.files_with_regex('*scalar.dat', folder)
    for floc in flist:
      fdat = os.path.basename(floc)
      meta = mole.interpret_qmcpack_fname(fdat)
      meta['path'] = folder
      meta['fdat'] = fdat
      gsdata.append(meta)
    # end for
  # end for
  gsdf = pd.DataFrame(gsdata).sort_values('group')
  return gsdf

def get_prefix_from_path(path, proj_dir, sep='_'):
  task_dir = path.replace(proj_dir, '')
  prefix = sep.join([seg for seg in task_dir.split('/')
    if seg not in ['.', '..']]).strip(sep)
  return prefix

def meta_from_path(path0):
  ipbe = path0.find('pbe')
  if ipbe < 0:
    raise RuntimeError('unknown path %s' % path0)
  path = path0[ipbe:]
  tokens = path.split('/')
  ttrst = tokens[1]
  ntict = tokens[2]
  tt, rst = ttrst.split('-')
  nt, ict = ntict.split('-')
  temp = int(tt.replace('t', ''))
  rs = float(rst.replace('rs', ''))
  natom = int(nt.replace('h', ''))
  iconf = int(ict.replace('i', ''))
  entry = {'temp': temp, 'rs': rs, 'natom': natom, 'iconf': iconf}
  return entry

# ====================== level 1: parse input =======================


def get_axes_pos(doc):
  from qharv.seed import xml
  axes = xml.get_axes(doc)
  pos = xml.get_pos(doc)
  entry = {'axes':axes.tolist(), 'pos':pos.tolist()}
  return entry


def get_density(doc):
  from qharv.inspect import axes_pos
  entry = get_axes_pos(doc)
  volume = axes_pos.volume(entry['axes'])
  natom = len(entry['pos'])
  rho = natom/volume
  rs = (3./(4*np.pi*rho))**(1./3)
  return {'rs':rs, 'volume':volume, 'rho':rho}

# =================== level 2: read QMC database ====================

def get_force_columns(cols, name='force', idx_dim=-1):
  fcols = [c.replace('_mean', '') for c in cols
           if c.startswith(name) and c != '%s_mean' % name
           and c.endswith('_mean')]
  # sort columns
  def iatom_idim(c):
    tokens = c.split('_')
    iatom = int(tokens[idx_dim-1])
    idim = int(tokens[idx_dim])
    return {'iatom': iatom, 'idim': idim}
  fdf = pd.DataFrame([iatom_idim(c) for c in fcols], dtype=int)
  fdf.iatom = fdf.iatom.astype(int)
  fdf.idim = fdf.idim.astype(int)
  fdf['force'] = fcols
  fcols = fdf.sort_values(['iatom', 'idim']).force.values.tolist()
  return fcols
