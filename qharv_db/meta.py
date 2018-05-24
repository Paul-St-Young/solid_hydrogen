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
