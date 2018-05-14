import os
import pandas as pd
from qharv.reel import mole


def get_pcdirs(
    tmp_dir='inp_text',
    pd_name='proj_dir.dat',
    fdl_name='folder_list.dat'
  ):
  """ read project dir and calc dirs from inputs """

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
  """ collect input text into a database """
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
