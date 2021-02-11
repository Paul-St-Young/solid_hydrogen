import pandas as pd

def mix_extrap_forces(
  sdf, fcols, dseries, vseries=0,
  msuffix='_mean', esuffix='_error',
  **kwargs
):
  """2*DMC-VMC forces

  Args:
    sdf (pd.DataFrame): scalar database, must contain ['path', 'series']
    fcols (list): force column names, e.g. ['force_0_0', 'force_0_1']
    dseries (int): series index for DMC, e.g. if s001, then 1
    vseries (int, optional): series index for VMC, default 0
    msuffix (str, optional): default '_mean'
    esuffix (str, optional): default '_error'
  Return:
    pd.DataFrame: 2*DMC-VMC forces
  """
  from qharv.sieve.mean_df import linex  # linear extrapolation
  from qharv_db.meta import get_force_columns
  # mix extrap
  mfc = [c+msuffix for c in fcols]
  efc = [c+esuffix for c in fcols]
  pdf = sdf.groupby('path')[['series']+mfc+efc].apply(
    lambda x: linex(x, vseries, dseries, fcols, **kwargs))
  pdf.reset_index(inplace=True)
  pdf.drop(columns=['level_1'], inplace=True)  # series
  return pdf

def read_fsc_out(fout, chi2_tol=1e-12):
  from qharv.reel import ascii_out
  mm = ascii_out.read(fout)
  # check chi^2
  idxl = ascii_out.all_lines_with_tag(mm, 'fitpn: Chi^2 =')
  for idx in idxl:
    mm.seek(idx)
    chi2 = ascii_out.name_sep_val(mm, 'fitpn: Chi^2', '=')
    assert chi2 < chi2_tol
  # rewind
  mm.seek(0)
  # get nelec
  nelec = ascii_out.name_sep_val(mm, 'nparts', '=', dtype=int)
  # read potential
  idx = mm.find(b'Potential energy correction')
  mm.seek(idx)
  idx = mm.find(b'using optimized potential')
  mm.seek(idx)
  dv = ascii_out.name_sep_val(mm, 'dV/N', '=')
  # read kinetic
  mm = ascii_out.read(fout)
  idx = mm.find(b'Kinetic energy correction')
  mm.seek(idx)
  idx = mm.find(b'using optimized potential')
  mm.seek(idx)
  dt = ascii_out.name_sep_val(mm, 'dT/N', '=')
  mm.close()
  return dv*nelec, dt*nelec
