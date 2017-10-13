# this is a dump
import pandas as pd

def collect_scf_sims(scf_sims):
  dft_data = []
  for scf in scf_sims:
    pa = scf.load_analyzer_image()
    dft_data.append( pa.to_dict() )
  # end for
  dft_df = pd.DataFrame(dft_data)
  return dft_df
# end def

def collect_qmc_sims(qmc_sims):
  qmc_data = []
  for qmc in qmc_sims:
    qa = qmc.load_analyzer_image()
    for iqmc in qa['qmc'].keys():
      qmc_data.append( qa['qmc'][iqmc]['scalars'].to_dict() )
    # end for
  # end for
  qmc_df = pd.DataFrame(qmc_data)
  return qmc_df
# end def

