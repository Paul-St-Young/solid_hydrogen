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
      entry = qa['qmc'][iqmc]['scalars'].to_dict()
      qmc_data.append( entry )
    # end for
  # end for
  qmc_df = pd.DataFrame(qmc_data)
  return qmc_df
# end def

def orb_nscf_input(sdmc):
  """ find the scf inputs used to generate sdmc """

  # step 1: find the p2q simulation id
  p2q_id = None
  for key in sdmc['dependencies'].keys():
    if sdmc['dependencies'][key].result_names[0] == 'orbitals':
      p2q_id = key
    # end if
  # end for dep

  # step 2: find the nscf simulation id
  nscf_id_list = sdmc['dependencies'][p2q_id]['sim']['dependencies'].keys()
  assert len(nscf_id_list) == 1
  nscf_id = nscf_id_list[0]

  # this may actually be the scf simulation
  nscf = sdmc['dependencies'][p2q_id]['sim']['dependencies'][nscf_id]
  nscf_inputs = nscf['sim']['input']
  return nscf_inputs
  # nscf_inputs should be good enough
  ###if (nscf_inputs['control']['calculation'] == 'scf'):
  ###  return nscf_inputs
  #### end if

  ###raise NotImplementedError('sdmc p2q_id=%d nscf_id=%d, need to find scf_id'%(p2q_id,nscf_id))
  #### if nscf is not the scf, then we need to go deeper

# end def

def qmc_scf_path_map(qmc_sims):
  data = []
  for qmc in qmc_sims:
    scf_inputs = orb_nscf_input(qmc)
    entry = {'qmc_path':qmc.path,'scf_inputs':scf_inputs}
  # end for qmc
  df = pd.DataFrame(data)
  return df
# end def qmc_scf_path_map

