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
      entry['path'] = qmc.path
      qmc_data.append( entry )
    # end for
  # end for
  qmc_df = pd.DataFrame(qmc_data)
  return qmc_df
# end def

def orb_scf_input(sdmc):
  """ find the scf inputs used to generate sdmc """
  myinputs = None # this is the goal
  sdep = 'dependencies' # string representation of the dependencies entry

  # step 1: find the p2q simulation id
  p2q_id = None
  for key in sdmc[sdep].keys():
    if sdmc[sdep][key].result_names[0] == 'orbitals':
      p2q_id = key
    # end if
  # end for dep

  # step 2: find the nscf simulation
  nscf_id_list = sdmc[sdep][p2q_id]['sim'][sdep].keys()
  assert len(nscf_id_list) == 1
  nscf_id = nscf_id_list[0]
  nscf = sdmc[sdep][p2q_id]['sim'][sdep][nscf_id]
  myinputs = nscf['sim']['input']

  # step 3: find the scf simulation
  calc = myinputs['control']['calculation']
  if (calc=='scf'): # scf may actually be the scf simulation
    pass # myinputs is already set
  elif (calc=='nscf'): # if nscf is not the scf, then we need to go deeper
    scf_id = nscf['sim'][sdep].keys()[0] 
    scf = nscf['sim'][sdep][scf_id]
    myinputs = scf['sim']['input'] # this is it!
    scalc = myinputs['control']['calculation']
    if scalc != 'scf':
      RuntimeError('nscf depends on %s instead of scf'%scalc)
    # end if
  else:
    raise RuntimeError('unknown simulation type %s'%calc)
  # end if
  
  return myinputs.to_dict()
# end def orb_scf_input

def qmc_scf_path_map(qmc_sims):
  data = []
  for qmc in qmc_sims:
    myinputs = orb_scf_input(qmc)
    entry = {'qmc_path':qmc.path,'nscf_inputs':dict(myinputs)}
    data.append(entry)
  # end for qmc
  df = pd.DataFrame(data).set_index('qmc_path')
  return df
# end def qmc_scf_path_map
