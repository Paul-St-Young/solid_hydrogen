# guide the flow
import os
from nexus import obj

def nscf_input_from_scf_input(scf,scf_inputs,suffix='-scf'):
  """ take an scf simulation object along with the inputs for it, create an nscf simulation to generate orbitals only at kpoints needed for wavefunction. 
  Args: 
    scf (nexus.Pwscf): scf simulation object.
    scf_inputs (nexus.obj): inputs for scf simulation (it would be ideal to extract this object from scf, but I do not know how yet).
    suffix (str,optional): suffix in the identifier and path of the scf simulation, these need to be edited to create a new run in nexus. Default is '-scf'.
  Returns:
    nexus.obj: inputs necessary for the nscf run.
  """
  # copy scf inputs and make nscf inputs have unique identifier and path
  nscf_inputs = scf_inputs.copy()
  nscf_inputs.identifier = scf_inputs.identifier.replace(suffix,'-nscf')
  nscf_inputs.path = scf_inputs.path.replace(suffix.strip('-'),'nscf')

  # let nexus figure out kgrid for super cell and twists
  nscf_inputs['nosym'] = True
  nscf_inputs.kgrid = None 

  # set dependency
  nscf_inputs.dependencies = (scf,'charge-density')
  return nscf_inputs
# end def

def p2q_input_from_nscf(nscf,suffix='-nscf',cusp_corr=False):
  """ take an nscf simulation object and create a p2q run to generate wavefunction.
  Args: 
    nscf (nexus.Pwscf): nscf simulation object. An scf simulation object can also be used here.
    suffix (str,optional): suffix in the identifier and path of the nscf simulation. Default is '-nscf'.
    cusp_corr (bool,optional): apply cusp correction, namely divide DFT orbitals by RPA Jastrows. This option requires a custom version of quantum espresso. Default=False.
  Returns:
    nexus.obj: inputs necessary for the p2q run.
  """
  p2q_input = obj(
    identifier   = nscf.identifier.replace(suffix,'-p2q'),
    path         = nscf.path,
    outdir       = nscf.input.control.outdir,
    write_psir   = False,
    dependencies = (nscf,'orbitals')
  )
  if cusp_corr:
    p2q_input['cusp_corr'] = True
    
  return p2q_input
# end def

def gamma_opt_input(p2q,system,init_jas=None):
  from nexus import loop, linear

  myid = p2q.identifier.replace('-p2q','-opt')
  nscf_dir = os.path.basename(p2q.path)
  mypath = p2q.path.replace(nscf_dir,'opt')

  linopt = obj(
    minmethod   = 'OneShiftOnly',
    energy = 0.95,
    reweightedvariance = 0.05,
    unreweightedvariance = 0.0,
    warmupsteps =  40,
    blocks      = 128,
    substeps    =   3,
    timestep    = 0.8,
    samples     = 8192,
  )
  calcs = [loop(max=5,qmc=linear(**linopt))]

  myjas = init_jas
  if init_jas is None:
    myjas = [('J1','size',8,'cusp',1),('J2','size',8)]
  # end if

  opt_inputs  = obj(
    identifier  = myid,
    path        = mypath,
    input_type  = 'basic',
    system      = system,
    bconds      = 'ppp',    # periodic in xyz directions
    calculations = calcs,
    twistnum    = 0,
    estimators   = [],
    jastrows     = myjas,
    pseudos      = [],
    dependencies = [(p2q,'orbitals')]
  )
  return opt_inputs
# end def gamma_opt_input
