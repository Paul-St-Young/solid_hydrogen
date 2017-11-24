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
  nscf_inputs['input_type'] = 'nscf'
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

def get_zero_backflow(rcut):
  from qmcpack_input import generate_transformation1, generate_transformation2
  tr1 = generate_transformation1(['H'],rcut)
  tr2 = generate_transformation2([('u','u'),('u','d')],rcut)
  from qmcpack_input import backflow,collection
  bf = backflow( transformations=collection([tr1,tr2]) )
  return bf
# end def get_zero_backflow

def bopt_input_from_opt(opt,opt_inputs,suffix='-opt',ts_reduce=15.,wts_reduce=5.,rcut=None):
  bopt_inputs = opt_inputs.copy()

  # assumption 1: opt has Jastrows
  #  have bopt depend on optimize Jastrows
  deps = opt_inputs.dependencies
  bopt_inputs.dependencies = deps + [(opt,'jastrow')]

  # assumption 2: opt has optmization <loop>
  #  edit VMC in optimization loop for backflow
  calcs = bopt_inputs.calculations
  for loop in calcs:
    calc = loop['qmc']
    # reduce timestep for all-particle move
    calc['move'] = 'wbyw'
    calc['timestep'] /= ts_reduce
    # further reduce timestep for warmup
    calc['warmupsteps']    = 128
    calc['warmuptimestep'] = calc['timestep']/wts_reduce
  # end for

  # make a unique simulation 
  bopt_inputs.identifier = opt_inputs.identifier.replace(suffix,'-bopt')
  bopt_inputs.path = opt_inputs.path.replace(suffix.strip('-'),'bopt')

  # add backflow
  
  bf = get_zero_backflow(rcut)
  bopt_inputs.backflow = bf
  bopt_inputs['precision'] = 'double'

  return bopt_inputs
# end def bopt_input_from_opt

def vmc_wbyw_input_from_p2q(p2q,system,suffix='-p2q'):
  """ construct vmc (walker-by-walker) inputs as nexus.obj
   assume orbitals come from a mean-field calculation "p2q"

   The returned nexus.obj can be edited at will even after return
    thus, the "system" input is not actually mandatory. I put it 
    there simply as a reminder.

   The "job" attribute still needs to be filled in after return.
   "job" is not assigned in this function to improve tranferability 
   among machines.
  Args:
   p2q (nexus.Pw2qmcpack): may also be any other Simulation object 
    having 'orbitals' in 'application_results'.
   system (nexus.PhysicalSystem): system to simulate.
   suffix (str,optional): suffix to the p2q simulation identifier and path.
  """
  from nexus import vmc

  # determine ID and path from p2q simulation
  myid    = p2q.identifier.replace(suffix,'-vmc')
  p2q_dir = os.path.basename(p2q.path)
  mypath  = p2q.path.replace(p2q_dir,'vmc')

  # write default inputs
  vmc_block = obj(
    warmuptimestep = 0.01,
    move        = 'not_pbyp_or_whatever',
    warmupsteps = 128,
    blocks      =  64,
    steps       =   4,
    substeps    =   4,
    timestep    = 0.03,
    walkers     = 16,
  )
  calcs = [vmc(**vmc_block)]
  vmc_inputs = obj(
    identifier = myid,
    path       = mypath,
    system     = system,
    input_type = 'basic',
    bconds     = 'ppp',
    calculations = calcs,
    dependencies = [(p2q,'orbitals')]
  )
  return vmc_inputs
# end def vmc_wbyw_input_from_p2q

def dmc_wbyw_input_from_p2q(p2q,system,nwalker=512
  ,tss=[0.006,0.002],corr=0.4,suffix='-p2q'):
  from nexus import dmc

  # get a quick start from VMC defaults
  dmc_inputs = vmc_wbyw_input_from_p2q(p2q,system)

  # change ID and path
  myid    = p2q.identifier.replace(suffix,'-dmc')
  p2q_dir = os.path.basename(p2q.path)
  mypath  = p2q.path.replace(p2q_dir,'dmc')
  dmc_inputs.identifier = myid
  dmc_inputs.path       = mypath

  # ask VMC for samples
  calcs = dmc_inputs.calculations
  assert len(calcs) == 1 # expect 1 VMC calculation
  vcalc = calcs[0]
  vcalc['samples'] = nwalker

  # add DMC calculations
  for ts in tss:
    steps = int(round(corr/ts))
    dmc_block = obj(
      move        = 'not_pbyp_or_whatever',
      blocks      = 64,
      steps       = steps,
      timestep    = ts,
      targetwalkers = nwalker
    )
    calcs += [dmc(**dmc_block)]
  # end for ts
  return dmc_inputs
# end def dmc_wbyw_input_from_p2q

def vmc_pbyp_input_from_p2q(p2q,system,suffix='-p2q'):
  from nexus import vmc

  # determine ID and path from p2q simulation
  myid    = p2q.identifier.replace(suffix,'-vmc')
  p2q_dir = os.path.basename(p2q.path)
  mypath  = p2q.path.replace(p2q_dir,'vmc')

  # write default inputs
  vmc_block = obj(
    move        = 'pbyp',
    warmupsteps =  16,
    blocks      =  64,
    steps       =   4,
    substeps    =   4,
    timestep    = 0.4,
    walkers     = 16,
    checkpoint  = 0
  )
  calcs = [vmc(**vmc_block)]
  vmc_inputs = obj(
    identifier = myid,
    path       = mypath,
    system     = system,
    input_type = 'basic',
    bconds     = 'ppp',
    calculations = calcs,
    dependencies = [(p2q,'orbitals')]
  )
  return vmc_inputs
# end def vmc_pbyp_input_from_p2q

def hydrogen_estimators(nbin=128): # 128 grid points over 4 bohr is good spacing
  from qmcpack_input import pressure,gofr,sk,csk#,skall,structurefactor
  pres = pressure({'type':'Pressure'})
  # species-resolved rho(k)
  #sksp = structurefactor({'type':'structurefactor','name':'sksp'})
  ## species-summed S(k) and rho(k)
  #skall= skall({'type':'skall','name':'skall','hdf5':False,'source':'ion0'})
  # charged structure factor (create on kylin branch 1d4d188e53939)
  csk = csk({'type':'csk','name':'csk'})
  # species-summed S(k); rho(k) hacked on kylin branch (21b33fb78c44172ec5dd)
  skest= sk({'type':'sk','name':'sk','hdf5':True})
  gofr = gofr({'type':'gofr','name':'gofr','num_bin':nbin})
  return [csk,skest,gofr,pres]#,sksp,skall]
# end def hydrogen_estimators
