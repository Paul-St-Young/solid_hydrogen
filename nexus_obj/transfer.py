# guide the flow
import os
import subprocess as sp
from nexus import obj
from lxml import etree

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
  # charged structure factor (create on kylin branch 1d4d188e53939)
  csk = csk({'type':'csk','name':'csk'})
  gofr = gofr({'type':'gofr','name':'gofr','num_bin':nbin})
  return [csk,gofr,pres]
# end def hydrogen_estimators

def dynamic_wbyw_opt(identifier,qsyst):
  """ assemble custom input using xml text 
   the returned xml template is meant to be customized further
  Args:
    identifier (str): QMCPACK project id (not essential)
    qsyst (str): string representation of <qmcsystem>
  Returns:
    lxml.etree.Element: <simulation>
  """
  from qharv.seed import xml,xml_examples
  pnode = etree.Element('project',{'id':identifier,'series':'0'})
  qnode = xml.parse(qsyst)
  cnode = xml_examples.wbyw_optimize()
  cnode.set('max','5')
  vnode = cnode.find('.//qmc')
  vnode.append( etree.Element('fokker_planck',{'nfp':'1'}) )
  vnode.append( etree.Element('ts_boost',{'group':'p','factor':'3'}) ) # boost proton diffusion
  xml.set_param(vnode,'warmupsteps','512')
  xml.set_param(vnode,'timestep','0.03') # 0.03 a.u. for 67% acceptance
  xml.set_param(vnode,'substeps','25') # important to use many steps because of slow protons
  # turn off latdev per_xyz
  qnode.find('.//estimator[@type="latticedeviation"]').set('per_xyz','no')

  snode = etree.Element('simulation')
  for node in [pnode,qnode,cnode]:
    snode.append(node)
  # end for
  return snode
# end def dynamic_wbyw_opt

def nexus_infile(myid,itwist):
  """ nexus style input name
  Args:
    myid (str): prefix
    itwist (int): twist index
  Returns:
    str: input filename
  """
  prefix = myid.split('.')[0]
  infile = '.'.join([prefix,'g%s'%str(itwist).zfill(3),'twistnum_%d'%itwist,'in','xml'])
  return infile
# end def nexus_infile

def twist_input(myid,calc_dir,itwist,qsyst,nwalker):
  """ construct an example QMCPACK input file, using a database of <qmcsystem>
   specialized for dynamic-proton hydrogen DMC
   while the constructed input is specialized, it can be edited later.
   The returned lxml document should be considered as a template.
   This function:
    1. checks the consistency of input against wf hdf5
    2. checks the consistency of input name against twist
    3. set the number of walkers produced by VMC consistent with DMC need

  Args: 
    myid (str): simulation identifier
    calc_dir(str): simulation path
    itwist (int): twist-averaged boundary condition index
    qsyst (str): text representation of <qmcsystem>
    nwalker (int): 
  Returns:
    lxml.etree.Document: doc, lxml document containing the constructed input
  """
  from qharv.seed import xml, xml_examples
  from qharv.inspect import inp_xml

  calc_name = os.path.basename(calc_dir)

  # step 1: check wf.h5 and axes are consistent
  qnode = xml.parse(qsyst)
  if not inp_xml.check_wf_hdf5(qnode,calc_dir,folded=False):
    raise RuntimeError('wavefunction h5 axes mismatch')
  # end if
  bb = qnode.find('.//sposet_builder')
  h5_fname = os.path.basename( bb.get('href') )
  if h5_fname != 'pwscf.pwscf.h5': # make sure this is the electronic wavefunction
    raise RuntimeError('failed to find electronic spo builder')
  if (itwist != int(bb.get('twistnum'))): # ensure the twist is correct
    raise RuntimeError('twist mismatch')

  # step 2: setup "nbdmc" folder
  if not os.path.isdir(calc_dir):
    sp.check_call(['mkdir','-p',calc_dir])
  # end if

  # !!!! double-check inputs for Titan before submitting
  # step 3: assemble QMCPACK input
  # edit <wavefunction> 
  #  use <ionwf>
  wf_node = qnode.find('.//wavefunction')
  # get proton exponent
  proton_exp = float(wf_node.find('.//radfunc').get('exponent'))
  pwidth = 1./(2*proton_exp)**0.5
  # remove proton spline builder
  pbb = wf_node.find('.//sposet_builder[@type="mo"]')
  pbb.getparent().remove(pbb)
  # remove hartree_product
  hp_node = wf_node.find('hartree_product')
  hp_node.getparent().remove(hp_node)

  # put in <ionwf>
  src_pset = 'wf_centers'
  pos    = xml.get_pos(qnode,pset=src_pset) # this line will fail if src_pset does not exist
  natom  = len(pos)
  widths = ['-1.0']*natom + [str(pwidth)]*natom # !!!! assume electrons come first
  iwf_node = etree.Element('ionwf',name='proton_wf',source=src_pset,width=' '.join(widths))
  wf_node.append(iwf_node)

  # unncessary, but move <backflow> for aesthetics
  bf_node = wf_node.find('.//backflow')
  bparent = bf_node.getparent()
  bparent.remove(bf_node)
  bparent.insert(0,bf_node)

  # edit <hamiltonian>
  hnode = xml_examples.dynamic_ae_ham()
  hnode0= qnode.find('.//hamiltonian')
  hidx  = hnode0.getparent().index(hnode0)
  hnode0.getparent().insert(hidx,hnode)
  hnode0.getparent().remove(hnode0)

  # add <qmc> sections
  vnode = xml_examples.wbyw_vmc()
  vnode.set('checkpoint','0')
  xml.set_param(vnode,'blocks','256')
  xml.set_param(vnode,'timestep','0.03')
  xml.set_param(vnode,'warmuptimestep','0.01')
  xml.set_param(vnode,'warmupsteps','16')
  xml.set_param(vnode,'samples',str(nwalker))
  vnode.append( etree.Element('fokker_planck',{'nfp':'1'}) )
  vnode.append( etree.Element('ts_boost',{'group':'p','factor':'3'}) )

  dnode1 = xml_examples.wbyw_dmc()
  xml.set_param(dnode1,'targetwalkers',str(nwalker),new=True)
  xml.set_param(dnode1,'blocks','64')
  xml.set_param(dnode1,'timestep','0.005')
  xml.set_param(dnode1,'steps','80')
  xml.set_param(dnode1,'feedback','0.1',new=True)
  xml.set_param(dnode1,'maxcpusecs','2400',new=True)

  dnode2 = xml_examples.wbyw_dmc()
  xml.set_param(dnode2,'targetwalkers',str(nwalker),new=True)
  xml.set_param(dnode2,'blocks','64')
  xml.set_param(dnode2,'timestep','0.003')
  xml.set_param(dnode2,'steps','133')
  xml.set_param(dnode2,'feedback','0.05',new=True)
  xml.set_param(dnode2,'maxcpusecs','3600',new=True)
  dnode2.append( etree.Element('dumpconfig',{'stride':'20'}) )

  dnode3 = xml_examples.wbyw_dmc()
  xml.set_param(dnode3,'targetwalkers',str(nwalker),new=True)
  xml.set_param(dnode3,'blocks','64')
  xml.set_param(dnode3,'timestep','0.0015')
  xml.set_param(dnode3,'steps','267')
  xml.set_param(dnode3,'feedback','0.05',new=True)
  xml.set_param(dnode3,'maxcpusecs','7200',new=True)
  dnode3.append( etree.Element('dumpconfig',{'stride':'20'}) )

  calc_nodes = [vnode,dnode1,dnode2,dnode3]
  snode = etree.Element('simulation')
  pnode = etree.Element('project',{'id':myid+'-'+calc_name,'series':'0'})
  for node in [pnode,qnode]+calc_nodes:
    snode.append(node)
  # end for

  doc = etree.ElementTree(snode)
  return doc
# end def twist_input
