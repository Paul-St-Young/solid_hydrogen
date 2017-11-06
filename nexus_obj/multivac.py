# the one machine
import os

def apply_machine_settings(machine,run_dir,account='',nk=1,**kwargs):
  """ apply default machines settings for hydrogen simulations
   return a dictionary of jobs for ['dft','p2q','opt','dmc'] 
  Args:
    machine (str): one of ['golub','titan','ws4','ws8',...]. 'ws' stands for workstation.
    run_dir (str): directory to run the jobs in.
    account (str,optional): default is to depend on the machine's default in nexus.
    nk (int,optional): the npool option in pw.x, default is 1 because it always runs. For large number of kpoints, set nk to be as large as possible for speed.
    kwargs (dict): extra keyword arguments to be passed to nexus.settings.
  Returns:
    dict: jobs, a dictionary of nexus.obj objects for ['dft','p2q','opt','dmc'], each can be passed to create nexus.Job objects using Job(**jobs['dft']), for example.
    """
  if (machine not in ['golub','titan'])  and (not machine.startswith('ws')):
    raise NotImplementedError('cannot handle machine=%s yet'%machine)
  # end if
  
  from nexus import settings, obj
  if machine == 'titan':
    account    = 'mat158'
    pseudo_dir = '/ccs/home/yyang173/scratch/hsolid/pseudo'
    vdw_table  = '/ccs/home/yyang173/scratch/hsolid/vdw/vdW_kernel_table'
    qedir      = '~/soft/espresso-5.3.0/bin'
    ccdir      = '~/soft/qmcpack-espresso-5.3.0/bin'
    qmcdir     = '~/soft/kylin_qmcpack'
  elif machine == 'golub':
    pseudo_dir = '/home/yyang173/scratch/hsolid/pseudo'
    vdw_table  = '/home/yyang173/scratch/hsolid/vdw/vdW_kernel_table'
    qedir      = '/projects/physics/yyang/soft/espresso-5.3.0/bin'
    ccdir      = '/projects/physics/yyang/soft/qmcpack-espresso-5.3.0/bin'
    qmcdir     = '~/soft/kylin_qmcpack'
  else: # workstation, defaults should do
    pseudo_dir = '/home/yyang173/Desktop/hsolid/pseudo'
    vdw_table  = '/home/yyang173/Desktop/hsolid/vdw/vdW_kernel_table'
    qedir      = '~/soft/espresso-5.3.0/bin'
    ccdir      = '~/soft/qmcpack-espresso-5.3.0/bin'
    qmcdir     = '~/soft/kylin_qmcpack'
  # end if
  settings(
    runs       = run_dir,
    machine    = machine,
    account    = account,
    pseudo_dir = pseudo_dir,
    vdw_table  = vdw_table,
    **kwargs
  )

  pw_bin   = os.path.join(qedir,'pw.x')
  cc_bin   = os.path.join(ccdir,'pw2qmcpack.x')
  qmc_bin  = os.path.join(qmcdir,"qmcpack_cpu_real")
  tabc_bin = os.path.join(qmcdir,"qmcpack_cpu_comp")

  # assign jobs
  if machine == 'titan':
    dft_job  = obj(nodes=1,minutes=10,app_options="-nk %d"%nk,app=pw_bin)
    p2q_job  = obj(nodes=1,serial=True,minutes=30,app=cc_bin)
    opt_job  = obj(nodes=4,threads=8,hours=2,app=qmc_bin)
    dmc_job  = obj(nodes=8,threads=8,hours=2,app=qmc_bin)
  elif machine == 'golub':
    dft_job  = obj(nodes=1,hours=4,app=pw_bin,app_options='-nk %d'%nk)
    p2q_job  = obj(nodes=1,serial=True,minutes=30,app=cc_bin)
    opt_job  = obj(nodes=1,threads=8,hours=4,app=qmc_bin)
    dmc_job  = obj(nodes=1,threads=8,hours=4,app=qmc_bin)
  else:
    dft_job  = obj(app_options='-nk %d'%nk,app=pw_bin)
    p2q_job  = obj(serial=True,app=cc_bin)
    opt_job  = obj(app=qmc_bin,threads=8)
    dmc_job  = obj(app=qmc_bin,threads=8)
  # end if
  tabc_job     = dmc_job.copy()
  tabc_job.app = tabc_bin
  jobs = {'dft':dft_job,'p2q':p2q_job,'opt':opt_job,'dmc':dmc_job,'tabc':tabc_job}

  return jobs
# end def apply_machine_settings

