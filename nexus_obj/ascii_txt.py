import os

def dsk_out(fout,kvecs,mdsk,edsk):
  """ output fluctuating structure factor dS(k) in energy.pl format 
  Args:
    fout (str): output filename
    kvecs (list): a list of 3D vectors
    mdsk (list): mean <dS(k)>, one value per kvector
    edsk (list): error, one value per kvector
  Returns:
    bool: success
  """
  fp   = open(fout,'w')
  fp.write('# kx  ky  kz  dsk_mean  dsk_error\n') # write header
  for ik in range(len(kvecs)):
    kvec = kvecs[ik]
    dskm = mdsk[ik]
    dske = edsk[ik]
    line = '%8.4f  %8.4f  %8.4f  %10.6f  %10.6f\n' % (kvec[0],kvec[1],kvec[2],dskm,dske)
    fp.write(line)
  # end for
  fp.close()
  return True
# end def np_dsk_ascii

def qsub_file(fnames,nmpi=64,title='title',hours=2):
  header = """#!/bin/bash
#PBS -N %s
#PBS -l walltime=0%d:00:00
#PBS -l nodes=%d

#PBS -A mat158
#PBS -j oe
#PBS -k n
cd ${PBS_O_WORKDIR}
export OMP_NUM_THREADS=8

BIN=~/soft/kylin_qmcpack/qmcpack_cpu_comp\n\n""" % (
  title,
  hours,
  len(fnames)*nmpi/2
  )

  body = 'cwd=`pwd`\n'
  for floc in fnames:
    fname  = os.path.basename(floc)
    rundir = os.path.dirname(floc)
    move_cmd = 'cd '+rundir
    run_cmd  = 'aprun -n %d -d 8 -S 1 $BIN '%nmpi + fname + ' > out 2> err&'
    body += '\n'.join([move_cmd,run_cmd,'cd $cwd']) + '\n'
  # end for fname
  body += '\nwait'

  text = header + body
  return text
# end def qsub_file

def bw_qsub_file(fnames,nmpi=64,title='title',hours=2,queue='normal'):
  qs = ['low','normal','high']
  if queue not in qs:
    raise RuntimeError('unknown queue %s; choose from'%queue+str(qs))
  # end if
  header = """#!/bin/bash
#PBS -N %s
#PBS -l walltime=0%d:00:00
#PBS -l nodes=%d:ppn=32:xe
#PBS -q %s

. /opt/modules/default/init/bash
module load cray-hdf5-parallel libxml2

cd ${PBS_O_WORKDIR}
export OMP_NUM_THREADS=8

BIN=~/soft/kylin_qmcpack/qmcpack_cpu_comp\n\n""" % (
  title,
  hours,
  len(fnames)*nmpi/4,
  queue
  )

  body = 'cwd=`pwd`\n'
  for floc in fnames:
    fname  = os.path.basename(floc)
    rundir = os.path.dirname(floc)
    move_cmd = 'cd '+rundir
    run_cmd  = 'aprun -n %d -d 8 -S 1 $BIN '%nmpi + fname + ' > out 2> err&'
    body += '\n'.join([move_cmd,run_cmd,'cd $cwd']) + '\n'
  # end for fname
  body += '\nwait'

  text = header + body
  return text
# end def qsub_file


