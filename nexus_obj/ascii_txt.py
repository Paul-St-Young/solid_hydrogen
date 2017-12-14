import os

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
