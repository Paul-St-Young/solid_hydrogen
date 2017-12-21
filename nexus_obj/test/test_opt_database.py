#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qharv.inspect import crystal

import static_correlation as sc
from nexus_obj import opt_database as odat
import qe_reader as qer

if __name__ == '__main__':
  frac = 0.1

  axes = np.loadtxt('axes.dat')
  pos  = np.loadtxt('pos.dat')
  pos1,com = odat.stretch_dimers(axes,pos,frac)

  fig = plt.figure()
  ax  = fig.add_subplot(1,1,1,projection='3d')
  crystal.draw_cell(ax,axes)
  crystal.draw_atoms(ax,pos,ms=5)
  crystal.draw_atoms(ax,pos1,ms=10,alpha=0.4)
  crystal.draw_atoms(ax,com,ms=1,c='r')
  plt.show()

# end __main__
