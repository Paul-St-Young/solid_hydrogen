# Fit energy per particle (epp) to a polynomial in inverse volume per particle (vpp).
# Each energy model has a pmodel corresponding to its pressure.

# Assume:
#  1. Raw data are stored in a pandas dataframe (df).
#  2. Rows to use in the fit are given by a boolean array selector (sel).

import numpy as np
import pandas as pd
import scipy.optimize as op
linear = lambda x,a,b:a/x+b
plinear = lambda x,a,b:a/x**2.
quad   = lambda x,a,b,c:a/x**2.+b/x+c
pquad   = lambda x,a,b,c:2.*a/x**3.+b/x**2.
cubic  = lambda x,a,b,c,d:a/x**3.+b/x**2.+c/x+d
pcubic  = lambda x,a,b,c,d:3*a/x**4.+2.*b/x**3.+c/x**2.
models  = {1:linear,2:quad,3:cubic}
pmodels = {1:plinear,2:pquad,3:pcubic}

# ---- level 0: based on model and pmodel ---- #

def check_order(order):
  if order not in models.keys():
    raise RuntimeError('order=%d not supported, must be one of 1,2,3'%order)
  # end if
# end def check_order

def eos_no_error(volpp,epp,order):
  model = models[order]
  popt,pcov = op.curve_fit(model,volpp,epp)
  perr = np.sqrt(np.diag(pcov))
  return popt,perr
# end def

def eos_with_error(volpp,eppm,eppe,order):
  model = models[order]
  popt,pcov = op.curve_fit(model,volpp,eppm,sigma=eppe,absolute_sigma=True)
  perr = np.sqrt(np.diag(pcov))
  return popt,perr
# end def

def fpeos_with_error(volpp,eppm,eppe,order):
  """ energy and pressure v.s. volume functions
  Args:
    volpp (np.array): volume per particle
    eppm  (np.array): energy per particle mean
    eppe  (np.array): energy per particle error
  Returns:
    tuple: (feos,peos), each a function a volpp
  """
  popt,perr = eos_with_error(volpp,eppm,eppe,order)                           
  feos = lambda volpp:models[order]( volpp,*popt)                                   
  peos = lambda volpp:pmodels[order](volpp,*popt)                                   
  return feos,peos                                                              
# end def

def eos(df,sel,volpp_name,epp_name,order,with_error=False):
  """ fit energy per particle (epp) to inverse volume per particle (volpp)
  Args:
    df (pd.DataFrame): must have columns [volpp_name,epp_name]
    sel (np.array): boolean array for selection rows
    volpp_name (str): name of volpp columns
    epp_name (str): name of epp columns
    order (int): order of polynomial, support 1,2,3
    with_error (bool,optional): fit with error, if True, df must have mean and error columns. Assume mean and error have names [epp_name+'_mean',epp_name+'_error'].
  Returns:
    tuple: (popt,perr), optimal polynomial coefficients and associated errors
  """
  # step 1: validate input
  check_order(order)

  # step 2: get data
  myx = df.loc[sel,volpp_name]

  if not with_error:
    myy  = df.loc[sel,epp_name]
  else:
    myym = df.loc[sel,epp_name+'_mean']
    myye = df.loc[sel,epp_name+'_error']
  # end if

  # step 3: get fit
  if not with_error:
    popt,perr = eos_no_error(myx,myy,order)
  else:
    popt,perr = eos_with_error(myx,myym,myye,order)
  # end if

  return popt,perr
# end def eos

def get_eos(df,sel,order,volpp_name='volpp',epp_name='Epp',with_error=False):
  """ get energy and pressure as a function of volume from df&sel
  Args:
    df (pd.Series): dataframe
    sel (np.array): boolean array i.e. selector
    order (int): polynomial order
  Returns:
    tuple: (feos,peos)
  """
  check_order(order)
  popt,perr = eos(df,sel,volpp_name,epp_name,order,with_error=with_error)
  model = models[order]
  pmodel = pmodels[order]
  feos = lambda x: model(x,*popt)
  peos = lambda x:pmodel(x,*popt)
  return feos,peos
# end def get_eos

def resample1d(myx,nx=100):
  finex = np.linspace(min(myx),max(myx),nx)
  return finex
# end def

def mutual_range(myx0,myx):
  nx = min(len(myx0),len(myx))
  xmin = max(min(myx0),min(myx))
  xmax = min(max(myx0),max(myx))
  return np.linspace(xmin,xmax,nx)
# end def

def relative_energy(volpp,feos0,feos1):
  """ evaluate the relative energy between two eos at selected volpp 
  Args:
    volpp (np.array): a list of volume per particle to evaluate eos at
    feos0 (function): reference eos
    feos1 (function): target eos
  Returns:
    np.array: de, the energy difference at volpp
  """
  e0 = feos0(volpp)
  e1 = feos1(volpp)
  de = e1-e0
  return de
# end def relative_energy

def volpp_at_pressures(myp,peos,vol0=9.):
  """ find volpp corresponding the pressures myp
  Args:
    myp (np.array): a list of pressures in Hartree atomic units
    peos (function): pressure eos
    vol0 (float): initial guess for volpp
  Returns:
    np.array: volpp
  """
  volpp = np.array([op.fsolve(lambda x:peos(x)-p,vol0)[0] for p in myp])
  return volpp
# end def volpp_at_pressures

def absolute_enthalpy(myp,feos,peos):
  """ evaluate the absolute enthalpy using energy & pressure v.s. volume
   functions
  Args:
    myp (np.array): a list of pressures in Hartree atomic units
    feos (function): energy eos
    peos (function): pressure eos
  Returns:
    np.array: h, a list of enthalpies difference at myp
  """
  v  = volpp_at_pressures(myp,peos)
  e = feos(v)
  p = peos(v)
  h = e+p*v
  return h
# end def absolute_enthalpy

def relative_enthalpy(myp,feos0,peos0,feos1,peos1):
  """ evaluate the relative enthalpy between two eos at selected myp
  Args:
    myp (np.array): a list of pressures in Hartree atomic units
    feos0 (function): reference eos
    peos0 (function): reference pressure eos
    feos1 (function): target eos
    peos1 (function): target pressure eos
  Returns:
    np.array: dh, the enthalpy difference at myp
  """

  # get reference enthalpies
  v0 = volpp_at_pressures(myp,peos0)
  e0 = feos0(v0)
  p0 = peos0(v0)
  h0 = e0+p0*v0

  # get target enthalpies
  v1 = volpp_at_pressures(myp,peos1)
  e1 = feos1(v1)
  p1 = peos1(v1)
  h1 = e1+p1*v1

  dh = h1-h0
  return dh
# end def relative_enthalpy

# ---- level 1: based on eos_df ---- #
def get_eos_df(df,order,volpp_name='volpp',epp_name='Epp',with_error=False):
  """ obtain eos_df for df 
  Args:
    df (pd.DataFrame): raw data, must have columns [volpp,epp]
    order (int): polynomial order, must be one of 1,2,3
  """
  data = []
  snamel = df.sname.unique()
  for sname in snamel:
    sel = df.sname == sname
    # get eos
    feos,peos = get_eos(df,sel,order,volpp_name=volpp_name,epp_name=epp_name,with_error=with_error)
    # get eos validity range
    myx  = df.loc[sel,volpp_name]
    entry = {'sname':sname,'volpp_min':min(myx),'volpp_max':max(myx)
      ,'feos':feos,'peos':peos}
    data.append(entry)
  # end for sname
  eos_df = pd.DataFrame(data)
  return eos_df
# end def get_eos_df

def eos_df_volpp(eos_df,esel,nx=32):
  xmin = eos_df.loc[esel,'volpp_min'].squeeze()
  xmax = eos_df.loc[esel,'volpp_max'].squeeze()
  volpp= np.linspace(xmin,xmax,nx)
  return volpp
# end def

def abs_epp_vs_volpp_eos(ax,eos_df,esel,nx=32,**kwargs):
  import plot_structures as ps
  sname = eos_df.loc[esel,'sname'].squeeze()
  feos = eos_df.loc[esel,'feos'].squeeze()
  myx  = eos_df_volpp(eos_df,esel)
  line = ax.plot(myx,feos(myx)
    ,c=ps.struct_colors[sname],lw=3,**kwargs)
  return line
# end def abs_epp_vs_volpp_eos

def abs_hpp_vs_volpp_eos(ax,eos_df,esel,nx=32,**kwargs):
  import plot_structures as ps
  sname = eos_df.loc[esel,'sname'].squeeze()
  feos = eos_df.loc[esel,'feos'].squeeze()
  peos = eos_df.loc[esel,'peos'].squeeze()
  myx  = eos_df_volpp(eos_df,esel)
  mye  = feos(myx)
  myp  = peos(myx)
  myy  = mye+myp*myx
  line = ax.plot(myx,myy
    ,c=ps.struct_colors[sname],lw=3,**kwargs)
  return line
# end def abs_hpp_vs_volpp_eos

def abs_epp_vs_volpp_raw(ax,df,sel,volpp_name='volpp',epp_name='Epp'
  ,with_error=False,**kwargs):
  import plot_structures as ps
  snamel = df.loc[sel,'sname'].unique()
  assert len(snamel) == 1
  sname = snamel[0]
  if not with_error:
    line = ax.plot(df.loc[sel,'volpp'],df.loc[sel,epp_name],ls='',ms=10
      ,c=ps.struct_colors[sname],marker=ps.struct_markers[sname],label=sname,**kwargs)
  else:
    mname = epp_name + '_mean'
    ename = epp_name + '_error'
    line = ax.errorbar(df.loc[sel,'volpp'],df.loc[sel,mname],yerr=df.loc[sel,ename]
      ,ls='',ms=10
      ,c=ps.struct_colors[sname],marker=ps.struct_markers[sname],label=sname,**kwargs)
  # end if
  return line
# end def abs_epp_vs_volpp_raw

# ---- level 2: relative quantities based on eos_df ---- #
