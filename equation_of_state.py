import numpy as np
import scipy.optimize as op
linear = lambda x,a,b:a/x+b
plinear = lambda x,a,b:a/x**2.
quad   = lambda x,a,b,c:a/x**2.+b/x+c
pquad   = lambda x,a,b,c:2.*a/x**3.+b/x**2.
cubic  = lambda x,a,b,c,d:a/x**3.+b/x**2.+c/x+d
pcubic  = lambda x,a,b,c,d:3*a/x**4.+2.*b/x**3.+c/x**2.
models  = {1:linear,2:quad,3:cubic}
pmodels = {1:plinear,2:pquad,3:pcubic}

def check_order(order):
  if order not in models.keys():
    raise RuntimeError('order=%d not supported, must be one of 1,2,3'%order)
  # end if
# end def check_order

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

  # step 1: decide which polynomial to use
  check_order(order)
  model = models[order]

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
    popt,pcov = op.curve_fit(model,myx,myy)
  else:
    popt,pcov = op.curve_fit(model,myx,myym,sigma=myye,absolute_sigma=True)
  # end if

  # step 4: estimate fit error
  perr = np.sqrt(np.diag(pcov))

  return popt,perr
# end def eos

def resample1d(myx,nx=100):
  finex = np.linspace(min(myx),max(myx),nx)
  return finex
# end def
