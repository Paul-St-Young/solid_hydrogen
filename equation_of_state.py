import numpy as np
import scipy.optimize as op
linear = lambda x,a,b:a/x+b
quad   = lambda x,a,b,c:a/x**2.+b/x+c
cubic  = lambda x,a,b,c,d:a/x**3.+b/x**2.+c/x+d
models = {1:linear,2:quad,3:cubic}

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
  if order not in models.keys():
    raise RuntimeError('order=%d not supported, must be one of 1,2,3'%order)
  # end if
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

def abs_epp_vs_vpp(vpp,popt,nx=100):
  # step 1: decide which polynomial to use
  order = len(popt)-1
  if order not in models.keys():
    raise RuntimeError('order=%d not supported, must be one of 1,2,3'%order)
  # end if
  model = models[order]

  # step 2: get eos
  feos = lambda x:model(x,*popt)

  # step 3: evaluate eos
  finex = np.linspace(min(vpp),max(vpp),nx)
  finey = feos(finex)
  return finex,finey
# end def abs_epp_vs_vpp
