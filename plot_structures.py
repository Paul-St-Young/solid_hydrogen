import numpy as np
import pandas as pd
# define some static values
struct_markers = {
    'cmca4':'v',
    'cmca12':'^',
    'c2c':'d',
    'p6122':'h',
    'i41amd':'*',
    'pc':'.',
    'c2c-12':'+',
    'p63m':'x'
}

func_lines = {
    'pbe':'-.',
    'blyp':'--',
    'vdw-df':'-'
}

struct_colors = {
    'cmca4':'r',
    'cmca12':'m',
    'c2c':'b',
    'p6122':'black',
    'i41amd':'gray',
    'pc':'y',
    'c2c-12':'g',
    'p63m':'violet'
}

space_group_id = {
  'cmca':64,
  'c2c':15,
  'i41amd':141,
}

# CODATA 2018
#bohr = 0.52917721067e-10 # m
bohr = 0.529177210903  # A
ha   = 27.211386245988   # ev
joule= 6.241509126e18    # ev

gpa = 29421
gpa = ha/joule/(bohr*1e-10)**3./1e9
mev = ha*1000.


def xy(df, sel, xname, yname):
  myx = df.loc[sel, xname].values
  myy = df.loc[sel, yname].values
  idx = np.argsort(myx)
  return myx[idx], myy[idx]


def xyye(df,sel,xname,yname):
  ymean = yname+'_mean'
  yerr  = yname+'_error'

  myx = df.loc[sel,xname].values
  myy = df.loc[sel,ymean].values
  myye= df.loc[sel,yerr].values
  idx = np.argsort(myx)
  return myx[idx],myy[idx],myye[idx]


def xxeyye(df,sel,xname,yname):
  xmean = xname+'_mean'
  xerr  = xname+'_error'
  ymean = yname+'_mean'
  yerr  = yname+'_error'

  myx = df.loc[sel,xmean].values
  myxe= df.loc[sel,xerr].values
  myy = df.loc[sel,ymean].values
  myye= df.loc[sel,yerr].values
  idx = np.argsort(myx)
  return myx[idx], myxe[idx],myy[idx],myye[idx]


def show_sel_xyname(ax,sel,xname,yname,df,**kwargs):
  myx,myy,myye = xyye(df,sel,xname,yname)
  line = ax.errorbar(myx,myy,myye,**kwargs)
  return line
# end def show_sel_xyname

def plot_hydrogen_solids(ax,plotdf,xname='myx',yname='myy',draw_scatter=True,draw_line=True
    ,candidates=['cmca4','cmca12','c2c'],funcs=['pbe']):
    """ given a pandas dataframe 'plotdf' that uses (candidate_name,functional) as index,
     and contains the columns 'xname' and 'yname', plot yname against xname on ax
     returns four lists: 1.scatter plots 2. lines
      3. scatter legend 4. line legend """
    
    scatter_plots = []
    line_plots = []
    lines = []
    istruct = 0
    for sname in candidates:
        ifunc   = 0
        for func in funcs:

            if (sname,func) not in plotdf.index:
                continue
            # end if

            mydf = plotdf.loc[(sname,func),:]
            mydf = mydf.sort_values(xname)
            myx  = mydf[xname]
            myy  = mydf[yname]

            # plot
            if draw_scatter:
                markers, = ax.plot(myx,myy
                    ,marker=struct_markers[sname],ls='',label=sname if ifunc==0 else ''
                    ,color=struct_colors[sname],ms=5)
                if not markers.get_label().startswith('_'):
                    scatter_plots.append(markers)
                # end if
            # end if

            if draw_line:
                line, = ax.plot(myx,myy
                    ,ls=func_lines[func],color=struct_colors[sname]
                    ,label=func if istruct==0 else '')
                if not line.get_label().startswith('_'):
                    lines.append(line)
                # end if
                line_plots.append(line)
            # end if draw_line
            
            ifunc += 1
        # end for func

        istruct += 1
    # end for sname

    if draw_scatter:
        scatter_leg = ax.legend(handles=scatter_plots,numpoints=1,loc='best')
        ax.add_artist(scatter_leg)
    else:
        scatter_leg = None
    # end if

    if draw_line:
        line_leg = ax.legend(handles=lines,loc='upper center')
        for handle in line_leg.legendHandles:
            handle.set_color('black')
        # end for handle
        ax.add_artist(line_leg)
    else:
        line_leg = None
    # end if
    return scatter_plots,line_plots,scatter_leg,line_leg
# end plot_hydrogen_solids

def plot_against_sname(ax,df,xcol,ycol,slist,yerr=False,**kwargs):
  if yerr:
    ymean = ycol + '_mean'
    yerror= ycol + '_error'
  # end if

  lines = []
  for sname in slist:
    sel  = df.sname==sname

    mydf = df.loc[sel].sort_values(xcol)
    myx  = mydf.loc[:,xcol]
    if yerr:
      myy  = mydf.loc[:,ymean]
      myye = mydf.loc[:,yerror]
    else:
      myy  = mydf.loc[:,ycol]
      myye = 0.0
    # end if

    line = ax.errorbar(myx,myy,yerr=myye
      ,marker=struct_markers[sname],c=struct_colors[sname]
      ,label=sname
      ,**kwargs)

    lines.append(line)
  # end for
  return lines
# end def plot_against_sname

def plot_against_structs(ax,df,xcol,ycol,func,no_error=False,slist=[],lw=2,ms=10,alpha=1.0):
  """df must have columns ["struct","func","press"] + [xcol,ycol]. 
  only one functional 'func' will be selected, then plot ycol vs. xcol
  return: lines, line_leg """
  
  if no_error:
    xmean = xcol
    ymean = ycol
  else:
    xmean = xcol + '_mean'
    xerror= xcol + '_error'
    ymean = ycol + '_mean'
    yerror= ycol + '_error'
  # end if

  pdf = df
  
  lines = []

  if len(slist) == 0:
    slist = pdf['struct'].unique()
  # end if

  for sname in slist:
    sel = (pdf['struct'] == sname) #& (pdf['func']==func)
    if no_error:
      cols = ['press',xcol,ycol,xmean,ymean]
    else:
      cols = ['press',xcol,ycol,xmean,xerror,ymean,yerror]

    mydf = pdf.loc[sel,]
    mydf = mydf.sort_values(['press'])

    if no_error:
      line = ax.plot(mydf[xcol],mydf[ycol]
          ,marker=struct_markers[sname],c=struct_colors[sname]
          ,ls='-',lw=lw,ms=ms,label=sname,alpha=alpha)
    else:
      line = ax.errorbar(mydf[xmean],mydf[ymean]
          ,xerr=mydf[xerror],yerr=mydf[yerror]
          ,marker=struct_markers[sname],c=struct_colors[sname]
          ,ls='-',lw=lw,ms=ms,label=sname,alpha=alpha)
    # end if
    lines.append(line)
  # end for sname

  line_leg = ax.legend(loc=0)#'lower right')
  
  return lines,line_leg
# end def plot_against_structs

def plot_against_funcs(ax,df,xcol,ycol,sname):
    """df must have columns ["struct","func","press"] + [xcol,ycol]. 
    only one structure 'sname' will be selected, then plot ycol vs. xcol """
    
    xmean = xcol + '_mean'
    xerror= xcol + '_error'
    ymean = ycol + '_mean'
    yerror= ycol + '_error'
    
    lines = []
    for func in df['func'].unique():
        sel = (df['struct'] == sname) & (df['func']==func)

        mydf = df.loc[sel,['press',xmean,xerror,ymean,yerror]]
        mydf = mydf.sort_values(['press'])

        line = ax.errorbar(mydf[xmean],mydf[ymean]
            ,xerr=mydf[xerror],yerr=mydf[yerror]
            ,marker=struct_markers[sname],c='black'
            ,ls=func_lines[func],label=func)
        lines.append(line)
    # end for sname

    line_leg = ax.legend(loc='lower right')
    
    return lines,line_leg
# end def plot_against_funcs

def interpret_subdir(subdir,sep='_'):
    """ find (struct,func,press) from subdir of form cmca4-pbe-2000 """
    tokens = subdir.split(sep)
    if len(tokens) == 3:
        struct,func,press = tokens
    elif len(tokens) == 4: # extra 'if' needed when sep='-' and vdw-df
        struct,func1,func2,press = tokens
        func = '-'.join([func1,func2])
    else:
        raise RuntimeError('cannot interpret %s'%subdir)
    # end if
    press = int(press)
    return struct,func,press
# end def

def sfp_label(subdir,sep='_'):
  struct,func,press = interpret_subdir(subdir,sep)
  return pd.Series({'struct':struct,'func':func,'press':press})

def sfr_label(subdir,sep='_'):
  struct,func,rs100 = interpret_subdir(subdir,sep)
  return pd.Series({'struct':struct,'func':func,'rs100':rs100})

def sra_label(subdir,sep='-'):
  tokens = subdir.split(sep)
  sname  = tokens[0]
  rst    = tokens[1]
  cat    = tokens[2]
  rs = float(rst.replace('rs',''))
  ca = float(cat.replace('ca',''))
  return {'sname':sname,'rs':rs,'ca':ca}
# end def sra_label

def srab_label(subdir,sep='-'):
  sname,rst,cat,rbt = subdir.split(sep)
  rs = float(rst.replace('rs',''))
  ca = float(cat.replace('ca',''))
  rb = float(rbt.replace('rb',''))
  return {'sname':sname,'rs':rs,'ca':ca,'rb':rb}
# end def srab_label

def dbond_dn(moldir,sep='-'):
  tokens = moldir.split(sep)
  dbond  = float(tokens[0].replace('dbond',''))
  dn     = int(tokens[1].replace('dn',''))
  return {'dbond':dbond,'dn':dn}
# end def dbond_dn

def cell_volume(axes):
  return np.dot(axes[0],np.cross(axes[1],axes[2]))
def vol2rs(vol):
    return (3.*vol/(4*np.pi))**(1./3)
def rs2vol(rs):
    return 4*np.pi/3*rs**3.

def rho2rs(rho, nprot):
  """ convert density (g/cm3) to rs """
  amu = 1.6605390666e-27  # kg
  mh = 1.00784*amu
  cm3 = (10**8/bohr)**3  # cm^3 to bohr^3

  nh = rho*1e-3/mh/nprot
  vol = cm3/nh
  rs = vol2rs(vol)
  return rs

def rs2rho(rs, nprot):
  """ convert rs to density (g/cm3) """
  amu = 1.6605390666e-27  # kg
  mh = 1.00784*amu
  cm3 = (10**8/bohr)**3  # cm^3 to bohr^3

  vol = rs2vol(rs)
  rho = nprot*mh/vol
  rho *= 1e3  # kg to g
  return rho * cm3

def hug(em, pm, rs, e0, p0, rs0):
  mye = em/ha  # convert eV to ha
  myp = pm/gpa # convert GPa to ha/bohr^3
  vol = rs2vol(rs)
  mye0 = e0/ha
  myp0 = p0/gpa
  vol0 = rs2vol(rs0)
  de = mye-mye0
  dv = vol-vol0
  dp = myp-myp0
  return de+0.5*dv*dp

def add_cp_top(ax,xlabel='C$_p$ (bohr$^-2$)'):
  """ add Cp as top xlabel when actual xlabel is 1/Cp.
   Cp is the proton orbital gaussian exponent.  """
  ax1 = ax.twiny()
  ax1.set_xticks( ax.get_xticks() )
  ax1.set_xlim( ax.get_xlim() )
  ax1.set_xlabel(xlabel)

  xlabels = []
  for tick in ax1.get_xticks():
    inv = 1./tick
    xlabels.append("%3.1f"%inv)
  # end for tick
  ax1.get_xaxis().set_ticklabels( xlabels )
  return ax1
# end def

def add_diag_line(ax):
  ax.plot(ax.get_xlim(),ax.get_ylim(),c='k',ls='--')
# end def

def check_rs(axes,natom,rs0,rs_tol=1e-6):
  from qharv.inspect import axes_pos
  rs = axes_pos.rs(axes,natom)
  if not np.isclose(rs,rs0,atol=rs_tol):
    raise RuntimeError('rs mismatch')
  # end if
# end def check_rs

def check_ca(axes0,ca0,aidx=0,cidx=2,ca_tol=5e-3):
  from qharv.inspect import axes_pos
  abc= axes_pos.abc(axes0)
  ca = abc[cidx]/abc[aidx]
  if not np.isclose(ca,ca0,atol=ca_tol):
    raise RuntimeError('ca mismatch')
  # end if
# end def check_ca

def check_rb(axes,pos,rb0,rmax=1.6,rb_tol=1e-3):
  from qharv.inspect import axes_pos
  upair,udist = axes_pos.dimer_pairs_and_dists(axes,pos,rmax)
  if not len(udist) == len(pos)/2:
    raise RuntimeError('wrong number of molecules %d' % len(udist))
  # end if

  rb = udist.mean()
  if not np.isclose(rb,rb0,atol=rb_tol):
    raise RuntimeError('rb mismatch')
  # end if
# end def check_rb


# ---------------- database I/O ----------------


def add_columns(df, ae=True):
  """ add Potential and Virial columns
  ['axes', 'pos'] must have been added already
  """
  from qharv.inspect import axes_pos
  df['natom'] = df['pos'].apply(len)
  df['volume'] = df['axes'].apply(axes_pos.volume)
  df['rho'] = df['natom']/df['volume']
  df['rs'] = df['rho'].apply(lambda x:vol2rs(1./x))
  df['rs100'] = df['rs'].apply(lambda x:int(round(100*x)))

  """ # Potential for dynamic is simply ElecElec; decide outside of this func.!
  if ae:
    df['Potential_mean'] = df['ElecElec_mean']\
                         + df['ElecIon_mean']\
                         + df['IonIon_mean']
    df['Potential_error'] = np.sqrt(df['ElecElec_error']**2.\
                         +df['ElecIon_error']**2)
  else:
    df['Potential_mean'] = df['LocalPotential_mean']
    df['Potential_error'] = df['LocalEnergy_error']
  # end if ae
  """

  # check E - (T+V), expect 0
  df['EmTV_mean'] = df['LocalEnergy_mean'] \
                  - (df['Kinetic_mean'] + df['Potential_mean'])
  df['EmTV_error'] = np.sqrt(df['Kinetic_error']**2. \
                           + df['Potential_error']**2.)  # correlated error

  # check Virial, expect 0
  df['Virial_mean'] = 2*df['Kinetic_mean']\
                    + df['Potential_mean']\
                    - 3*df['Pressure_mean']*df['volume']
  df['Virial_error'] = np.sqrt(
    2*df['Kinetic_error']**2. + df['Potential_error']**2\
  + 3*df['Pressure_error']**2.*df['volume']
  )

  # enthalpy
  df['Enthalpy_mean'] = df['LocalEnergy_mean'] + df['Pressure_mean']*df['volume']
  df['Enthalpy_error'] = np.sqrt(df['LocalEnergy_error']**2. +\
    (df['Pressure_error']*df['volume'])**2.)


def add_per_atom_columns(df):
  """ energy, kinetic, potential
  ['natom', 'Potential'] must have been added already
  """

  # add per-proton (pp) columns with error
  xname = 'natom'
  obsl = ['LocalEnergy', 'Kinetic', 'Potential'
    , 'EmTV', 'Virial', 'Enthalpy', 'Variance']
  for obs in obsl:
    ymean = obs+'_mean'
    yerror = obs+'_error'
    obs1 = obs+'_pp'
    ymean1 = obs1+'_mean'
    yerror1 = obs1+'_error'
    df[ymean1] = df[ymean]/df[xname]
    df[yerror1] = df[yerror]/df[xname]

  # add per-proton (pp) columns without error
  df['volpp'] = df['volume']/df[xname]

  df['Epp_vint_mean'] = df['LocalEnergy_pp_mean'] + df['vint']/1e3
  df['Epp_vint_error'] = df['LocalEnergy_pp_error']


def add_unit_columns(df):
  """ add different units of columns
  """
  df['Pressure_gpa_mean'] = df['Pressure_mean']*gpa
  df['Pressure_gpa_error'] = df['Pressure_error']*gpa

def add_sname_legend(ax, snames, extra_style=None,
  marker=False, **kwargs):
  from qharv.plantation import kyrt
  if extra_style is None:
    extra_style = {}
  styles = []
  for sname in snames:
    style = extra_style.copy()
    style.update({'c': struct_colors[sname]})
    if marker:
      style.update({'marker': struct_markers[sname]})
    styles.append(style)
  leg = kyrt.create_legend(ax, styles, snames, **kwargs)
  ax.add_artist(leg)
