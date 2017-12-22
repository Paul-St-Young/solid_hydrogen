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

bohr = 0.52917721067e-10 # m
ha   = 27.21138602       # ev
joule= 6.241509126e18    # ev

gpa = ha/joule/bohr**3./1e9
mev = ha*1000.

def xyye(df,sel,xname,yname):
  ymean = yname+'_mean'
  yerr  = yname+'_error'

  myx = df.loc[sel,xname]
  myy = df.loc[sel,ymean]
  myye= df.loc[sel,yerr]
  return myx,myy,myye
# end def xyye

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
  sname,rst,cat = subdir.split(sep)
  rs = float(rst.replace('rs',''))
  ca = float(cat.replace('ca',''))
  return pd.Series({'sname':sname,'rs':rs,'ca':ca})
# end def sra_label

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
# end def vol2rs
def rs2vol(rs):
    return 4*np.pi/3*rs**3.
# end def

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
