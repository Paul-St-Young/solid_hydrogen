# define some static values
struct_markers = {
    'cmca4':'v',
    'cmca12':'^',
    'c2c':'d',
    'p6122':'h',
    'i41amd':'*',
    'pc':'.'
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
    'pc':'y'
}

bohr = 0.52917721067e-10 # m
ha   = 27.21138602 # ev
joule= 6.241509126e18 # ev

gpa = ha/joule/bohr**3./1e9

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
