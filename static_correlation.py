#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import pymatgen as mg

def gofr_snapshot(axes,pos,rmax,rmin=0,nbin=40,gofr_norm=None):
    """ calculate the pair correlation function of a snapshot of a crystal structure given the 'axes' of the simulation cell and the positions ('pos') of atoms inside the cell.
    'rmax' is the maximum pair distance to histogram. 'rmin','rmax', and 'nbin' are passed to numpy.histogram to generate the results.
    return 3 lists: r, g(r), and g(r) normalization.
    Note: if a gofr_norm is given, then it will not be recalculated. This can save time if the simulation box does not change. """
    elem = ['H']*len(pos)
    struct = mg.Structure(axes,elem,pos,coords_are_cartesian=True)
    dtable = struct.distance_matrix
    nptcl  = struct.num_sites

    # upper triagular distance matrix (i.e. unique pair distances)
    i_triu = np.triu_indices(nptcl,m=nptcl,k=1)
    pair_dists = dtable[i_triu]

    counts,ticks = np.histogram(pair_dists,range=(rmin,rmax),bins=nbin)
    myx = (ticks[:-1] + ticks[1:])/2.
    dx  = myx[1] - myx[0]
    
    if gofr_norm is None:
        gofr_norm = 4*np.pi*myx**2.*dx * nptcl*(nptcl-1)/2./struct.volume
    # end if
    myy = counts/gofr_norm
    
    return myx,myy,gofr_norm
# end def 

def sofk_snapshot(axes,pos,nkmax=5,legal_kvecs=None):
    """ calculate the structure factor of a snapshot of a crystal structure given 'axes' and 'pos' of atoms.
    'nkmax' is the number of kvectors to include in each of x,y,z directions. 
    return 2 lists: legal kvectors, and S(k) of each kvector """
    
    rho = lambda kvec: np.exp(1j* np.dot(pos,kvec) ).sum()
    elem = ['H']*len(pos)
    
    struct = mg.Structure(axes,elem,pos,coords_are_cartesian=True)
    reclat = struct.lattice.reciprocal_lattice
    
    if legal_kvecs is None:
        import itertools
        cube_pts = np.array( [a for a in itertools.product(
            range(nkmax),repeat=3)] ) # 3 dimensions
        legal_kvecs = np.dot(cube_pts,reclat.matrix)
    # end if
    
    sk_arr = np.array([(rho(kvec)*rho(-kvec)).real/len(pos) 
        for kvec in legal_kvecs])
    
    return legal_kvecs, sk_arr
# end def

def shell_avg_sofk(legal_kvecs,sk_arr,smear_fac=100):
    """ average S(k) if multiple k-vectors have the same magnitude,
     legal_kvecs.shape should be (Nk,Ndim)
     sk_arr.shape should be (Nk) """

    # determine similarity of k-vectors by converting magnitude to int
    kmags = np.linalg.norm(legal_kvecs,axis=1)
    kids  = map(int,map(round,kmags*smear_fac))

    # pick unique kshells according to index
    unique_kid   = np.unique(kids)

    # loop over each shell and average
    unique_kmags = np.zeros(len(unique_kid))
    unique_sk    = np.zeros(len(unique_kid))
    for iukmag in range(len(unique_kid)):
        kid = unique_kid[iukmag] # shell ID
        sel = kids == kid # select this shell
        unique_sk[iukmag] = np.mean(sk_arr[sel])
        unique_kmags[iukmag] = float(kid)/smear_fac
    # end for iukmag
    return unique_kmags,unique_sk
# end def 

def gr2sk(k,grx,gry,rho):
    """ return spherical S(k) at given 'k' value, assuming spherical g(r)=(grx,gry)
      'rho' is density N/Vol. """ 
    integrand = grx**2. * (gry-1.0) * np.sin(k*grx)/(k*grx)
    val = 4*np.pi*rho*sum(integrand*(grx[1]-grx[0]))
    return 1.+val
# end def

def plot_sofk(ax,legal_kvecs,sk_arr):
    ax.set_xlabel('k (1/bohr)')
    ax.set_ylabel('S(k)')

    # shell average
    unique_kmags,unique_sk = shell_avg_sofk(legal_kvecs,sk_arr)
     
    # visualize
    line, = ax.plot(unique_kmags[1:],unique_sk[1:])
    return line
# end def

def cubic_rws(entry):
    axes = entry['axes']
    alat = axes[0][0]
    if not np.allclose( axes, alat*np.eye(3) ):
        raise RuntimeError('non-cubic cell, please provide -rws')
    # end if
    return alat/2
# end def

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='calculate g(r) and S(k) from a database of crystal structures')
    parser.add_argument('config_json', type=str, help='json file containing axes and pos of atomic configurations')
    parser.add_argument('-rws','--r_wigner_seizs', type=float, default=None, required=False, help='Wigner-Seitz radius, used to cutoff g(r)')
    parser.add_argument('-is','--istart', type=int, default=0, help='index of first sample')
    parser.add_argument('-p','--plot', action='store_true', help='plot data')
    parser.add_argument('-s','--save', action='store_true', help='save plot')
    parser.add_argument('-f','--force', action='store_true', help='force analysis, ignore gofr.dat and sofk.dat in local directory')
    args = parser.parse_args()

    # read a list of configurations (axes and pos)
    #cdf = pd.read_json('i4-pbe-4800-48-solid_configs.json')
    cdf = pd.read_json(args.config_json)
    prefix = args.config_json.replace('.json','')

    # get the first sample
    istart = args.istart
    entry = cdf[istart]
    axes = entry['axes']
    pos  = entry['pos']

    # calculate Wigner-Seitz radius (only for cubic cell)
    if args.r_wigner_seizs is not None:
        rws = args.r_wigner_seizs
    else:
        rws = cubic_rws(entry)
    # end if

    # initialize gofr_norm with the first sample
    myx,myy,gofr_norm = gofr_snapshot(axes,pos,rmax=rws)
    legal_kvecs,sk_arr = sofk_snapshot(axes,pos)

    if (not (os.path.isfile('gofr.dat') and os.path.isfile('sofk.dat'))) or (args.force):
        # analyze the rest of the samples
        nsample  = len(cdf.columns)
        all_gofr = np.zeros([nsample,len(myy)])
        all_sofk = np.zeros([nsample,len(legal_kvecs)])
        for isample in range(istart,istart+nsample):
            entry = cdf[isample]
            axes = entry['axes']
            pos  = entry['pos']
            
            # accumulate gofr
            myx,myy,junk = gofr_snapshot(axes,pos,rmax=rws
                ,gofr_norm=gofr_norm)
            all_gofr[isample-istart,:] = myy

            # accumulate sofk
            junk,sk_arr = sofk_snapshot(axes,pos
                ,legal_kvecs=legal_kvecs)
            all_sofk[isample-istart,:] = sk_arr
        # end for isample

        # save results
        gofr = all_gofr.mean(axis=0)
        np.savetxt('gofr.dat',gofr)

        sofk = all_sofk.mean(axis=0)
        np.savetxt('sofk.dat',sofk)
    else:
        gofr = np.loadtxt('gofr.dat')
        sofk = np.loadtxt('sofk.dat')
    # end if

    if args.plot:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2)

        # plot g(r)
        ax[0].set_xlabel('r')
        ax[0].set_ylabel('g(r)')
        ax[0].plot(myx,gofr)

        # plot S(k)
        ax[1].set_xlabel('k')
        ax[1].set_ylabel('S(k)')
        ax[1].get_yaxis().tick_right()
        ax[1].get_yaxis().set_label_position('right')
        plot_sofk(ax[1],legal_kvecs,sofk)
        if args.save:
            fig.savefig(prefix + '.png',dpi=200)
        # end if
        plt.show()
    # end if

# end __main__
