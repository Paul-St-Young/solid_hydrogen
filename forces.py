#!/usr/bin/env python
def plot_forces(ax,epl_df,force_prefix='force_'):
    forces = epl_df[ epl_df['name'].apply(lambda x:x.startswith(force_prefix)) ]
        
    import matplotlib.pyplot as plt
    ax.errorbar(range(len(forces)),forces['val'],forces['err'])
    return
# end def

if __name__ == '__main__':

    import argparse
    # parse command line input for trace file name
    parser = argparse.ArgumentParser(description='analyze an output of energy.pl')
    parser.add_argument('epl_out', type=str, help='output of energy.pl')
    parser.add_argument('-fpre','--force_prefix', type=str, default='force_', help='force row label prefix')
    args = parser.parse_args()
    epl_out = args.epl_out
    fpre    = args.force_prefix

    import qmcpack_reader as qpr
    epl_df  = qpr.epl_val_err(epl_out)
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1)
    plot_forces(ax,epl_df,force_prefix=fpre)
    plt.show()

# end __main__
