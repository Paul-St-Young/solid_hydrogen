#!/usr/bin/env python

import pandas as pd
def epl_val_err(epl_out):
    """ epl_out is expected to be an output of energy.pl from QMCPACK
     It simply has to have the format {name:22c}={val:17.3f} +/- {err:26.4f}.
     rows with forces will be recognized with 'force_prefix' """
    tab = pd.read_table(epl_out,delimiter='=',names=['name','text'])
    tab = tab.dropna()
    
    def text2val(text):
        tokens = text.split('+/-')
        if len(tokens) != 2:
            raise NotImplementedError('unrecognized value '+text)
        # end if
        val,err = map(float,tokens)
        return pd.Series({'val':val,'err':err})
    # end def text2val
    
    df = pd.concat( [tab.drop('text',axis=1),tab['text'].apply(text2val)],axis=1 )
    return df

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

    epl_df  = epl_val_err(epl_out)
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1)
    plot_forces(ax,epl_df,force_prefix=fpre)
    plt.show()

# end __main__
