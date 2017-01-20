#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import qe_reader as qer

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='generate configurations from a QE DFT-MD output')
    parser.add_argument('md_out',type=str,help='QE DFT-MD output')
    parser.add_argument('-pre','--prefix',type=str,default='default',help='configuration filename prefix {prefix}_configs.json')
    parser.add_argument('-e','--nequil',type=int,default=0)
    parser.add_argument('-n','--nsample',type=int,default=50)
    args = parser.parse_args()

    target_json = '%s_configs.json' % args.prefix
    if os.path.isfile(target_json):
        raise RuntimeError('%s exists'%target_json)
    # end if
    print "generating %s from %s" % (target_json,args.md_out)

    all_axes,all_pos = qer.available_structures(args.md_out)
    nstruct = len(all_pos)-1

    idx = [int(round(x)) for x in np.linspace(args.nequil,nstruct,args.nsample)]
    print "using md steps " + " ".join( map(str,idx) )

    data = []
    for istep in idx:
        axes = all_axes[istep]
        pos  = all_pos[istep]
        entry = {'istep':istep,'axes':axes,'pos':pos}
        data.append(entry)
    # end for istep

    cdf = pd.DataFrame(data)
    cdf.T.to_json(target_json)
    
# end __main__
