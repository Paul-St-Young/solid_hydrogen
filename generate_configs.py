#!/usr/bin/env python
import os
import pandas as pd
import qe_reader as qer
from pymbar.timeseries import detectEquilibration

if __name__ == '__main__':

    md_out = 'scf.out'
    target_json = 'i4-pbe-4800-48-solid_configs.json'
    if os.path.isfile(target_json):
        raise IOError('%s exists'%target_json)
    # end if

    edf = pd.DataFrame( qer.md_traces(md_out) )
    nequil = detectEquilibration(edf['total energy'].values[:-1])[0]
    all_axes,all_pos = qer.available_structures(md_out)

    data = []
    for istep in range(len(all_axes)):
	if istep > nequil:
	    axes = all_axes[istep]
	    pos  = all_pos[istep]
	    entry = {'istep':istep,'axes':axes,'pos':pos}
	    data.append(entry)
	# end if
    # end for istep

    cdf = pd.DataFrame(data)
    cdf.T.to_json(target_json)
    
# end __main__
