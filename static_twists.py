#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import subprocess as sp
import nexus_addon as na

def locate_bundle_input(path):
    # locate bunled QMCPACK input
    fidx = 0
    out_tokens = sp.check_output('ls %s/*.in' %path
            ,shell=True).split('\n')[:-1]

    good_in = 0
    if len(out_tokens) > 1:
        for i in range(len(out_tokens)):
            fname = out_tokens[i]
            if not fname.endswith('.qsub.in'):
                good_in += 1
                fidx = i
            # end if
        # end for
    # end if

    if (len(out_tokens) != 1) and (good_in != 1) :
        raise NotImplementedError(
            '%d inputs found in %s, can only handle 1' % (len(out_tokens),path) 
        )
    # end if
    return out_tokens[fidx]
# end def

def collect_raw_data(paths,skip_failed=False,db_name='twists.json',verbose=True):
    failed = False

    # initialize analyzer
    from qmca import QBase
    options = {"equilibration":"auto"}
    QBase.options.transfer_from(options)

    for path in paths:
        if verbose:
            print "getting raw data from %s" % path
        # end if

        target_json = os.path.join(path,db_name)
        if os.path.isfile(target_json):
            continue
        # end if
        
        bundled_input = locate_bundle_input(path)
        igroup_dict = {}
        with open(bundled_input,'r') as f:
            igroup = 0
            for line in f:
                infile = line.strip('\n')
                igroup_dict[infile] = igroup
                igroup += 1
            # end for line
        # end with open

        # make a database of all the scalar files
        data = []
        for qmc_input in igroup_dict.keys():
            entry = na.scalars_from_input(os.path.join(path,qmc_input)
                    ,skip_failed=skip_failed,igroup=igroup_dict[qmc_input]) 
            data.append(entry)
        # end for
        df = pd.DataFrame([entry for sublist in data for entry in sublist])
        
        # save raw data in local directory
        if len(df)!=0:
            pd.concat([df,df['settings'].apply(pd.Series)],axis=1).to_json(target_json)
        else:
            failed=True
        # end if

    # end for path
    return failed
# end def collect_raw_data

def average_twists(paths,src_db_name='twists.json',tar_db_name='scalars.json',manual_twists=None,verbose=True):
    failed = False

    for path in paths:
        if verbose:
            print "averaging twists in %s" % path
        # end if

        target_json = os.path.join(path,tar_db_name)
        if os.path.isfile(target_json):
            continue
        # end if

        source_json = os.path.join(path,src_db_name)
        # load local data
        if not os.path.exists(source_json):
            raise IOError('cannot locate %s, is it collected?' % source_json)
        # end if
        df_all = pd.read_json(source_json)

        df = df_all # may select twists
        if manual_twists is not None:
            sel = df_all['twistnum'].apply(lambda x:x in manual_twists)
            df  = df_all[sel]
            if len(manual_twists) != len(df):
                raise NotImplementedError('missing twists')
            # end if
        # end if
        
        # exclude columns that don't need to be averaged, add more as needed
        special_colmns = ['iqmc','method','path','settings','vol_unit','volume']
        columns_to_average = df.drop(special_colmns,axis=1).columns

        mean_names = []
        error_names= []
        for col_name in columns_to_average:
            if col_name.endswith('_mean'):
                mean_names.append(col_name)
            elif col_name.endswith('_error'):
                error_names.append(col_name)
            # end if
        # end for col_name

        col_names   = []
        for iname in range(len(mean_names)):
            mname = mean_names[iname].replace('_mean','')
            ename = error_names[iname].replace('_error','')
            assert mname == ename
            col_names.append(mname)
        # end for i
        
        # perform twist averaging
        new_means  = df.groupby('iqmc')[mean_names].apply(np.mean)
        ntwists = len(df[df['iqmc']==0]) # better way to determine ntwists?
        new_errors = df.groupby('iqmc')[error_names].apply(
            lambda x:np.sqrt((x**2.).sum())/ntwists)
        
        # make averaged database
        dfev = pd.merge(new_means.reset_index(),new_errors.reset_index())
        extras = df[special_colmns].groupby('iqmc').apply(lambda x:x.iloc[0])
        newdf = pd.merge( extras.drop('iqmc',axis=1).reset_index(), dfev)
        
        newdf.to_json(target_json)
        
    # end for

    return failed
# end def average_twists


if __name__ == '__main__':

    import argparse

    # parse command line input for trace file name
    parser = argparse.ArgumentParser(description='collect DMC data from a directory of QMCPACK runs')
    parser.add_argument('src_dir', type=str, help='directory containing QMCPACK runs')
    parser.add_argument('final_target_json', type=str, help='json file to save collected data')
    parser.add_argument('-rid','--rundir_id', type=str,default='dmc_444',help='run directory identifier, default dmc_444')
    parser.add_argument('-r','--only_real', action='store_true', help='only use real twists out of 64 twists')
    parser.add_argument('-v','--verbose', action='store_true', help='report progress')
    args = parser.parse_args()

    # parsed inputs
    #  specify source data directory and target database file
    src_dir = args.src_dir
    final_target_json = args.final_target_json
    only_real = args.only_real
    rundir_id = args.rundir_id
    verbose   = args.verbose

    # hard-coded inputs
    sfile_name = 'scalars.json'
    if only_real:
        sfile_name = 'new_real_' + sfile_name
    # end if
    if only_real:
        final_target_json = 'real_' + final_target_json
    # end if

    # get twist run locations
    proc = sp.Popen(['find',src_dir,'-name',rundir_id]
        ,stdout=sp.PIPE,stderr=sp.PIPE)
    out,err = proc.communicate()
    paths = out.split('\n')[:-1]

    # collect raw data in local directories
    failed = collect_raw_data(paths)
    if failed:
        raise NotImplementedError('raw data collection failed')
    # end if

    # store twist-averaged data in local directories
    if only_real:
        failed = average_twists(paths,tar_db_name=sfile_name,manual_twists=[0,2,8,10,32,34,40,42])
    else: # average over all twists
        failed = average_twists(paths,tar_db_name=sfile_name)
    # end if
    if failed:
        raise NotImplementedError('twist average failed')
    # end if

    # analyze data
    import dmc_databse_analyzer as dda
    data = []
    for path in paths:
        print "analyzing %s" % path
        jfile = os.path.join(path,sfile_name)
        if not os.path.exists(jfile):
            raise IOError('failed to find %s' % jfile)
        # end if
        local_scalars = pd.read_json(jfile)
        extrap_scalars= dda.process_dmc_data_frame(local_scalars)
        data.append(extrap_scalars)
    # end for path
    df = pd.concat(data).reset_index().drop('index',axis=1)
    df.to_json(final_target_json)

# end __main__
