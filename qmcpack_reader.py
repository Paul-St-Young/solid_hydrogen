#!/usr/bin/env python
from mmap import mmap
def get_value(qmcout,keyword='Madelung',delimiter='=',val_type=float,val_loc=-1):
    """ find the value of the line 'keyword = value' in qmcout """
    
    # open file
    fhandle = open(qmcout,'r+')
    mm = mmap(fhandle.fileno(),0)
    
    # find first line with the keyword
    idx = mm.find(keyword)
    if idx == -1:
        raise IOError(keyword + ' not found')
    # end 
    
    # go to line and read
    mm.seek(idx)
    line = mm.readline().strip('\n')
    val_text = line.split(delimiter)[val_loc]
    
    # try to obtain the value of keyword
    val = val_type(val_text) # rely on internal exception
    
    # close file
    fhandle.close()
    
    return val
# end def get_madelung

import os
import h5py
def retrieve_psig(h5_file,only_occupied=False,occupations=None):
    """ return a list dictionaries of DFT orbital coefficients in PW basis by reading an hdf5 file written by pw2qmcpack. If only_occupied=True and a database of occupied orbitals are given, then only read orbitals that are occupied. """
    if only_occupied and (occupations is None):
        raise NotImplementedError("no occupation database is given")
    # end if

    ha = 27.21138602 # ev from 2014 CODATA

    orbitals = []

    h5handle = h5py.File(h5_file)
    electron = h5handle['electrons']

    kpt_labels = []
    for key in electron.keys():
        if key.startswith('kpoint'):
            kpt_labels.append(key)
        # end if
    # end for key

    nk = electron['number_of_kpoints'].value
    assert nk==len(kpt_labels)

    for label in kpt_labels:

        # get kpoint index
        kpt_idx = int( label.split('_')[-1] )

        # get plane wave wave numbers
        if kpt_idx == 0:
            mypath = os.path.join(label,'gvectors')
            gvecs = electron[mypath].value
        # end if

        # verify eigenstates at this kpoint
        kpt_ptr = electron[os.path.join(label,'spin_0')]
        nbnd = kpt_ptr['number_of_states'].value

        evals = kpt_ptr['eigenvalues'].value

        # compare to nscf output (eigenvalues and occupation)
        if occupations is not None:
            mydf = occupations[occupations['ik']==kpt_idx]
            myval= mydf['eval'].values[0]
            myocc= mydf['occ'].values[0]
            assert nbnd == len(myval), "expect %d bands, nscf has %d bands" % (nbnd,len(myval))
            assert np.allclose(evals*ha,myval,atol=1e-4), str(evals*ha-myval)
        # end if

        for iband in range(nbnd):
            if only_occupied and (np.isclose(myocc[iband],0.0)):
                continue
            # end if
            psig  = kpt_ptr['state_%d/psi_g'%iband].value
            entry = {'ik':kpt_idx,'iband':iband,'eval':evals[iband],'psig':psig}
            orbitals.append(entry)
        # end for iband

    # end for label

    h5handle.close()
    return orbitals
# end def retrieve_psig

def retrieve_system(h5_file):
    h5handle = h5py.File(h5_file)
    lattice  = h5handle['supercell/primitive_vectors'].value
    #elem     = h5handle['atoms/species_ids'].value # why so complicated?
    pos      = h5handle['atoms/positions'].value
    gvecs    = h5handle['electrons/kpoint_0/gvectors'].value
    h5handle.close()
    return {'axes':lattice,'pos':pos,'gvecs':gvecs}
# end def

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
# end def epl_val_err

def sk_from_fs_out(fs_out):
    """ extract fluctuating S(k) from qmcfinitesize output
      returns: kmag,sk,vk,spk,spsk
       kmag: magnitude of kvectors, sk: raw fluc. S(k), vk: long-range potential after break-up
       spk: kmags for splined S(k), spsk: splined S(k) """

    import reader
    bint = reader.BlockInterpreter()
    sfile = reader.SearchableFile(fs_out)

    # read raw data
    block_text = sfile.block_text('#SK_RAW_START#','#SK_RAW_STOP#')
    kmag,sk,vk = bint.matrix(block_text[block_text.find('\n')+1:]).T

    # read splined S(k)
    block_text = sfile.block_text('#SK_SPLINE_START#','#SK_SPLINE_STOP#')
    spk,spsk = bint.matrix(block_text[block_text.find('\n')+1:]).T

    return kmag,sk,vk,spk,spsk
# end def

import numpy as np
from copy import deepcopy
def read_jastrows(jas_node):
    """ 'jas_node' should be an xml node containing bspline jastrows
     put coefficients and attributes into a list of dictionaries """
    
    if (jas_node.attrib["type"] != "Two-Body"): # works for one-body! miracle!
        pass#raise TypeError("input is not a two-body Jastrow xml node")
    elif (jas_node.attrib["function"].lower() != "bspline"):
        raise NotImplementedError("can only handle bspline Jastrows for now")
    # end if
    
    data = []
    for corr in jas_node.xpath('./correlation'):
        coeff = corr.xpath('./coefficients')[0]
        entry = deepcopy( corr.attrib )
        entry.update(coeff.attrib)
        entry['coeff'] = np.array(coeff.text.split(),dtype=float)
        entry['type']  = jas_node.attrib['type']
        data.append(entry)
    # end for corr
    
    return data
# end def read_jastrows

from lxml import etree
def extract_jastrows(qmcpack_input,json_name='jas.json',warn=True,force_refresh=False):
    """ given a QMCPACK input that contains linear optimization, extract all printed Jastrows and store in a local database
     1. parse 'qmcpack_input' for the qmc[@metho="linear"] section
     2. for each *.opt.xml, parse if it exists
     3. parse each opt.xml and make local database """

    subdir = os.path.dirname(qmcpack_input)
    target_json = os.path.join(subdir,json_name)
    if os.path.isfile(target_json) and (not force_refresh):
        if warn:
            print "skipping %s" % subdir
        # end if
        return 0 # skip ths file
    # end if

    parser = etree.XMLParser(remove_blank_text=True)

    # get prefix
    xml = etree.parse(qmcpack_input,parser)
    proj = xml.xpath("//project")[0]
    prefix = proj.attrib['id']

    # determine number of optimization loops
    all_qmc_sections = xml.xpath('.//qmc[@method="linear"]')
    all_iopt = 0 # track multiple 'linear' sections
    data = []
    for qmc_section in all_qmc_sections:
        # for each linear optimization:

        # find the number of loops
        nopt = 1
        loop = qmc_section.getparent()
        if loop.tag == 'loop':
            nopt = int(loop.attrib['max'])
        # end if

        # collect all jastrow coefficients
        for iopt in range(nopt):
            # get optimization file
            opt_file = prefix + ".s%s.opt.xml" % str(iopt).zfill(3)
            opt_xml  = os.path.join(subdir + opt_file)
            if not os.path.isfile(opt_xml):
                if warn:
                    print "skipping %d in %s" % (iopt,subdir)
                # end if
                continue
            # end if

            # parse optimization file
            opt = etree.parse(opt_xml,parser)
            jnodes = opt.xpath('//jastrow')
            for jas_node in jnodes:
                entries = read_jastrows(jas_node)
                for entry in entries:
                    entry['iopt'] = all_iopt
                # end for entry
                data.append(entry)
            # end for
            all_iopt += 1
        # end for iopt
    # end for qmc_section
    df = pd.DataFrame( data )

    df.to_json(target_json)

    return 0
# end def extract_jastrows
