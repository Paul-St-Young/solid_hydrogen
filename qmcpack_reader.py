#!/usr/bin/env python
import os
import pandas as pd

def epl_val_err(epl_out):
  """ convert epl_out to a pandas DataFrame. 
  epl_out is expected to be an output of energy.pl from QMCPACK
  It simply has to have the format {name:22c}={val:17.3f} +/- {err:26.4f}.
  rows with forces will be recognized with 'force_prefix'
  Args:
    epl_out (str): energy.pl output filename
  Returns:
    pd.DataFrame: df contains columns ['name','val','err']
  """
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

    failed = False

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
            opt_file = prefix + ".s%s.opt.xml" % str(all_iopt).zfill(3)
            opt_xml  = os.path.join(subdir,opt_file)
            if not os.path.isfile(opt_xml):
                if warn:
                    print "skipping %d in %s" % (all_iopt,subdir)
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
    if len(data) == 0:
        failed = True
    else:
        df = pd.DataFrame( data )
        df.to_json(target_json)
    # end if

    return failed
# end def extract_jastrows

def extract_best_jastrow_set(opt_input,opt_json='opt_scalar.json',nequil='auto',force_refresh=False):
    import nexus_addon as na
    subdir = os.path.dirname(opt_input)

    # locally create jas.json 
    extract_jastrows(opt_input,force_refresh=force_refresh)

    # locally create opt_scalar.json
    scalar_json = os.path.join(subdir,opt_json)
    if (not os.path.isfile(scalar_json)) or force_refresh:
        # initialize analyzer
        from qmca import QBase
        options = {"equilibration":nequil}
        QBase.options.transfer_from(options)

        entry = na.scalars_from_input(opt_input)
        pd.DataFrame(entry).to_json(scalar_json)
    # end if

    # get best jastrow set
    best_jas = collect_best_jastrow_set(subdir)
    return best_jas
# end def extract_best_jastrow_set

def collect_best_jastrow_set(subdir,jas_json='jas.json',opt_json='opt_scalar.json'
        ,rval_weight=0.75,rerr_weight=0.25):
    """ find best set of jastrows in 'subdir', assume files:
     1. jas.json: a database of QMCPACK bspline jastrows with 'iopt' column 
     2. opt_scalar.json: a database of QMCPACK scalars including 'LocalEnergy_mean', 'LocalEnergy_error', 'Variance_mean', and 'Variance_error' """
    from dmc_database_analyzer import div_columns

    jfile = os.path.join(subdir,jas_json)
    if not os.path.isfile(jfile):
        raise RuntimeError('%s not found in %s' % (jfile,subdir))
    # end if
    
    ofile = os.path.join(subdir,opt_json)
    if not os.path.isfile(ofile):
        raise RuntimeError('%s not found in %s' % (ofile,subdir))
    # end if
    
    jdf = pd.read_json(jfile) # jastrows
    sdf = pd.read_json(ofile) # scalars
    
    # same units for stddev and LocalEnergy
    sdf['stddev_mean'] = sdf['Variance_mean'].apply(np.sqrt)
    sdf['stddev_error'] = sdf['Variance_error'].apply(np.sqrt)
    
    # make ratios
    ratio_mean, ratio_error = div_columns(['stddev','LocalEnergy'],sdf)
    
    # take both value and error into account
    rv_cost = ratio_mean/ratio_mean.mean()
    re_cost = ratio_error/ratio_error.mean()
    
    # make a cost function
    cost = rv_cost*rval_weight + re_cost*rerr_weight
    
    # minimize cost function
    idx  = np.argmin(cost)
    
    # grab winner jastrows
    best_jas = jdf[jdf['iopt']==idx].copy()
    return best_jas
# end def collect_best_jastrow_set
