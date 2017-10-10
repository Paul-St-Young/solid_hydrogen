import numpy as np
from mmap import mmap

def value_by_label_sep_pos(mm,label,sep=b'=',pos=-1,dtype=float,from_start=False):
    """ find the value of the line 'keyword = value' in memory map 'mm' by default """
    
    if from_start:
        mm.seek(0)
    # end if
    
    # accomodate python3
    if type(label) is str:
        label = label.encode()
    elif type(sep) is str:
        sep = label.encode()
    # end if
    
    idx = mm.find(label)
    if idx == -1:
        raise RuntimeError(label+' not found')
    # end if
    
    mm.seek(idx)
    line = mm.readline()
    tokens = line.split(sep)
    
    val_text = tokens[pos]
    val = dtype(val_text)
    return val

# end def value_by_label_sep_pos

def read_first_energy(scf_out):
    with open(scf_out,'r+') as f:
        mm = mmap(f.fileno(),0)
    # end with
    idx = mm.find('!')
    mm.seek(idx)
    eline = mm.readline()
    energy = float( eline.split()[-2] )
    return energy
# end def

def read_forces(scf_out,ndim=3,which='total'):
    """ read the forces in a pwscf output, assume only one force block
     'which' decides which block of forces to read, choices are:
         ['total', 'non-local', 'local', 'ionic', 'core', 'Hubbard', 'scf']
     !!!! assuming QE uses Ry, will convert to Ha """
    Ry = 0.5 # Ha
    begin_tag_dict = {
        'total':'Forces acting on atoms',
        'non-local':'The non-local contrib.  to forces',
        'ionic':'The ionic contribution  to forces',
        'local':'The local contribution  to forces',
        'core':'The core correction contribution to forces',
        'Hubbard':'The Hubbard contrib.    to forces',
        'scf':'The SCF correction term to forces'
    }
    end_tag_dict = {
        'total':'The non-local contrib.  to forces',
        'non-local':'The ionic contribution  to forces',
        'ionic':'The local contribution  to forces',
        'local':'The core correction contribution to forces',
        'core':'The Hubbard contrib.    to forces',
        'Hubbard':'The SCF correction term to forces',
        'scf':'Total force ='
    }

    fhandle = open(scf_out,'r+')
    mm = mmap(fhandle.fileno(),0)
    
    natom = value_by_label_sep_pos(mm,'number of atoms',dtype=int)

    # locate force block
    begin_tag = begin_tag_dict[which]
    end_tag   = end_tag_dict[which]

    begin_idx = mm.find(begin_tag.encode())
    end_idx   = mm.find(end_tag.encode())
    if begin_idx == -1:
        raise RuntimeError('cannot locate %s'%begin_tag)
    elif end_idx == -1:
        # maybe verbosity='low'
        end_idx = mm.find('Total force =')
        if end_idx == -1:
            raise RuntimeError('cannot locate %s'%end_tag)
        # end if
    # end if
    force_block = mm[begin_idx:end_idx]
    
    # parse force block for forces
    forces = np.zeros([natom,ndim])
    iatom = 0
    for line in force_block.split(b'\n'):
        if line.strip().startswith(b'atom'):
            tokens = line.split()
            if len(tokens)==9: # found an atom
                myforce = np.array(tokens[-3:],dtype=float)
                forces[iatom,:] = tokens[-3:]
                iatom += 1
            # end if
        # end if
    # end for
    if iatom != natom:
        raise RuntimeError('found %d forces for %d atoms'%(iatom,natom))
    # end if
    
    fhandle.close()
    
    return forces*Ry
# end def

def retrieve_occupations(nscf_outfile, max_nbnd_lines=10):
    """ read the eigenvalues and occupations of DFT orbitals at every available kpoint in an non-scf output produced by pwscf """ 

    fhandle = open(nscf_outfile,'r+')
    mm = mmap(fhandle.fileno(),0)

    # read number of k points
    nk_prefix = "number of k points="
    idx = mm.find(nk_prefix)
    mm.seek(idx)

    nk_line = mm.readline()
    nk = int( nk_line.strip(nk_prefix).split()[0] )

    # skip to the end of band structure calculation
    idx = mm.find('End of band structure calculation')
    mm.seek(idx)

    # read the eigenvalues and occupations at each kpoint
    kpt_prefix = "k ="
    data = []
    for ik in range(nk):
        idx = mm.find(kpt_prefix)
        mm.seek(idx)
        kpt_line = mm.readline()
        kpt = map(float,kpt_line.strip(kpt_prefix).split()[:3])

        mm.readline() # skip empty line
        eval_arr = np.array([])
        for iline in range(max_nbnd_lines):
            tokens = mm.readline().split()
            if len(tokens)==0:
                break
            # end if
            eval_arr = np.append(eval_arr, map(float,tokens))
        # end for iline
        
        idx = mm.find('occupation numbers')
        mm.seek(idx)
        mm.readline() # skip current line
        occ_arr = np.array([])
        for iline in range(4):
            tokens = mm.readline().split()
            if len(tokens)==0:
                break
            # end if
            occ_arr = np.append(occ_arr, map(float,tokens))
        # end for iline

        entry = {'ik':ik,'kpt':list(kpt),'eval':list(eval_arr),'occ':list(occ_arr)}
        data.append(entry)
    # end for
    mm.close()
    fhandle.close()
    return data
# end def

import subprocess as sp
def find_pwscf_io(path,infile_subfix='-scf.in',outfile_subfix='.out',use_last=False):
    # assuming there is only 1 pair of pw.x input and output in path
    #  return the names of the input and output files
    out = sp.check_output(['ls',path])
    
    infile  = ''
    outfile = ''
    
    found_in  = False
    found_out = False
    for fname in out.split('\n')[:-1]:
        if fname.endswith(infile_subfix):
            if found_in and not use_last:
                raise NotImplementedError('multiple inputs found in %s'%path)
            # end if
            infile   = fname
            found_in = True
        elif fname.endswith(outfile_subfix):
            if found_out and not use_last:
                raise NotImplementedError('multiple outputs found in %s'%path)
            # end if
            outfile   = fname
            found_out = True
        # end if
    # end for fname
    
    if not found_in:
        raise IOError('infile not found in %s'%path)
    elif not found_out:
        raise IOError('outfile not found in %s'%path)
    # end if
    
    return infile,outfile
# end def find_pwscf_io

def all_lines_with_tag(mm,tag,nline_max):
    """ return a list of memory indices pointing to the start of tag """
    mm.seek(0) # rewind file
    all_idx = []
    for iline in range(nline_max):
        idx = mm.find(tag)
        if idx == -1:
            break
        # end if
        mm.seek(idx)
        all_idx.append(idx)
        mm.readline()
    # end for iline
    
    # guard
    if iline >= nline_max-1:
        raise NotImplementedError('may need to increase nline_max')
    # end if
    return all_idx
# end def all_lines_with_tag

import struct
def available_structures(pw_out,nstruct_max=10000,natom_max=1000,ndim=3
        ,variable_cell=False):
    """ find all available structures in a pwscf output """
    fhandle = open(pw_out,'r+')
    mm = mmap(fhandle.fileno(),0)

    idx = mm.find('lattice parameter')
    mm.seek(idx)
    lat_line = mm.readline()
    alat = float( lat_line.split()[-2] )
    
    # locate all axes
    axes_tag = 'CELL_PARAMETERS ('.encode()
    axes_starts = all_lines_with_tag(mm,axes_tag,nstruct_max)
    naxes = len(axes_starts)
    if (naxes != 0) and (not variable_cell):
        raise NotImplementedError('CELL_PARAMETERS found, are you sure this is not a variable cell run?')
    # end if

    # crystal coords
    crystal_pos = False
    
    # locate all atomic cd positions
    pos_tag    = 'ATOMIC_POSITIONS'.encode()
    pos_starts = all_lines_with_tag(mm,pos_tag,nstruct_max)
    npos = len(pos_starts)
    
    if variable_cell and (npos != naxes):
        raise NotImplementedError('expect same number of cells as atomic positions in a variable cell calculation. got (naxes,npos)=(%d,%d)'%(naxes,npos))
    # end if
    
    # count number of atoms
    mm.seek(pos_starts[0])
    mm.readline() # skip tag line
    natom = 0
    for iatom in range(natom_max):
        line = mm.readline()
        tokens = line.split()
        if len(tokens) != 4:
            break
        # end if
        natom += 1
    # end for iatom
    
    # read initial crystal axes
    axes = np.zeros([ndim,ndim])
    if not variable_cell: 
        idx = all_lines_with_tag(mm,'crystal axes'.encode(),nstruct_max)[0]
        mm.seek(idx)
        tag_line = mm.readline()
        unit_text= tag_line.split()[-1].strip('()')
        for idim in range(ndim):
            line = mm.readline()
            axes[idim,:] = line.split()[3:3+ndim]
            if 'alat' in unit_text:
                axes[idim,:] *= alat
            else:
                raise NotImplementedError('crystal axes: what unit is %s?'%unit_text)
            # end if
        # end for 
    # end if
    
    bohr = 0.52917721067 # angstrom (CODATA 2014)
    nstructs = max(naxes,npos)
    all_axes = np.zeros([nstructs,ndim,ndim])
    all_pos  = np.zeros([nstructs,natom,ndim])
    for istruct in range(nstructs):
        
        if variable_cell: # read cell parameters
            cell_idx = axes_starts[istruct]
            mm.seek(cell_idx)
            tag_line = mm.readline() # get unit from tag line
            axes_unit = tag_line.split('(')[-1].replace(')','')
            if not axes_unit.startswith('alat'):
              raise RuntimeError('unknown CELL_PARAMETERS unit %s'%axes_unit)
            # end if
            alat = float(axes_unit.split('=')[-1])
            
            axes_text = ''
            for idim in range(ndim):
              axes[idim,:] = mm.readline().split()
            # end for idim
            axes *= alat
        # end if variable_cell
        
        all_axes[istruct,:,:] = axes
        
        pos_idx = pos_starts[istruct]
        mm.seek(pos_idx)
        tag_line = mm.readline()
        unit_text= tag_line.split()[-1]
        au2unit = 1. # !!!! assume bohr
        if 'angstrom' in unit_text:
            au2unit = 1./bohr
        elif 'bohr' in unit_text:
            au2unit = 1.
        elif 'alat' in unit_text:
            au2unit = alat
        elif 'crystal' in unit_text:
            crystal_pos = True
        else:
            raise NotImplementedError('what unit is this? %s' % unit_text)
        # end if
        
        for iatom in range(natom):
            line = mm.readline()
            name = line.split()[0]
            pos_text = line.strip(name)
            try:
              name,xpos,ypos,zpos = struct.unpack('4sx14sx14sx13s',pos_text)
              pos = np.array([xpos,ypos,zpos],dtype=float) * au2unit
              if crystal_pos:
                pos = np.dot(pos,axes)
              all_pos[istruct,iatom,:] = pos
            except:
              print 'failed to read (istruct,iatom)=(%d,%d)' % (istruct,iatom)
            # end try
        # end for iatom
        
    # end for istruct
        
    fhandle.close()
    return all_axes,all_pos
# end def available_structures

def md_traces(md_out,nstep=2000):
    """ extract scalar traces from pwscf md output md_out 
     look for tags defined in line_tag_map """
    fhandle = open(md_out,'r+')
    mm = mmap(fhandle.fileno(),0)

    line_tag_map = { # unique identifier of the line that contains each key
        'fermi energy':'the Fermi energy is',
        'total energy':'!',
        'kinetic energy':'kinetic energy',
        'temperature':'temperature',
        'econst':'Ekin + Etot'
    }
    val_idx_map  = {} # assume -2
    val_type_map = {} # assume float

    mm.seek(0)
    data = []
    for istep in range(nstep):
        if mm.tell() >= mm.size():
            break
        # end if
        found_stuff = False
        
        entry = {'istep':istep}
        for label in line_tag_map.keys():
            
            # locate line with value for label
            idx = mm.find(line_tag_map[label].encode())
            if idx == -1:
                continue
            # end if
            found_stuff = True
            mm.seek(idx)
            line = mm.readline()
            
            # locate value in line
            rval_idx = -2 # assume patten "label = value unit"
            if label in val_idx_map.keys():
                rval_idx = val_idx_map[label]
            # end if
            rval = line.split()[rval_idx]
            
            # convert value
            val_type = float
            if label in val_type_map.keys():
                val_type = val_type_map[key]
            # end if
            value = val_type(rval)
            
            entry[label] = value # !!!! assume float value
        # end for
        
        if found_stuff:
            data.append(entry)
        else:
            break
        # end if
    # end for istep
    if istep >= nstep-1:
        print "WARNING: %d structures found, nstep may need to be increased" %istep
    # end if
    fhandle.close()
    return data
# end def md_traces

def pos_in_box(pos,axes):
    """ return atomic positions 'pos' in simulation box specified by 'axes' """
    # convert positions to fractional coordinates
    inv_axes = np.linalg.inv(axes)
    upos = np.dot(pos,inv_axes)
    upos -= np.floor(upos)

    # convert back
    newpos = np.dot(upos,axes)
    return newpos
# end def

def input_structure(scf_in,put_in_box=True):
    ndim = 3 # assume 3 dimensions
    with open(scf_in,'r+') as f:
        mm = mmap(f.fileno(),0)
    # end with

    ntyp = value_by_label_sep_pos(mm,'ntyp',dtype=int)
    if ntyp != 1:
        raise NotImplementedError('only support 1 type of atom for now')
    # end if

    # read lattice
    mm.seek(0)
    idx = mm.find('ibrav')
    mm.seek(idx)
    ibrav_line = mm.readline()
    ibrav = int(ibrav_line.split('=')[-1])
    if ibrav != 0:
        raise NotImplementedError('only ibrav = 0 is supported')
    # end if
    idx = mm.find('CELL_PARAMETERS')
    mm.seek(idx)
    header = mm.readline()
    unit = header.split()[-1]
    axes = np.zeros([ndim,ndim])
    for idim in range(ndim):
        line = mm.readline()
        axes[idim,:] = map(float,line.split())
    # end for
    cell = {'unit':unit,'axes':axes}

    # read atomic positions
    mm.seek(0) # rewind
    idx = mm.find('nat')
    mm.seek(idx)
    nat_line = mm.readline()
    nat = int(nat_line.split('=')[-1])

    idx = mm.find('ATOMIC_POSITIONS')
    mm.seek(idx)
    header = mm.readline()
    unit = header.split()[-1]

    pos = np.zeros([nat,ndim])
    for iat in range(nat):
        line = mm.readline()
        pos[iat,:] = map(float,line.split()[-3:])
    # end for iat

    try:
        line = mm.readline()
        float(line.split()[-3:])
        raise RuntimeError('next lines looks like positions too!\n%s'%line)
    except:
        pass # expect to see an empty line
    # end try
    if put_in_box:
	atpos = {'pos_unit':unit,'pos':pos_in_box(np.array(pos),np.array(axes)).tolist()}
    else:
	atpos = {'pos_unit':unit,'pos':pos}
    # end if

    entry = {'infile':scf_in}
    entry.update(cell)
    entry.update(atpos)

    return entry
# end def input_structure

def read_stress(pw_out,stress_tag = 'total   stress  (Ry/bohr**3)',nstruct_max=4096):
  """ read all stress tensors from a quantum espresso output
  Args:
    pw_out (str): output filename
    stress_tag (str): tag at the beginning of each text block containing the stress tensor
    nstruct_max (int): maximum number of blocks to look for 
  Returns:
    (list[np.array],list[np.array]): (au_mat_list,kbar_mat_list), lists of stress tensors read 
  """
  with open(pw_out,'r+') as f:
    mm = mmap(f.fileno(),0)
  # end with
  au_mat_list   = []
  kbar_mat_list = []
  stress_starts = all_lines_with_tag(mm,stress_tag.encode(),nstruct_max)
  for idx in stress_starts:
    mm.seek(idx)
    header = mm.readline()
    tokens = header.split()
    # make sure we are about to read the correct block of text
    assert tokens[2].strip('()') == 'Ry/bohr**3'
    assert tokens[3].strip('()') == 'kbar'
    idx = header.find('P=')
    press = float(header[idx:].strip('P=')) # average pressure in kbar, used for checking only
    au_mat   = [] # pressure in Ry/bohr**3
    kbar_mat = [] # pressure in kbar
    for idim in range(3): # assume 3 dimensions
      line = mm.readline()
      tokens = line.split()
      assert len(tokens) == 6
      au_mat.append(tokens[:3])
      kbar_mat.append(tokens[3:])
    # end for idim
    kbar_mat = np.array(kbar_mat,dtype=float)
    assert np.isclose(np.diagonal(kbar_mat).mean(),press)
    kbar_mat_list.append(kbar_mat)
    au_mat_list.append(np.array(au_mat,dtype=float))
  # end for idx
  return au_mat_list,kbar_mat_list
# end def read_stress

def vc_relax_output(fout):
  all_axes,all_pos = available_structures(fout,variable_cell=True)
  amats,kmats      = read_stress(fout)
  data = []
  for i in range(len(all_axes)):
    axes = all_axes[i]
    pos  = all_pos[i]
    entry = {'istep':i,'axes':axes,'pos':pos,
      'amat':amats[i],'kmat':kmats[i]}
    data.append(entry)
  # end for i
  return data
# end def vc_relax_output

def relax_forces(fout,nstruct_max=4096):
  """ read all force blocks from a relax output (may also work on md output) 
  Args:
    fout (str): quantum espresso output, expected scf='relax'
    nstruct_max (int): maximum number of force blocks to be read
  Return:
    np.array: shape (nstep,natom,ndim), forces on atoms at each optimization step
  """

  nheader_before_forces = 2
  """ e.g.      Forces acting on atoms (Ry/au): # header line 1
                                                # header line 2
           atom    1 type  1   force =    -0.00000000   -0.00012993   -0.00008628
  """

  # get a memory map of the file
  fhandle = open(fout,'r+')
  mm = mmap(fhandle.fileno(),0)

  # decide on array size
  ndim = 3 # !!!! assume 3 dimensions 
  natom = value_by_label_sep_pos(mm,'number of atoms',dtype=int)
  idx_list = all_lines_with_tag(mm,'Forces acting on atoms (Ry/au)',nstruct_max)
  nstep = len(idx_list)

  forces = np.zeros([nstep,natom,ndim])

  # go through each force block
  for istep in range(nstep):
    mm.seek( idx_list[istep] )
    for iheader in range(nheader_before_forces):
      mm.readline() # skip headers
    for iatom in range(natom):
      line = mm.readline()
      tokens = line.split()
      if len(tokens) != 9:
        raise RuntimeError('invalid force block %s' % line)
      # end if
      forces[istep,iatom,:] = map(float,tokens[-3:])
    # end for iatom
  # end for istep

  # check that all forces have been read
  line = mm.readline()
  if line.startswith('atom'):
    raise RuntimeError('extra force line %s before memory idx %d'%(line,mm.tell()))
  # end if

  return forces
# end def relax_forces

def relax_output(fout):
  all_axes,all_pos = available_structures(fout,variable_cell=False)
  forces = relax_forces(fout)
  data = []
  assert len(forces) == len(all_axes)
  for i in range(len(all_axes)):
    axes = all_axes[i]
    pos  = all_pos[i]
    entry = {'istep':i,'axes':axes,'pos':pos,'forces':forces[i]}
    data.append(entry)
  # end for i
  return data
# end def relax_output
