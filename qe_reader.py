import numpy as np
from mmap import mmap

from qharv.reel.ascii_out import read, name_sep_val, all_lines_with_tag

def read_first_energy(scf_out):
    with open(scf_out,'r+') as f:
        mm = mmap(f.fileno(),0)
    # end with
    idx = mm.find(b'!')
    mm.seek(idx)
    eline = mm.readline().decode()
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
    
    natom = name_sep_val(mm,'number of atoms',dtype=int)

    # locate force block
    begin_tag = begin_tag_dict[which]
    end_tag   = end_tag_dict[which]

    begin_idx = mm.find(begin_tag.encode())
    end_idx   = mm.find(end_tag.encode())
    if begin_idx == -1:
        raise RuntimeError('cannot locate %s'%begin_tag)
    elif end_idx == -1:
        # maybe verbosity='low'
        end_idx = mm.find(b'Total force =')
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
    from qharv.reel import ascii_out
    span = 7
    def scanf_7f(line, n):
      """ implement scanf("%7.*f") """
      numl = []
      for i in range(n):
        token = line[span*i:span*(i+1)]
        num = float(token)
        numl.append(num)
      return numl

    fhandle = open(nscf_outfile,'r+')
    mm = mmap(fhandle.fileno(),0)

    # read number of k points
    nk_prefix = b"number of k points="
    idx = mm.find(nk_prefix)
    mm.seek(idx)

    nk_line = mm.readline()
    nk = int( nk_line.strip(nk_prefix).split()[0] )

    # skip to the end of band structure calculation
    idx = mm.find(b'End of self-consistent calculation')
    idx = mm.find(b'End of band structure calculation')
    mm.seek(idx)

    # read the eigenvalues and occupations at each kpoint
    kpt_prefix = "k ="
    data = []
    for ik in range(nk):
        idx = mm.find(kpt_prefix.encode())
        mm.seek(idx)
        kpt_line = mm.readline()
        kxkykz = ascii_out.lr_mark(kpt_line, '=', '(')
        kpt = scanf_7f(kxkykz, 3)

        mm.readline() # skip empty line
        eval_arr = np.array([])
        for iline in range(max_nbnd_lines):
            tokens = mm.readline().split()
            if len(tokens)==0:
                break
            # end if
            eval_arr = np.append(eval_arr, map(float,tokens))
        # end for iline
        
        idx = mm.find(b'occupation numbers')
        mm.seek(idx)
        mm.readline() # skip current line
        occ_arr = np.array([])
        for iline in range(100):
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

import struct
def available_structures(pw_out,nstruct_max=10000,natom_max=1000,ndim=3
        ,variable_cell=False):
    """ find all available structures in a pwscf output """
    fhandle = open(pw_out,'r+')
    mm = mmap(fhandle.fileno(),0)

    idx = mm.find(b'lattice parameter')
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
              msg = 'failed to read (istruct, iatom)=(%d, %d)' %\
                (istruct,iatom)
              print(msg)
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
        msg = "WARNING: %d/%d structures found," % (istep, nstep)
        msg += " nstep may need to be increased"
        print(msg)
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

    from qharv.reel.ascii_out import name_sep_val
    ntyp = name_sep_val(mm, 'ntyp', dtype=int)
    if ntyp != 1:
        raise NotImplementedError('only support 1 type of atom for now')
    # end if

    # read lattice
    mm.seek(0)
    idx = mm.find(b'ibrav')
    mm.seek(idx)
    ibrav_line = mm.readline()
    ibrav = int(ibrav_line.split('=')[-1])
    if ibrav != 0:
        raise NotImplementedError('only ibrav = 0 is supported')
    # end if
    idx = mm.find(b'CELL_PARAMETERS')
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
    idx = mm.find(b'nat')
    mm.seek(idx)
    nat_line = mm.readline()
    nat = int(nat_line.split('=')[-1])

    idx = mm.find(b'ATOMIC_POSITIONS')
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
  stress_starts = all_lines_with_tag(mm,stress_tag,nstruct_max)
  for idx in stress_starts:
    mm.seek(idx)
    header = mm.readline().decode()
    tokens = header.split()
    # make sure we are about to read the correct block of text
    assert tokens[2].strip('()') == 'Ry/bohr**3'
    assert tokens[3].strip('()') == 'kbar'
    idx = header.find(b'P=')
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

def get_axsf_normal_mode(faxsf,imode):
  """ extract the first normal mode labeled by 'PRIMCOORD {imode:d}' 
  assume the following format:

  PRIMCOORD  1
    16   1
  H      0.00000   0.00000   1.50303  -0.00000   0.00000   0.02501
  H      0.63506   0.63506   0.00000   0.00000  -0.00000   0.02500
  ...

  Args:
    faxsf (str): name of axsf file
    imode (int): index of normal mode
  Return:
    tuple: (elem,data), elem is a list of atomic symbols,
     data is a np.array of floats (6 columns in above example).
  """
  from qharv.reel import ascii_out
  mm = ascii_out.read(faxsf)

  # search through all modes for requested imode
  all_idx = ascii_out.all_lines_with_tag(mm,'PRIMCOORD')
  found = False
  for idx in all_idx:
    mm.seek(idx)
    line = mm.readline()
    myi  = int(line.split()[1])
    if myi != imode: continue
    
    # found imode
    found = True

    # get number of atoms
    line = mm.readline()
    natom = int(line.split()[0])

    # get atomic symbols, positions and normal mode
    elem = []
    data = []
    for iatom in range(natom):
      line = mm.readline()
      tokens = line.split()
      elem.append(tokens[0])
      data.append(map(float,tokens[1:]))
    # end for iatom

    # check that the next line is either next mode or empty
    line = mm.readline()
    expected = (line == '') or (line.startswith('PRIMCOORD'))
    if not expected:
      raise RuntimeError('failed to read mode %d correctly'%imode)
    # end if
    break
  # end for idx

  if not found:
    raise RuntimeError('failed to find mode %d in %s'%(imode,faxsf))
  # end if

  return elem,np.array(data)
# end def get_axsf_normal_mode


def parse_output(floc):
  """ get energy, volume and pressure from QE output """
  etot = read_first_energy(floc)
  entry = {'energy':etot/2.}  # Ry to ha

  mm  = read(floc)

  label_map = {
    'volume':'unit-cell volume',
    'natom':'number of atoms/cell'
  }

  for key in label_map.keys():
    val = name_sep_val(mm, label_map[key])
    entry[key] = val
  # end for
  au_stressl,kbar_stressl = read_stress(floc)
  assert len(au_stressl) == 1
  au_stress = au_stressl[0]
  entry['pressure'] = np.diag(au_stress).mean()/2.  # Ry to ha
  entry['stress'] = au_stress/2.  # Ry to ha

  return entry
# end def parse_output


def parse_bands_out(bout, max_evline=1024):
  fp = open(bout, 'r')
  header = fp.readline()
  nbnd, nks = [int(keyval.split('=')[1].strip('\n').strip('/'))
    for keyval in header.split(',')]

  kvecs = []
  etable = []
  for iks in xrange(nks):
    kline = fp.readline()
    kvecs.append( map(float, kline.split()) )

    evl = []
    mynbnd = 0
    for i in xrange(max_evline):
      bline = fp.readline()
      nums = map(float, bline.split())
      evl.append( nums )
      mynbnd += len(nums)
      if mynbnd >= nbnd: break
    # end for
    eva = [a for b in evl for a in b]

    if not len(eva) == nbnd:
      raise RuntimeError('increase max_evline')
    etable.append(eva)
  # end for

  if len(fp.readline()) != 0:
    raise RuntimeError('wrong nbnd')

  fp.close()

  return np.array(kvecs), np.array(etable)
# end def parse_bands_out


def parse_nscf_bands(nscf_out, span=7, trailer='occupation numbers'):
  data = {}  # build a dictionary as return value
  def scanf_7f(line, n):
    """ implement scanf("%7.*f") """
    numl = []
    for i in range(n):
      token = line[span*i:span*(i+1)]
      num = float(token)
      numl.append(num)
    return numl
  def parse_float_body(body):
    """ parse a blob of floats """
    lines = body.split('\n')
    numl = []
    for line in lines:
      if len(line) == 0: continue
      numl += map(float, line.split())
    return numl
  from qharv.reel import ascii_out
  ndim = 3
  mm = ascii_out.read(nscf_out)
  alat = ascii_out.name_sep_val(mm, 'lattice parameter (alat)')
  blat = 2*np.pi/alat

  # find the beginnings of each band
  bhead = ' k ='
  idxl = ascii_out.all_lines_with_tag(mm, bhead)
  nkpt = len(idxl)
  data['nkpt'] = nkpt

  # estimate the end of the last band
  idx1 = ascii_out.all_lines_with_tag(mm, trailer)[-1]

  # trick to use no if statement in the loop
  idxl = idxl + [idx1]

  kvecs = []  # (nkpt, ndim)
  mat = []    # (nkpt, nbnd)
  for ikpt in range(nkpt):
    # specify beginning and end of the band output
    idx0 = idxl[ikpt]
    idx1 = idxl[ikpt+1]

    # parse band output
    #  first read header
    mm.seek(idx0)
    header = mm.readline()
    if not 'bands (ev)' in header: continue
    kxkykz = ascii_out.lr_mark(header, '=', '(')
    kvec = scanf_7f(kxkykz, ndim)
    kvecs.append(kvec)
    #  then read body
    body = mm[mm.tell():idx1].strip('\n')
    if trailer in body:
      idx2 = mm.find(trailer.encode())
      body = mm[mm.tell():idx2].strip('\n')
    row = parse_float_body(body)
    mat.append(row)
  # end for ikpt
  data['kvecs'] = blat*np.array(kvecs)
  data['bands'] = np.array(mat)
  return data

def parse_kline(line, ik=None):
  from qharv.reel import ascii_out
  assert 'k(' in line
  ikt, kvect, wkt = line.split('=')
  myik = int(ascii_out.lr_mark(ikt, '(', ')'))
  if ik is not None:  # check k index
    assert ik == myik-1  # fortran 1-based indexing
  wk = float(wkt)
  klist = ascii_out.lr_mark(kvect, '(', ')').split()
  kvec = np.array(klist, dtype=float)
  return kvec, wk

def read_kpoints(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)

  # get lattice units
  alat = ascii_out.name_sep_val(mm, 'lattice parameter (alat)')
  blat = 2*np.pi/alat

  # start parsing k points
  idx = mm.find(b'number of k points')
  mm.seek(idx)

  # read first line
  #  e.g. number of k points=    32  Fermi-Dirac smearing ...
  line = mm.readline()
  nk = int(line.split('=')[1].split()[0])

  # confirm units in second line
  line = mm.readline()
  assert '2pi/alat' in line

  # start parsing kvectors
  data = np.zeros([nk, 4])  # ik, kx, ky, kz, wk
  for ik in range(nk):
    line = mm.readline()
    kvec, wk = parse_kline(line, ik=ik)
    data[ik, :3] = kvec*blat
    data[ik, 3] = wk
  return data

def read_kfracs(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  # get number of kpoints
  idx = mm.find(b'number of k points')
  mm.seek(idx)
  line = mm.readline()
  nk = int(line.split('=')[1].split()[0])
  # find first line
  idx = mm.find(b'cryst. coord.')
  mm.seek(idx)
  mm.readline()
  # read kpoints and weights
  data = np.zeros([nk, 4])
  for ik in range(nk):
    line = mm.readline()
    kvec, wk = parse_kline(line)
    data[ik, :3] = kvec
    data[ik, 3] = wk
  return data

def parse_scf_conv(scf_out):
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)

  idxl = ascii_out.all_lines_with_tag(mm, 'iteration #')
  data = []
  for idx in idxl:
    mm.seek(idx)

    # read iteration number
    iternow = ascii_out.name_sep_val(mm, 'iteration', sep='#', dtype=int)

    # find total energy and other info (!!!! must be in order)
    try:
      time = ascii_out.name_sep_val(mm, 'cpu time spent up to now', sep='is')
      enow = ascii_out.name_sep_val(mm, 'total energy')
    except:
      continue

    entry = {'istep':iternow, 'energy':enow, 'time':time}
    data.append(entry)

  return data


def get_efermi(fout):
  from qharv.reel import ascii_out
  mm = ascii_out.read(fout)
  efermi = ascii_out.name_sep_val(mm, 'the Fermi energy', sep='is')
  return efermi


def get_gc_occ(mat, efermi):
  """ get grand canonical occupation vector

  example:
    data = qer.parse_nscf_bands(scf_out)
    kvecs = data['kvecs']
    bands = np.array(data['bands'])

    mm = ascii_out.read(scf_out)
    efermi = ascii_out.name_sep_val(mm, 'the Fermi energy', sep='is')

    norbs = get_gc_occ(bands, efermi)

  Args:
    mat (np.array): Kohn-Sham eigenvalues (nkpt, nband)
    efermi (float): Fermi energy
  Return:
    np.array: number of occupied orbitals at each kpoint
  """
  norbl = []
  nkpt, nbnd = mat.shape
  for ikpt in range(nkpt):
    row = mat[ikpt]
    sel = row<=efermi
    norb = len(row[sel])
    norbl.append(norb)
  # end for
  norbs = np.array(norbl)
  return norbs


def get_occ_df(kvecs, norbs):
  """ save grand canonical occupation vector with twists

  Args:
    kvecs (np.array): twist vectors, user-defined units
    norbs (np.array): a list of integers
  """
  import pandas as pd
  cols = ('kmag', 'norb', 'kx', 'ky', 'kz')
  kmags = np.linalg.norm(kvecs, axis=1)
  data = np.zeros([len(norbs), len(cols)])
  data[:, 0] = kmags
  data[:, 1] = norbs
  data[:, 2:] = kvecs
  mydf = pd.DataFrame(data, columns=cols)
  mydf['norb'] = mydf['norb'].astype(int)
  mydf['group'] = mydf.index
  return mydf

def read_cell(scf_in, ndim=3):
  with open(scf_in,'r+') as f:
    mm = mmap(f.fileno(), 0)
  idx = mm.find(b'CELL_PARAMETERS')
  mm.seek(idx)
  header = mm.readline()
  unit = header.split()[-1]
  mat = np.zeros([ndim, ndim])
  for idim in range(ndim):
    line = mm.readline()
    vec = np.array(line.split(), dtype=float)
    mat[idim, :] = vec
  data = {
    'unit': str(unit),
    'axes': mat
  }
  return data

def read_out_cell(scf_out, ndim=3):
  axes = np.zeros([ndim, ndim])
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)
  idx = mm.find(b'crystal axes')
  mm.seek(idx)
  mm.readline()
  for idim in range(ndim):
    line = mm.readline()
    right = line.split('=')[-1]
    text = ascii_out.lr_mark(right, '(', ')')
    axes[idim, :] = map(float, text.split())
  return axes


def get_occupation_numbers(nscf_out, nmax=1024):
  from qharv.reel import ascii_out
  mm = ascii_out.read(nscf_out)
  idx = ascii_out.all_lines_with_tag(mm, 'occupation numbers')
  occl = []
  for i in idx:
    mm.seek(i)
    mm.readline()
    occ = []
    for j in range(nmax):
      line = mm.readline()
      tokens = line.split()
      if len(tokens) == 0:
        break
      occ += map(float, tokens)
    next_line = mm.readline()
    occl.append(occ)
  return np.array(occl)

def read_sym_ops(scf_out, ndim=3):
  """ read symmetry operators

  Args:
    scf_out (str): QE output file
    ndim (int, optional): number of spatial dimensions, default is 3
  Return:
    list: all symmetry operators, each is represented as a dictionary
      isym is index, name is description, vec is shift, mat is rotation
  """
  from qharv.reel import ascii_out
  mm = ascii_out.read(scf_out)

  # find starting location of symmetry operator output
  idx = mm.find(b'Sym. Ops.')
  if idx == -1:
    msg = 'no symmetry operations printed in %s. Is verbosity high?' % scf_out
    raise RuntimeError(msg)
  # rewind to beginning of line
  idx0 = mm.rfind(b'\n', 0, idx)
  mm.seek(idx0+1)
  header = mm.readline().decode()
  nsym = int(header.split()[0])

  # check the number of symmetry outputs
  idxl = ascii_out.all_lines_with_tag(mm, 'isym = ')
  if len(idxl) != nsym:
    raise RuntimeError('found %d symm. expected %d' % (len(idxl), nsym))

  # parse symmetry operators
  symops = []
  for idx in idxl:
    mm.seek(idx)

    # read symmetry index and name: isym, name
    line0 = mm.readline().decode()
    text0 = line0.split('=')[1]
    tokens0 = text0.split()
    isym = int(tokens0[0])
    name = ' '.join(tokens0[1:])

    # read translation vector: vec
    vec = [0]*ndim
    if 'cart. axis' in name:
      vect = ascii_out.lr_mark(line0, '[', ']')
      vec[:] = list(map(float, vect.split(',')))

    # read rotation matrix: mat
    mat = []
    idx = mm.find(b'cryst.')
    mm.readline()  # skip empty line
    for idim in range(ndim):
      line = mm.readline().decode()
      if 'cryst.' in line:
        line = line.split('=')[1]
      text = ascii_out.lr_mark(line, '(', ')')
      mat.append(list(map(float, text.split())))
    entry = {
      'isym': isym,
      'name': name,
      'vec': vec,
      'mat': mat
    }
    symops.append(entry)
  mm.close()
  return symops

def get_weights(nscf_out, remove_copy=False, atol=1e-10):
  from qharv.reel import ascii_out
  mm = ascii_out.read(nscf_out)
  idx = ascii_out.all_lines_with_tag(mm, 'wk =')
  lines = ascii_out.all_lines_at_idx(mm, idx)
  weights = []
  for line in lines:
    wt = float(line.strip('\n').split('wk =')[-1])
    weights.append(wt)
  mm.close()
  nt = len(weights)
  if remove_copy:
    weights = weights[:nt/2]
  wtot = sum(weights)
  if not np.isclose(wtot, 2.0, atol=atol):
    raise RuntimeError('wrong weight sum %3.2f; expected 2.0' % wtot)
  return np.array(weights)

def get_gc_occ(bands, efermi):
  norbl = []
  nkpt, nbnd = bands.shape
  for ikpt in range(nkpt):
    row = bands[ikpt]
    sel = row<=efermi
    norb = len(row[sel])
    norbl.append(norb)
  norbs = np.array(norbl)
  return norbs

def get_tgrid_tshift(nscf_in):
  from qharv.reel import ascii_out
  mm = ascii_out.read(nscf_in)
  idx = mm.find(b'K_POINTS automatic')
  mm.seek(idx)
  mm.readline()
  kline = mm.readline()
  mm.close()

  nums = list(map(int, kline.split()))
  tgrid = np.array(nums[:3])
  tshift = np.array(nums[3:])
  return tgrid, tshift

def get_axes(nscf_in, ndim=3):
  from qharv.reel import ascii_out
  mm = ascii_out.read(nscf_in)
  idx = mm.find(b'CELL_PARAMETERS')
  mm.seek(idx)
  mm.readline()
  cell = []
  for idim in range(ndim):
    line = mm.readline().decode()
    nums = list(map(float, line.split()))
    cell.append(nums)
  mm.close()
  axes = np.array(cell)
  return axes

def get_tgrid_raxes(nscf_in, ndim=3):
  from qharv.inspect import axes_pos
  tgrid, tshift = get_tgrid_tshift(nscf_in)
  axes = get_axes(nscf_in, ndim=ndim)
  raxes = axes_pos.raxes(axes)
  return tgrid, raxes

def get_elem_pos(nscf_in):
  from qharv.reel import ascii_out
  mm = ascii_out.read(nscf_in)
  natom = ascii_out.name_sep_val(mm, 'nat', '=', dtype=int)
  idx = mm.find(b'ATOMIC_POSITIONS')
  mm.seek(idx)
  header = mm.readline().decode()
  eleml = []
  posl = []
  for iatom in range(natom):
    line = mm.readline()
    tokens = line.split()
    eleml.append(tokens[0])
    posl.append(tokens[1:])
  mm.close()
  elem = np.array(eleml, dtype=str)
  pos = np.array(posl, dtype=float)
  return elem, pos, header
