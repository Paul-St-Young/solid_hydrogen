#!/usr/bin/env python

class MoldenNormalMode:
    
    def __init__(self,filename,nmode=None,natom=None):
        self.fhandle   = open(filename,"r+")
        self.mm        = mmap(self.fhandle.fileno(),0)
        self.sections  = None
        self.nmode     = nmode
        self.natom     = natom
    # end def
    
    def find_sections(self,start_marker="[",nsection_max=20):
    
        self.mm.seek(0)
        sections = {}
        for isection in range(nsection_max):
            # each section is marked by a []
            idx = self.mm.find(start_marker)
            if idx == -1:
                break
            # end if

            self.mm.seek(idx)
            section_name = self.mm.readline().strip()[1:-1]
            sections[section_name] = idx
        # end for
        
        self.sections = sections
        return sections
    # end def find_sections
    
    def section_text(self,section_label,nline_max = 2048):
        if self.sections is None:
            raise NotImplementedError("call find_sections() first")
        # end if
        
        self.mm.seek( self.sections[section_label] )
        self.mm.readline() # skip section label line
        block = ''
        for iat in range(nline_max):
            line = self.mm.readline()
            if line.startswith("[") or self.mm.size() == self.mm.tell():
                break
            # end if
            block += line
        # end for
        return block
    # end def
    
    def find_modes(self, nmode_max = 256, start_marker = "vibration"):
        
        self.mm.seek(0)
        
        vib_start_idx = []
        for imode in range(nmode_max):
            idx = self.mm.find(start_marker)
            vib_start_idx.append(idx)
            if idx == -1:
                break
            # end if
            self.mm.seek(idx)
            self.mm.readline()
        # end for

        return vib_start_idx
    # end def
    
    def read_modes(self,ndim=3):
        
        sections = self.find_sections()
        atom_lines = self.section_text("FR-COORD").split("\n")[:-1] 
        natom = len(atom_lines)
        self.natom = natom
        
        vib_start_idx = self.find_modes()
        nmode = len(vib_start_idx)-1
        self.nmode = nmode

        normal_modes = np.zeros([nmode,natom,ndim])
        for imode in range(nmode):
            vibration_block = self.mm[vib_start_idx[imode]:vib_start_idx[imode+1]]
            disp_vecs_texts = vibration_block.split("\n")[1:]
            disp_vecs = [map(float,line.split()) for line in disp_vecs_texts]
            if len(disp_vecs) == natom +1:
                disp_vecs = disp_vecs[:-1]
            # end if
            normal_modes[imode,:,:] = disp_vecs
        # end for imode
        return normal_modes
    # end def

    def read_freqs(self):
        if self.nmode is None:
          self.read_modes()
        # end if

        freq_lines = self.section_text("FREQ").split("\n")
        if len(freq_lines) == self.nmode+1:
            freq_lines = freq_lines[:-1]
        # end if
        
        freqs = map(float,freq_lines)
        return freqs
    # end def
# end class MoldenNormalMode

if __name__ == '__main__':

    import sys
    fname = sys.argv[1]

    mreader = MoldenNormalMode(fname)
    print "normal mode frequencies: ",mreader.read_freqs()

# end __main__
    
