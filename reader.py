import numpy as np
from mmap import mmap
# Aldous Huxley

class SearchableFile:

    def __init__(self,fname):
        self.fhandle = open(fname,'r+')
        self.mm = mmap(self.fhandle.fileno(),0)
    # end def __init__

    def __del__(self):
        self.fhandle.close()
    # end def

    def rewind(self):
        self.mm.seek(0)
    # end def

    def find(self,string):
        idx = self.mm.find(string.encode())
        return idx
    # end def

    def find_first(self,string):
        self.rewind()
        idx = self.mm.find(string.encode())
        return idx
    # end def

    def locate_block(self,header,trailer):
        begin_idx = self.mm.find(header.encode())
        end_idx   = self.mm.find(trailer.encode())
        return begin_idx,end_idx
    # end def

    def block_text(self,header,trailer,skip_header=True,skip_trailer=True):
        bidx,eidx = self.locate_block(header,trailer)
        if skip_header:
            self.mm.seek(bidx)
            self.mm.readline()
            bidx = self.mm.tell()
        # end if
        if not skip_trailer:
            self.mm.seek(eidx)
            self.mm.readline()
            eidx = self.mm.tell()
        # end if
        return self.mm[bidx:eidx]
    # end def block_text

# end class SearchableFile

class BlockInterpreter:

    def __init__(self):
        pass
    # end def

    def matrix(self,text,columns_begin=0,columns_end=-1):
        rows = text.split('\n')
        if len(rows[-1])==0:
            rows.pop()
        # end if
        if columns_end == -1:
            matrix_text = [row.split()[columns_begin:] for row in rows]
        else: 
            matrix_text = [row.split()[columns_begin:columns_end] for row in rows]
        # end if
        return np.array(matrix_text,dtype=float)
    # end def matrix

# end class BlockInterpreter
