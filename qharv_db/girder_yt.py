import os

def gc_init(api_url='https://girder.hub.yt/api/v1', api_key=None):
  from girder_client import GirderClient
  if api_key is None:
    try:
      api_key = os.environ['GIRDER_API_KEY']
    except KeyError as err:
      msg = 'please define GIRDER_API_KEY environment variable\n'
      msg += ' or pass api_key kwargs.'
      raise KeyError(msg)
  gc = GirderClient(apiUrl=api_url)
  gc.authenticate(apiKey=api_key)
  return gc

def find_folder(gc, path, root, verbose=False):
  tokens = path.split('/')
  # initialize
  target = None
  folder_id = root
  new_path = []  # track search path
  # perform search
  for tok in tokens:
    folders = gc.listFolder(folder_id)
    for folder in folders:
      name = folder['name']
      folder_id = folder['_id']
      if name == tok:
        target = folder
        new_path.append(folder['name'])
        break
  # check search
  if len(new_path) != len(tokens):
    target = None
    msg = 'search ended at %s\n' % '/'.join(new_path)
    msg += ' expected %s' % path
    raise RuntimeError(msg)
  return target

def ls(gc, folder_id):
  from itertools import chain
  gen = chain(gc.listItem(folder_id), gc.listFolder(folder_id))
  return gen

def same_atoms(atoms0, atoms1, atol=1e-8, verbose=False):
  import numpy as np
  same_natom = len(atoms0) == len(atoms1)
  same_cell = np.allclose(atoms0.get_cell(), atoms1.get_cell(), atol=atol)
  same_pbc = np.allclose(atoms0.get_pbc(), atoms1.get_pbc(), atol=atol)
  same_pos = np.allclose(atoms0.get_positions(), atoms1.get_positions(), atol=atol)
  same = same_natom and same_cell and same_pbc and same_pos
  if (not same) and verbose:
    names = ['natom', 'cell', 'pbc', 'pos']
    sames = [same_natom, same_cell, same_pbc, same_pos]
    msg = ''
    for name, s1 in zip(names, sames):
      msg += '%s=%d; ' % (name, s1)
    print(msg)
  return same

def hash_atoms(atoms, ndig, ndig_pos=None):
  from hashlib import sha512
  h = sha512()
  axes = atoms.get_cell()
  h.update(axes.round(ndig).tobytes())
  if ndig_pos is not None:
    pos = atoms.get_positions()
    h.update(pos.round(ndig_pos).tobytes())
  return h.hexdigest()
