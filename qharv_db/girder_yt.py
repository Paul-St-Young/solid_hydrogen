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
