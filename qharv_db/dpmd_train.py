def descriptor(rcut=4, desc_type='se_ar'):
  if desc_type == 'se_ar':
    smth_frac = 0.85
    mneiba = 150
    rmult = 1.5
    mneibr = 500
    ar_smth = rcut*smth_frac
    desc = {
      'type': 'se_ar',
      'a': {
        'sel': [mneiba],
        'rcut_smth': ar_smth,
        'rcut': rcut,
        'neutron': [10, 20, 40],
        'resnet_dt': False,
        'axis_neuron': 4,
        'seed': 1,
      },
      'r': {
        'sel': [mneibr],
        'rcut_smth': ar_smth*rmult,
        'rcut': rcut*rmult,
        'neuron': [5, 10, 20],
        'resnet_dt': False,
        'seed': 1
      }
    }
  else:
    msg = 'please add inputs for descriptor type %s' % desc_type
    raise RuntimeError(msg)
  return desc

def fitting_net():
  fn = {
    'neuron': [240, 240, 240],
    'resnet_dt': True,
    'seed': 1
  }
  return fn

def loss_function():
  loss = {
    'start_pref_e': 0.02,
    'limit_pref_e': 1,
    'start_pref_f': 1000,
    'limit_pref_f': 1,
    'start_pref_v': 1000,
    'limit_pref_v': 1
  }
  return loss

def calc_decay_steps(stop_batch, start_lr, stop_lr, decay_rate):
  import numpy as np
  decay = np.log(stop_lr/start_lr)/np.log(decay_rate)
  decay_steps = int(round(stop_batch/decay))
  return decay_steps

def learning_rate(stop_batch, start_lr=5e-3, stop_lr=5e-8,
  decay_rate=0.95):
  decay_steps = calc_decay_steps(stop_batch)
  lr = {
    'type': 'exp',
    'start_lr': start_lr,
    'stop_lr': stop_lr,
    'decay_steps': decay_steps,
    'decay_rate': decay_rate
  }
  return lr

def training(stop_batch, batch_size):
  tr = {
    'seed': 1,
    'systems': ['.'],
    'set_prefix': 'set',
    'batch_size': batch_size,
    'stop_batch': stop_batch,
  }
  display = {
    'disp_file': 'lcurve.out',
    'disp_freq': 1000,
    'numb_test': 64,
    'disp_training': True,
    'time_training': True,
    'profiling': False,
    'profiling_file': 'timeline.json',
  }
  checkpoint = {
    'save_ckpt': 'model.ckpt',
    'load_ckpt': 'model.ckpt',
    'save_freq': 10000,
  }
  tr.update(display)
  tr.update(checkpoint)
  return tr

def default_input(stop_batch=100000, batch_size=32):
  dpmd_input = {
    'model': {
      'type_map': ['H'],
      'descriptor': descriptor(),
      'fitting_net': fitting_net(),
    },
    'loss': loss_function(),
    'learning_rate': learning_rate(stop_batch),
    'training': training(stop_batch, batch_size)
  }
  return dpmd_input
