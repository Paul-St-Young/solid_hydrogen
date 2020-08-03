import numpy as np

def lmp_table_text(rang, eev, forces=None):
  if forces is None:
    from scipy.interpolate import CubicSpline
    ecs = CubicSpline(rang, eev)
    fcs = ecs.derivative()
    forces = -fcs(rang)
  nr = len(rang)
  text = 'N %d\n\n' % nr

  line_fmt = '{ir:12d} {r:>20.6e} {e:>20.6e} {f:>20.6e}\n'
  ir = 1
  for r, e, f in zip(rang, eev, forces):
    line = line_fmt.format(
      ir = ir, r=r, e=e, f=f
    )
    ir += 1
    text += line
  return text

def sg1978(rsep, noc9=False):
  def fcut(r):  # attenuate long-range multipole terms
    bohr = 0.529177210903  # # CODATA 2018
    pre = 1.28
    rm = 3.41/bohr
    fr = np.ones(len(r))
    sel = r < pre*rm
    fr[sel] = np.exp(-(pre*rm/r[sel]-1)**2)
    return fr
  expos = [-6, -8, -9, -10]
  if noc9:
    expos = [-6, -8, -10]
  coeff_map = {
    -6: -12.14,
    -8: -215.2,
    -9:  143.1,
   -10: -4813.9
  }
  pot = sum([coeff_map[e]*rsep**e for e in expos])
  alpha = 1.713
  beta = 1.5671
  gamma = 0.00993
  sr = np.exp(alpha-beta*rsep-gamma*rsep**2)
  lr = fcut(rsep)*pot
  return sr + lr

def sg_lr(rsep, rstar=8.248276588766563, expo_coeffs=None):
  if expo_coeffs is None:
    expo_coeffs = {
      -6: -12.14,
      -8: -215.2,
      -9:  143.1,
     -10: -4813.9
    }
  pot = sum([expo_coeffs[e]*rsep**e for e in expo_coeffs])
  fr = np.ones(len(rsep))
  sel = rsep < rstar
  fr[sel] = np.exp(-(rstar/rsep[sel]-1)**2)
  return fr*pot

def sg_sr(rsep, params=None):
  if params is None:
    params = [1.713, 1.5671, 0.00993]
  alpha, beta, gamma = params
  sr = np.exp(alpha-beta*rsep-gamma*rsep**2)
  return sr
