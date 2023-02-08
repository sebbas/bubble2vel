#!/usr/bin/env python3

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import h5py as h5

PRINT_DEBUG = 1
IMG_DEBUG = 0
VID_DEBUG = 0
FILE_IO   = 0

SRC_FLOWNET = 1
SRC_FLASHX  = 2

MODEL_NAME = 'bubble2vel'

nDim = 2

# Values for experimental dataset (flownet, water)
# Reference quantities for non-dimensionalization
V         = 0.625               # [meter/second],   Reference velocity
L         = 0.02                # [meter],          Reference length
T         = L / V               # [second],         Reference time
_nu       = 2.938e-7            # [meter^2/second], Kinematic viscosity at 100 Celsius
Re_fn     = (V * L) / _nu       # dimensionless,    Reynolds number
# World space quantities
fps       = 400                  # [frames/sec]
pixelSize = 1.922e-5             # [meters] at resolution 1024px
worldSize = pixelSize * 1024     # [meters], must mult with res that the pixel size was captured at
# Domain space quantities
imageSize = 512


# Values for experimental dataset (flownet, water)
# Reference quantities for non-dimensionalization
V_fx         = 0.0868                 # [meter/second],     Reference velocity
L_fx         = 7e-4                   # [meter],            Reference length
T_fx         = L_fx / V_fx            # [second],           Reference time
_rho_fx      = 1620                   # [meter^2/second],   Density
_mu_fx       = 4e-4                   # [N*second/meter^2], Dynamic viscosity
_nu_fx       = _mu_fx / _rho_fx       # [meter^2/second],   Kinematic viscosity
Re_fx        = (V_fx * L_fx) / _nu_fx # dimensionless,      Reynolds number
# World space quantities
fps_fx       = 125
worldSize_fx = 0.0168 # 16.8 millimeters
# Domain space quantities
imageSize_fx = 384


def _ensure_exists(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
  return dir


def save_image(src, subdir, name, frame, i=0, origin='lower', cmap='gray'):
  imgDir = _ensure_exists('../img')
  dir = _ensure_exists(os.path.join(imgDir, subdir))
  fname = '%s_i%02d_%04d.png' % (name, i, frame) if i>0 else '%s_%04d.png' % (name, frame)
  plt.imsave(os.path.join(dir, fname), src, origin=origin, cmap=cmap)
  plt.clf()


def save_array(src, subdir, name, frame, size):
  grid = np.zeros((size[0], size[1]), dtype=int)
  for pos in src:
    grid[int(pos[1]), int(pos[0])] = 1
  save_image(grid, subdir, name, frame)


def save_figure_plot(fig, subdir, name, frame, colorbar=False, plot=None, clip=True, cmin=0, cmax=1):
  imgDir = _ensure_exists('../img')
  dir = _ensure_exists(os.path.join(imgDir, subdir))
  fname = '%s_%04d.png' % (name, frame)

  if colorbar:
    if clip:
      plot.set_clim(vmin=cmin, vmax=cmax)
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    fig.colorbar(plot, orientation='vertical', cax=cax)

  plt.savefig(os.path.join(dir, fname))
  plt.close(fig)


def save_figure_dataframe(df, fig, subdir, name):
  imgDir = _ensure_exists('../img')
  dir = _ensure_exists(os.path.join(imgDir, subdir))
  fig.savefig(os.path.join(dir, name))
  plt.close(fig)


def save_video(subdir, name, imgDir, fps=5):
  vidDir = _ensure_exists('../vid')
  finalDir = os.path.join(vidDir, subdir)
  dir = _ensure_exists(finalDir)
  imgName = imgDir + name + '_%04d.png'
  vidName = os.path.join(finalDir, name + '_%02dfps.mp4' % fps)
  command = f'ffmpeg -y -r ' + str(fps) + ' -f image2 -s 512x512 -i ' + imgName + ' -vcodec libx264 -crf 5 -pix_fmt yuv420p ' + vidName
  subprocess.call(command, shell=True)


def save_velocity(src, subdir, name, frame, size=(512,512,1), invertY=False, \
                  type='stream', arrow_res=1, cmin=0.0, cmax=5.0, density=4.5, \
                  filterZero=False, cmap='jet'):
  fig, ax = plt.subplots(1, 1, figsize=(10,10))
  x = np.arange(0, int(size[0]), arrow_res)
  y = np.arange(0, int(size[1]), arrow_res)
  X, Y = np.meshgrid(x, y, indexing='xy')
  U, V = src[:,:,0], src[:,:,1]
  M = np.sqrt(np.square(U) + np.square(V))
  colorbar = False
  if type == 'stream':
    plot = ax.streamplot(X, Y, U, V, density=density, linewidth=1.0)#, color='#A23BEC')
  elif type == 'quiver':
    colorbar = True
    U = U[::arrow_res, ::arrow_res]
    V = V[::arrow_res, ::arrow_res]
    M = M[::arrow_res, ::arrow_res]
    if filterZero:
      maskFilterZero = M > 0
      U = U[maskFilterZero]
      V = V[maskFilterZero]
      M = M[maskFilterZero]
      X = X[maskFilterZero]
      Y = Y[maskFilterZero]
    plot = ax.quiver(X, Y, U, V, M, cmap='winter', scale_units = 'xy', angles='xy')
  elif type == 'mag':
    colorbar = True
    cmin = np.min(M)
    cmax = np.max(M)
    plot = plt.imshow(M, cmap=cmap)

  ax.set_xlim(0, size[0])
  ax.set_ylim(0, size[1])
  if invertY:
    ax.invert_yaxis()
  save_figure_plot(fig, subdir, name, frame, colorbar=colorbar, plot=plot, cmin=cmin, cmax=cmax)


def save_velocity_bins(src, subdir, name, frame, bmin, bmax, bstep):
  srcFlat = src.flatten()
  bins = np.arange(bmin, bmax, bstep)
  fig, ax = plt.subplots()
  pt = [2.5, 97.5] # percentile
  ax.set_xlabel('Bins')
  ax.set_ylabel('# Occurences')
  ax.set_title(name)
  plot = plt.hist(srcFlat, bins=bins)
  subtitle = 'mean(): {:.2f}, \
              mean(abs()): {:.2f}, \
              min(): {:.2f}, \
              max(): {:.2f},\n \
              median(): {:.2f}, \
              median(abs()): {:.2f}, \
              var(): {:.2f}, \
              percentile({}, {}): [{:.2f}, {:.2f}]' \
              .format(np.mean(srcFlat), \
                      np.mean(np.abs(srcFlat)), \
                      np.min(srcFlat), \
                      np.max(srcFlat), \
                      np.median(srcFlat), \
                      np.median(np.abs(srcFlat)), \
                      np.var(srcFlat), \
                      *pt, *np.percentile(srcFlat, pt))
  fig.text(.5, .02, subtitle, ha='center')
  fig.set_size_inches(7, 9, forward=True)
  save_figure_plot(fig, subdir, name, frame, colorbar=False, plot=plot)


def save_plot(src, subdir, name, frame, size=(512,512,1), invertY=False, cmin=0.0, cmax=5.0, cmap=None):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.set_xlim(0, int(size[0]))
    ax.set_ylim(0, int(size[1]))
    if invertY:
      ax.invert_yaxis()
    plot = plt.imshow(src, cmap=cmap)
    save_figure_plot(fig, subdir, name, frame, colorbar=True, plot=plot, cmin=cmin, cmax=cmax)


def print_progress(i, n):
    j = (i + 1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-40s] %d%%" % ('='*int(40*j), 100*j))
    sys.stdout.flush()
    if i+1 == n: print('\n')


def compute_relative_error(true, pred, positions, epsMin, epsMax):
  nCell, rerr, res = 0, 0, 0
  for pos in positions:
    i, j = int(pos[0]), int(pos[1])
    absTrue = np.abs(true[j,i])
    if absTrue > epsMin and absTrue < epsMax:
      nCell += 1
      residual = np.abs(true[j,i] - pred[j,i])
      rerr  += np.abs(residual / true[j,i])
      res   += residual
  return rerr / nCell, res / nCell


def get_arch_string(arch):
  archStr   = ''
  l0        = arch[0]
  nSameSize = 1
  for l in arch[1:]:
    if l == l0:
      nSameSize += 1
    else:
      if nSameSize == 1:
        archStr = archStr + '-' + repr(l0)
      else:
        archStr = archStr + '-{}x{}'.format(l0, nSameSize)
      l0        = l
      nSameSize = 1
  if nSameSize == 1:
    archStr = archStr + '-' + repr(l0)
  else:
    archStr = archStr + '-{}x{}'.format(l0, nSameSize)
  return archStr


def get_list_string(list, delim=''):
  return delim.join(map(str, list))


def vel_domain_to_world(input, worldSize, fps):
  return input * (fps * (worldSize / imageSize))


def pos_domain_to_world(input, worldSize, imageSize):
  return input * (worldSize / imageSize)


def time_domain_to_world(input, fps):
  return input * (1 / fps)


def vel_world_to_dimensionless(input, V):
  return input / V


def pos_world_to_dimensionless(input, L):
  return input / L


def time_world_to_dimensionless(input, T):
  return input / T


def vel_dimensionless_to_world(input, V):
  return input * V


def vel_world_to_domain(input, worldSize, fps):
  return input / (fps * (worldSize / imageSize))


def get_pixelsize_dimensionless(worldSize, imageSize, L):
  return (worldSize / imageSize) / L


def get_reynolds_number(source):
  rn = None
  if source == SRC_FLOWNET: rn = Re_fn
  if source == SRC_FLASHX: rn = Re_fx
  assert rn is not None
  return rn


def print_info(string):
  print('-----------------------------------------------')
  print(string)


def save_array_hdf5(arrays, dir='../data/', filePrefix='bdata', size=[512,512], \
                    frame=0, compType='gzip', compLevel=9, dtype='float64'):
    if not os.path.exists(dir):
      os.makedirs(dir)

    fname = os.path.join(dir, filePrefix + '_{}_{:04}.h5'.format(size[0], frame))
    dFile = h5.File(fname, 'w')
    dFile.attrs['size']  = size
    dFile.attrs['frame'] = frame

    for key, value in arrays.items():
      print(value.shape)
      dFile.create_dataset(key, size, compression=compType, compression_opts=compLevel, \
                           dtype=dtype, chunks=True, data=value)

    dFile.close()
    print('Saved arrays to file {}'.format(fname))


def read_array_hdf5(arrays, fname):
    if not os.path.exists(fname):
      sys.exit('File {} does not exist'.format(fname))

    dFile = h5.File(fname, 'r')

    for key, array in arrays.items():
      print('Reading array {}'.format(key))
      array[:] = np.array(dFile.get(key))

    dFile.close()
    print('Read arrays from file {}'.format(fname))
