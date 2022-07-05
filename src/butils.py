#!/usr/bin/env python3

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess


PRINT_DEBUG = 0
IMG_DEBUG = 1


def _ensure_exists(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
  return dir


def save_image(src, subdir, name, frame, origin='upper'):
  imgDir = _ensure_exists('../img')
  dir = _ensure_exists(os.path.join(imgDir, subdir))
  fname = '%s_%04d.png' % (name, frame)
  plt.imsave(os.path.join(dir, fname), src, origin=origin)
  plt.clf()


def save_figure_plot(fig, subdir, name, frame, colorbar=False, plot=None, cmin=0, cmax=1):
  imgDir = _ensure_exists('../img')
  dir = _ensure_exists(os.path.join(imgDir, subdir))
  fname = '%s_%04d.png' % (name, frame)

  if colorbar:
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


def print_progress(i, n):
    j = (i + 1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-40s] %d%%" % ('='*int(40*j), 100*j))
    sys.stdout.flush()
    if i+1 == n: print('\n')


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
