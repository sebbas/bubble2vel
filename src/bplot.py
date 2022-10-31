#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# Model being used for plots
parser.add_argument('-n', '--name', help='name of model that is used for plots',
  default='bubble2vel-200x8_d10000_c10000_a1.0-1.0_b0.0-0.0-0.0_g0.0-0.0-0.0_lr0.0005_p200')

args = parser.parse_args()

# Read log file
history = pd.read_csv(args.name + '.log', sep = ',', engine='python')

nEpoch = history.shape[0] - 1
batchSize = 128
activation= 'tanh'
architecture = [200, 200, 200, 200, 200, 200, 200, 200]

# Short name describing model params
descString = 'e-{}_b-{}_a-{}'.format(nEpoch, batchSize, activation)

nRows = 6
nColumns = 2

plt.style.use('ggplot')

titleStr = '=== Training (left) & validation (right) ===\n' \
           'Model: {}\n'\
           'Architecture: {}\n' \
           'Activation: {}\n' \
           .format(args.name, architecture, activation)

def plot_history(s, e):
  hist = history[s:e]
  print(hist)
  title = titleStr + 'Epochs: {:03}  - {:03}'.format(s, e)

  fig = plt.figure(figsize=(nColumns*5, nRows*3), dpi=120)
  fig.suptitle(title, fontsize=14)
  fig.subplots_adjust(hspace=.5)
  fig.subplots_adjust(wspace=.5)

  ax = fig.add_subplot(nRows, nColumns, 1)
  ax.set_yscale('log')
  plt.plot(hist['loss'])
  plt.xlabel('epoch')
  ax.title.set_text('loss')
  legend = ['train']
  plt.legend(legend, loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 2)
  ax.set_yscale('log')
  plt.plot(hist['val_loss'])
  plt.xlabel('epoch')
  ax.title.set_text('val loss')
  legend = ['val']
  plt.legend(legend, loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 3)
  ax.set_yscale('log')
  plt.plot(hist['pde0'])
  plt.plot(hist['pde1'])
  plt.plot(hist['pde2'])
  plt.xlabel('epoch')
  ax.title.set_text('MSE PDE')
  plt.legend(['pde0', 'pde1', 'pde2'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 4)
  ax.set_yscale('log')
  plt.plot(hist['val_pde0'])
  plt.plot(hist['val_pde1'])
  plt.plot(hist['val_pde2'])
  plt.xlabel('epoch')
  ax.title.set_text('val MSE PDE')
  plt.legend(['val_pde0', 'val_pde1', 'val_pde2'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 5)
  ax.set_yscale('log')
  plt.plot(hist['uMse'])
  plt.plot(hist['vMse'])
  plt.xlabel('epoch')
  ax.title.set_text('MSE u, v')
  plt.legend(['uMse', 'vMse', 'pde2'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 6)
  ax.set_yscale('log')
  plt.plot(hist['val_uMse'])
  plt.plot(hist['val_vMse'])
  plt.xlabel('epoch')
  ax.title.set_text('val MSE u, v')
  plt.legend(['val_uMse', 'val_vMse'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 7)
  ax.set_yscale('log')
  plt.plot(hist['uMae'])
  plt.plot(hist['vMae'])
  plt.xlabel('epoch')
  ax.title.set_text('MAE u, v')
  plt.legend(['uMae', 'vMae'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 8)
  ax.set_yscale('log')
  plt.plot(hist['val_uMae'])
  plt.plot(hist['val_vMae'])
  plt.xlabel('epoch')
  ax.title.set_text('val MAE u, v')
  plt.legend(['val_uMae', 'val_vMae'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 9)
  ax.set_yscale('log')
  plt.plot(hist['uMseWalls'])
  plt.plot(hist['vMseWalls'])
  plt.plot(hist['pMseWalls'])
  plt.xlabel('epoch')
  ax.title.set_text('MSE Walls u, v, p')
  plt.legend(['uMseWalls', 'vMseWalls', 'pMseWalls'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 10)
  ax.set_yscale('log')
  plt.plot(hist['val_uMseWalls'])
  plt.plot(hist['val_vMseWalls'])
  plt.plot(hist['val_pMseWalls'])
  plt.xlabel('epoch')
  ax.title.set_text('val MAE u, v, p')
  plt.legend(['val_uMseWalls', 'val_vMseWalls', 'val_pMseWalls'], loc='upper right')

  ax = fig.add_subplot(nRows, nColumns, 11)
  ax.set_yscale('log')
  plt.plot(hist['lr'])
  plt.xlabel('epoch')
  ax.title.set_text('lr')
  plt.legend(['lr'], loc='upper right')

  fig.savefig(args.name + '_f{:03}-{:03}_{}.png'.format(s, e, descString), bbox_inches='tight')
  plt.close(fig)

# Plot history in multiple steps
plot_history(s=0, e=nEpoch)

