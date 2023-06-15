#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# Model being used for plots
parser.add_argument('-n', '--name', default=None, \
                    help='name of model that will be plotted plots')

args = parser.parse_args()

assert args.name is not None, 'Must supply model name'

# Read log file
history = pd.read_csv(args.name + '.log', sep = ',', engine='python')

nEpoch = history.shape[0] - 1
batchSize = 64
activation= 'tanh'

# Short name describing model params
descString = 'e-{}_b-{}_a-{}'.format(nEpoch, batchSize, activation)

# Plot or skip: Overall loss, data loss, pde loss, wall loss, MAE, lr
plotRows = [1, 1, 1, 1, 1, 1]

nRows = sum(plotRows)
nColumns = 4

plt.style.use('ggplot')

titleStr = '=== Training (left) & validation (right) ===\n' \
           'Model: {}\n'\
           'Activation: {}\n' \
           .format(args.name, activation)

def plot_history(s, e):
  hist = history[s:e]
  print(hist)
  title = titleStr + 'Epochs: {:03}  - {:03}'.format(s, e)
  cnt = 0

  fig = plt.figure(figsize=(nColumns*5, nRows*3), dpi=120)
  #fig.suptitle(title, fontsize=14)
  fig.subplots_adjust(hspace=.5)
  fig.subplots_adjust(wspace=.5)

  if plotRows[0]:
    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['loss'])
    plt.xlabel('epoch')
    ax.title.set_text('Overall loss')
    legend = ['train']
    plt.legend(legend, loc='upper right')

    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['val_loss'])
    plt.xlabel('epoch')
    ax.title.set_text('val loss')
    legend = ['val']
    plt.legend(legend, loc='upper right')

  if plotRows[1]:
    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['uMse'])
    plt.plot(hist['vMse'])
    plt.xlabel('epoch')
    ax.title.set_text('MSE Interface')
    plt.legend(['uMse', 'vMse', 'pde2'], loc='upper right')

    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['val_uMse'])
    plt.plot(hist['val_vMse'])
    plt.xlabel('epoch')
    ax.title.set_text('val MSE Interface')
    plt.legend(['val_uMse', 'val_vMse'], loc='upper right')

  if plotRows[2]:
    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['pde0'])
    plt.plot(hist['pde1'])
    plt.plot(hist['pde2'])
    plt.xlabel('epoch')
    ax.title.set_text('MSE PDE')
    plt.legend(['pde0', 'pde1', 'pde2'], loc='upper right')

    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['val_pde0'])
    plt.plot(hist['val_pde1'])
    plt.plot(hist['val_pde2'])
    plt.xlabel('epoch')
    ax.title.set_text('val MSE PDE')
    plt.legend(['val_pde0', 'val_pde1', 'val_pde2'], loc='upper right')

  if plotRows[3]:
    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['uMseWalls'])
    plt.plot(hist['vMseWalls'])
    #plt.plot(hist['pMseWalls'])
    plt.xlabel('epoch')
    ax.title.set_text('MSE Boundary')
    plt.legend(['uMseWalls', 'vMseWalls', 'pMseWalls'], loc='upper right')

    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['val_uMseWalls'])
    plt.plot(hist['val_vMseWalls'])
    #plt.plot(hist['val_pMseWalls'])
    plt.xlabel('epoch')
    ax.title.set_text('val MSE Boundary')
    plt.legend(['val_uMseWalls', 'val_vMseWalls', 'val_pMseWalls'], loc='upper right')

  if plotRows[4]:
    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['uMae'])
    plt.plot(hist['vMae'])
    plt.xlabel('epoch')
    ax.title.set_text('MAE Interface')
    plt.legend(['uMae', 'vMae'], loc='upper right')

    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['val_uMae'])
    plt.plot(hist['val_vMae'])
    plt.xlabel('epoch')
    ax.title.set_text('val MAE Interface')
    plt.legend(['val_uMae', 'val_vMae'], loc='upper right')

  if plotRows[5]:
    cnt += 1
    ax = fig.add_subplot(nRows, nColumns, cnt)
    ax.set_yscale('log')
    plt.plot(hist['lr'])
    plt.xlabel('epoch')
    ax.title.set_text('lr')
    plt.legend(['lr'], loc='upper right')

  fig.savefig(args.name + '_f{:03}-{:03}_{}.png'.format(s, e, descString), bbox_inches='tight')
  plt.close(fig)

# Plot history in multiple steps
plot_history(s=0, e=nEpoch)

