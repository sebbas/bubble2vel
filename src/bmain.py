#!/usr/bin/env python3

import sys
sys.path.append('../src')
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

# reduce number of threads
#os.environ['TF_NUM_INTEROP_THREADS'] = '1'
#os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf

np.random.seed(2022)
tf.random.set_seed(2022)

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

import bdataset as BD
import bmodel as BM
import butils as Util

keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
### Architecture
parser.add_argument('-l', '--architecture', type=int, nargs='*',
                    default=[100, 100, 100, 100, 50, 50, 50, 50],
                    help='size of each hidden layer')

# Regularizer coefficients
parser.add_argument('-r', '--reg', type=float, nargs='*', default=None,
                    help='l2 regularization')
parser.add_argument('-alpha', '--alpha', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help='coefficients for data loss')
parser.add_argument('-beta', '--beta', type=float, nargs=3, default=[1e-4, 1e-4, 1e-4],
                    help='coefficients for pde residual')

### Points for data, boundary and unknown fluid areas
parser.add_argument('-d', '--nDataPoint', type=int, default=1000,
                    help='number of data points in training')
parser.add_argument('-b', '--nBcPoint', type=int, default=250,
                    help='number of boundary points in training')
parser.add_argument('-c', '--nColPoint', type=int, default=2000,
                    help='number of collocation points in training')

### Epochs, checkpoints
parser.add_argument('-f', '--file', default='../data/bdata_512_56389.h5',
                    help='the file(s) containing flow velocity training data')
parser.add_argument('-tf', '--totalFrames', type=int, default=400,
                    help='number of frames to load from dataset')
parser.add_argument('-name', '--name', default='BubblePINN', help='model name prefix')
parser.add_argument('-ie', '--initTrain', type=int, default=0,
                    help='initial train epochs')
parser.add_argument('-e', '--nEpoch', default=10000, type=int, help='epochs')
parser.add_argument('-bs', '--batchSize', type=int, default=128,
                    help='batch size in training')
parser.add_argument('-restart', '--restart', default=False, action='store_true',
                    help='restart from checkpoint')
parser.add_argument('-ckpnt', '--checkpoint', default=None, help='checkpoint name')
parser.add_argument('-p', '--predictionFrames', type=int, default=1,
                    help='predict all frames from start frame up to this frame')

# Learning rate
parser.add_argument('-lr0', '--lr0', type=float, default=5e-4, help='init learning rate')
parser.add_argument('-lrmin', '--lrmin', type=float, default=1e-7, help='min learning rate')
parser.add_argument('-pa', '--patience', type=int, default=200,
                    help='patience for reducing learning rate')
parser.add_argument('-lr',  '--restartLr', type=float, default=None,
                     help='learning rate to restart training')

# Save more info
parser.add_argument('-g', '--saveGradStat', default=False, action='store_true',
                    help='save gradient statistics')
parser.add_argument('-ci', '--cachedInput', default=False, action='store_true',
                    help='extract boundary values from cached numpy arrays')
                    
# Plotting options
parser.add_argument('-pd', '--plotDomain', default=False, action='store_true',
                    help='plot prediction velocities for the entire domain')
parser.add_argument('-ev', '--exportVideo', default=False, action='store_true',
                    help='export image data to video (using ffmpeg)')

args = parser.parse_args()

nDim = 2
archStr = Util.get_arch_string(args.architecture)

# Load external data
dataSet = BD.BubbleDataSet(args.file, args.totalFrames, args.nDataPoint, args.nBcPoint, args.nColPoint, dim=nDim)

if not args.cachedInput:
  if not dataSet.load_data():
    sys.exit()
  dataSet.extract_domain_bc(walls=[0,0,0,1])
  dataSet.extract_bubble_bc(velEps=0.01)
  dataSet.extract_collocation_points()
  dataSet.save()
  dataSet.summary()
else:
  dataSet.restore(args.file)

# Ensure correct output size at end of input architecture
args.architecture.append(nDim)

### Create training and validation
nSamples = dataSet.get_num_bc() + dataSet.get_num_col()
nValid   = int(nSamples * 0.1)
nTrain   = nSamples - nValid

print('{} samples in training, {} in validation'.format(nTrain, nValid))
assert nValid > 0, 'Number of validation samples must be greater than 0'

normalizeXyt = True
onlyFluid = not args.plotDomain

### Generators
trainGen = dataSet.generate_trainval_pts(0, nTrain, normalizeXyt=normalizeXyt, batchSize=args.batchSize)
validGen = dataSet.generate_trainval_pts(nTrain, nSamples, normalizeXyt=normalizeXyt, batchSize=args.batchSize)

### Create model
modelName = args.name + archStr + '_c{}'.format(args.nColPoint)

#with BM.strategy.scope():
bubbleNet = BM.BubblePINN(width=args.architecture, reg=args.reg, alpha=args.alpha, beta=args.beta)
bubbleNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr0))

bubbleNet.preview()

### Callbacks
nsCB = [keras.callbacks.ModelCheckpoint(filepath='./'+modelName+'/checkpoint', \
                                        monitor='val_loss', save_best_only=True,\
                                        verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, min_delta=0.01,\
                                          patience=args.patience, min_lr=args.lrmin),
        keras.callbacks.CSVLogger(modelName+'.log', append=True),
        keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)]

### Load checkpoints if restart
if args.restart:
  ckpntName = modelName if args.checkpoint == None else args.checkpoint
  bubbleNet.load_weights(tf.train.latest_checkpoint(ckpntName))
  if args.restartLr != None:
    keras.backend.set_value(bubbleNet.optimizer.learning_rate, args.restartLr)

### Training
bubbleNet.fit(trainGen, validation_data=validGen,
          initial_epoch=args.initTrain, epochs=args.nEpoch,
          steps_per_epoch=nTrain//args.batchSize,
          validation_steps=nValid//args.batchSize,
          verbose=2, callbacks=nsCB)
#bubbleNet.summary()

### Loss plot
losses = pd.DataFrame(bubbleNet.history.history)
fig = losses.plot().get_figure()
Util.save_figure_dataframe(losses, fig, 'stats', 'losses')

### Prediction

assert args.predictionFrames <= args.totalFrames, 'Cannot use more prediction frames than total frames'
assert args.predictionFrames > 0, 'Must use at least 1 prediction frame'

size    = dataSet.get_size()

predStart, predEnd = 0, args.predictionFrames
predVels = None
predGen = dataSet.generate_predict_pts(predStart, predEnd, onlyFluid=onlyFluid)
predVels = bubbleNet.predict(predGen)

nExpectedPred = dataSet.get_num_fluid(0, args.predictionFrames) if onlyFluid else dataSet.get_num_cells() * args.predictionFrames
assert len(predVels) == nExpectedPred, 'Prediction cell count mismatch, expected {} vs {}'.format(nExpectedPred, len(predVels))

### Plots

predVelOffset = 0 # just a helper var to find beginning of next frame in predVels array
for frame in range(predStart, predEnd):

  xyBc    = dataSet.get_xy_bc(frame)
  xyFluid = dataSet.get_xy_fluid(frame)
  bc      = dataSet.get_bc(frame)

  ### Original vel plots

  arrow_res = 2
  x = np.arange(0, int(size[0]), arrow_res)
  y = np.arange(0, int(size[1]), arrow_res)
  X, Y = np.meshgrid(x, y, indexing='xy')

  origvelGrid = np.zeros((size[0], size[1], nDim))
  origvelMag = np.zeros((size))
  for pos, vel in zip(xyBc, bc):
    x, y = int(pos[0]), int(pos[1])
    origvelMag[x, y] = np.sqrt(vel[0]**2 + vel[1]**2)
    origvelGrid[x, y, :] = vel

  if Util.PRINT_DEBUG:
    print('origvel max u {}, max v {}'.format(max( origvelGrid[:,:,0].min(), origvelGrid[:,:,0].max(), key=abs ),
                                              max( origvelGrid[:,:,1].min(), origvelGrid[:,:,1].max(), key=abs )))
  Util.save_image(src=origvelMag, subdir='origvels', name='origvels_pts', frame=frame)

  velsU = origvelGrid[::arrow_res,::arrow_res,0]
  velsV = origvelGrid[::arrow_res,::arrow_res,1]
  origvelMag = origvelMag[::arrow_res,::arrow_res]

  mask = origvelMag > 0
  velsU = velsU[mask]
  velsV = velsV[mask]
  X = X[mask]
  Y = Y[mask]
  velsMag = origvelMag[mask]

  fig, ax = plt.subplots(1, 1, figsize=(10,10))
  plot = ax.quiver(X, Y, velsU, velsV, velsMag, cmap='winter', scale_units = 'xy', angles='xy')
  ax.set_xlim(0, size[0])
  ax.set_ylim(0, size[1])
  ax.invert_yaxis()
  Util.save_figure_plot(fig, 'origvels', 'origvels', frame, colorbar=True, plot=plot, cmin=0, cmax=0.8)

  ### Prediction vel plots

  if predVels is not None:
    arrow_res = 16
    x = np.arange(0, int(size[0]), arrow_res)
    y = np.arange(0, int(size[1]), arrow_res)
    X, Y = np.meshgrid(x, y, indexing='xy')

    predvelGrid = np.zeros((size[0], size[1], nDim))
    predvelMag = np.zeros((size))

    gridXy = [(x,y,frame) for x in range(size[0]) for y in range(size[1])]
    loopArray = xyFluid if onlyFluid else gridXy
    cnt = 0
    for xi, yi, _ in loopArray:
      xi, yi = int(xi), int(yi)
      vel = predVels[cnt + predVelOffset, :]
      predvelMag[xi, yi] = np.sqrt(vel[0]**2 + vel[1]**2)
      predvelGrid[xi, yi, :] = vel
      cnt += 1

    if onlyFluid:
      nExpectedFluid = dataSet.get_num_fluid(frame, frame+1)
      assert cnt == nExpectedFluid, 'Fluid cell count mismatch, expected {} vs {}'.format(nExpectedFluid, cnt)

    # Start of the predvels of next frame
    predVelOffset += cnt

    if Util.PRINT_DEBUG:
      print('predvel max u {}, max v {}'.format(max( predvelGrid[:,:,0].min(), predvelGrid[:,:,0].max(), key=abs ),
                                                max( predvelGrid[:,:,1].min(), predvelGrid[:,:,1].max(), key=abs )))
    Util.save_image(src=predvelMag, subdir='predvels', name='predvels_pts', frame=frame)

    predVelsU = predvelGrid[::arrow_res, ::arrow_res, 0]
    predVelsV = predvelGrid[::arrow_res, ::arrow_res, 1]
    predvelMag = predvelMag[::arrow_res, ::arrow_res]

    mask = predvelMag > 0.0

    predVelsU = predVelsU[mask]
    predVelsV = predVelsV[mask]
    X = X[mask]
    Y = Y[mask]
    predVelsMag = predvelMag[mask]

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    plot = ax.quiver(X, Y, predVelsU, predVelsV, predVelsMag, cmap='winter', scale_units = 'xy', angles='xy')
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.invert_yaxis()
    Util.save_figure_plot(fig, 'predvels', 'predvels', frame, colorbar=True, plot=plot, cmin=0, cmax=0.4)

    # Plot with matplotlib streamplot
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    x = np.arange(0, int(size[0]), 1)
    y = np.arange(0, int(size[1]), 1)
    X, Y = np.meshgrid(x, y, indexing='xy')
    predVelsU = predvelGrid[:,:,0]
    predVelsV = predvelGrid[:,:,1]
    plot = ax.streamplot(X, Y, predVelsU, predVelsV, density=2.5, linewidth=1.0)#, color='#A23BEC')
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.invert_yaxis()
    Util.save_figure_plot(fig, 'predvels', 'predvels_stream', frame)

# Optional video export with ffmpeg
if args.exportVideo:
  Util.save_video(subdir='origvels', name='origvels', imgDir='../img/origvels/', fps=15)
  Util.save_video(subdir='predvels', name='predvels', imgDir='../img/predvels/', fps=15)
  Util.save_video(subdir='origvels', name='origvels_pts', imgDir='../img/origvels/', fps=15)
  Util.save_video(subdir='predvels', name='predvels_pts', imgDir='../img/predvels/', fps=15)
  Util.save_video(subdir='predvels', name='predvels_stream', imgDir='../img/predvels/', fps=15)


