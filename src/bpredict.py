#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

import bdataset as BD
import bmodel as BM
import butils as Util

parser = argparse.ArgumentParser()

### Architecture
parser.add_argument('-l', '--architecture', type=int, nargs='*',
                    default=[100, 100, 100, 100, 50, 50, 50, 50],
                    help='size of each hidden layer')

# Regularizer coefficients
parser.add_argument('-r', '--reg', type=float, nargs='*', default=None,
                    help='l2 regularization')

parser.add_argument('-f', '--file', default='../data/bdata_512_56389.h5',
                    help='the file(s) containing flow velocity training data')
parser.add_argument('-p', '--predictionFrames', type=int, default=10,
                    help='predict all frames from start frame up to this frame')

# Plotting options
parser.add_argument('-pd', '--plotDomain', default=False, action='store_true',
                    help='plot prediction velocities for the entire domain')
parser.add_argument('-ev', '--exportVideo', default=False, action='store_true',
                    help='export image data to video (using ffmpeg)')

# Model being used for predictions
parser.add_argument('-n', '--name', default='bubble2vel-100x4-50x4_c800000',
                    help='name of model that is used to make predictions')

args = parser.parse_args()

nDim = 2
onlyFluid = not args.plotDomain

# Ensure correct output size at end of input architecture
args.architecture.append(nDim + 1)

dataSet = BD.BubbleDataSet()
dataSet.restore(args.file)

assert args.predictionFrames <= dataSet.get_num_frames(), 'Cannot use more prediction frames than total frames'
assert args.predictionFrames > 0, 'Must use at least 1 prediction frame'

bubbleNet = BM.BubblePINN(width=args.architecture, reg=args.reg)
bubbleNet.load_weights(tf.train.latest_checkpoint(args.name)).expect_partial()

size = dataSet.get_size()

predStart, predEnd = 0, args.predictionFrames
predVels = None
predGen = dataSet.generate_predict_pts(predStart, predEnd, onlyFluid=onlyFluid)
predVels = bubbleNet.predict(predGen)

print(predVels.shape)

nExpectedPred = dataSet.get_num_fluid(0, args.predictionFrames) if onlyFluid else dataSet.get_num_cells() * args.predictionFrames
assert len(predVels) == nExpectedPred, 'Prediction cell count mismatch, expected {} vs {}'.format(nExpectedPred, len(predVels))

# Plots

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
    predPGrid = np.zeros((size))

    gridXy = [(x,y,frame) for x in range(size[0]) for y in range(size[1])]
    loopArray = xyFluid if onlyFluid else gridXy
    cnt = 0
    for xi, yi, _ in loopArray:
      xi, yi = int(xi), int(yi)
      vel = predVels[cnt + predVelOffset, :nDim]
      predvelMag[xi, yi] = np.sqrt(vel[0]**2 + vel[1]**2)
      predvelGrid[xi, yi, :nDim] = vel
      p = predVels[cnt + predVelOffset, nDim]
      predPGrid[xi, yi] = p
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
    Util.save_image(src=predPGrid, subdir='predvels', name='predp_pts', frame=frame)

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

