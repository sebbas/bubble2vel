#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import copy

import bdataset as BD
import bmodel as BM
import butils as UT

parser = argparse.ArgumentParser()

### Architecture
parser.add_argument('-l', '--architecture', type=int, nargs='*',
                    default=[150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
                    help='size of each hidden layer')

# Regularizer coefficients
parser.add_argument('-r', '--reg', type=float, nargs='*', default=None,
                    help='l2 regularization')

# Plotting options
parser.add_argument('-b', '--onlyBc', default=False, action='store_true',
                    help='plot prediction velocities only at data point positions')

# Model being used for predictions
parser.add_argument('-n', '--name', default=None,
                    help='name of model that is used to make predictions')
parser.add_argument('-p', '--predFrames', type=int, default=100,
                    help='number of frames to predict')
parser.add_argument('-s', '--startFrame', type=int, default=0,
                    help='first frame to predict')
parser.add_argument('-e', '--endFrame', type=int, default=100,
                    help='last frame to predict')
parser.add_argument('-f', '--file', default='../data/bdata.h5',
                    help='the file(s) containing flow velocity training data')

args = parser.parse_args()

assert args.name is not None, 'Must supply model name'
assert args.startFrame <= args.endFrame, 'Start frame must be smaller/equal than/to end frame'

# Ensure correct output size at end of input architecture
args.architecture.append(UT.nDim + 1)

dataSet = BD.BubbleDataSet()
dataSet.restore(args.file)

bubbleNet = BM.BubblePINN(width=args.architecture, reg=args.reg)
bubbleNet.load_weights(tf.train.latest_checkpoint(args.name)).expect_partial()

size                   = dataSet.get_size()
source                 = dataSet.get_source()
sourceName             = dataSet.get_source_name()
numPredFrames          = args.predFrames
startFrame, endFrame   = args.startFrame, args.endFrame
cmin, cmax             = 0.0, 15.0
relErrULst, relErrVLst = [], []

assert source in [UT.SRC_FLOWNET, UT.SRC_FLASHX], 'Invalid dataset source'
if source == UT.SRC_FLOWNET:
  worldSize, imageSize, fps, V, L, T = UT.worldSize, UT.imageSize, UT.fps, UT.V, UT.L, UT.T
if source == UT.SRC_FLASHX:
  worldSize, imageSize, fps, V, L, T = UT.worldSize_fx, UT.imageSize_fx, UT.fps_fx, UT.V_fx, UT.L_fx, UT.T_fx

# Generators
predGen = dataSet.generate_predict_pts(0, numPredFrames, worldSize, imageSize, fps, L, T, onlyBc=args.onlyBc)
uvpPred = bubbleNet.predict(predGen)
predVels = uvpPred[:, :UT.nDim]
predP = copy.deepcopy(uvpPred[:, 2])

# Convert predvels from dimensionless to world to domain space
if source == UT.SRC_FLOWNET:
  print('--- Dimensionless pred vels ---')
  print(predVels)
  print('Min / max of predVel')
  print('  u  {}, {}'.format(np.min(predVels[:,0]), np.max(predVels[:,0])))
  print('  v  {}, {}'.format(np.min(predVels[:,1]), np.max(predVels[:,1])))
  print('--- World pred vels ---')
  predVels = UT.vel_dimensionless_to_world(predVels, UT.V)
  print(predVels)
  print('Min / max of predVel')
  print('  u  {}, {}'.format(np.min(predVels[:,0]), np.max(predVels[:,0])))
  print('  v  {}, {}'.format(np.min(predVels[:,1]), np.max(predVels[:,1])))
  print('--- Domain pred vels ---')
  predVels = UT.vel_world_to_domain(predVels, UT.worldSize, UT.fps)
  print(predVels)
  print('Min / max of predVel')
  print('  u  {}, {}'.format(np.min(predVels[:,0]), np.max(predVels[:,0])))
  print('  v  {}, {}'.format(np.min(predVels[:,1]), np.max(predVels[:,1])))

# Plots

predVelOffset = 0 # just a helper var to find beginning of next frame in predVels array
for f in range(0, numPredFrames):

  frame = f + startFrame
  ### Original vel plots

  plotOrig = True
  if plotOrig:
    xyBc     = dataSet.get_xy_bc(f)
    xyFluid  = dataSet.get_xy_fluid(f)
    uvp      = dataSet.get_bc(f)
    uvpFluid = dataSet.get_uvp_fluid(f)

    arrow_res = 2 if args.onlyBc else 8

    origvelGrid = np.zeros((size[0], size[1], UT.nDim + 1))
    origvelMag  = np.zeros((size))

    '''
    # Plot everything
    xyOrig = np.concatenate((xyBc, xyFluid))
    uvpOrig = np.concatenate((uvp, uvpFluid))
    for xyt, uvp in zip(xyOrig, uvpOrig):
      xi, yi = int(xyt[0]), int(xyt[1])
      origvelMag[yi, xi] = np.sqrt(np.square(uvp[0]) + np.square(uvp[1]))
      origvelGrid[yi, xi] = uvp[:]
    UT.save_plot(origvelGrid[:,:,2], '{}/origvels/plots'.format(sourceName), 'origphi_plt', frame, size=size, cmin=-1.0, cmax=1.0, cmap='jet')
    '''

    xyOrig = xyBc if args.onlyBc else xyFluid
    uvpOrig = uvp if args.onlyBc else uvpFluid

    # Only use points within boundaries
    mask = dataSet.get_wall_mask(xyOrig, useAll=True)
    xyMasked = xyOrig[mask]
    uvpMasked = uvpOrig[mask]
    for xyt, uvp in zip(xyMasked, uvpMasked):
      xi, yi = int(xyt[0]), int(xyt[1])
      origvelMag[yi, xi] = np.sqrt(np.square(uvp[0]) + np.square(uvp[1]))
      origvelGrid[yi, xi] = uvp[:]

    UT.save_image(origvelGrid[:,:,0], '{}/origvels/raw'.format(sourceName), 'origvelsx_raw', frame, cmap='jet')
    UT.save_image(origvelGrid[:,:,1], '{}/origvels/raw'.format(sourceName), 'origvelsy_raw', frame, cmap='jet')
    UT.save_image(origvelMag, '{}/origvels/raw'.format(sourceName), 'origvelsmag_raw', frame)

    if UT.PRINT_DEBUG:
      print('origvel u [{}, {}], v [{}, {}]'.format( \
            np.min(origvelGrid[:,:,0]), np.max(origvelGrid[:,:,0]), \
            np.min(origvelGrid[:,:,1]), np.max(origvelGrid[:,:,1])))

    UT.save_plot(origvelMag, '{}/origvels/plots'.format(sourceName), 'origvelsmag_plt', frame, size=size, cmin=cmin, cmax=cmax)
    UT.save_plot(origvelGrid[:,:,2], '{}/origvels/plots'.format(sourceName), 'origphi_plt', frame, size=size, cmin=-1.0, cmax=1.0, cmap='jet')

    UT.save_velocity(origvelGrid, '{}/origvels/plots'.format(sourceName), 'velocity_stream', frame, size=size, type='stream')
    UT.save_velocity(origvelGrid, '{}/origvels/plots'.format(sourceName), 'velocity_vector', frame, size=size, type='quiver', arrow_res=arrow_res, cmin=cmin, cmax=cmax, filterZero=True)

  ### Prediction vel plots

  if predVels is not None:
    xyBc    = dataSet.get_xy_bc(f)
    xyFluid = dataSet.get_xy_fluid(f)

    arrow_res = 2 if args.onlyBc else 8

    predvelGrid = np.zeros((size[0], size[1], UT.nDim))
    predvelMag  = np.zeros((size))
    predPGrid   = np.zeros((size))

    xyPred = xyBc if args.onlyBc else xyFluid

    # Only use points within boundaries
    mask = dataSet.get_wall_mask(xyPred, useAll=True)
    xyMasked = xyPred[mask]

    cnt = 0
    for xyt in xyMasked:
      xi, yi = int(xyt[0]), int(xyt[1])
      i, j = xi, yi
      vel = predVels[cnt + predVelOffset, :UT.nDim]
      predvelMag[yi, xi] = np.sqrt(np.square(vel[0]) + np.square(vel[1]))
      predvelGrid[yi, xi] = vel
      predPGrid[yi, xi] = predP[cnt + predVelOffset]
      cnt += 1

    # Start of the predvels of next frame
    predVelOffset += cnt

    # Relative error for this frame
    if 0:
      epsMin, epsMax = 0.001, 50.0
      relErrU, residualsU = UT.compute_relative_error(origvelGrid[:,:,0], predvelGrid[:,:,0], xyMasked, epsMin, epsMax)
      relErrV, residualsV = UT.compute_relative_error(origvelGrid[:,:,1], predvelGrid[:,:,1], xyMasked, epsMin, epsMax)
      print('Frame {}: relErrU {}, residualsU: {}'.format(frame, relErrU, residualsU))
      print('Frame {}: relErrV {}, residualsV: {}'.format(frame, relErrV, residualsV))

      relErrULst.append(relErrU)
      relErrVLst.append(relErrV)

    UT.save_image(predvelMag, '{}/predvels/raw'.format(sourceName), 'predvels_raw', frame)
    UT.save_image(predPGrid, '{}/predvels/raw'.format(sourceName), 'predp_raw', frame)

    if UT.PRINT_DEBUG:
      print('predvel u [{}, {}], v [{}, {}]'.format( \
            np.min(predvelGrid[:,:,0]), np.max(predvelGrid[:,:,0]), \
            np.min(predvelGrid[:,:,1]), np.max(predvelGrid[:,:,1])))
      print('pred pressure [{}, {}]'.format(np.min(predPGrid[:,:]), np.max(predPGrid[:,:])))

    UT.save_plot(predvelMag, '{}/predvels/plots'.format(sourceName), 'predvelsmag_plt', frame, size=size, cmin=cmin, cmax=cmax)
    UT.save_plot(predPGrid, '{}/predvels/plots'.format(sourceName), 'predp_plt', frame, size=size, cmin=cmin, cmax=cmax)

    UT.save_velocity(predvelGrid, '{}/predvels/plots'.format(sourceName), 'velocity_stream', frame, size=size, type='stream')
    UT.save_velocity(predvelGrid, '{}/predvels/plots'.format(sourceName), 'velocity_vector', frame, size=size, type='quiver', arrow_res=arrow_res, filterZero=True)

    if UT.FILE_IO:
      # Save predictions
      dictSave = {'velx': predvelGrid[:,:,0], 'vely': predvelGrid[:,:,1]}
      UT.save_array_hdf5(arrays=dictSave, filePrefix='Pred_data_INS_Pool_Boiling', size=size, frame=frame)

      '''
      # Data can later be read and plotted like this:

      # Read predictions
      velsP = np.zeros((size[0], size[1], UT.nDim))
      dictRead = {'velx': velsP[:,:,0], 'vely': velsP[:,:,1]}
      UT.read_array_hdf5(arrays=dictRead, fname='../data/predVels_384_{:04}.h5'.format(frame))

      # Plot predictions that were just read
      UT.save_image(velsP[:,:,0], '../data', 'velPx_read', frame)
      UT.save_velocity(velsP, '../data', 'velP_read', frame, size=size, type='quiver', arrow_res=arrow_res, cmin=cmin, cmax=cmax, filterZero=False)
      UT.save_velocity(velsP, '../data', 'velP_read', frame, size=size, type='mag')
      '''

#print('Mean residual U: {}'.format(np.mean(relErrULst)))
#print('Mean residual V: {}'.format(np.mean(relErrVLst)))


# Optional video export with ffmpeg
if UT.VID_DEBUG:
  fps = 5
  UT.save_video(subdir=origVelDir, name='origvels', imgDir='../img/origvels/', fps=fps)
  UT.save_video(subdir=predVelDir, name='predvels', imgDir='../img/predvels/', fps=fps)
  UT.save_video(subdir=origVelDir, name='origvels_pts', imgDir='../img/origvels/', fps=fps)
  UT.save_video(subdir=predVelDir, name='predvels_pts', imgDir='../img/predvels/', fps=fps)
  UT.save_video(subdir=predVelDir, name='predvels_stream', imgDir='../img/predvels/', fps=fps)

