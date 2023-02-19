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
import butils as UT

keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
# Architecture
parser.add_argument('-l', '--architecture', type=int, nargs='*',
                    default=[150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
                    help='size of each hidden layer')
parser.add_argument('-w', '--walls', type=int, nargs=4, default=[0,0,1,0],
                    help='domain walls, [top, right, bottom, left]')

# Regularizer coefficients
parser.add_argument('-r', '--reg', type=float, nargs='*', default=None,
                    help='l2 regularization')
parser.add_argument('-alpha', '--alpha', type=float, nargs=3, default=[1.0, 1.0, 0.0],
                    help='coefficients for data loss')
parser.add_argument('-beta', '--beta', type=float, nargs=3, default=[1e-2, 1e-2, 1e-2],
                    help='coefficients for pde residual')
parser.add_argument('-gamma', '--gamma', type=float, nargs=3, default=[1e-4, 1e-4, 1e-4],
                    help='coefficients for domain wall loss')

# Epochs, checkpoints
parser.add_argument('-f', '--file', default='../data/bdata_512_56389.h5',
                    help='file containing training and validation points')
parser.add_argument('-n', '--name', default=None, \
                    help='use custom model name, by default one will be generated based on params')
parser.add_argument('-ie', '--initTrain', type=int, default=0,
                    help='initial train epochs')
parser.add_argument('-e', '--nEpoch', default=10000, type=int, help='epochs')
parser.add_argument('-bs', '--batchSize', type=int, default=64,
                    help='batch size in training')
parser.add_argument('-restart', '--restart', default=False, action='store_true',
                    help='restart from checkpoint')
parser.add_argument('-ckpnt', '--checkpoint', default=None, help='checkpoint name')

# Learning rate
parser.add_argument('-lr0', '--lr0', type=float, default=5e-4, help='init learning rate')
parser.add_argument('-lrmin', '--lrmin', type=float, default=1e-7, help='min learning rate')
parser.add_argument('-p', '--patience', type=int, default=200,
                    help='patience for reducing learning rate')
parser.add_argument('-lr',  '--restartLr', type=float, default=None,
                     help='learning rate to restart training')

# Numbers of data, collocation and wall points
parser.add_argument('-b', '--nWallPnt', type=int, default=-1,
                    help='number of boundary points in training, use all by default')
parser.add_argument('-c', '--nColPoint', type=int, default=-1,
                    help='number of collocation points in training')
parser.add_argument('-d', '--nDataPoint', type=int, default=-1,
                    help='number of data points in training, use all by default')

# Save more info
parser.add_argument('-g', '--saveGradStat', default=False, action='store_true',
                    help='save gradient statistics')

parser.add_argument('-src', '--source', type=int, default=0,
                    help='type of training data source, either 1: experiment or 2: simulation')
parser.add_argument('-rt', '--resetTime', default=False, action='store_true',
                    help='start dataset time at zero')

args = parser.parse_args()

archStr = UT.get_arch_string(args.architecture)

# Restore dataset with data from generated .h5 file
assert args.file.endswith('.h5')
dataSet = BD.BubbleDataSet(wallPoints=args.nWallPnt, \
                           colPoints=args.nColPoint, dataPoints=args.nDataPoint)
dataSet.restore(args.file)
dataSet.prepare_batch_arrays(zeroInitialCollocation=True, resetTime=args.resetTime, zeroMean=0)

# Ensure correct output size at end of input architecture
args.architecture.append(UT.nDim + 1)

# Create training and validation (data + collocation points)
nSamples = dataSet.get_num_data_pts() + dataSet.get_num_col_pts()
nTrain   = int(nSamples * 0.85)
# Ensure training samples fit evenly
nTrain   = args.batchSize * round(nTrain / args.batchSize)
nValid   = nSamples - nTrain

print('{} data / collocation points in training, {} in validation'.format(nTrain, nValid))

assert args.source in [UT.SRC_FLOWNET, UT.SRC_FLASHX], 'Invalid dataset source'
if args.source == UT.SRC_FLOWNET:
  worldSize, imageSize, fps, V, L, T = UT.worldSize, UT.imageSize, UT.fps, UT.V, UT.L, UT.T
if args.source == UT.SRC_FLASHX:
  worldSize, imageSize, fps, V, L, T = UT.worldSize_fx, UT.imageSize_fx, UT.fps_fx, UT.V_fx, UT.L_fx, UT.T_fx

# Generators
trainGen = dataSet.generate_train_valid_batch(0, nTrain, \
                                              worldSize, imageSize, fps, V, L, T, \
                                              batchSize=args.batchSize)
validGen = dataSet.generate_train_valid_batch(nTrain, nSamples, \
                                              worldSize, imageSize, fps, V, L, T, \
                                              batchSize=args.batchSize)

# Create model
nameStr, paramsStr = args.name, ''
if args.name is None:
  nameStr = UT.MODEL_NAME
  paramsStr = '_d{}_c{}_a{}_b{}_g{}_lr{}_p{}'.format( \
    dataSet.get_num_data_pts(), dataSet.get_num_col_pts(), \
    UT.get_list_string(args.alpha, delim='-'), \
    UT.get_list_string(args.beta, delim='-'), \
    UT.get_list_string(args.gamma, delim='-'), args.lr0, args.patience)
modelName = nameStr + archStr + paramsStr

#with BM.strategy.scope():
bubbleNet = BM.BModel(width=args.architecture, reg=args.reg,
                          alpha=args.alpha, beta=args.beta, gamma=args.gamma, \
                          Re=UT.get_reynolds_number(args.source))
bubbleNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr0))

bubbleNet.preview()

# Callbacks
bubbleCB = [keras.callbacks.ModelCheckpoint(filepath='./' + modelName + '/checkpoint', \
                                            monitor='val_loss', save_best_only=True,\
                                            save_weights_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.01,\
                                              patience=args.patience, min_lr=args.lrmin),
            keras.callbacks.CSVLogger(modelName+'.log', append=True)]

# Load checkpoints if restart
if args.restart:
  ckpntName = modelName if args.checkpoint == None else args.checkpoint
  bubbleNet.load_weights(tf.train.latest_checkpoint(ckpntName))
  if args.restartLr != None:
    keras.backend.set_value(bubbleNet.optimizer.learning_rate, args.restartLr)

# Training
bubbleNet.fit(
      trainGen,
      initial_epoch=args.initTrain,
      epochs=args.nEpoch,
      steps_per_epoch=nTrain//args.batchSize,
      validation_data=validGen,
      validation_steps=nValid//args.batchSize,
      verbose=True,
      callbacks=bubbleCB)
bubbleNet.summary()

# Loss plot
#losses = pd.DataFrame(bubbleNet.history.history)
#fig = losses.plot().get_figure()
#UT.save_figure_dataframe(losses, fig, 'stats', 'losses')


