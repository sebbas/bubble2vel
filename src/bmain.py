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

# Learning rate
parser.add_argument('-lr0', '--lr0', type=float, default=5e-4, help='init learning rate')
parser.add_argument('-lrmin', '--lrmin', type=float, default=1e-7, help='min learning rate')
parser.add_argument('-p', '--patience', type=int, default=200,
                    help='patience for reducing learning rate')
parser.add_argument('-lr',  '--restartLr', type=float, default=None,
                     help='learning rate to restart training')

# Save more info
parser.add_argument('-g', '--saveGradStat', default=False, action='store_true',
                    help='save gradient statistics')
parser.add_argument('-ci', '--cachedInput', default=False, action='store_true',
                    help='extract boundary values from cached numpy arrays')

args = parser.parse_args()

nDim = 2
archStr = Util.get_arch_string(args.architecture)

# Load external data
dataSet = BD.BubbleDataSet(args.file, args.totalFrames, args.nDataPoint, args.nBcPoint, args.nColPoint, dim=nDim)

if not args.cachedInput:
  if not dataSet.load_data():
    sys.exit()
  dataSet.extract_domain_bc(walls=[0,0,1,0])
  dataSet.extract_bubble_bc(velEps=0.01)
  dataSet.extract_collocation_points()
  dataSet.save()
  dataSet.summary()
else:
  dataSet.restore(args.file)
dataSet.combine_data_colloc_points()

# Ensure correct output size at end of input architecture
args.architecture.append(nDim)

### Create training and validation
nSamples = dataSet.get_num_bc() + dataSet.get_num_col()
nValid   = int(nSamples * 0.1)
nTrain   = nSamples - nValid

print('{} samples in training, {} in validation'.format(nTrain, nValid))
assert nValid > 0, 'Number of validation samples must be greater than 0'

normalizeXyt = True

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
bubbleCB = [keras.callbacks.ModelCheckpoint(filepath='./' + modelName + '/checkpoint', \
                                            monitor='val_loss', save_best_only=True,\
                                            save_weights_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.01,\
                                              patience=args.patience, min_lr=args.lrmin),
            keras.callbacks.CSVLogger(modelName+'.log', append=True)]

### Load checkpoints if restart
if args.restart:
  ckpntName = modelName if args.checkpoint == None else args.checkpoint
  bubbleNet.load_weights(tf.train.latest_checkpoint(ckpntName))
  if args.restartLr != None:
    keras.backend.set_value(bubbleNet.optimizer.learning_rate, args.restartLr)

### Training
bubbleNet.fit(
      trainGen,
      initial_epoch=args.initTrain,
      epochs=args.nEpoch,
      steps_per_epoch=nTrain//args.batchSize,
      validation_data=validGen,
      validation_steps=nValid//args.batchSize,
      verbose=True,
      callbacks=bubbleCB)
#bubbleNet.summary()

### Loss plot
losses = pd.DataFrame(bubbleNet.history.history)
fig = losses.plot().get_figure()
Util.save_figure_dataframe(losses, fig, 'stats', 'losses')


