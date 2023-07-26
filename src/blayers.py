#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

np.random.seed(2022)
tf.random.set_seed(2022)

import tensorflow.keras.layers as KL
import tensorflow.keras.regularizers as KR


class DenseLayers(KL.Layer):
  def __init__(self, width=[64,64,64], act='tanh', prefix='dense_',
               reg=None, last_linear=False, **kwargs):
    super(DenseLayers, self).__init__(**kwargs)
    assert len(width) > 0

    # Dense layer generation loop below expects a list of regularizers
    if reg != None:
      if len(reg) == 1:
          tmp = reg[0]
          reg = np.zeros(len(width))
          reg[:] = tmp
      else:
          assert len(reg) == len(width)
    else:
      reg = np.zeros(len(width))

    # Create list of dense layers
    self.layers = []
    for i, w in enumerate(width):
      is_last = (i == len(width)-1)
      activation = 'linear' if is_last and last_linear else act

      if not is_last:
        self.layers.append(KL.Dense(w, activation=activation,
                           kernel_regularizer=KR.l2(reg[i]),
                           name=prefix+repr(i)))
      else:
        self.layers.append(KL.Dense(w, activation=activation,
                           use_bias=False, kernel_regularizer=KR.l2(reg[i]),
                           name=prefix+repr(i)))


  def call(self, inputs):
    nLayers = len(self.layers)

    # Normalize inputs and then shift to [-1, 1] range
    xNorm = (inputs[:,0:1] / UT.dimlessMax) * 2 - 1
    yNorm = (inputs[:,1:2] / UT.dimlessMax) * 2 - 1
    t = inputs[:,2:3] / UT.timeMax
    xyt = KL.Concatenate(axis=-1)([xNorm, yNorm, t])

    uvp = self.layers[0](xyt)

    for i in range(1, nLayers-1):
      dense = self.layers[i]
      uvp = dense(uvp)

    return self.layers[nLayers-1](uvp)
