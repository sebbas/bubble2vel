#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

np.random.seed(2022)
tf.random.set_seed(2022)

from tensorflow import keras
from tensorflow.keras.layers import Dense


class DenseLayers(keras.layers.Layer):
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

      self.layers.append(keras.layers.Dense(w, activation=activation,
                         kernel_regularizer=keras.regularizers.l2(reg[i]),
                         name=prefix+repr(i)))


  def call(self, inputs):
    dense = inputs
    for layer in self.layers:
      dense = layer(dense)
    return dense
