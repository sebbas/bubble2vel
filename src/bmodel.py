#!/usr/bin/env python3

import h5py as h5
import tensorflow as tf
import numpy as np

np.random.seed(2022)
tf.random.set_seed(2022)

from tensorflow import keras
from blayers import *
import butils as Util


''' --- TODO ---
strategy = tf.distribute.MirroredStrategy()
'''

class BubblePINN(keras.Model):
  def __init__(self, width=[256, 256, 256, 128, 128, 128, 64, 32, 2],\
               alpha = [1.0, 1.0, 1.0], beta = [1e-4, 1e-4, 1e-4], gamma = [1.0, 1.0],\
               reg=None, saveGradStat=False, **kwargs):
    super(BubblePINN, self).__init__(**kwargs)
    self.width = width
    self.reg   = reg
    if reg == None:
      self.mlp = DenseLayers(width=width, prefix='bc', last_linear=True)
    else:
      self.mlp = DenseLayers(width=width, reg=reg, prefix='bc', last_linear=True)
    # Coefficient for data and pde loss
    self.alpha = alpha
    self.beta  = beta
    self.gamma = gamma

    # ---- dicts for metrics and statistics ---- #
    # Save gradients' statistics per layer
    self.saveGradStat = saveGradStat
    # Create metrics
    self.trainMetrics = {}
    self.validMetrics = {}
    # add metrics
    names = ['loss', 'uMse', 'vMse', 'pde0', 'pde1', 'pde2', 'uMae', 'vMae', 'uMseWalls', 'vMseWalls']
    for key in names:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)
    ## add metrics for layers' weights, if save_grad_stat is required
    ## i even for weights, odd for bias
    if self.saveGradStat:
      for i in range(len(width)):
        for prefix in ['u_', 'v_', 'pde0_', 'pde1_', 'pde2_', 'uWalls_', 'vWalls_']:
          for suffix in ['w_avg', 'w_std', 'b_avg', 'b_std']:
            key = prefix + repr(i) + suffix
            self.trainMetrics[key] = keras.metrics.Mean(name='train '+key)
    # Statistics
    self.trainStat = {}
    self.validStat = {}


  def call(self, inputs):
    '''
    inputs: [xy, t, w]
    '''
    xyt = tf.concat([inputs[0], inputs[1]], axis=-1)
    return self.mlp(xyt)


  def record_layer_gradient(self, grads, baseName):
    '''
    record the average and standard deviation of each layer's
    weights and biases
    '''
    for i, g in enumerate(grads):
      if g != None:
        l = i // 2
        parameter = 'w' if i%2==0 else 'b'
        prefix = baseName + '_{:d}{}_'.format(l,parameter)
        gAbs = tf.abs(g)
        gAvg = tf.reduce_mean(gAbs)
        gStd = tf.reduce_mean(tf.square(gAbs - gAvg))
        self.trainMetrics[prefix+'avg'].update_state(gAvg)
        self.trainMetrics[prefix+'std'].update_state(gStd)


  def compute_losses(self, xy, t, w, uv, xyWalls, tWalls, uvWalls):
    # Track computation for 2nd derivatives for u, v
    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(xy)
      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(xy)
        tape1.watch(t)
        uvPred = self([xy, t])
        uPred  = uvPred[:,0]
        vPred  = uvPred[:,1]
        pPred  = uvPred[:,2]
      # 1st order derivatives
      u_grad = tape1.gradient(uPred, xy)
      v_grad = tape1.gradient(vPred, xy)
      p_grad = tape1.gradient(pPred, xy)
      u_x, u_y = u_grad[:,0], u_grad[:,1]
      v_x, v_y = v_grad[:,0], v_grad[:,1]
      p_x, p_y = p_grad[:,0], p_grad[:,1]
      u_t = tape1.gradient(uPred,  t)
      v_t = tape1.gradient(vPred,  t)
      del tape1
    # 2nd order derivatives
    u_xx = tape2.gradient(u_x, xy)[:,0]
    u_yy = tape2.gradient(u_y, xy)[:,1]
    v_xx = tape2.gradient(v_x, xy)[:,0]
    v_yy = tape2.gradient(v_y, xy)[:,1]
    del tape2

    # Compute data loss
    w = tf.squeeze(w)
    w.set_shape([None])
    uMse = keras.losses.mean_squared_error(tf.boolean_mask(uv[:,0],w), tf.boolean_mask(uvPred[:,0],w))
    vMse = keras.losses.mean_squared_error(tf.boolean_mask(uv[:,1],w), tf.boolean_mask(uvPred[:,1],w))

    # Compute pde loss, 0 continuity, 1-2 NS
    ww      = 1.0 - w
    nu      = 0.002   # kinematic viscosity
    rho     = 0.96333 # density at 200F
    pde0    = tf.square(u_x + v_y)
    pde1    = tf.square(u_t + uPred*u_x + vPred*u_y + p_x*(1/rho) - (u_xx + u_yy)*nu)
    pde2    = tf.square(v_t + uPred*v_x + vPred*v_y + p_y*(1/rho) - (v_xx + v_yy)*nu)
    ww.set_shape([None])
    pdeMse0 = keras.losses.mean_squared_error(0, tf.boolean_mask(pde0,ww))
    pdeMse1 = keras.losses.mean_squared_error(0, tf.boolean_mask(pde1,ww))
    pdeMse2 = keras.losses.mean_squared_error(0, tf.boolean_mask(pde2,ww))

    # Compute domain wall loss
    uvWallsPred = self([xyWalls, tWalls])
    uMseWalls = keras.losses.mean_squared_error(uvWalls[:,0], uvWallsPred[:,0])
    vMseWalls = keras.losses.mean_squared_error(uvWalls[:,1], uvWallsPred[:,1])

    return uvPred, uMse, vMse, pdeMse0, pdeMse1, pdeMse2, uMseWalls, vMseWalls


  def train_step(self, data):
    xy       = data[0][0]
    t        = data[0][1]
    w        = data[0][2]
    xyWalls  = data[0][3]
    tWalls   = data[0][4]
    bcDomain = data[0][5]
    uv       = data[1]

    with tf.GradientTape(persistent=True) as tape0:
      # Compute the data loss for u, v and pde losses for
      # continuity (0) and NS (1-2)
      uvPred, uMse, vMse, pdeMse0, pdeMse1, pdeMse2, uMseWalls, vMseWalls = \
        self.compute_losses(xy, t, w, uv, xyWalls, tWalls, bcDomain)
      # replica's loss, divided by global batch size
      loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse
              + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2
              + self.gamma[0]*uMseWalls + self.gamma[1]*vMseWalls )
      loss += tf.add_n(self.losses)
      loss  = loss #/ strategy.num_replicas_in_sync
    # update gradients
    if self.saveGradStat:
      uMseGrad      = tape0.gradient(uMse,      self.trainable_variables)
      vMseGrad      = tape0.gradient(vMse,      self.trainable_variables)
      pdeMse0Grad   = tape0.gradient(pdeMse0,   self.trainable_variables)
      pdeMse1Grad   = tape0.gradient(pdeMse1,   self.trainable_variables)
      pdeMse2Grad   = tape0.gradient(pdeMse2,   self.trainable_variables)
      uMseWallsGrad = tape0.gradient(uMseWalls, self.trainable_variables)
      vMseWallsGrad = tape0.gradient(vMseWalls, self.trainable_variables)
    lossGrad = tape0.gradient(loss, self.trainable_variables)
    del tape0

    # ---- Update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- Update metrics and statistics ---- #
    # Track loss and mae
    self.trainMetrics['loss'].update_state(loss)#*strategy.num_replicas_in_sync)
    self.trainMetrics['uMse'].update_state(uMse)
    self.trainMetrics['vMse'].update_state(vMse)
    self.trainMetrics['pde0'].update_state(pdeMse0)
    self.trainMetrics['pde1'].update_state(pdeMse1)
    self.trainMetrics['pde2'].update_state(pdeMse2)
    self.trainMetrics['uMseWalls'].update_state(uMseWalls)
    self.trainMetrics['vMseWalls'].update_state(vMseWalls)
    w = tf.squeeze(w)
    w.set_shape([None])
    uMae = tf.keras.metrics.mean_absolute_error(tf.boolean_mask(uv[:,0],w), tf.boolean_mask(uvPred[:,0],w))
    vMae = tf.keras.metrics.mean_absolute_error(tf.boolean_mask(uv[:,1],w), tf.boolean_mask(uvPred[:,1],w))
    self.trainMetrics['uMae'].update_state(uMae)
    self.trainMetrics['vMae'].update_state(vMae)
    # track gradients coefficients
    if self.saveGradStat:
      self.record_layer_gradient(uMseGrad, 'u_')
      self.record_layer_gradient(vMseGrad, 'v_')
      self.record_layer_gradient(pdeMse0Grad, 'pde0_')
      self.record_layer_gradient(pdeMse1Grad, 'pde1_')
      self.record_layer_gradient(pdeMse2Grad, 'pde2_')
      self.record_layer_gradient(uMseWallsGrad, 'uWalls_')
      self.record_layer_gradient(vMseWallsGrad, 'vWalls_')
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat


  def test_step(self, data):
    xy       = data[0][0]
    t        = data[0][1]
    w        = data[0][2]
    xyWalls  = data[0][3]
    tWalls   = data[0][4]
    bcDomain = data[0][5]
    uv       = data[1]

    # Compute the data and pde losses
    uvPred, uMse, vMse, pdeMse0, pdeMse1, pdeMse2, uMseWalls, vMseWalls = \
      self.compute_losses(xy, t, w, uv, xyWalls, tWalls, bcDomain)
    # replica's loss, divided by global batch size
    loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse \
            + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2 \
            + self.gamma[0]*uMseWalls + self.gamma[1]*vMseWalls )

    # Track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pde0'].update_state(pdeMse0)
    self.validMetrics['pde1'].update_state(pdeMse1)
    self.validMetrics['pde2'].update_state(pdeMse2)
    self.validMetrics['uMseWalls'].update_state(uMseWalls)
    self.validMetrics['vMseWalls'].update_state(vMseWalls)
    w = tf.squeeze(w)
    w.set_shape([None])
    uMae = tf.keras.metrics.mean_absolute_error(tf.boolean_mask(uv[:,0],w), tf.boolean_mask(uvPred[:,0],w))
    vMae = tf.keras.metrics.mean_absolute_error(tf.boolean_mask(uv[:,1],w), tf.boolean_mask(uvPred[:,1],w))
    self.validMetrics['uMae'].update_state(uMae)
    self.validMetrics['vMae'].update_state(vMae)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat


  def reset_metrics(self):
    for key in self.trainMetrics:
      self.trainMetrics[key].reset_states()
    for key in self.validMetrics:
      self.validMetrics[key].reset_states()


  @property
  def metrics(self):
    return [self.trainMetrics[key] for key in self.trainMetrics] \
         + [self.validMetrics[key] for key in self.validMetrics]


  def summary(self):
    nVar = 0
    for t in self.trainable_variables:
      print(t.name, t.shape)
      nVar += tf.reduce_prod(t.shape)
    print('{} trainable variables'.format(nVar))


  def preview(self):
    print('--------------------------------')
    print('Model preview')
    print('--------------------------------')
    print('Fully connected network: {}'.format(self.width))
    #print(self.width)
    print('Layer regularization: {}'.format(self.reg))
    #print(self.reg)
    print('Coefficients for data loss {} {} {}'.format(\
          self.alpha[0], self.alpha[1], self.alpha[2]))
    print('Coefficients for pde residual {} {} {}'.format(\
          self.beta[0], self.beta[1], self.beta[2]))
    print('Coefficients for domain wall loss {} {}'.format(\
          self.gamma[0], self.gamma[1]))
    print('--------------------------------')
