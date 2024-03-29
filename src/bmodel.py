#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

np.random.seed(2022)
tf.random.set_seed(2022)

from tensorflow import keras
from blayers import *

import butils as UT

''' --- TODO ---
strategy = tf.distribute.MirroredStrategy()
'''

class BModel(keras.Model):
  def __init__(self, width=[150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 3],\
               alpha=[1.0, 1.0, 0.0], beta=[1e-2, 1e-2, 1e-2], gamma=[1e-4, 1e-4, 0.0],\
               reg=None, saveGradStat=False, Re=3.5e4, initialCondition=False, **kwargs):
    super(BModel, self).__init__(**kwargs)
    print('Creating Model with alpha={}, beta={}, gamma={}, Re={}'.format( \
          alpha, beta, gamma, Re))

    self.width = width
    self.reg   = reg
    if reg == None:
      self.mlp = DenseLayers(width=width, prefix='bc', act='tanh', last_linear=True)
    else:
      self.mlp = DenseLayers(width=width, reg=reg, prefix='bc', act='tanh', last_linear=True)
    # Coefficient for data and pde loss
    self.alpha = alpha
    self.beta  = beta
    self.gamma = gamma
    self.Re    = Re
    self.initialCondition = initialCondition
    # ---- dicts for metrics and statistics ---- #
    # Save gradients' statistics per layer
    self.saveGradStat = saveGradStat
    # Create metrics
    self.trainMetrics = {}
    self.validMetrics = {}
    # add metrics
    names = ['loss', 'uMse', 'vMse', 'pMse', 'pde0', 'pde1', 'pde2', 'uMae', 'vMae', 'pMae', 'uMseWalls', 'vMseWalls', 'pMseWalls']
    for key in names:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)
    ## add metrics for layers' weights, if save_grad_stat is required
    ## i even for weights, odd for bias
    if self.saveGradStat:
      for i in range(len(width)):
        for prefix in ['u_', 'v_', 'p_', 'pde0_', 'pde1_', 'pde2_', 'uWalls_', 'vWalls_', 'pWalls_']:
          for suffix in ['w_avg', 'w_std', 'b_avg', 'b_std']:
            key = prefix + repr(i) + suffix
            self.trainMetrics[key] = keras.metrics.Mean(name='train '+key)
    # Statistics
    self.trainStat = {}
    self.validStat = {}


  def call(self, inputs, training=False):
    '''
    inputs: [xy, t, uvp]
    '''
    xy = inputs[0]
    t = inputs[1]
    xyt = tf.concat([xy, t], axis=-1)
    uvp = self.mlp(xyt)
    bcUvp = inputs[2]

    initCond = 0
    omitP = 1 # TODO: Must be enabled for now
    if initCond:
      select = tf.cast(tf.math.equal(t, 0), tf.float32)
      remove = tf.cast(tf.math.not_equal(t, 0), tf.float32)
      if omitP:
        uv, p = uvp[:,:2], uvp[:,2]
        bcUv, bcP = bcUvp[:,:2], bcUvp[:,2]
        p        = tf.expand_dims(p, axis=-1)
        uv = bcUvp[:,:2] * select + uv[:,:2] * remove 
        uvp   = tf.concat([uv, p], axis=-1)

      '''
      colMask = tf.cast(tf.math.equal(w, 0), tf.float32)

      bcUv, bcP = bcUvp[:,:2], bcUvp[:,2]
      bcMask = tf.cast(tf.math.less(d, 1), tf.float32)
      g = bcUvp * bcMask
      if omitPBc:
        g = g[:,:2]

      omitP = 0
      if omitP:
        uv, p = uvp[:,:2], uvp[:,2]
        p     = tf.expand_dims(p, axis=-1)
        uv    = g + d * uv
        #tf.print(uv.shape, summarize=64)
        uvp   = tf.concat([uv, p], axis=-1)
      else:
        uvp = g + d * uvp
      '''

    filter = 0
    if filter:
      xy   = inputs[0]
      x, y = xy[:,0], xy[:,1]

      # The dimensionless size of a single pixel and the domain
      pixelSize = UT.get_pixelsize_dimensionless(UT.worldSize_fx, UT.imageSize_fx, UT.L_fx)
      domainSize = pixelSize * (UT.imageSize_fx-1)

      # Set how many pixels from border to use when filtering (left, top, right, bottom)
      wallPixels = [1, 0, 1, 1]
      filterSize = [pixelSize*pixelCnt for pixelCnt in wallPixels]

      facLeft   = tf.clip_by_value(y/filterSize[0], 0, 1) if filterSize[0] else 1
      facTop    = tf.clip_by_value((domainSize-y)/filterSize[1], 0, 1) if filterSize[1] else 1
      facRight  = tf.clip_by_value((domainSize-x)/filterSize[2], 0, 1) if filterSize[2] else 1
      facBottom = tf.clip_by_value(x/filterSize[3], 0, 1) if filterSize[3] else 1
      #tf.print(facBottom, summarize=100)

      # Function d filters network effect near walls
      leftBottom = tf.math.minimum(facLeft, facTop)
      rightTop = tf.math.minimum(facRight, facBottom)
      d = tf.expand_dims(tf.math.minimum(leftBottom, rightTop), axis=-1)
      #d += 1e-10 # Avoid zero factors

      omitPBc = 1
      # Function g contains exact data from boundary cells
      g = 0
      if 1: #training:
        #bcUvp = inputs[2]
        bcUv, bcP = bcUvp[:,:2], bcUvp[:,2]
        bcMask = tf.cast(tf.math.less(d, 1), tf.float32) # 1s in walls, 0s in fluid
        fluidMask = tf.cast(tf.math.greater_equal(d, 1), tf.float32) # 0s in walls, 1s in fluid
        #tf.print(d, summarize=64)
        g = bcUvp * bcMask
        if omitPBc:
          g = g[:,:2]
        #tf.print(g, summarize=64)

      if omitPBc:
        uv, p = uvp[:,:2], uvp[:,2]
        p     = tf.expand_dims(p, axis=-1)
        uv    = g + fluidMask * uv
        #tf.print(uv.shape, summarize=64)
        uvp   = tf.concat([uv, p], axis=-1)
      else:
        uvp   = g + d * uvp

    return uvp


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


  def compute_losses(self, xy, t, w, phi, uv):
    # Track computation for 2nd derivatives for u, v
    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(xy)
      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(xy)
        tape1.watch(t)
        uvpPred = self([xy, t, uv], training=True)
        uPred   = uvpPred[:,0]
        vPred   = uvpPred[:,1]
        pPred   = uvpPred[:,2]
      # 1st order derivatives
      u_grad = tape1.gradient(uPred, xy)
      v_grad = tape1.gradient(vPred, xy)
      p_grad = tape1.gradient(pPred, xy)
      u_x, u_y = u_grad[:,0], u_grad[:,1]
      v_x, v_y = v_grad[:,0], v_grad[:,1]
      p_x, p_y = p_grad[:,0], p_grad[:,1]
      u_t = tape1.gradient(uPred, t)
      v_t = tape1.gradient(vPred, t)
      del tape1
    # 2nd order derivatives
    u_xx = tape2.gradient(u_x, xy)[:,0]
    u_yy = tape2.gradient(u_y, xy)[:,1]
    v_xx = tape2.gradient(v_x, xy)[:,0]
    v_yy = tape2.gradient(v_y, xy)[:,1]
    del tape2

    # Compute data loss
    w = tf.squeeze(w)
    dataMask = tf.cast(tf.equal(w, 1), tf.float32)
    nDataPoint = tf.reduce_sum(dataMask) + 1.0e-10
    uMse = tf.reduce_sum(tf.square(uv[:,0] - uPred) * dataMask) / nDataPoint
    vMse = tf.reduce_sum(tf.square(uv[:,1] - vPred) * dataMask) / nDataPoint
    pMse = tf.reduce_sum(tf.square(uv[:,2] - pPred) * dataMask) / nDataPoint

    # Compute PDE loss (2D Navier Stokes: 0 continuity, 1-2 momentum)
    pdeTrue = 0.0
    Re      = self.Re
    pde0    = u_x + v_y
    pde1    = u_t + uPred*u_x + vPred*u_y + p_x - (1/Re)*(u_xx + u_yy)
    pde2    = v_t + uPred*v_x + vPred*v_y + p_y - (1/Re)*(v_xx + v_yy)

    if 0:
      # tInit: The timestamp (dimensionless number after zero-mean shift) where the initial condition is taken from
      tInit = 0.0
      isInitCond  = tf.cast(tf.math.equal(t, tInit), tf.float32)
      notInitCond = tf.cast(tf.math.not_equal(t, tInit), tf.float32)

      #isColInit      = tf.cast(tf.math.logical_and(isInitCond, colMask), tf.float32)
      nPdePoint = tf.reduce_sum(colMask * notInitCond) + 1.0e-10
      nInitCondPoint = tf.reduce_sum(colMask * isInitCond) + 1.0e-10

      #tf.print(nPdePoint, summarize=64)
      #a = tf.reduce_sum(colMask * notInitCond)
      #tf.print(tf.math.equal(nPdePoint, a))
      pdeMse0  = tf.reduce_sum(tf.square(pdeTrue - pde0) * colMask * notInitCond) / nPdePoint
      pdeMse1  = tf.reduce_sum(tf.square(pdeTrue - pde1) * colMask * notInitCond) / nPdePoint
      pdeMse2  = tf.reduce_sum(tf.square(pdeTrue - pde2) * colMask * notInitCond) / nPdePoint
      #tf.print(tf.reduce_sum(colMask * notInitCond))
      uMse += tf.reduce_sum(tf.square(uv[:,0] - uPred) * colMask * isInitCond) / nInitCondPoint
      vMse += tf.reduce_sum(tf.square(uv[:,1] - vPred) * colMask * isInitCond) / nInitCondPoint
      pMse += tf.reduce_sum(tf.square(uv[:,2] - pPred) * colMask * isInitCond) / nInitCondPoint


    # Compute initial condition loss
    if self.initialCondition:
      # Add initial condition loss to data loss (it's using the collocation points but with MSE)
      initCondMask = tf.cast(tf.equal(w, 3), tf.float32)
      nInitCondPoint = tf.reduce_sum(initCondMask) + 1.0e-10
      uMse += tf.reduce_sum(tf.square(uv[:,0] - uPred) * initCondMask) / nInitCondPoint
      vMse += tf.reduce_sum(tf.square(uv[:,1] - vPred) * initCondMask) / nInitCondPoint
      pMse += tf.reduce_sum(tf.square(uv[:,2] - pPred) * initCondMask) / nInitCondPoint

    # Compute informed collocation loss
    if 0:
      # Add initial condition loss to data loss (it's using the collocation points but with MSE)
      colDataMask = tf.cast(tf.equal(w, 4), tf.float32)
      nColDataPoint = tf.reduce_sum(colDataMask) + 1.0e-10
      uMse += tf.reduce_sum(tf.square(uv[:,0] - uPred) * colDataMask) / nColDataPoint
      vMse += tf.reduce_sum(tf.square(uv[:,1] - vPred) * colDataMask) / nColDataPoint
      pMse += tf.reduce_sum(tf.square(uv[:,2] - pPred) * colDataMask) / nColDataPoint

    # Compute PDE loss
    if 1:
      #phi = tf.squeeze(phi)
      #phiInv = 1 / tf.abs(phi)
      colMask = tf.cast(tf.equal(w, 0), tf.float32)
      nPdePoint = tf.reduce_sum(colMask) + 1.0e-10
      pdeMse0 = tf.reduce_sum(tf.square(pdeTrue - pde0) * colMask) / nPdePoint
      pdeMse1 = tf.reduce_sum(tf.square(pdeTrue - pde1) * colMask) / nPdePoint
      pdeMse2 = tf.reduce_sum(tf.square(pdeTrue - pde2) * colMask) / nPdePoint

    # Compute domain wall loss
    wallMask = tf.cast(tf.equal(w, 2), tf.float32)
    nWallsPoint = tf.reduce_sum(wallMask) + 1.0e-10
    uMseWalls = tf.reduce_sum(tf.square(uv[:,0] - uPred) * wallMask) / nWallsPoint
    vMseWalls = tf.reduce_sum(tf.square(uv[:,1] - vPred) * wallMask) / nWallsPoint
    pMseWalls = tf.reduce_sum(tf.square(uv[:,2] - pPred) * wallMask) / nWallsPoint

    return uvpPred, uMse, vMse, pMse, pdeMse0, pdeMse1, pdeMse2, uMseWalls, vMseWalls, pMseWalls


  def train_step(self, data):
    xy  = data[0][0]
    t   = data[0][1]
    uv  = data[0][2]
    w   = data[0][3]
    phi = data[0][4]

    with tf.GradientTape(persistent=True) as tape0:
      # Compute the data loss for u, v and pde losses for
      # continuity (0) and NS (1-2)
      uvpPred, uMse, vMse, pMse, pdeMse0, pdeMse1, pdeMse2, uMseWalls, vMseWalls, pMseWalls = \
        self.compute_losses(xy, t, w, phi, uv)
      # replica's loss, divided by global batch size
      loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse + self.alpha[2]*pMse \
              + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2 \
              + self.gamma[0]*uMseWalls + self.gamma[1]*vMseWalls + self.gamma[2]*pMseWalls)
      loss += tf.add_n(self.losses)
      loss  = loss #/ strategy.num_replicas_in_sync
    # update gradients
    if self.saveGradStat:
      uMseGrad      = tape0.gradient(uMse,      self.trainable_variables)
      vMseGrad      = tape0.gradient(vMse,      self.trainable_variables)
      pMseGrad      = tape0.gradient(pMse,      self.trainable_variables)
      pdeMse0Grad   = tape0.gradient(pdeMse0,   self.trainable_variables)
      pdeMse1Grad   = tape0.gradient(pdeMse1,   self.trainable_variables)
      pdeMse2Grad   = tape0.gradient(pdeMse2,   self.trainable_variables)
      uMseWallsGrad = tape0.gradient(uMseWalls, self.trainable_variables)
      vMseWallsGrad = tape0.gradient(vMseWalls, self.trainable_variables)
      pMseWallsGrad = tape0.gradient(pMseWalls, self.trainable_variables)
    lossGrad = tape0.gradient(loss, self.trainable_variables)
    del tape0

    # ---- Update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- Update metrics and statistics ---- #
    # Track loss and mae
    self.trainMetrics['loss'].update_state(loss)#*strategy.num_replicas_in_sync)
    self.trainMetrics['uMse'].update_state(uMse)
    self.trainMetrics['vMse'].update_state(vMse)
    self.trainMetrics['pMse'].update_state(pMse)
    self.trainMetrics['pde0'].update_state(pdeMse0)
    self.trainMetrics['pde1'].update_state(pdeMse1)
    self.trainMetrics['pde2'].update_state(pdeMse2)
    self.trainMetrics['uMseWalls'].update_state(uMseWalls)
    self.trainMetrics['vMseWalls'].update_state(vMseWalls)
    self.trainMetrics['pMseWalls'].update_state(pMseWalls)
    w = tf.squeeze(w)
    nDataPoint = tf.reduce_sum(w) + 1.0e-10
    uMae = tf.reduce_sum(tf.abs((uvpPred[:,0] - uv[:,0]) * w)) / nDataPoint
    vMae = tf.reduce_sum(tf.abs((uvpPred[:,1] - uv[:,1]) * w)) / nDataPoint
    pMae = tf.reduce_sum(tf.abs((uvpPred[:,2] - uv[:,2]) * w)) / nDataPoint
    self.trainMetrics['uMae'].update_state(uMae)
    self.trainMetrics['vMae'].update_state(vMae)
    self.trainMetrics['pMae'].update_state(pMae)
    # track gradients coefficients
    if self.saveGradStat:
      self.record_layer_gradient(uMseGrad, 'u_')
      self.record_layer_gradient(vMseGrad, 'v_')
      self.record_layer_gradient(pMseGrad, 'p_')
      self.record_layer_gradient(pdeMse0Grad, 'pde0_')
      self.record_layer_gradient(pdeMse1Grad, 'pde1_')
      self.record_layer_gradient(pdeMse2Grad, 'pde2_')
      self.record_layer_gradient(uMseWallsGrad, 'uWalls_')
      self.record_layer_gradient(vMseWallsGrad, 'vWalls_')
      self.record_layer_gradient(pMseWallsGrad, 'pWalls_')
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat


  def test_step(self, data):
    xy  = data[0][0]
    t   = data[0][1]
    uv  = data[0][2]
    w   = data[0][3]
    phi = data[0][4]

    # Compute the data and pde losses
    uvpPred, uMse, vMse, pMse, pdeMse0, pdeMse1, pdeMse2, uMseWalls, vMseWalls, pMseWalls = \
      self.compute_losses(xy, t, w, phi, uv)
    # replica's loss, divided by global batch size
    loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse + self.alpha[2]*pMse \
            + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2 \
            + self.gamma[0]*uMseWalls + self.gamma[1]*vMseWalls + self.gamma[2]*pMseWalls )

    # Track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pMse'].update_state(pMse)
    self.validMetrics['pde0'].update_state(pdeMse0)
    self.validMetrics['pde1'].update_state(pdeMse1)
    self.validMetrics['pde2'].update_state(pdeMse2)
    self.validMetrics['uMseWalls'].update_state(uMseWalls)
    self.validMetrics['vMseWalls'].update_state(vMseWalls)
    self.validMetrics['pMseWalls'].update_state(pMseWalls)
    w = tf.squeeze(w)
    nDataPoint = tf.reduce_sum(w) + 1.0e-10
    uMae = tf.reduce_sum(tf.abs((uvpPred[:,0] - uv[:,0]) * w)) / nDataPoint
    vMae = tf.reduce_sum(tf.abs((uvpPred[:,1] - uv[:,1]) * w)) / nDataPoint
    pMae = tf.reduce_sum(tf.abs((uvpPred[:,2] - uv[:,2]) * w)) / nDataPoint
    self.validMetrics['uMae'].update_state(uMae)
    self.validMetrics['vMae'].update_state(vMae)
    self.validMetrics['pMae'].update_state(pMae)

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
    print('Coefficients for domain wall loss {} {} {}'.format(\
          self.gamma[0], self.gamma[1], self.gamma[2]))
    print('--------------------------------')
