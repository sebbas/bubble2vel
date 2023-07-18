#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import math

np.random.seed(2022)
tf.random.set_seed(2022)

from tensorflow import keras
from blayers import *

import butils as UT

''' --- TODO ---
strategy = tf.distribute.MirroredStrategy()
'''

class BModel(keras.Model):
  def __init__(self, width=[150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 4],\
               alpha=[1.0, 1.0, 0.0, 1.0], beta=[1e-2, 1e-2, 1e-2, 1e-2], gamma=[1e-4, 1e-4, 0.0], delta=[1.0, 1.0, 1.0], \
               reg=None, saveGradStat=False, Re=3.5e4, Pr=7, hardBc=False, **kwargs):
    super(BModel, self).__init__(**kwargs)
    print('Creating Model with alpha={}, beta={}, gamma={}, delta={}, Re={}'.format( \
          alpha, beta, gamma, delta, Re))

    self.width = width
    self.reg   = reg
    act = 'tanh'
    if reg == None:
      self.mlp = DenseLayers(width=width, act=act, last_linear=True)
    else:
      self.mlp = DenseLayers(width=width, reg=reg, act=act, last_linear=True)
    # Coefficient for data and pde loss
    self.alpha = alpha
    self.beta  = beta
    self.gamma = gamma
    self.delta = delta
    self.Re    = Re #tf.Variable(Re, trainable=True)
    self.Pe    = Re * Pr
    self.hardBc = hardBc
    # ---- dicts for metrics and statistics ---- #
    # Save gradients' statistics per layer
    self.saveGradStat = saveGradStat
    # Create metrics
    self.trainMetrics = {}
    self.validMetrics = {}
    # add metrics
    names = ['loss', 'uMse', 'vMse', 'pMse', 'cMse', 'pde0', 'pde1', 'pde2', 'pde3', 'uMae', 'vMae', 'pMae', 'uMseWalls', 'vMseWalls', 'pMseWalls', 'uMseInit', 'vMseInit', 'cMseInit', 'Re']
    for key in names:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)
    ## add metrics for layers' weights, if save_grad_stat is required
    ## i even for weights, odd for bias
    if self.saveGradStat:
      for i in range(len(width)):
        for prefix in ['u_', 'v_', 'p_', 'c_', 'pde0_', 'pde1_', 'pde2_', 'pde3_', 'uWalls_', 'vWalls_', 'pWalls_', 'uInit_', 'vInit_', 'cInit_']:
          for suffix in ['w_avg', 'w_std', 'b_avg', 'b_std']:
            key = prefix + repr(i) + suffix
            self.trainMetrics[key] = keras.metrics.Mean(name='train '+key)
    # Statistics
    self.trainStat = {}
    self.validStat = {}


    alpha = 0.999
    temperature = 0.01
    rho = 0.99
    self.alphaR = alpha
    self.temperature = temperature
    self.rho = rho
    self.call_count = tf.Variable(0, trainable=False, dtype=tf.int16)

    # Using only uMse, vMse, pde0, pde1, pde2 for relobralo loss for now
    self.numTerms = 5#len(self.alpha) + len(self.beta) + len(self.gamma)
    self.lambdas = [tf.Variable(1., trainable=False) for _ in range(self.numTerms)]
    self.last_losses = [tf.Variable(1., trainable=False) for _ in range(self.numTerms)]
    self.init_losses = [tf.Variable(1., trainable=False) for _ in range(self.numTerms)]

    self.iter0 = 0
    self.iter1 = 1


  def _getG(self, xy, xyBc, bc, eps=1e-10, exp=2):
    xyExp = tf.expand_dims(xy, axis=1)       # [nBatch, 1, nDim]
    dist = tf.square(xyExp - xyBc)           # [nBatch, nXyBc, nDim]
    dist = tf.reduce_sum(dist, axis=-1)      # [nBatch, nXyBc]
    #dist = tf.math.sqrt(dist)               # [nBatch, nXyBc]
    wi = tf.pow(1.0 / (dist + eps), exp)     # [nBatch, nXyBc]
    wi = tf.expand_dims(wi, axis=-1)         # [nBatch, nXyBc, 1]
    denom = tf.reduce_sum(wi, axis=1)        # [nBatch, 1, nDim]
    numer = tf.reduce_sum(wi * bc, axis=1)   # [nBatch, 1, dimBc]
    g = numer / denom                        # [nBatch, dimBc]
    return g


  def _getPhi(self, x, y, walls):
    phi = 1.0
    if walls[0]: phi *= x     # left
    if walls[1]: phi *= (1-y) # top
    if walls[2]: phi *= (1-x) # right
    if walls[3]: phi *= y     # bottom
    return phi


  def call(self, inputs, training=False):
    '''
    inputs: [xy, t, uvp]
    '''
    xy = inputs[0]
    t = inputs[1]
    uvpOrig = inputs[2]

    xyt = tf.concat([xy, t], axis=1)
    uvp = self.mlp(xyt)

    # The dimensionless size of a single pixel and the domain
    #pixelSize = UT.get_pixelsize_dimensionless(UT.worldSize_fx, UT.imageSize_fx, UT.L_fx)
    #domainSize = pixelSize * (UT.imageSize_fx-1)

    # Use hard boundary condition
    # Reproduces exact values in boundary and filter networks effects near boundaries
    if self.hardBc:
      xy    = inputs[0]
      xy /= UT.dimlessMax # Normalize to [0,1] for IDW calculation
      x, y  = xy[:,0], xy[:,1]

      uvpBc = inputs[4]
      idsBc = inputs[5]

      # (1) Construct g's. Gives exact bc in boundary cells, IDW interpolation value at every interior point
      # (1.1) Construct function g for velocity bc
      xyBc  = inputs[3]
      xyBc /= UT.dimlessMax
      uvBc  = uvpBc[:,:,:2] # [nBatch, nXyBc, 2]

      # Get all bc points where vel bc is enfored (filter out other bc points)
      walls  = tf.cast(tf.equal(idsBc, 20), tf.float32) # left
      walls += tf.cast(tf.equal(idsBc, 22), tf.float32) # right
      walls += tf.cast(tf.equal(idsBc, 23), tf.float32) # bottom
      mask   = tf.cast(walls[0,:], tf.bool)
      xyBc   = tf.boolean_mask(xyBc, mask, axis=1)
      uvBc   = tf.boolean_mask(uvBc, mask, axis=1)
      gVel   = self._getG(xy, xyBc, uvBc)

      # (1.2) Construct function g for pressure bc
      xyBc  = inputs[3]
      xyBc /= UT.dimlessMax
      pBc   = uvpBc[:,:,2:3] # [nBatch, nXyBc, 1]

      # Get all bc points where pressure bc is enfored (filter out other bc points)
      walls = tf.cast(tf.equal(idsBc, 21), tf.float32) # top
      mask  = tf.cast(walls[0,:], tf.bool)
      xyBc  = tf.boolean_mask(xyBc, mask, axis=1)
      pBc   = tf.boolean_mask(pBc, mask, axis=1)
      gPres = self._getG(xy, xyBc, pBc)

      # (2) Construct phi's. Zero in boundary cells, getting more positive at interior points
      # (2.1) Construct function phi for velocity bc
      #phiVel = x * (1-x) * y # Filter function for left, right, and bottom domain boundaries
      phiVel = self._getPhi(x, y, [1,0,1,1]) # Filter at left, right, and bottom domain boundaries

      # (2.2) Construct function phi for pressure bc
      #phiPres = (1-y)
      phiPres = self._getPhi(x, y, [0,1,0,0]) # Filter at top domain boundary

      withInitCond = 0
      if withInitCond:
        w = inputs[6]
        uvp = tf.where(tf.equal(w, 3), uvpOrig, uvp)

      # Assemble g and phi with uvp
      uv, p, c = uvp[:,:2], uvp[:,2:3], uvp[:,3:4]
      phiVel   = tf.expand_dims(phiVel, axis=-1)
      phiPres  = tf.expand_dims(phiPres, axis=-1)
      uv       = gVel + uv * phiVel               # [nBatch, nDim]
      p        = gPres + p * phiPres              # [nBatch, nDim]
      uvp      = tf.concat([uv, p, c], axis=-1)   # [nBatch, nDim + 1]

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


  def compute_losses(self, xy, t, w, phi, uv, xyBc, uvBc, validBc):
    # Track computation for 2nd derivatives for u, v
    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(xy)
      with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape1, \
           tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape0:
        tape1.watch(xy)
        tape0.watch(t)
        uvpPred = self([xy, t, uv, xyBc, uvBc, validBc, w, phi], training=True)
        uPred   = uvpPred[:,0]
        vPred   = uvpPred[:,1]
        pPred   = uvpPred[:,2]
        cPred   = uvpPred[:,3]
      # 1st order derivatives
      u_grad = tape1.gradient(uPred, xy)
      v_grad = tape1.gradient(vPred, xy)
      p_grad = tape1.gradient(pPred, xy)
      c_grad = tape1.gradient(cPred, xy)
      u_x, u_y = u_grad[:,0], u_grad[:,1]
      v_x, v_y = v_grad[:,0], v_grad[:,1]
      p_x, p_y = p_grad[:,0], p_grad[:,1]
      c_x, c_y = c_grad[:,0], c_grad[:,1]
      u_t = tape0.gradient(uPred, t)
      v_t = tape0.gradient(vPred, t)
      c_t = tape0.gradient(cPred, t)
      del tape1
      del tape0
    # 2nd order derivatives
    u_xx = tape2.gradient(u_x, xy)[:,0]
    u_yy = tape2.gradient(u_y, xy)[:,1]
    v_xx = tape2.gradient(v_x, xy)[:,0]
    v_yy = tape2.gradient(v_y, xy)[:,1]
    c_xx = tape2.gradient(c_x, xy)[:,0]
    c_yy = tape2.gradient(c_y, xy)[:,1]
    del tape2

    # Match shape of u_x, u_y, etc (i.e. horizontal row vec)
    u_t = tf.squeeze(tf.transpose(u_t))
    v_t = tf.squeeze(tf.transpose(v_t))

    # Compute data loss
    w = tf.squeeze(w)
    dataMask = tf.cast(tf.equal(w, 1), tf.float32)
    nDataPoint = tf.reduce_sum(dataMask) + 1.0e-10
    uMse = tf.reduce_sum(tf.square(uv[:,0] - uPred) * dataMask) / nDataPoint
    vMse = tf.reduce_sum(tf.square(uv[:,1] - vPred) * dataMask) / nDataPoint
    pMse = tf.reduce_sum(tf.square(uv[:,2] - pPred) * dataMask) / nDataPoint
    cTrue = tf.sqrt(tf.square(uv[:,0]) + tf.square(uv[:,1]))
    cMse = tf.reduce_sum(tf.square(cTrue - cPred) * dataMask) / nDataPoint

    # Compute PDE loss (2D Navier Stokes: 0 continuity, 1-2 momentum)
    pdeTrue = 0.0
    Re, Pe  = self.Re, self.Pe
    pde0    = u_x + v_y
    pde1    = u_t + uPred*u_x + vPred*u_y + p_x - (1/Re)*(u_xx + u_yy)
    pde2    = v_t + uPred*v_x + vPred*v_y + p_y - (1/Re)*(v_xx + v_yy)
    pde3    = c_t + uPred*c_x + vPred*c_y - (1/Pe)*(c_xx + c_yy)

    # Compute data loss for initial condition
    initCondMask = tf.cast(tf.equal(w, 3), tf.float32)
    nInitCondPoint = tf.reduce_sum(initCondMask) + 1.0e-10
    uMseInit = tf.reduce_sum(tf.square(uv[:,0] - uPred) * initCondMask) / nInitCondPoint
    vMseInit = tf.reduce_sum(tf.square(uv[:,1] - vPred) * initCondMask) / nInitCondPoint
    cMseInit = tf.reduce_sum(tf.square(cTrue - cPred) * initCondMask) / nInitCondPoint

    # Compute PDE loss
    #colMask = tf.cast(tf.equal(w, 0), tf.float32)
    #nPdePoint = tf.reduce_sum(colMask) + 1.0e-10
    pdeMse0 = tf.reduce_sum(tf.square(pdeTrue - pde0))# * colMask) / nPdePoint
    pdeMse1 = tf.reduce_sum(tf.square(pdeTrue - pde1))# * colMask) / nPdePoint
    pdeMse2 = tf.reduce_sum(tf.square(pdeTrue - pde2))# * colMask) / nPdePoint
    pdeMse3 = tf.reduce_sum(tf.square(pdeTrue - pde3))# * colMask) / nPdePoint

    # Compute domain wall loss for velocity
    wallMask = tf.cast(tf.equal(w, 20), tf.float32)  # left
    wallMask += tf.cast(tf.equal(w, 22), tf.float32) # right
    wallMask += tf.cast(tf.equal(w, 23), tf.float32) # bottom
    nWallsPoint = tf.reduce_sum(wallMask) + 1.0e-10
    # Impose ground truth velocity bc
    velocityTrue = 0
    uMseWalls = tf.reduce_sum(tf.square(uv[:,0] - uPred) * wallMask) / nWallsPoint
    vMseWalls = tf.reduce_sum(tf.square(uv[:,1] - vPred) * wallMask) / nWallsPoint

    # Compute domain wall loss for pressure
    wallMask = tf.cast(tf.equal(w, 21), tf.float32) # top
    nWallsPoint = tf.reduce_sum(wallMask) + 1.0e-10
    # Impose zero-pressure bc
    pressureTrue = 0
    pMseWalls = tf.reduce_sum(tf.square(pressureTrue - pPred) * wallMask) / nWallsPoint

    return uvpPred, uMse, vMse, pMse, cMse, pdeMse0, pdeMse1, pdeMse2, pdeMse3, uMseWalls, vMseWalls, pMseWalls, uMseInit, vMseInit, cMseInit


  @tf.function
  def getAlpha1(self):
    if tf.equal(self.call_count, self.iter1):
      return 0.
    else:
      return self.alphaR 
  @tf.function
  def getAlpha(self):
    if tf.equal(self.call_count, self.iter0):
      return 1.
    else:
      return self.getAlpha1()

  @tf.function
  def getRho1(self):
    if tf.equal(self.call_count, self.iter1):
      return 1.
    else:
      return tf.cast(tf.random.uniform(shape=()) < self.rho, dtype=tf.float32)
  @tf.function
  def getRho(self):
    if tf.equal(self.call_count, self.iter0):
      return 1.
    else:
      return self.getRho1()


  #@tf.function
  def train_step(self, data):
    # A batch of points has ...
    xy      = data[0][0] # xy positions
    t       = data[0][1] # timestamps
    uv      = data[0][2] # uv velocities
    xyBc    = data[0][3] # xy positions of all bc points
    uvpBc   = data[0][4] # uvp of all bc points
    validBc = data[0][5] # binary value indicating if bc points is valid
    w       = data[0][6] # id of the point (e.g. collocation, wall, ...)
    phi     = data[0][7] # SDF value

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape0:

      tape0.watch(self.trainable_variables)

      uvpPred, uMse, vMse, pMse, cMse, pdeMse0, pdeMse1, pdeMse2, pdeMse3, uMseWalls, vMseWalls, pMseWalls, uMseInit, vMseInit, cMseInit = \
        self.compute_losses(xy, t, w, phi, uv, xyBc, uvpBc, validBc)

      #tf.print(len(self.trainable_variables))

      usingReLoBRaLoLoss = 0
      if usingReLoBRaLoLoss:
        losses = [uMse, vMse, pdeMse0, pdeMse1, pdeMse2]

        EPS = 1e-7
        # in first iteration (self.call_count == 0), drop lambda_hat and use init lambdas, i.e. lambda = 1
        #   i.e. alpha = 1 and rho = 1
        # in second iteration (self.call_count == 1), drop init lambdas and use only lambda_hat
        #   i.e. alpha = 0 and rho = 1
        # afterwards, default procedure (see paper)
        #   i.e. alpha = self.alpha and rho = Bernoully random variable with p = self.rho
        #alpha = tf.cond(tf.equal(self.call_count, 0),
        #        lambda: 1.,
        #        lambda: tf.cond(tf.equal(self.call_count, 1),
        #                        lambda: 0.,
        #                        lambda: self.alpha))

        alpha = self.getAlpha()
        rho = self.getRho()

        # compute new lambdas w.r.t. the losses in the previous iteration
        lambdas_hat = [losses[i] / (self.last_losses[i] * self.temperature + EPS) for i in range(len(losses))]
        lambdas_hat = tf.nn.softmax(lambdas_hat - tf.reduce_max(lambdas_hat)) * tf.cast(len(losses), dtype=tf.float32)

        # compute new lambdas w.r.t. the losses in the first iteration
        init_lambdas_hat = [losses[i] / (self.init_losses[i] * self.temperature + EPS) for i in range(len(losses))]
        init_lambdas_hat = tf.nn.softmax(init_lambdas_hat - tf.reduce_max(init_lambdas_hat)) * tf.cast(len(losses), dtype=tf.float32)

        # use rho for deciding, whether a random lookback should be performed
        new_lambdas = [(rho * alpha * self.lambdas[i] + (1 - rho) * alpha * init_lambdas_hat[i] + (1 - alpha) * lambdas_hat[i]) for i in range(len(losses))]
        for var, lam in zip(self.lambdas, new_lambdas):
          var.assign(tf.stop_gradient(lam))

        # store current losses in self.last_losses to be accessed in the next iteration
        for var, loss in zip(self.last_losses, losses):
          var.assign(tf.stop_gradient(loss))

        # in first iteration, store losses in self.init_losses to be accessed in next iterations
        first_iteration = tf.cast(self.call_count < self.iter1, dtype=tf.float32)
        for var, loss in zip(self.init_losses, losses):
          var.assign(tf.stop_gradient(loss * first_iteration + var * (1 - first_iteration)))

        # compute weighted loss
        loss = tf.reduce_sum([lam * loss for lam, loss in zip(self.lambdas, losses)])

        #tf.print(self.lambdas)
        self.call_count.assign_add(1)
      else:
        loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse + self.alpha[2]*pMse + self.alpha[3]*cMse \
                + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2 + self.beta[3]*pdeMse3 \
                + self.gamma[0]*uMseWalls + self.gamma[1]*vMseWalls + self.gamma[2]*pMseWalls \
                + self.delta[0]*uMseInit + self.delta[1]*vMseInit + self.delta[2]*cMseInit)

      loss += tf.add_n(self.losses)
    # update gradients
    if self.saveGradStat:
      uMseGrad      = tape0.gradient(uMse,      self.trainable_variables)
      vMseGrad      = tape0.gradient(vMse,      self.trainable_variables)
      pMseGrad      = tape0.gradient(pMse,      self.trainable_variables)
      cMseGrad      = tape0.gradient(cMse,      self.trainable_variables)
      pdeMse0Grad   = tape0.gradient(pdeMse0,   self.trainable_variables)
      pdeMse1Grad   = tape0.gradient(pdeMse1,   self.trainable_variables)
      pdeMse2Grad   = tape0.gradient(pdeMse2,   self.trainable_variables)
      pdeMse3Grad   = tape0.gradient(pdeMse3,   self.trainable_variables)
      uMseWallsGrad = tape0.gradient(uMseWalls, self.trainable_variables)
      vMseWallsGrad = tape0.gradient(vMseWalls, self.trainable_variables)
      pMseWallsGrad = tape0.gradient(pMseWalls, self.trainable_variables)
      uMseInitGrad  = tape0.gradient(uMseInit,  self.trainable_variables)
      vMseInitGrad  = tape0.gradient(vMseInit,  self.trainable_variables)
      cMseInitGrad  = tape0.gradient(cMseInit,  self.trainable_variables)
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
    self.trainMetrics['cMse'].update_state(cMse)
    self.trainMetrics['pde0'].update_state(pdeMse0)
    self.trainMetrics['pde1'].update_state(pdeMse1)
    self.trainMetrics['pde2'].update_state(pdeMse2)
    self.trainMetrics['pde3'].update_state(pdeMse3)
    self.trainMetrics['uMseWalls'].update_state(uMseWalls)
    self.trainMetrics['vMseWalls'].update_state(vMseWalls)
    self.trainMetrics['pMseWalls'].update_state(pMseWalls)
    self.trainMetrics['uMseInit'].update_state(uMseInit)
    self.trainMetrics['vMseInit'].update_state(vMseInit)
    self.trainMetrics['cMseInit'].update_state(cMseInit)
    self.trainMetrics['Re'].update_state(self.Re)
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
      self.record_layer_gradient(cMseGrad, 'c_')
      self.record_layer_gradient(pdeMse0Grad, 'pde0_')
      self.record_layer_gradient(pdeMse1Grad, 'pde1_')
      self.record_layer_gradient(pdeMse2Grad, 'pde2_')
      self.record_layer_gradient(pdeMse3Grad, 'pde3_')
      self.record_layer_gradient(uMseWallsGrad, 'uWalls_')
      self.record_layer_gradient(vMseWallsGrad, 'vWalls_')
      self.record_layer_gradient(pMseWallsGrad, 'pWalls_')
      self.record_layer_gradient(uMseInitGrad, 'uInit_')
      self.record_layer_gradient(vMseInitGrad, 'vInit_')
      self.record_layer_gradient(cMseInitGrad, 'cInit_')
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat


  def test_step(self, data):
    xy      = data[0][0]
    t       = data[0][1]
    uv      = data[0][2]
    xyBc    = data[0][3]
    uvpBc   = data[0][4]
    validBc = data[0][5]
    w       = data[0][6]
    phi     = data[0][7]

    # Compute the data and pde losses
    uvpPred, uMse, vMse, pMse, cMse, pdeMse0, pdeMse1, pdeMse2, pdeMse3, uMseWalls, vMseWalls, pMseWalls, uMseInit, vMseInit, cMseInit = \
      self.compute_losses(xy, t, w, phi, uv, xyBc, uvpBc, validBc)
    # replica's loss, divided by global batch size
    loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse + self.alpha[2]*pMse + self.alpha[3]*cMse \
            + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2 + self.beta[3]*pdeMse3 \
            + self.gamma[0]*uMseWalls + self.gamma[1]*vMseWalls + self.gamma[2]*pMseWalls \
            + self.delta[0]*uMseInit + self.delta[1]*vMseInit + self.delta[2]*cMseInit)

    # Track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pMse'].update_state(pMse)
    self.validMetrics['cMse'].update_state(cMse)
    self.validMetrics['pde0'].update_state(pdeMse0)
    self.validMetrics['pde1'].update_state(pdeMse1)
    self.validMetrics['pde2'].update_state(pdeMse2)
    self.validMetrics['pde3'].update_state(pdeMse3)
    self.validMetrics['uMseWalls'].update_state(uMseWalls)
    self.validMetrics['vMseWalls'].update_state(vMseWalls)
    self.validMetrics['pMseWalls'].update_state(pMseWalls)
    self.validMetrics['uMseInit'].update_state(uMseInit)
    self.validMetrics['vMseInit'].update_state(vMseInit)
    self.validMetrics['cMseInit'].update_state(cMseInit)
    self.validMetrics['Re'].update_state(self.Re)
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
    print('Coefficients for data loss {} {} {} {}'.format(\
          self.alpha[0], self.alpha[1], self.alpha[2], self.alpha[3]))
    print('Coefficients for pde residual {} {} {} {}'.format(\
          self.beta[0], self.beta[1], self.beta[2], self.beta[3]))
    print('Coefficients for domain wall loss {} {} {}'.format(\
          self.gamma[0], self.gamma[1], self.gamma[2]))
    print('Coefficients for initial condition loss {} {} {}'.format(\
          self.delta[0], self.delta[1], self.delta[2]))
    print('--------------------------------')
