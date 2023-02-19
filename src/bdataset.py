#!/usr/bin/env python3

import h5py as h5
import matplotlib.pyplot as plt
import os.path
import sys
import copy
import numpy as np
import random as rn
import tensorflow as tf
import warnings
try:
  import boxkit.api as boxkit
  haveBoxkit = True
except ImportError:
  warnings.warn('Boxkit not found. Creating datasets from flashx source will not be possible.')
  haveBoxkit = False

import butils as UT

rn.seed(2022)
np.random.seed(2022)

class BubbleDataSet:

  FLAG_FLUID = 0
  FLAG_BUBBLE = 1
  FLAG_VISITED = 1

  def __init__(self, fName='', startFrame=0, endFrame=399, dim=2, \
               wallPoints=-1, colPoints=-1, dataPoints=-1, walls=[1,1,1,1], \
               interface=1, source=UT.SRC_FLOWNET):
    assert dim == 2, "Only supporting 2D datasets"
    self.fName        = fName
    self.dim          = dim
    self.size         = np.zeros(self.dim, dtype=int)
    self.isLoaded     = False
    self.startFrame   = startFrame
    self.endFrame     = endFrame
    self.nTotalFrames = endFrame - startFrame + 1 # Include start and end frame
    self.walls        = walls
    self.interface    = interface
    self.source       = source
    self.sourceName   = ''
    self.phiInit      = 99 # Initial levelset value
    # Lists to count number of cells per frame
    self.nBcBubble = []
    self.nFluid    = []
    self.nWalls    = []
    # Requested number of points
    self.nColPnt  = colPoints
    self.nWallPnt = wallPoints
    self.nDataPnt = dataPoints
    # Array to store ground truth data (after processing .flo input)
    self.vel      = None # [frames, width, height, dim + 1]
    # Arrays to store processed data (after processing ground truth)
    self.bc       = None # [nSamples, dim + 1]
    self.xyBc     = None # [nSamples, dim + 1]
    self.xyFluid  = None # [nSamples, dim + 1]
    self.uvpFluid = None # [nSamples, dim + 1]
    self.bcDomain = None # [nSamples, dim + 1]
    self.xyDomain = None # [nSamples, dim + 1]
    # Arrays to store selection of points
    self.xyCol    = None # [nColPnt,  dim + 1]
    self.uvpCol   = None # [nColPnt,  dim + 1]
    self.xyData   = None # [nDataPnt, dim + 1]
    self.uvData   = None # [nDataPnt, dim + 1]
    self.xyWalls  = None # [nWallPnt, dim + 1]
    self.uvWalls  = None # [nWallPnt, dim + 1]
    # Arrays to store (shuffled) mix of data and collocation points
    self.labels   = None
    self.xyt      = None
    self.id       = None
    # Resolution of collocation points (only use every other points as col point)
    self.colRes = 1
    # FlashX
    self.pshape = [] # Num of points per block in each dimension
    self.bshape = [] # Num of blocks in each dimension


  def _load_flashx(self):
    assert haveBoxkit, 'Boxkit has not been imported. Cannot load FlashX data'

    frame = self.startFrame
    fNameExact = self.fName % frame
    cnt = 0 # Frame counter
    allocArrays = True

    while os.path.exists(fNameExact) and frame <= self.endFrame:
      UT.print_progress(cnt, self.nTotalFrames)

      dataset = boxkit.read.dataset(fNameExact, source="flash")

      # Block count in each dimension (z-y-x)
      bshapeZ = int(dataset.zmax-dataset.zmin)
      if not bshapeZ: bshapeZ = 1 # Use z=1 if in 2D
      bshape = (bshapeZ, int(dataset.ymax-dataset.ymin), int(dataset.xmax-dataset.xmin))
      # Points per block in each dimension (z-y-x)
      self.pshape = (dataset.nzb, dataset.nyb, dataset.nxb)
      # Overall size (x-y)
      self.size = np.array([bshape[2]*self.pshape[2], bshape[1]*self.pshape[1]])

      # Only allocate arrays once (assume same dimension for every frame)
      if allocArrays:
        temperature = np.zeros((self.nTotalFrames, self.size[1], self.size[0]), dtype=float)
        pressure = np.zeros((self.nTotalFrames, self.size[1], self.size[0]), dtype=float)
        self.vel = np.zeros((self.nTotalFrames, self.size[1], self.size[0], self.dim + 1), dtype=float)
        allocArrays = False
      else:
        assert temperature.shape[1] == self.size[1] and temperature.shape[2] == self.size[0]
        assert pressure.shape[1] == self.size[1] and pressure.shape[2] == self.size[0]
        assert self.vel.shape[1] == self.size[1] and self.vel.shape[2] == self.size[0]

      bcnt = 0 # Current block number
      for bk in range(bshape[0]):
        for bj in range(bshape[1]):
          for bi in range(bshape[2]):
            for pk in range(self.pshape[0]):
              for pj in range(self.pshape[1]):
                # Get one block line for every attribute
                temp = dataset.blocklist[bcnt]['temp'][pk][pj]
                pres = dataset.blocklist[bcnt]['pres'][pk][pj]
                dfun = dataset.blocklist[bcnt]['dfun'][pk][pj]
                velx = dataset.blocklist[bcnt]['velx'][pk][pj]
                vely = dataset.blocklist[bcnt]['vely'][pk][pj]
                # Write block line into arrays
                sx = bi * self.pshape[2] # start in x dimension
                ex = sx + self.pshape[2] # end in x dimension
                sy = bj * self.pshape[1] + pj # start in y dimension
                temperature[cnt, sy, sx:ex] = temp[0:self.pshape[2]]
                pressure[cnt, sy, sx:ex] = pres[0:self.pshape[2]]
                self.vel[cnt, sy, sx:ex, 0] = velx[0:self.pshape[2]]
                self.vel[cnt, sy, sx:ex, 1] = vely[0:self.pshape[2]]
                self.vel[cnt, sy, sx:ex, 2] = dfun[0:self.pshape[2]]
            bcnt += 1

      if 0 and UT.PRINT_DEBUG:
        tempMin, tempMax = temperature[cnt].min(), temperature[cnt].max()
        print('Frame {}: Min/Max temp [{},{}]'.format(frame, tempMin, tempMax))
        presMin, presMax = pressure[cnt].min(), pressure[cnt].max()
        print('Frame {}: Min/Max pressure [{},{}]'.format(frame, presMin, presMax))

        velx = abs(max(self.vel[cnt,:,:,0].min(), self.vel[cnt,:,:,0].max(), key = abs))
        vely = abs(max(self.vel[cnt,:,:,1].min(), self.vel[cnt,:,:,1].max(), key = abs))
        print('Frame {}: Maximum vel [{}, {}]'.format(frame, velx, vely))
        dfunMin, dfunMax = self.vel[cnt,:,:,2].min(), self.vel[cnt,:,:,2].max()
        print('Frame {}: Min/Max levelset [{},{}]'.format(frame, dfunMin, dfunMax))

      if UT.IMG_DEBUG:
        # Raw grids
        UT.save_image(temperature[cnt], '{}/raw'.format(self.sourceName), 'flashx_temp_raw', frame, cmap='jet')
        UT.save_image(pressure[cnt], '{}/raw'.format(self.sourceName), 'flashx_pres_raw', frame, cmap='jet')
        UT.save_image(self.vel[cnt,:,:,0], '{}/raw'.format(self.sourceName), 'flashx_velx_raw', frame, cmap='hot')
        UT.save_image(self.vel[cnt,:,:,1], '{}/raw'.format(self.sourceName), 'flashx_vely_raw', frame, cmap='hot')
        UT.save_image(self.vel[cnt,:,:,2], '{}/raw'.format(self.sourceName), 'flashx_phi_raw', frame, cmap='jet', vmin=-0.5, vmax=0.5)
        # Velocity matplotlib plots
        UT.save_velocity(self.vel[cnt], '{}/plots'.format(self.sourceName), 'flashx_vel_stream', frame, size=self.size, type='stream', density=5.0)
        UT.save_velocity(self.vel[cnt], '{}/plots'.format(self.sourceName), 'flashx_vel_vector', frame, size=self.size, type='quiver', arrow_res=8, cmin=0.0, cmax=5.0)
        # Grid matplotlib plots
        UT.save_plot(temperature[cnt], '{}/plots'.format(self.sourceName), 'flashx_temp_plt', frame, size=self.size, cmin=0.0, cmax=1.0, cmap='jet')
        UT.save_plot(pressure[cnt], '{}/plots'.format(self.sourceName), 'flashx_pres_plt', frame, size=self.size, cmin=0.0, cmax=1.0, cmap='jet')
        UT.save_plot(self.vel[cnt,:,:,2], '{}/plots'.format(self.sourceName), 'flashx_phi_plt', frame, size=self.size, cmin=-1.0, cmax=1.0, cmap='jet')
        # Distribution velocities in bins
        UT.save_velocity_bins(self.vel[cnt,:,:,0], '{}/histograms'.format(self.sourceName), 'flashx_velx_bins', frame, bmin=-1.0, bmax=1.0, bstep=0.05)
        UT.save_velocity_bins(self.vel[cnt,:,:,1], '{}/histograms'.format(self.sourceName), 'flashx_vely_bins', frame, bmin=-1.0, bmax=1.0, bstep=0.05)

      # Next array index
      cnt += 1
      # Next frame index
      frame += 1
      fNameExact = self.fName % frame

    if UT.VID_DEBUG:
      # Ffmpeg videos
      UT.save_video(subdir='simulation', name='velocity_stream', imgDir='../img/simulation/plots/', fps=15)
      UT.save_video(subdir='simulation', name='velocity_vector', imgDir='../img/simulation/plots/', fps=15)
      UT.save_video(subdir='simulation', name='temperature_plt', imgDir='../img/simulation/plots/', fps=15)
      UT.save_video(subdir='simulation', name='pressure_plt', imgDir='../img/simulation/plots/', fps=15)
      UT.save_video(subdir='simulation', name='levelset_plt', imgDir='../img/simulation/plots/', fps=15)

    return cnt == (self.endFrame - self.startFrame + 1)


  def _load_flownet(self, shuffle=True):
    frame = self.startFrame
    fNameExact = self.fName % frame
    cnt = 0 # Frame counter
    allocArrays = True

    while os.path.exists(fNameExact) and frame <= self.endFrame:
      UT.print_progress(cnt, self.nTotalFrames)

      dataset = open(fNameExact, 'rb')
      magic = np.fromfile(dataset, np.float32, count = 1)
      sizeFromFile = np.fromfile(dataset, np.int32, count = self.dim)
      dataFromFile = np.fromfile(dataset, np.float32, count = self.dim * sizeFromFile[0] * sizeFromFile[1])
      dataFromFile = np.resize(dataFromFile, (sizeFromFile[1], sizeFromFile[0], self.dim))

      # Only allocate arrays once (assume same dimension for every frame)
      if allocArrays:
        self.vel = np.zeros((self.nTotalFrames, sizeFromFile[1], sizeFromFile[0], self.dim + 1), dtype=float)
        # Initialize 3rd dimension with negative phi init value
        self.vel[:,:,:,2] = -1.0 * self.phiInit
        allocArrays = False
      else:
        assert sizeFromFile[0] == self.size[0], 'Width in dataset does not match width from files, {} vs {}'.format(self.size[0], sizeFromFile[0])
        assert sizeFromFile[1] == self.size[1], 'Height in dataset does not match width from files, {} vs {}'.format(self.size[1], sizeFromFile[1])

      # Copy read data into data structures
      self.size = sizeFromFile
      assert (self.size[0] + self.size[1]) * 2 * self.nTotalFrames >= self.nWallPnt, "Maximum number of bc domain points exceeded"

      # Use origin 'lower' format
      dataFromFile = dataFromFile[::-1,:,:]
      dataFromFile[:,:,1] *= -1

      self.vel[cnt,:,:,:self.dim] = dataFromFile

      if UT.IMG_DEBUG:
        # Raw grids
        UT.save_image(dataFromFile[:,:,0], '{}/raw'.format(self.sourceName), 'flownet_velx_raw', frame, cmap='hot')
        UT.save_image(dataFromFile[:,:,1], '{}/raw'.format(self.sourceName), 'flownet_vely_raw', frame, cmap='hot')
        # Velocity matplotlib plots
        UT.save_velocity(dataFromFile, '{}/plots'.format(self.sourceName), 'flownet_vel_stream', frame, size=self.size, type='stream', density=5.0)
        UT.save_velocity(dataFromFile, '{}/plots'.format(self.sourceName), 'flownet_vel_vector', frame, size=self.size, type='quiver', arrow_res=8, cmin=0.0, cmax=5.0)
        # Distribution velocities in bins
        UT.save_velocity_bins(dataFromFile[:,:,0], '{}/histograms'.format(self.sourceName), 'flownet_velx_bins', frame, bmin=-3.0, bmax=3.0, bstep=0.1)
        UT.save_velocity_bins(dataFromFile[:,:,1], '{}/histograms'.format(self.sourceName), 'flownet_vely_bins', frame, bmin=-3.0, bmax=3.0, bstep=0.1)

      # Next array index
      cnt += 1
      # Next frame index
      frame += 1
      fNameExact = self.fName % frame

    return cnt == (self.endFrame - self.startFrame + 1)


  def load_data(self, source):
    UT.print_info('Loading training / validation data files')

    self.source = source
    sourceName, isLoaded = None, False
    if source == UT.SRC_FLOWNET:
      self.sourceName = 'flownet'
      isLoaded = self._load_flownet()
    elif source == UT.SRC_FLASHX:
      self.sourceName = 'flashx'
      isLoaded = self._load_flashx()

    loadSuccess = 'Loaded {} dataset'.format(self.sourceName) if isLoaded else 'Could not load {} dataset'.format(self.sourceName)
    print(loadSuccess)

    self.isLoaded = isLoaded
    return isLoaded, self.sourceName


  def summary(self):
    if not self.isLoaded:
      print('Bubble DataSet has not been loaded yet, end summary')
      return
    print('--------------------------------')
    print('Total Frames: {}'.format(self.nTotalFrames))
    print('Domain size: [{}, {}]'.format(self.size[0], self.size[1]))
    print('Min / max in entire dataset:')
    print('  u  {}, {}'.format(np.amin(self.vel[:,:,:,0]), np.amax(self.vel[:,:,:,0])))
    print('  v  {}, {}'.format(np.amin(self.vel[:,:,:,1]), np.amax(self.vel[:,:,:,1])))
    print('  p  {}, {}'.format(np.amin(self.vel[:,:,:,2]), np.amax(self.vel[:,:,:,2])))
    print('--------------------------------')


  # Obtain mask that filters points on walls
  def get_wall_mask(self, xy, offset=[0,0,0,0], useAll=False):
    assert xy.shape[1] >= 2, "xy array must have at least 2 dimensions"
    bWLeft, bWTop = self.walls[0] + offset[0], self.walls[1] + offset[1]
    bWRight, bWBottom = self.walls[2] + offset[2], self.walls[3] + offset[3]
    xMask = np.logical_and(xy[:,0] >= bWLeft, xy[:,0] < self.size[0] - bWRight)
    yMask = np.logical_and(xy[:,1] >= bWBottom, xy[:,1] < self.size[1] - bWTop)
    mask = np.logical_and(xMask, yMask)

    if useAll:
      mask = np.logical_or(mask, True)
    return mask


  def generate_predict_pts(self, begin, end, worldSize, imageSize, fps, L, T, xyPred=[1,1,1], resetTime=True, zeroMean=True):
    print('Generating prediction points')

    for f in range(begin, end):
      uvpData  = self.get_bc(f)
      xytData  = self.get_xy_bc(f)
      uvpFluid = self.get_uvp_fluid(f)
      xytFluid = self.get_xy_fluid(f)
      uvpWalls = self.get_bc_walls(f)
      xytWalls = self.get_xy_walls(f)

      # Get ground truth xyt and uvp of data, collocation, and / or wall points
      xytTarget, uvpTarget = np.empty(shape=(0, self.dim + 1)), np.empty(shape=(0, self.dim + 1))
      print(xytData.shape)
      if xyPred[0]:
        xytTarget = np.concatenate((xytTarget, xytData))
        uvpTarget = np.concatenate((uvpTarget, uvpData))
      if xyPred[1]:
        xytTarget = np.concatenate((xytTarget, xytFluid))
        uvpTarget = np.concatenate((uvpTarget, uvpFluid))
      if xyPred[2]:
        xytTarget = np.concatenate((xytTarget, xytWalls))
        uvpTarget = np.concatenate((uvpTarget, uvpWalls))

      # Only use points within boundaries
      mask = self.get_wall_mask(xytTarget, useAll=True)
      xytTargetMasked = xytTarget[mask]
      uvpTargetMasked = uvpTarget[mask]

      nGridPnt = len(xytTargetMasked)

      # Arrays to store batches
      xy    = np.zeros((nGridPnt, self.dim), dtype=float)
      t     = np.zeros((nGridPnt, 1), dtype=float)
      # Dummies, unused for pred pts
      label = np.zeros((nGridPnt, self.dim + 1),  dtype=float)
      w     = np.zeros((nGridPnt, 1),         dtype=float)

      # Fill batch arrays
      xy[:, :] = xytTargetMasked[:, :self.dim]
      t[:, 0]  = xytTargetMasked[:, self.dim]
      label[:, :] = uvpTargetMasked[:, :]

      # Shift time range to start at zero
      if resetTime:
        print('min, max xyt[0]: [{}, {}]'.format(np.min(t[:,0]), np.max(t[:,0])))
        t[:,0] -= self.startFrame
        if UT.PRINT_DEBUG:
          print(self.startFrame)

      # Zero mean for time range (use -1 to account for 0 in center)
      if zeroMean:
        print(self.nTotalFrames)
        xy[:,0] -= (self.size[0]-1) / 2
        xy[:,1] -= (self.size[1]-1) / 2
        #t[:,0]  -= (self.nTotalFrames-1) / 2

        if UT.PRINT_DEBUG:
          print('min, max xyt[0]: [{}, {}]'.format(np.min(t[:,0]), np.max(t[:,0])))
          print('min, max xyt[1]: [{}, {}]'.format(np.min(xy[:,0]), np.max(xy[:,0])))
          print('min, max xyt[2]: [{}, {}]'.format(np.min(xy[:,1]), np.max(xy[:,1])))

      # Convert from domain space to world space
      pos  = UT.pos_domain_to_world(xy, worldSize, imageSize)
      time = UT.time_domain_to_world(t, fps)

      # Convert from world space to dimensionless quantities
      pos  = UT.pos_world_to_dimensionless(pos, L)
      time = UT.time_world_to_dimensionless(time, T)

      yield [pos, time, label]


  def prepare_batch_arrays(self, zeroInitialCollocation=False, resetTime=True, zeroMean=True):
    print('Preparing samples for batch generator')
    rng = np.random.default_rng(2022)

    # Extract selection of data points (otherwise use all)
    useAllDataPts = (self.nDataPnt < 0)
    self.select_data_points(useAllDataPts)

    # Extract selection of collocation points
    useDefaultColPts = (self.nColPnt < 0)
    self.select_collocation_points(useDefaultColPts)

    # Extract selection of wall points (otherwise use all)
    useAllWallPts = (self.nWallPnt < 0)
    self.select_wall_points(useAllWallPts)

    # TODO: Add option to use old collocation point label init
    if 0:
      # Init collocation point labels with 0.0
      if zeroInitialCollocation:
        colPntLabel = 0.0
        uvCol = np.full((self.nColPnt, self.dim + 1), colPntLabel)
      else: # Init collocation point label with random values in range of min/max of data point label
        minU, maxU = np.min(self.bc[:,0]), np.max(self.bc[:,0])
        minV, maxV = np.min(self.bc[:,1]), np.max(self.bc[:,1])
        samplesU = np.random.uniform(low=minU, high=maxU, size=self.nColPnt)
        samplesV = np.random.uniform(low=minV, high=maxV, size=self.nColPnt)
        samplesU = np.expand_dims(samplesU, axis=-1)
        samplesV = np.expand_dims(samplesV, axis=-1)
        samplesP = np.zeros(self.nColPnt)
        samplesP = np.expand_dims(samplesP, axis=-1)
        uvCol = np.concatenate((samplesU, samplesV, samplesP), axis=-1)

    uvpSamples = np.concatenate((self.uvData, self.uvpCol, self.uvWalls))
    xytSamples = np.concatenate((self.xyData, self.xyCol, self.xyWalls))
    dataId     = np.full((len(self.xyData), 1), 1)
    colId      = np.full((len(self.xyCol), 1), 0)
    wallId     = np.full((len(self.xyWalls), 1), 2)
    idSamples  = np.concatenate((dataId, colId, wallId))

    assert len(uvpSamples) == len(xytSamples)
    assert len(uvpSamples) == len(idSamples)

    # Shuffle the combined point arrays. Shuffle all arrays with same permutation
    perm = np.random.permutation(len(uvpSamples))
    self.labels = uvpSamples[perm]
    self.xyt    = xytSamples[perm]
    self.id     = idSamples[perm]

    # Shift time range to start at zero
    if resetTime:
      self.xyt[:,2] -= self.startFrame

    # Zero mean for pos range (use -1 to account for 0 in center)
    if zeroMean:
      self.xyt[:,0] -= (self.size[0]-1) / 2
      self.xyt[:,1] -= (self.size[1]-1) / 2
      #self.xyt[:,2] -= (self.nTotalFrames-1) / 2

      print('min, max xyt[0]: [{}, {}]'.format(np.min(self.xyt[:,0]), np.max(self.xyt[:,0])))
      print('min, max xyt[1]: [{}, {}]'.format(np.min(self.xyt[:,1]), np.max(self.xyt[:,1])))
      print('min, max xyt[2]: [{}, {}]'.format(np.min(self.xyt[:,2]), np.max(self.xyt[:,2])))

    initialCond = 0
    # Use the collocation points at the initial timestamp to set initial condition
    if initialCond:
      tInit   = np.min(self.xyt[:,2])         # Use min timestamp as the initial time
      isTInit = (self.xyt[:,2] == tInit)      # Select all t's at initial time
      isCol   = (self.id[:,0] == 0)           # Select all collocation points
      isInit = np.logical_and(isTInit, isCol) # The initial condition points are here
      self.id[isInit] = 3                     # Assign new id to these points

    dataColPts = 0
    if dataColPts:
      indices = np.arange(0, self.xyCol.shape[0])
      nTakeCol = int(self.nColPnt * 0.5)
      randIndices = rng.choice(indices, size=nTakeCol, replace=False)
      self.id[randIndices] = 4

      if UT.IMG_DEBUG:
        UT.save_array(self.xyCol[randIndices, :], '{}/collocation'.format(self.sourceName), 'colptslabelled_select', 30, self.size)

    assert len(self.labels) == len(self.xyt)
    assert len(self.labels) == len(self.id)


  def generate_train_valid_batch(self, begin, end, worldSize, imageSize, fps, \
                                 V, L, T, batchSize=64, shuffle=True):
    generatorType = 'training' if begin == 0 else 'validation'
    UT.print_info('\nGenerating {} sample {} batches'.format(batchSize, generatorType))

    # Arrays to store data + collocation + wall point batch
    uv     = np.zeros((batchSize, self.dim + 1), dtype=float)
    xy     = np.zeros((batchSize, self.dim), dtype=float)
    t      = np.zeros((batchSize, 1),        dtype=float)
    id     = np.zeros((batchSize, 1),        dtype=float)
    phi    = np.zeros((batchSize, 1),        dtype=float)

    s = begin
    while True:
      if s + batchSize > end:
        s = begin
      e = s + batchSize

      # Fill batch arrays
      uv[:, :self.dim] = self.labels[s:e, :self.dim]
      phi[:, 0]        = self.labels[s:e, self.dim] # phi for now
      xy[:, :]         = self.xyt[s:e, :self.dim]
      t[:, 0]          = self.xyt[s:e, self.dim]
      id[:, 0]         = self.id[s:e, 0]

      s  += batchSize

      # Shuffle batch
      if shuffle:
        assert len(xy) == batchSize
        perm = np.random.permutation(batchSize)
        xy   = xy[perm]
        t    = t[perm]
        id   = id[perm]
        uv   = uv[perm]
        phi  = phi[perm]

      # Convert from domain space to world space
      pos  = UT.pos_domain_to_world(xy, worldSize, imageSize)
      time = UT.time_domain_to_world(t, fps)

      # Convert from world space to dimensionless quantities
      pos  = UT.pos_world_to_dimensionless(pos, L)
      time = UT.time_world_to_dimensionless(time, T)

      # Only non-dimensionalize velocities from flownet dataset
      vel = uv
      if self.source == UT.SRC_FLOWNET:
        vel = UT.vel_domain_to_world(uv, worldSize, fps)
        vel = UT.vel_world_to_dimensionless(vel, V)

      yield [pos, time, vel, id, phi]


  # Define domain border locations + attach bc
  def extract_wall_points(self, useDataBc=False):
    UT.print_info('Extracting domain wall points')

    rng = np.random.default_rng(2022)

    if not sum(self.walls):
      if UT.PRINT_DEBUG: print('No walls defined. Not extracting any wall points')
      return

    bcFrameLst = [] # Domain boundary condition for every frame
    xyFrameLst = [] # Domain boundary condition xy for every frame

    sizeX, sizeY = self.size

    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      hasLeftWall   = self.walls[0] > 0
      hasTopWall    = self.walls[1] > 0
      hasRightWall  = self.walls[2] > 0
      hasBottomWall = self.walls[3] > 0

      # Boundary width position offset, bwidth=1 => 0 offset (therefore the -1)
      bWLeft   = self.walls[0]-1 if self.walls[0] else 0
      bWTop    = self.walls[1]-1 if self.walls[1] else 0
      bWRight  = self.walls[2]-1 if self.walls[2] else 0
      bWBottom = self.walls[3]-1 if self.walls[3] else 0

      assert bWLeft >= 0 and bWTop >= 0 and bWRight >= 0 and bWBottom >= 0, 'Wall position out scope, cannot be negative'

      # Zero boundary condition
      uvpZero = [0, 0, 0]

      # xyt positions, using origin 'lower' (x from left to right, y from bottom to top)
      xyLst = []
      cellCnt = 0

      # Walls:
      # Left domain wall
      for i in range(self.walls[0]):
        numCells = sizeY - bWBottom - bWTop - 2 # -2 to exclude corners
        x = np.full((numCells,), bWLeft-i, dtype=float)
        y = np.linspace(bWBottom+1, sizeY-1-bWTop-1, num=numCells, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        xyLst.extend(list(zip(x,y,t)))
        cellCnt += numCells

      # Top domain wall
      for i in range(self.walls[1]):
        numCells = sizeX - bWLeft - bWRight -2
        x = np.linspace(bWLeft+1, sizeX-1-bWRight-1, num=numCells, dtype=float)
        y = np.full((numCells,), sizeY-1-bWTop+i, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        xyLst.extend(list(zip(x,y,t)))
        cellCnt += numCells

      # Right domain wall
      for i in range(self.walls[2]):
        numCells = sizeY - bWTop - bWBottom - 2
        x = np.full((numCells,), sizeX-1-bWRight+i, dtype=float)
        y = np.linspace(sizeY-1-bWTop-1, bWBottom+1, num=numCells, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        xyLst.extend(list(zip(x,y,t)))
        cellCnt += numCells

      # Bottom domain wall
      for i in range(self.walls[3]):
        numCells = sizeX - bWRight - bWLeft - 2
        x = np.linspace(sizeY-1-bWRight-1, bWLeft+1, num=numCells, dtype=float)
        y = np.full((numCells,), bWBottom-i, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        xyLst.extend(list(zip(x,y,t)))
        cellCnt += numCells

      # Corners:
      # Top left corner
      if hasLeftWall or hasTopWall:
        acc = []
        for i in range(0, max(1, self.walls[0])): # left
          for j in range(min(sizeY-self.walls[1], sizeY-1), sizeY): # top
            acc.append((i,j,frame))
            cellCnt += 1
        xyLst.extend(acc)

      # Top right corner
      if hasRightWall or hasTopWall:
        acc = []
        for i in range(min(sizeX-self.walls[2], sizeX-1), sizeX): # right
          for j in range(min(sizeY-self.walls[1], sizeY-1), sizeY): # top
            acc.append((i,j,frame))
            cellCnt += 1
        xyLst.extend(acc)

      # Bottom left corner
      if hasLeftWall or hasBottomWall:
        acc = []
        for i in range(0, max(1, self.walls[0])): # left
          for j in range(0, max(1, self.walls[3])): # bottom
            acc.append((i,j,frame))
            cellCnt += 1
        xyLst.extend(acc)

      # Bottom right corner
      if hasRightWall or hasBottomWall:
        acc = []
        for i in range(min(sizeX-self.walls[2], sizeX-1), sizeX): # right
          for j in range(0, max(1, self.walls[3])): # bottom
            acc.append((i,j,frame))
            cellCnt += 1
        xyLst.extend(acc)

      # uvp labels
      bcLst = []

      # Using actual boundary values from dataset for wall uvp
      if useDataBc:
         # Override bc list and copy value from dataset
        for i, j, _ in xyLst:
          xi, yi = int(i), int(j)
          curUvp = self.vel[f,yi,xi,:]
          bcLst.append(curUvp)
      else:
        bcLst.extend(np.tile(uvpZero, (cellCnt, 1)))

      bcFrameLst.append(bcLst)
      xyFrameLst.append(xyLst)

      if UT.IMG_DEBUG:
        UT.save_array(xyLst, '{}/all'.format(self.sourceName), 'wallpts_all', frame, self.size)

      # Keep track of number of wall cells per frame
      self.nWalls.append(len(xyLst))

    if UT.PRINT_DEBUG:
      print('Total number of wall samples: {}'.format(np.sum(self.nWalls)))

    self.bcDomain = np.zeros((np.sum(self.nWalls), self.dim + 1))
    self.xyDomain = np.zeros((np.sum(self.nWalls), self.dim + 1))

    s = 0
    for f in range(self.nTotalFrames):
      assert len(bcFrameLst[f]) == len(xyFrameLst[f]), 'Number of velocity labels must match number of xy positions'

      n = len(bcFrameLst[f])
      e = s + n
      if n:
        uvp = np.asarray(bcFrameLst[f], dtype=float)
        xyt = np.asarray(xyFrameLst[f], dtype=float)
        self.bcDomain[s:e, :] = uvp
        self.xyDomain[s:e, :] = xyt
      s = e


  def _thicken_interface(self, n, flags, intersection, phi=None, fromInside=True):
    sizeX, sizeY = self.size
    if n <= 1: # Interface already has 1 pixel thickness
      return

    nz = np.nonzero(intersection)
    indices = list(zip(nz[0], nz[1]))
    for x in range(n): # Thicken interface by n-1 pixels
      nextIndices = []
      for idx in (indices):
        i, j = idx[0], idx[1]
        neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        for ng in neighbors:
          ni, nj = ng[0], ng[1]
          if ni < 0 or ni >= sizeX or nj < 0 or nj >= sizeY:
            continue
          # Thicken interface from the inside or outside
          if fromInside:
            if flags[ni, nj] == self.FLAG_FLUID: continue
          else:
            if flags[ni, nj] == self.FLAG_BUBBLE: continue
          if intersection[ni, nj] == 1:
            continue
          if x < n-1:
            intersection[ni, nj] = 1
          if phi is not None:
            phi[ni, nj] = (x+1) if fromInside else -(x+1)
          nextIndices.append((ni, nj))
      indices = nextIndices


  def _extract_flashx_points(self, phi, intersection, flags):
    sizeX, sizeY = self.size

    for j in range(sizeY):
      for i in range(sizeX):
        isBubble = phi[i,j] > 0
        isFluid = not isBubble
        # Safe grid indices
        iM, iP, jM, jP = max(i-1,0), min(i+1, sizeX-1), max(j-1,0), min(j+1, sizeY-1)
        
        isIntersection = ( phi[iM, j ] > 0 or phi[iM, j ] > 0 or \
                           phi[i,  jP] > 0 or phi[i,  jM] > 0 or \
                           phi[iP, jP] > 0 or phi[iM, jP] > 0 or \
                           phi[iP, jM] > 0 or phi[iM, jM] > 0) and phi[i,j] <= 0
        intersection[i, j] = isIntersection
        flags[i, j] = self.FLAG_BUBBLE if isBubble else self.FLAG_FLUID


  def _extract_flownet_points(self, mag, intersection, flags, velEps):
    sizeX, sizeY = self.size

    # Visited == grid with all cells that have been visited during flood-fill
    visited = np.zeros((sizeX, sizeY), dtype=int)

    cornerCells = [(0,0), (0,sizeY-1), (sizeX-1,sizeY-1), (sizeX-1,0)]
    # Initialize flood-fill search with corner cells of grid
    for i, j in cornerCells:
      if mag[i, j] < velEps:
        flags[i, j] = self.FLAG_FLUID
      # Skip cells that are already marked
      if (visited[i, j] == self.FLAG_VISITED):
        continue
      visited[i, j] = self.FLAG_VISITED

      stack = [(i, j)] # Init stack with current border cell coords
      while (len(stack) > 0):
        c = stack.pop()
        i, j = c[0], c[1]
        visited[i, j] = self.FLAG_VISITED # Mark current cell as visited
        neighbors = [(i+1, j  ), (i-1, j  ), (i  , j+1), (i  , j-1),
                     (i+1, j+1), (i-1, j+1), (i+1, j-1), (i-1, j-1)]
        for n in neighbors:
          # Only proceed if neigbor is valid, ie if cell is inside domain borders
          if n[0] < 0 or n[0] >= sizeX or n[1] < 0 or n[1] >= sizeY:
            continue
          ni, nj = n[0], n[1]
          # Also skip if neighbor cell has already been visited
          if (visited[ni, nj] == self.FLAG_VISITED):
            continue
          # Now mark neighbor cell as visited
          visited[ni, nj] = self.FLAG_VISITED
          # Cut-off based on velocity, ie if neighbor has no vel it is fluid
          if mag[ni, nj] < velEps:
            stack.append(n) # Add fluid cell to stack, it needs to be searched next
            flags[ni, nj] = self.FLAG_FLUID
          else:
            intersection[ni, nj] = 1 # Mark this cell in boundary condition mask

    # Smooth out single pixels
    for iter in range(1): # Number of smoothing passes
      for i in range(sizeX): # Loop over entire domain
        for j in range(sizeY):
          # Prevent out of range indexing
          if i <= 0 or i >= sizeX-1 or j <= 0 or j >= sizeY-1:
            continue
          # Skip fluid cells, only start smoothing from bubble cells
          if flags[i,j] == self.FLAG_FLUID:
            continue
          neighbors = [(i+1, j  ), (i-1, j  ), (i  , j+1), (i  , j-1),
                       (i+1, j+1), (i-1, j+1), (i+1, j-1), (i-1, j-1)]
          fluidNeighborCnt = 0
          for n in neighbors:
            if flags[n] == self.FLAG_FLUID: fluidNeighborCnt += 1
          if flags[i,j] == self.FLAG_BUBBLE and fluidNeighborCnt >= 5:
            flags[i,j] = self.FLAG_FLUID
            intersection[i,j] = 0


  def extract_fluid_points(self, velEps=0.1):
    UT.print_info('Extracting bubble and fluid points')

    bcFrameLst = []
    xyFrameLst = []
    uvpFluidFrameLst = []
    xyFluidFrameLst = []
    sizeX, sizeY = self.size

    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      # Intersection == grid with all cells at fluid / bubble intersection
      intersection = np.zeros((sizeX, sizeY), dtype=int)
      # Flags == grid with the type of the cell (fluid or bubble)
      flags = np.full((sizeX, sizeY), self.FLAG_BUBBLE)

      # Velocity magnitude per cell
      U, V = self.vel[f,:,:,0], self.vel[f,:,:,1]
      mag = np.sqrt(np.square(U) + np.square(V))

      # Thicken interface and generate levelset if needed (flownet only)
      assert self.source in [UT.SRC_FLOWNET, UT.SRC_FLASHX], "Unknown fluid points source"
      curPhi = self.vel[f,:,:,2]
      if self.source == UT.SRC_FLOWNET:
        self._extract_flownet_points(mag, intersection, flags, velEps)
        self._thicken_interface(self.interface, flags, intersection, phi=None, fromInside=True)
        curPhi[np.array(flags, dtype=bool)] = 0
        tmpIntersection = copy.deepcopy(intersection)
        self._thicken_interface(self.phiInit, flags, tmpIntersection, phi=curPhi, fromInside=False)
      elif self.source == UT.SRC_FLASHX:
        self._extract_flashx_points(curPhi, intersection, flags)
        self._thicken_interface(self.interface, flags, intersection, curPhi, fromInside=False)

      if UT.IMG_DEBUG:
        UT.save_image(intersection, '{}/all'.format(self.sourceName), 'datapts_all', frame, i=self.interface)
        UT.save_image(mag, '{}/all'.format(self.sourceName), 'magnitude_all', frame)
        UT.save_image(curPhi, '{}/all'.format(self.sourceName), 'phi_all', frame, cmap='jet')
        if self.source == UT.SRC_FLOWNET:
          UT.save_plot(self.vel[f,:,:,2], '{}/plots'.format(self.sourceName), 'flownet_phi_plt', frame, size=self.size, cmin=np.min(curPhi), cmax=np.max(curPhi), cmap='jet')

      # Add bubble border positions to xyBc list
      nzIndices = np.nonzero(intersection)
      bubbleBorderIndices = list(zip(nzIndices[1], nzIndices[0])) # np.nonzero returns data in [rows, columns], ie [y,x]
      xyFrameLst.append(bubbleBorderIndices)

      # Add bubble border velocities to bc list, same order as indices list
      bcLst = []
      for idx in zip(nzIndices[0], nzIndices[1]):
        i, j = idx[0], idx[1]
        curUvp = self.vel[f,i,j,:]
        bcLst.append(curUvp)
      bcFrameLst.append(bcLst)

      assert len(bcLst) == len(bubbleBorderIndices), 'Number of data point velocities must match number of indices'

      # Create fluid mask
      exclude = flags
      if self.source == UT.SRC_FLASHX:
        exclude += intersection
      fluidmask = 1 - exclude
      if UT.IMG_DEBUG:
        UT.save_image(fluidmask, '{}/all'.format(self.sourceName), 'colpts_all', frame)
      assert np.sum(fluidmask) == (sizeX*sizeY - np.sum(exclude) ), 'Fluid mask must match total size minus bubble mask'
      nzIndices = np.nonzero(fluidmask)
      fluidIndices = list(zip(nzIndices[1], nzIndices[0])) # np.nonzero returns data in [rows, columns], ie [y,x]
      xyFluidFrameLst.append(fluidIndices)

      # Add fluid properties to list, same order as indices list
      uvpLst = []
      for idx in zip(nzIndices[0], nzIndices[1]):
        i, j = idx[0], idx[1]
        curUvp = self.vel[f,i,j,:]
        uvpLst.append(curUvp)
      uvpFluidFrameLst.append(uvpLst)

      # Keep track of number of bubble border cells and fluid cells per frame
      self.nBcBubble.append(len(bubbleBorderIndices))
      self.nFluid.append(len(fluidIndices))

    if UT.PRINT_DEBUG:
      print('Total number of bubble points: {}'.format(np.sum(self.nBcBubble)))
      print('Total number of fluid points: {}'.format(np.sum(self.nFluid)))

    if UT.VID_DEBUG:
      UT.save_video(subdir='extract', name='flags_extract', imgDir='../img/extract/', fps=15)
      UT.save_video(subdir='extract', name='dataPts_extract_i{:02d}'.format(self.interface), imgDir='../img/extract/', fps=15)
      UT.save_video(subdir='extract', name='magnitude_extract', imgDir='../img/extract/', fps=15)

    # Allocate arrays for preprocessed data ...
    self.bc   = np.zeros((np.sum(self.nBcBubble), self.dim + 1), dtype=float) # dim + 1 for p
    self.xyBc = np.zeros((np.sum(self.nBcBubble), self.dim + 1), dtype=float) # dim + 1 for t
    self.xyFluid = np.zeros((np.sum(self.nFluid), self.dim + 1), dtype=float)
    self.uvpFluid = np.zeros((np.sum(self.nFluid), self.dim + 1), dtype=float)

    # ... and insert data from lists
    s, sFl = 0, 0
    for f in range(self.nTotalFrames):
      assert len(bcFrameLst[f]) == len(xyFrameLst[f]), 'Number of labels must match number of positions'
      assert len(uvpFluidFrameLst[f]) == len(xyFluidFrameLst[f]), 'Number of labels must match number of positions'

      frame = f + self.startFrame

      # Insertion of bc and xyBc lists
      n = len(bcFrameLst[f])
      t = np.full((n, 1), frame, dtype=float)
      e = s + n
      if n: # only insert if there is at least 1 cell
        uvp = np.asarray(bcFrameLst[f], dtype=float)
        xy  = np.asarray(xyFrameLst[f], dtype=float)
        self.bc[s:e, :] = uvp
        self.xyBc[s:e, :] = np.hstack((xy, t))
        if UT.IMG_DEBUG:
          UT.save_array(self.xyBc[s:e, :], '{}/positions'.format(self.sourceName), 'xyBc_extract', frame, self.size)

      if UT.PRINT_DEBUG:
        print('Min / max of positions')
        print('  x ({},{}), y ({},{})'.format(np.min(self.xyBc[s:e, 0]), np.max(self.xyBc[s:e, 0]),np.min(self.xyBc[s:e, 1]), np.max(self.xyBc[s:e, 1])))
      s = e

      # Insertion of xyFluid list
      nFl = len(xyFluidFrameLst[f])
      tFl = np.full((nFl, 1), frame, dtype=float)
      eFl = sFl + nFl
      if nFl: # only insert if there is at least 1 cell
        uvp = np.asarray(uvpFluidFrameLst[f], dtype=float)
        xy = np.asarray(xyFluidFrameLst[f], dtype=float)
        self.uvpFluid[sFl:eFl, :] = uvp
        self.xyFluid[sFl:eFl, :] = np.hstack((xy, tFl))
        if UT.IMG_DEBUG:
          UT.save_array(self.xyFluid[sFl:eFl, :], '{}/positions'.format(self.sourceName), 'xyFluid_extract', frame, self.size)
      sFl = eFl

    if UT.IMG_DEBUG:
      UT.save_velocity_bins(self.bc[:, 0], '{}/histograms'.format(self.sourceName), 'bc_x_bins', frame, bmin=-3.0, bmax=3.0, bstep=0.1)
      UT.save_velocity_bins(self.bc[:, 1], '{}/histograms'.format(self.sourceName), 'bc_y_bins', frame, bmin=-3.0, bmax=3.0, bstep=0.1)


  def select_data_points(self, all=False):
    UT.print_info('Selecting data points')
    rng = np.random.default_rng(2022)

    # Use all points that are available
    if all:
      self.xyData = copy.deepcopy(self.xyBc)
      self.uvData = copy.deepcopy(self.bc)

      # Only use points within boundaries
      mask = self.get_wall_mask(self.xyData)
      self.xyData = self.xyData[mask]
      self.uvData = self.uvData[mask]

      self.nDataPnt = len(self.xyData)
      print('Using {} data points'.format(self.nDataPnt))
      return

    # Else, use the exact number of points that was specified
    nDataPntPerFrame = self.nDataPnt // self.nTotalFrames
    # Update actual number of points
    self.nDataPnt = nDataPntPerFrame * self.nTotalFrames
    # Allocate arrays based on actual number of points
    self.xyData = np.zeros((self.nDataPnt, self.dim + 1))
    self.uvData = np.zeros((self.nDataPnt, self.dim + 1))

    print('Using {} data points'.format(self.nDataPnt))

    s = 0
    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      # Get all data point coords and vels for the current frame
      xyDataFrame = self.get_xy_bc(f)
      uvDataFrame = self.get_bc(f)

      # Only use points within boundaries
      mask = self.get_wall_mask(xyDataFrame)
      xyDataFrameMasked = xyDataFrame[mask]
      uvDataFrameMasked = uvDataFrame[mask]

      # Insert random selection of data point coords into data point array
      if UT.PRINT_DEBUG:
        print('Taking {} data points out of {} available'.format(nDataPntPerFrame, xyDataFrame.shape[0]))
      indices = np.arange(0, xyDataFrameMasked.shape[0])
      randIndices = rng.choice(indices, size=nDataPntPerFrame, replace=False)
      e = s + nDataPntPerFrame
      self.xyData[s:e, :] = xyDataFrameMasked[randIndices, :]
      self.uvData[s:e, :] = uvDataFrameMasked[randIndices, :]
      if UT.IMG_DEBUG:
        UT.save_array(self.xyData[s:e, :], '{}/data'.format(self.sourceName), 'datapts_select', frame, self.size)
      s = e


  def select_collocation_points(self, default=False):
    UT.print_info('Selecting collocation points')
    rng = np.random.default_rng(2022)

    # No specific number of collocation points supplied in cmd-line args
    if default:
      self.nColPnt = self.nDataPnt * 10

    nColPntPerFrame = self.nColPnt // self.nTotalFrames
    # Update actual number of points
    self.nColPnt = nColPntPerFrame * self.nTotalFrames
    # Allocate array based on actual number of points
    self.xyCol = np.zeros((self.nColPnt, self.dim + 1))
    self.uvpCol = np.zeros((self.nColPnt, self.dim + 1))

    print('Using {} collocation points'.format(self.nColPnt))

    # Return early if no collocation points requested
    if not self.nColPnt:
      return

    s = 0
    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      # Get all fluid coords for the current frame
      xyFluidFrame = self.get_xy_fluid(f)
      uvpFluidFrame = self.get_uvp_fluid(f)

      # Only use every other grid point (self.colRes == interval) as collocation point
      mask = np.logical_and(xyFluidFrame[:,0] % self.colRes == 0, xyFluidFrame[:,1] % self.colRes == 0)
      xyFluidFrameMasked = xyFluidFrame[mask]
      uvpFluidFrameMasked = uvpFluidFrame[mask]

      # Only use points within boundaries
      mask = self.get_wall_mask(xyFluidFrameMasked)
      xyFluidFrameMasked = xyFluidFrameMasked[mask]
      uvpFluidFrameMasked = uvpFluidFrameMasked[mask]

      # Insert random selection of fluid coords into collocation array
      if UT.PRINT_DEBUG:
        print('Taking {} collocation points out of {} available'.format(nColPntPerFrame, xyFluidFrameMasked.shape[0]))
      indices = np.arange(0, xyFluidFrameMasked.shape[0])
      randIndices = rng.choice(indices, size=nColPntPerFrame, replace=False)
      e = s + nColPntPerFrame
      self.xyCol[s:e, :] = xyFluidFrameMasked[randIndices, :]
      self.uvpCol[s:e, :] = uvpFluidFrameMasked[randIndices, :]
      s = e

      if UT.IMG_DEBUG:
        UT.save_array(xyFluidFrameMasked[randIndices, :], '{}/collocation'.format(self.sourceName), 'colpts_select', frame, self.size)


  def select_wall_points(self, all=False):
    UT.print_info('Selecting wall points')
    rng = np.random.default_rng(2022)

    # Use all points that are available
    if all:
      self.xyWalls = copy.deepcopy(self.xyDomain)
      self.uvWalls = copy.deepcopy(self.bcDomain)
      self.nWallPnt = len(self.xyWalls)
      print('Using {} wall points'.format(self.nWallPnt))
      return

    # Else, use the exact number of points that was specified
    nWallsPntPerFrame = self.nWallPnt // self.nTotalFrames
    # Update actual number of points
    self.nWallPnt = nWallsPntPerFrame * self.nTotalFrames
    # Allocate arrays based on actual number of points
    self.xyWalls = np.zeros((self.nWallPnt, self.dim + 1))
    self.uvWalls = np.zeros((self.nWallPnt, self.dim + 1))

    print('Using {} wall points'.format(self.nWallPnt))

    # Return early if no points requested
    if not self.nWallPnt:
      return

    s = 0
    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      # Get all data point coords and vels for the current frame
      xyWallsFrame = self.get_xy_walls(f)
      uvWallsFrame = self.get_bc_walls(f)

      # Insert random selection of data point coords into data point array
      if UT.PRINT_DEBUG:
        print('Taking {} wall points out of {} available'.format(nWallsPntPerFrame, xyWallsFrame.shape[0]))
      indices = np.arange(0, xyWallsFrame.shape[0])
      randIndices = rng.choice(indices, size=nWallsPntPerFrame, replace=False)
      e = s + nWallsPntPerFrame
      self.xyWalls[s:e, :] = xyWallsFrame[randIndices, :]
      self.uvWalls[s:e, :] = uvWallsFrame[randIndices, :]
      s = e

      if UT.IMG_DEBUG:
        UT.save_array(xyWallsFrame[randIndices, :], '{}/wall'.format(self.sourceName), 'wallpts_select', frame, self.size)


  def save(self, dir='../data/', filePrefix='bdata'):
    if not os.path.exists(dir):
      os.makedirs(dir)

    nSampleBc     = np.sum(self.nBcBubble)
    nSampleFluid  = np.sum(self.nFluid)
    nSampleWalls  = np.sum(self.nWalls)

    fname = os.path.join(dir, filePrefix + '_{}_r{}_t{:03}-{:03}_i{}_w{}.h5'.format( \
              self.size[0], self.colRes, self.startFrame, self.endFrame, \
              self.interface, UT.get_list_string(self.walls, delim='-')))
    dFile = h5.File(fname, 'w')
    dFile.attrs['size']         = self.size
    dFile.attrs['startFrame']   = self.startFrame
    dFile.attrs['endFrame']     = self.endFrame
    dFile.attrs['walls']        = self.walls
    dFile.attrs['interface']    = self.interface
    dFile.attrs['source']       = self.source
    dFile.attrs['sourceName']   = self.sourceName
    dFile.attrs['nBcBubble']    = np.asarray(self.nBcBubble)
    dFile.attrs['nFluid']       = np.asarray(self.nFluid)
    dFile.attrs['nWalls']       = np.asarray(self.nWalls)

    # Compression
    comp_type = 'gzip'
    comp_level = 9

    dFile.create_dataset('bc', (nSampleBc, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.bc)
    dFile.create_dataset('xyBc', (nSampleBc, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyBc)
    dFile.create_dataset('uvpFluid', (nSampleFluid, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.uvpFluid)
    dFile.create_dataset('xyFluid', (nSampleFluid, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyFluid)
    dFile.create_dataset('bcDomain', (nSampleWalls, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.bcDomain)
    dFile.create_dataset('xyDomain', (nSampleWalls, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyDomain)

    dFile.close()
    print('Saved dataset to file {}'.format(fname))


  def restore(self, fname):
    if not os.path.exists(fname):
      sys.exit('File {} does not exist'.format(fname))

    dFile = h5.File(fname, 'r')

    self.size         = dFile.attrs['size']
    self.startFrame   = dFile.attrs['startFrame']
    self.endFrame     = dFile.attrs['endFrame']
    self.walls        = dFile.attrs['walls']
    self.interface    = dFile.attrs['interface']
    self.source       = dFile.attrs['source']
    self.sourceName   = dFile.attrs['sourceName']
    self.nBcBubble    = dFile.attrs['nBcBubble']
    self.nFluid       = dFile.attrs['nFluid']
    self.nWalls       = dFile.attrs['nWalls']

    self.bc       = np.array(dFile.get('bc'))
    self.xyBc     = np.array(dFile.get('xyBc'))
    self.uvpFluid = np.array(dFile.get('uvpFluid'))
    self.xyFluid  = np.array(dFile.get('xyFluid'))
    self.bcDomain = np.array(dFile.get('bcDomain'))
    self.xyDomain = np.array(dFile.get('xyDomain'))

    self.isLoaded = True
    self.nTotalFrames = self.endFrame - self.startFrame + 1
    dFile.close()
    print('Restored dataset from file {}'.format(fname))
    print('Dataset: size [{},{}], frames {}'.format(self.size[0], self.size[1], self.nTotalFrames))


  def get_num_wall_pts(self):
    return self.nWallPnt


  def get_num_data_pts(self):
    return self.nDataPnt


  def get_num_col_pts(self):
    return self.nColPnt


  def get_num_fluid(self, fromFrame, toFrame):
    return sum(self.nFluid[fromFrame:toFrame])


  def get_num_frames(self):
    return self.nTotalFrames


  def get_num_cells(self):
    return np.prod(list(self.size))


  def get_size(self):
    return self.size


  def get_dim(self):
    return self.dim


  def get_source(self):
    return self.source


  def get_source_name(self):
    return self.sourceName

  def get_xy_bc(self, f):
    s = sum(self.nBcBubble[:f])
    e = s + self.nBcBubble[f]
    return self.xyBc[s:e, ...]


  def get_xy_fluid(self, f):
    s = sum(self.nFluid[:f])
    e = s + self.nFluid[f]
    return self.xyFluid[s:e, ...]


  def get_bc(self, f):
    s = sum(self.nBcBubble[:f])
    e = s + self.nBcBubble[f]
    return self.bc[s:e, ...]


  def get_uvp_fluid(self, f):
    s = sum(self.nFluid[:f])
    e = s + self.nFluid[f]
    return self.uvpFluid[s:e, ...]


  def get_xy_walls(self, f):
    s = sum(self.nWalls[:f])
    e = s + self.nWalls[f]
    return self.xyDomain[s:e, ...]


  def get_bc_walls(self, f):
    s = sum(self.nWalls[:f])
    e = s + self.nWalls[f]
    return self.bcDomain[s:e, ...]


  '''
  def get_xy_data(self, f):
    ptsPerFrame = (self.nDataPnt // self.nTotalFrames)
    s = ptsPerFrame * f
    e = s + ptsPerFrame
    return self.xyData[s:e, ...]


  def get_xy_col(self, f):
    ptsPerFrame = (self.nColPnt // self.nTotalFrames)
    s = ptsPerFrame * f
    e = s + ptsPerFrame
    return self.xyCol[s:e, ...]


  def get_xy_bcdomain(self, f):
    ptsPerFrame = (self.nWallPnt // self.nTotalFrames)
    s = ptsPerFrame * f
    e = s + ptsPerFrame
    return self.xyDomain[s:e, ...]
  '''

