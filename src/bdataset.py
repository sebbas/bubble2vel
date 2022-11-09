#!/usr/bin/env python3

import h5py as h5
import matplotlib.pyplot as plt
import os.path
import sys
import copy
import numpy as np
import random as rn
import tensorflow as tf

import butils as UT

rn.seed(2022)
np.random.seed(2022)


class BubbleDataSet:

  FLAG_FLUID = 0
  FLAG_BUBBLE = 1
  FLAG_VISITED = 1

  def __init__(self, fName='', totalframes=0, startframe=0, dim=2, \
               wallPoints=-1, colPoints=-1, dataPoints=-1):
    assert dim == 2, "Only supporting 2D datasets"
    self.fName        = fName
    self.dim          = dim
    self.size         = np.zeros(self.dim, dtype=int)
    self.isLoaded     = False
    self.startframe   = startframe
    self.nTotalFrames = totalframes
    # Lists to count number of cells per frame
    self.nBcBubble = []
    self.nFluid    = []
    self.nWalls    = []
    # Requested number of points
    self.nColPnt  = colPoints
    self.nWallPnt = wallPoints
    self.nDataPnt = dataPoints
    # Array to store ground truth data (after processing .flo input)
    self.vel      = None # [frames, width, height, dim]
    # Arrays to store processed data (after processing ground truth)
    self.bc       = None # [nSamples, dim]
    self.xyBc     = None # [nSamples, dim + 1]
    self.xyFluid  = None # [nSamples, dim + 1]
    self.bcDomain = None # [nSamples, dim]
    self.xyDomain = None # [nSamples, dim + 1]
    # Arrays to store selection of points
    self.xyCol    = None # [nColPnt,  dim + 1]
    self.xyData   = None # [nDataPnt, dim + 1]
    self.uvData   = None # [nDataPnt, dim]
    self.xyWalls  = None # [nWallPnt, dim + 1]
    self.uvWalls  = None # [nWallPnt, dim]
    # Arrays to store (shuffled) mix of data and collocation points
    self.labels   = None
    self.xyt      = None
    self.id       = None
    self.labelsW  = None
    self.xytW     = None
    # Resolution of collocation points (only use every other points as col point)
    self.colRes = 2


  def load_data(self, normalize=True, shuffle=True):
    velLst = [] # List of vel arrays, one array per frame, later converted to np array

    frame = self.startframe
    fNameExact = self.fName % frame
    # Loop over request number of frames
    while os.path.exists(fNameExact) and frame < self.nTotalFrames:
      f = open(fNameExact, 'rb')
      magic = np.fromfile(f, np.float32, count = 1)
      sizeFromFile = np.fromfile(f, np.int32, count = self.dim)
      dataFromFile = np.fromfile(f, np.float32, count = self.dim * sizeFromFile[0] * sizeFromFile[1])
      dataFromFile = np.resize(dataFromFile, (sizeFromFile[1], sizeFromFile[0], self.dim))
      # After the 1st frame we know the size, so make sure the following files have same size
      if frame > 0:
        assert sizeFromFile[0] == self.size[0], 'Width in dataset does not match width from files'
        assert sizeFromFile[1] == self.size[1], 'Height in dataset does not match width from files'
      # Copy read data into data structures
      self.size = sizeFromFile
      assert (self.size[0] + self.size[1]) * 2 * self.nTotalFrames >= self.nWallPnt, "Maximum number of bc domain points exceeded"

      velLst.append(dataFromFile)

      # Prepare next file name
      frame += 1
      fNameExact = self.fName % frame

    if len(velLst):
      print('Read {} frames with size [{}, {}] (width, height)'.format(frame, self.size[0], self.size[1]))
      self.vel = np.asarray(velLst)
      self.nTotalFrames = frame
      self.isLoaded = True

      # Normalization. Remember: input is optical flow, i.e. pixel movement between frame i and i+1
      if normalize:
        normWithMaxVel = True
        # Normalize data with the overall maximum velocity
        if normWithMaxVel:
          maxVel = np.zeros(self.dim)
          maxVel[0] = abs(max(self.vel[:,:,:,0].min(), self.vel[:,:,:,0].max(), key = abs))
          maxVel[1] = abs(max(self.vel[:,:,:,1].min(), self.vel[:,:,:,1].max(), key = abs))
          if UT.PRINT_DEBUG:
            print('Max vel [{}, {}]'.format(maxVel[0], maxVel[1]))
          norm = maxVel
        # Normalize with image width and height
        else:
          norm = self.size

        if UT.PRINT_DEBUG:
          u = abs(max(self.vel[:,:,:,0].min(), self.vel[:,:,:,0].max(), key = abs))
          v = abs(max(self.vel[:,:,:,1].min(), self.vel[:,:,:,1].max(), key = abs))
          print('Before normalization max vel [{}, {}]'.format(u, v))

        # Apply normalization factor
        self.vel[:,:,:,0] /= norm[0]
        self.vel[:,:,:,1] /= norm[1]

        if UT.PRINT_DEBUG:
          u = abs(max(self.vel[:,:,:,0].min(), self.vel[:,:,:,0].max(), key = abs))
          v = abs(max(self.vel[:,:,:,1].min(), self.vel[:,:,:,1].max(), key = abs))
          print('After normalization max vel [{}, {}]'.format(u, v))
      else:
        if UT.PRINT_DEBUG:
          u = abs(max(self.vel[:,:,:,0].min(), self.vel[:,:,:,0].max(), key = abs))
          v = abs(max(self.vel[:,:,:,1].min(), self.vel[:,:,:,1].max(), key = abs))
          print('No normalization, max vel [{}, {}]'.format(u, v))
          print('u [{}, {}], v [{}, {}]'.format(np.min(self.vel[:,:,:,0]), np.max(self.vel[:,:,:,0]), np.min(self.vel[:,:,:,1]), np.max(self.vel[:,:,:,1])))
    else:
      print('No data read. Returning early. Does the dataset exist?')
      self.isLoaded = False

    return self.isLoaded


  def summary(self):
    if not self.isLoaded:
      print('Bubble DataSet has not been loaded yet, end summary')
      return
    print('--------------------------------')
    print('Total Frames: {}'.format(self.nTotalFrames))
    print('Domain size: [{}, {}]'.format(self.size[0], self.size[1]))
    print('Min / max of input velocity')
    print('  u  {}, {}'.format(np.amin(self.vel[:,:,:,0]), np.amax(self.vel[:,:,:,0])))
    print('  v  {}, {}'.format(np.amin(self.vel[:,:,:,1]), np.amax(self.vel[:,:,:,1])))
    print('--------------------------------')


  def generate_predict_pts(self, begin, end, worldSize, fps, L, T, onlyBc=False):
    print('Generating prediction points')

    for frame in range(begin, end):
      loopArray = self.get_xy_bc(frame) if onlyBc else self.get_xy_fluid(frame)
      tmpLst    = [pos for pos in loopArray]
      nGridPnt  = len(tmpLst)

      # Arrays to store batches
      xy    = np.zeros((nGridPnt, self.dim), dtype=float)
      t     = np.zeros((nGridPnt, 1), dtype=float)
      # Dummies, unused for pred pts
      label = np.zeros((nGridPnt, self.dim),  dtype=float)
      w     = np.zeros((nGridPnt, 1),         dtype=float)

      # Fill batch arrays
      tmpArr   = np.asarray(tmpLst, dtype=float)
      xy[:, :] = tmpArr[:, :self.dim]
      t[:, 0]  = tmpArr[:, self.dim]

      # Convert from domain space to world space
      pos  = UT.pos_domain_to_world(xy, worldSize)
      time = UT.time_domain_to_world(t, fps)

      # Convert from world space to dimensionless quantities
      pos  = UT.pos_world_to_dimensionless(pos, L)
      time = UT.time_world_to_dimensionless(time, T)

      yield [pos, time, w], label


  def prepare_batch_arrays(self, zeroInitialCollocation=False):
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

    # Init collocation point labels with 0.0
    if zeroInitialCollocation:
      colPntLabel = 0.0
      uvCol = np.full((self.nColPnt, self.dim), colPntLabel)
    else: # Init collocation point label with random values in range of min/max of data point label
      minU, maxU = np.min(self.bc[:,0]), np.max(self.bc[:,0])
      minV, maxV = np.min(self.bc[:,1]), np.max(self.bc[:,1])
      samplesU = np.random.uniform(low=minU, high=maxU, size=self.nColPnt)
      samplesV = np.random.uniform(low=minV, high=maxV, size=self.nColPnt)
      samplesU = np.expand_dims(samplesU, axis=-1)
      samplesV = np.expand_dims(samplesV, axis=-1)
      uvCol = np.concatenate((samplesU, samplesV), axis=-1)

    bcSamples  = np.concatenate((self.uvData, uvCol))
    xytSamples = np.concatenate((self.xyData, self.xyCol))
    ones       = np.full((len(self.xyData), 1), 1) # Indicates a data point
    zeros      = np.full((len(self.xyCol), 1), 0) # Indicates a collocation point
    idSamples  = np.concatenate((ones, zeros))

    assert len(bcSamples) == len(xytSamples)
    assert len(bcSamples) == len(idSamples)

    # Shuffle the combined point arrays. Shuffle all arrays with same permutation
    p = np.random.permutation(len(bcSamples))
    self.labels  = bcSamples[p]
    self.xyt     = xytSamples[p]
    self.id      = idSamples[p]

    assert len(self.labels) == len(self.xyt)
    assert len(self.labels) == len(self.id)

    # Shuffle domain wall arrays
    pp = np.random.permutation(len(self.xyWalls))
    self.xytW    = self.xyWalls[pp]
    self.labelsW = self.uvWalls[pp]

    assert len(self.labelsW) == len(self.xytW)


  def generate_train_valid_batch(self, begin, end, beginW, endW, worldSize, fps, \
                                 V, L, T, batchSize=64, shuffle=True):
    generatorType = 'training' if begin == 0 else 'validation'
    print('\nGenerating {} sample {} batches'.format(batchSize, generatorType))

    # Arrays to store data + collocation point batch
    label  = np.zeros((batchSize, self.dim), dtype=float)
    xy     = np.zeros((batchSize, self.dim), dtype=float)
    t      = np.zeros((batchSize, 1),        dtype=float)
    id     = np.zeros((batchSize, 1),        dtype=float)
    # Arrays to store domain wall batch
    xyW    = np.zeros((batchSize, self.dim), dtype=float)
    tW     = np.zeros((batchSize, 1),        dtype=float)
    labelW = np.zeros((batchSize, self.dim), dtype=float)

    s, sW = begin, beginW
    while True:
      if s + batchSize > end:
        s = begin
      if sW + batchSize > endW:
        sW = beginW
      e = s + batchSize
      eW = sW + batchSize

      # Fill batch arrays
      label[:, :] = self.labels[s:e, :]
      xy[:, :]    = self.xyt[s:e, :self.dim]
      t[:, 0]     = self.xyt[s:e, self.dim]
      id[:, 0]    = self.id[s:e, 0]
      if endW > 0:
        labelW[:, :] = self.labelsW[sW:eW, :]
        xyW[:, :]    = self.xytW[sW:eW, :self.dim]
        tW[:, 0]     = self.xytW[sW:eW, self.dim]

      s  += batchSize
      sW += batchSize

      # Shuffle batch
      if shuffle:
        assert len(xy) == batchSize
        p      = np.random.permutation(batchSize)
        xy     = xy[p]
        t      = t[p]
        id     = id[p]
        label  = label[p]
        xyW    = xyW[p]
        labelW = labelW[p]

      # Convert from domain space to world space
      vel   = UT.vel_domain_to_world(label, worldSize, fps)
      pos   = UT.pos_domain_to_world(xy, worldSize)
      time  = UT.time_domain_to_world(t, fps)
      velW  = UT.vel_domain_to_world(labelW, worldSize, fps)
      posW  = UT.pos_domain_to_world(xyW, worldSize)
      timeW = UT.time_domain_to_world(tW, fps)

      # Convert from world space to dimensionless quantities
      vel   = UT.vel_world_to_dimensionless(vel, V)
      pos   = UT.pos_world_to_dimensionless(pos, L)
      time  = UT.time_world_to_dimensionless(time, T)
      velW  = UT.vel_world_to_dimensionless(velW, V)
      posW  = UT.pos_world_to_dimensionless(posW, L)
      timeW = UT.time_world_to_dimensionless(timeW, T)

      yield [pos, time, id, posW, timeW, velW], vel


  # Define solid domain borders: walls = [top, right, bottom, left]
  def extract_wall_points(self, walls):
    print('Extracting domain wall points')
    rng = np.random.default_rng(2022)

    bcFrameLst = [] # Domain boundary condition for every frame
    xyFrameLst = [] # Domain boundary condition xy for every frame

    sizeX, sizeY = self.size

    for frame in range(self.nTotalFrames):
      UT.print_progress(frame, self.nTotalFrames)

      bcCondition = 0. # zero vel at boundary
      bcLst = []
      xyLst = []

      if not sum(walls):
        if UT.PRINT_DEBUG: print('Warning frame %d: All domain walls open. Returning early' % frame)
        return

      # Top domain wall
      if walls[0]:
        x = np.full((sizeY-2,), 0, dtype=int)
        y = np.linspace(1, sizeY-2, num=sizeY-2, dtype=int)
        bcLst.extend(np.full((sizeY-2,self.dim), bcCondition)) #, dtype=int))
        xyLst.extend(list(zip(x,y)))

      # Right domain wall
      if walls[1]:
        x = np.linspace(1, sizeX-2, num=sizeX-2, dtype=int)
        y = np.full((sizeX-2,), sizeX-1, dtype=int)
        bcLst.extend(np.full((sizeX-2,self.dim), bcCondition)) #, dtype=int))
        xyLst.extend(list(zip(x,y)))

      # Bottom domain wall
      if walls[2]:
        x = np.full((sizeY-2,), sizeY-1, dtype=int)
        y = np.linspace(1, sizeY-2, num=sizeY-2, dtype=int)[::-1]
        bcLst.extend(np.full((sizeY-2,self.dim), bcCondition)) #, dtype=int))
        xyLst.extend(list(zip(x,y)))

      # Left domain wall
      if walls[3]:
        x = np.linspace(1, sizeX-2, num=sizeX-2, dtype=int)[::-1]
        y = np.full((sizeX-2,), 0, dtype=int)
        bcLst.extend(np.full((sizeX-2,self.dim), bcCondition)) #, dtype=int))
        xyLst.extend(list(zip(x,y)))

      bcFrameLst.append(bcLst)
      xyFrameLst.append(xyLst)

      if UT.IMG_DEBUG:
        debugGrid = np.zeros((self.size), dtype=int)
        for x, y in xyLst:
          debugGrid[x,y] = 1
        UT.save_image(src=debugGrid, subdir='extract', name='domainPts_extract', frame=frame, origin='upper')

      # Keep track of number of wall cells per frame
      self.nWalls.append(len(xyLst))

    if UT.PRINT_DEBUG:
      print('Total number of wall samples: {}'.format(np.sum(self.nWalls)))

    self.bcDomain = np.zeros((np.sum(self.nWalls), self.dim))
    self.xyDomain = np.zeros((np.sum(self.nWalls), self.dim + 1))

    s = 0
    for frame in range(self.nTotalFrames):
      assert len(bcFrameLst[frame]) == len(xyFrameLst[frame]), 'Number of velocity labels must match number of xy positions'

      n = len(bcFrameLst[frame])
      t = np.full((n, 1), frame, dtype=float)
      e = s + n
      if n:
        uv = np.asarray(bcFrameLst[frame], dtype=float)
        xy = np.asarray(xyFrameLst[frame], dtype=float)
        self.bcDomain[s:e, :] = uv
        self.xyDomain[s:e, :] = np.hstack((xy, t))
      s = e


  def extract_fluid_points(self, velEps=0.1):
    print('Extracting bubble and fluid points')

    bcFrameLst = []
    xyFrameLst = []
    xyFluidFrameLst = []
    sizeX, sizeY = self.size

    for frame in range(self.nTotalFrames):
      UT.print_progress(frame, self.nTotalFrames)

      bcLst = []
      xyLst = []
      xyFluidLst = []

      # Intersection == grid with all cells at fluid / bubble intersection
      intersection = np.zeros((sizeX, sizeY), dtype=int)
      # Flags == grid with the type of the cell (fluid or bubble)
      flags = np.full((sizeX, sizeY), self.FLAG_BUBBLE)
      # Visited == grid with all cells that have been visited during flood-fill
      visited = np.zeros((sizeX, sizeY), dtype=int)
      # Velocity magnitude per cell
      mag = np.sqrt(self.vel[frame,:,:,0] ** 2 + self.vel[frame,:,:,1] ** 2)

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
        while(len(stack) > 0):
          c = stack.pop()
          i = c[0]
          j = c[1]
          visited[i, j] = self.FLAG_VISITED # Mark current cell as visited
          neighbors = [(i+1, j  ), (i-1, j  ), (i  , j+1), (i  , j-1),
                       (i+1, j+1), (i-1, j+1), (i+1, j-1), (i-1, j-1)]
          for n in neighbors:
            # Only proceed if neigbor is valid, ie if cell is inside domain borders
            if n[0] < 0 or n[0] >= sizeX or n[1] < 0 or n[1] >= sizeY:
              continue
            ni, nj = n[0], n[1]
            # Also skip if neighbor cell has already been visited
            if (visited[ni, nj] == 1):
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

      if UT.IMG_DEBUG:
        UT.save_image(src=flags, subdir='extract', name='flags_extract', frame=frame)
        UT.save_image(src=intersection, subdir='extract', name='bubblePts_extract', frame=frame)
        UT.save_image(src=mag, subdir='extract', name='magnitude_extract', frame=frame)

      # Add bubble border velocities to bc list
      curVels = self.vel[frame, :, :, :]
      bubbleBorderVels = curVels[np.array(intersection, dtype=bool)]
      bcLst.extend(bubbleBorderVels)
      bcFrameLst.append(bcLst)

      # Add bubble border positions to xyBc list
      nz = np.nonzero(intersection)
      bubbleBorderIndices = list(zip(nz[0], nz[1]))
      xyLst.extend(bubbleBorderIndices)
      xyFrameLst.append(xyLst)

      # Create fluid mask
      fluidmask = 1 - flags
      assert np.sum(fluidmask) == (sizeX*sizeY - np.sum(flags)), 'Fluid mask must match total size minus bubble mask'
      nz = np.nonzero(fluidmask)
      fluidIndices = list(zip(nz[0], nz[1]))
      xyFluidLst.extend(fluidIndices)
      xyFluidFrameLst.append(xyFluidLst)

      # Keep track of number of bubble border cells and fluid cells per frame
      self.nBcBubble.append(len(xyLst))
      self.nFluid.append(len(xyFluidLst))

    if UT.PRINT_DEBUG:
      print('Total number of bubble boundary samples: {}'.format(np.sum(self.nBcBubble)))
      print('Total number of fluid samples: {}'.format(np.sum(self.nFluid)))

    # Allocate arrays for preprocessed data ...
    self.bc   = np.zeros((np.sum(self.nBcBubble), self.dim),     dtype=float)
    self.xyBc = np.zeros((np.sum(self.nBcBubble), self.dim + 1), dtype=float) # dim + 1 for t
    self.xyFluid = np.zeros((np.sum(self.nFluid), self.dim + 1), dtype=float)

    # ... and insert data from lists
    s, sFl = 0, 0
    for frame in range(self.nTotalFrames):
      assert len(bcFrameLst[frame]) == len(xyFrameLst[frame]), 'Number of velocity labels must match number of xy positions'

      # Insertion of bc and xyBc lists
      n = len(bcFrameLst[frame])
      t = np.full((n, 1), frame, dtype=float)
      e = s + n
      if n: # only insert if there is at least 1 cell
        uv = np.asarray(bcFrameLst[frame], dtype=float)
        xy = np.asarray(xyFrameLst[frame], dtype=float)
        self.bc[s:e, :] = uv
        self.xyBc[s:e, :] = np.hstack((xy, t))
      s = e

      # Insertion of xyFluid list
      nFl = len(xyFluidFrameLst[frame])
      tFl = np.full((nFl, 1), frame, dtype=float)
      eFl = sFl + nFl
      if nFl: # only insert if there is at least 1 cell
        xy = np.asarray(xyFluidFrameLst[frame], dtype=float)
        self.xyFluid[sFl:eFl, :] = np.hstack((xy, tFl))
      sFl = eFl


  def select_data_points(self, all=False):
    print('Selecting data points')
    rng = np.random.default_rng(2022)

    # Use all points that are available
    if all:
      self.xyData = copy.deepcopy(self.xyBc)
      self.uvData = copy.deepcopy(self.bc)
      self.nDataPnt = len(self.xyData)
      return

    # Else get a selection of points
    self.xyData = np.zeros((self.nDataPnt, self.dim + 1))
    self.uvData = np.zeros((self.nDataPnt, self.dim))
    nDataPntPerFrame = self.nDataPnt // self.nTotalFrames
    s = 0
    for frame in range(self.nTotalFrames):
      UT.print_progress(frame, self.nTotalFrames)

      # Get all data point coords and vels for the current frame
      xyDataFrame = self.get_xy_bc(frame)
      uvDataFrame = self.get_bc(frame)

      # Insert random selection of data point coords into data point array
      if UT.PRINT_DEBUG:
        print('Taking {} data points out of {} available'.format(nDataPntPerFrame, xyDataFrame.shape[0]))
      indices = np.arange(0, xyDataFrame.shape[0])
      randIndices = rng.choice(indices, size=nDataPntPerFrame, replace=False)
      e = s + nDataPntPerFrame
      self.xyData[s:e, :] = xyDataFrame[randIndices, :]
      self.uvData[s:e, :] = uvDataFrame[randIndices, :]
      s = e

      if UT.IMG_DEBUG:
        sizeX, sizeY = self.size
        datapts = np.zeros((sizeX, sizeY), dtype=int)
        for pos in xyDataFrame[randIndices, :]:
          datapts[int(pos[0]), int(pos[1])] = 1
        UT.save_image(src=datapts, subdir='extract', name='dataPts_extract', frame=frame)


  def select_collocation_points(self, default=False):
    print('Selecting collocation points')
    rng = np.random.default_rng(2022)

    # No collocation points supplied. Use 5000 points per frame by default
    if default:
      self.nColPnt = self.nTotalFrames * 5000

    self.xyCol = np.zeros((self.nColPnt, self.dim + 1))
    nColPntPerFrame = self.nColPnt // self.nTotalFrames
    s = 0
    for frame in range(self.nTotalFrames):
      UT.print_progress(frame, self.nTotalFrames)

      # Get all fluid coords for the current frame
      xyFluidFrame = self.get_xy_fluid(frame)

      # Only use every other grid point (self.colRes == interval) as collocation point
      mask = np.logical_and(xyFluidFrame[:,0] % self.colRes == 0, xyFluidFrame[:,1] % self.colRes == 0)
      xyFluidFrameMasked = xyFluidFrame[mask]

      # Insert random selection of fluid coords into collocation array
      if UT.PRINT_DEBUG:
        print('Taking {} collocation points out of {} available'.format(nColPntPerFrame, xyFluidFrameMasked.shape[0]))
      indices = np.arange(0, xyFluidFrameMasked.shape[0])
      randIndices = rng.choice(indices, size=nColPntPerFrame, replace=False)
      e = s + nColPntPerFrame
      self.xyCol[s:e, :] = xyFluidFrameMasked[randIndices, :]
      s = e

      if UT.IMG_DEBUG:
        sizeX, sizeY = self.size
        colpts = np.zeros((sizeX, sizeY), dtype=int)
        for pos in xyFluidFrameMasked[randIndices, :]:
          colpts[int(pos[0]), int(pos[1])] = 1
        UT.save_image(src=colpts, subdir='extract', name='colPts_extract', frame=frame)


  def select_wall_points(self, all=False):
    print('Selecting wall points')
    rng = np.random.default_rng(2022)

    # Use all points that are available
    if all:
      self.xyWalls = copy.deepcopy(self.xyDomain)
      self.uvWalls = copy.deepcopy(self.bcDomain)
      self.nWallPnt = len(self.xyWalls)
      return

    # Else get a selection of points
    self.xyWalls = np.zeros((self.nWallPnt, self.dim + 1))
    self.uvWalls = np.zeros((self.nWallPnt, self.dim))
    nWallsPntPerFrame = self.nWallPnt // self.nTotalFrames
    s = 0
    for frame in range(self.nTotalFrames):
      UT.print_progress(frame, self.nTotalFrames)

      # Get all data point coords and vels for the current frame
      xyWallsFrame = self.get_xy_walls(frame)
      uvWallsFrame = self.get_bc_walls(frame)

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
        sizeX, sizeY = self.size
        wallpts = np.zeros((sizeX, sizeY), dtype=int)
        for pos in xyWallsFrame[randIndices, :]:
          wallpts[int(pos[0]), int(pos[1])] = 1
        UT.save_image(src=wallpts, subdir='extract', name='wallPts_extract', frame=frame)


  def save(self, dir='../data/', filePrefix='bdata', walls=[0,0,1,0]):
    if not os.path.exists(dir):
      os.makedirs(dir)

    nSampleBc     = np.sum(self.nBcBubble)
    nSampleFluid  = np.sum(self.nFluid)
    nSampleWalls  = np.sum(self.nWalls)

    fname = os.path.join(dir, filePrefix + '_{}_r{}_t{}_w{}.h5'.format( \
              self.size[0], self.colRes, self.nTotalFrames, UT.get_list_string(walls)))
    dFile = h5.File(fname, 'w')
    dFile.attrs['size']         = self.size
    dFile.attrs['frames']       = self.nTotalFrames
    dFile.attrs['nBcBubble']    = np.asarray(self.nBcBubble)
    dFile.attrs['nFluid']       = np.asarray(self.nFluid)
    dFile.attrs['nWalls']       = np.asarray(self.nWalls)

    # Compression
    comp_type = 'gzip'
    comp_level = 9

    dFile.create_dataset('bc', (nSampleBc, self.dim), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.bc)
    dFile.create_dataset('xyBc', (nSampleBc, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyBc)
    dFile.create_dataset('xyFluid', (nSampleFluid, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyFluid)
    dFile.create_dataset('bcDomain', (nSampleWalls, self.dim), compression=comp_type,
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
    self.nTotalFrames = dFile.attrs['frames']
    self.nBcBubble    = dFile.attrs['nBcBubble']
    self.nFluid       = dFile.attrs['nFluid']
    self.nWalls       = dFile.attrs['nWalls']

    self.bc       = np.array(dFile.get('bc'))
    self.xyBc     = np.array(dFile.get('xyBc'))
    self.xyFluid  = np.array(dFile.get('xyFluid'))
    self.bcDomain = np.array(dFile.get('bcDomain'))
    self.xyDomain = np.array(dFile.get('xyDomain'))

    self.isLoaded = True
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


  def get_xy_bc(self, frame):
    s = sum(self.nBcBubble[:frame])
    e = s + self.nBcBubble[frame]
    return self.xyBc[s:e, ...]


  def get_xy_fluid(self, frame):
    s = sum(self.nFluid[:frame])
    e = s + self.nFluid[frame]
    return self.xyFluid[s:e, ...]


  def get_bc(self, frame):
    s = sum(self.nBcBubble[:frame])
    e = s + self.nBcBubble[frame]
    return self.bc[s:e, ...]


  def get_xy_walls(self, frame):
    s = sum(self.nWalls[:frame])
    e = s + self.nWalls[frame]
    return self.xyDomain[s:e, ...]


  def get_bc_walls(self, frame):
    s = sum(self.nWalls[:frame])
    e = s + self.nWalls[frame]
    return self.bcDomain[s:e, ...]


  '''
  def get_xy_data(self, frame):
    ptsPerFrame = (self.nDataPnt // self.nTotalFrames)
    s = ptsPerFrame * frame
    e = s + ptsPerFrame
    return self.xyData[s:e, ...]


  def get_xy_col(self, frame):
    ptsPerFrame = (self.nColPnt // self.nTotalFrames)
    s = ptsPerFrame * frame
    e = s + ptsPerFrame
    return self.xyCol[s:e, ...]


  def get_xy_bcdomain(self, frame):
    ptsPerFrame = (self.nWallPnt // self.nTotalFrames)
    s = ptsPerFrame * frame
    e = s + ptsPerFrame
    return self.xyDomain[s:e, ...]
  '''

