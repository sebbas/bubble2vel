#!/usr/bin/env python3

import h5py as h5
import matplotlib.pyplot as plt
import os.path
import sys
import copy
import numpy as np
import random as rn
import tensorflow as tf

import butils as Util

rn.seed(2022)
np.random.seed(2022)


class BubbleDataSet:

  FLAG_FLUID = 0
  FLAG_BUBBLE = 1
  FLAG_VISITED = 1

  def __init__(self, fName='', totalframes=0, datapoints=0, bcpoints=0, colpoints=0, startframe=0, dim=2):
    assert dim == 2, "Only supporting 2D datasets"
    self.fName        = fName
    self.dim          = dim
    self.size         = np.zeros(self.dim, dtype=int)
    self.isLoaded     = False
    self.startframe   = startframe
    self.nTotalFrames = totalframes
    # Lists to count number of cells per frame
    self.nBcDomain = []
    self.nBcBubble = []
    self.nFluid    = []
    # Requested number of points
    self.nColPnt  = colpoints
    self.nBcPnt   = bcpoints
    self.nDataPnt = datapoints
    # Array to store ground truth data (after processing .flo input)
    self.vel      = None # [frames, width, height, dim]
    # Arrays to store processed data (after processing ground truth)
    self.bc       = None # [nSamples, dim]
    self.xyBc     = None # [nSamples, dim + 1]
    self.xyFluid  = None # [nSamples, dim + 1]
    self.bcDomain = None # [nSamples, dim]
    self.xyDomain = None # [nSamples, dim + 1]
    self.xyCol    = None # [nColPnt,  dim + 1]
    # Arrays to store (shuffled) mix of data and collocation points
    self.labels   = None
    self.xyt      = None
    self.id       = None
    # Resolution of collocation points (only use every other points as col point)
    self.colRes = 8


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
          if Util.PRINT_DEBUG:
            print('Max vel [{}, {}]'.format(maxVel[0], maxVel[1]))
          norm = maxVel
        # Normalize with image width and height
        else:
          norm = self.size

        if Util.PRINT_DEBUG:
          u = abs(max(self.vel[:,:,:,0].min(), self.vel[:,:,:,0].max(), key = abs))
          v = abs(max(self.vel[:,:,:,1].min(), self.vel[:,:,:,1].max(), key = abs))
          print('Before normalization max vel [{}, {}]'.format(u, v))

        # Apply normalization factor
        self.vel[:,:,:,0] /= norm[0]
        self.vel[:,:,:,1] /= norm[1]

        if Util.PRINT_DEBUG:
          u = abs(max(self.vel[:,:,:,0].min(), self.vel[:,:,:,0].max(), key = abs))
          v = abs(max(self.vel[:,:,:,1].min(), self.vel[:,:,:,1].max(), key = abs))
          print('After normalization max vel [{}, {}]'.format(u, v))
      else:
        if Util.PRINT_DEBUG:
          u = abs(max(self.vel[:,:,:,0].min(), self.vel[:,:,:,0].max(), key = abs))
          v = abs(max(self.vel[:,:,:,1].min(), self.vel[:,:,:,1].max(), key = abs))
          print('No normalization, max vel [{}, {}]'.format(u, v))
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


  def generate_predict_pts(self, begin, end, onlyFluid=True, normalizeXyt=True):
    print('Generating prediction points')
    sizeX, sizeY = self.size

    for frame in range(begin, end):
      gridXyt   = [(x,y,frame) for x in range(sizeX) for y in range(sizeY)]
      #loopArray = self.xyFluid[:self.nFluid[frame], :] if onlyFluid else gridXyt
      loopArray = self.get_xy_fluid(frame) if onlyFluid else gridXyt
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
      xy[:, :] =  tmpArr[:, :self.dim]
      t[:, 0]  = tmpArr[:, self.dim]

      # Normalization
      if normalizeXyt:
        xy[:,] /= self.size
        t[:,] /= self.nTotalFrames

      yield [xy, t, w], label


  def prepare_batch_arrays(self):
    print('Preparing samples for batch generator')
    rng = np.random.default_rng(2022)
    colPntLabel = 0.0 # Label for collocation points

    # Combine data points and collocation points arrays
    zeros2D    = np.full((len(self.xyCol), self.dim), colPntLabel)
    bcSamples  = np.concatenate((self.bc, zeros2D))
    xytSamples = np.concatenate((self.xyBc, self.xyCol))
    ones       = np.full((len(self.xyBc), 1), 1) # Indicates a data point
    zeros      = np.full((len(self.xyCol), 1), 0) # Indicates a collocation point
    idSamples   = np.concatenate((ones, zeros))

    assert len(bcSamples) == len(xytSamples)
    assert len(bcSamples) == len(idSamples)

    # Shuffle the combined point arrays. Shuffle all arrays with same permutation
    p = np.random.permutation(len(bcSamples))
    self.labels  = bcSamples[p]
    self.xyt     = xytSamples[p]
    self.id      = idSamples[p]

    # Shuffle domain wall arrays
    pp = np.random.permutation(len(self.xyDomain))
    self.xyDomain = self.xyDomain[pp]
    self.bcDomain = self.bcDomain[pp]


  def generate_train_valid_batch(self, begin, end, beginW, endW,
                                 normalizeXyt=True, batchSize=64, shuffle=True):
    generatorType = 'training' if begin == 0 else 'validation'
    print('\nGenerating {} sample {} batch'.format(batchSize, generatorType))

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
      # Fill batch arrays
      label[:, :] = self.labels[s:s+batchSize, :]
      xy[:, :]    = self.xyt[s:s+batchSize, :self.dim]
      t[:, 0]     = self.xyt[s:s+batchSize, self.dim]
      id[:, 0]    = self.id[s:s+batchSize, 0]

      labelW[:, :] = self.bcDomain[sW:sW+batchSize, :self.dim]
      xyW[:, :]    = self.xyDomain[sW:sW+batchSize, :self.dim]
      tW[:, 0]     = self.xyDomain[sW:sW+batchSize, self.dim]

      s  += batchSize
      sW += batchSize

      # Normalization
      if normalizeXyt:
        xy[:,]  /= self.size
        t[:,]   /= self.nTotalFrames
        xyW[:,] /= self.size
        tW[:,]  /= self.nTotalFrames

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

      yield [xy, t, id, xyW, tW, labelW], label


  # Define solid domain borders: walls = [top, right, bottom, left]
  def extract_wall_points(self, walls):
    print('Extracting domain wall points')

    bcFrameLst = [] # Domain boundary condition for every frame
    xyFrameLst = [] # Domain boundary condition xy for every frame

    sizeX, sizeY = self.size

    for frame in range(self.nTotalFrames):
      Util.print_progress(frame, self.nTotalFrames)

      bcCondition = 0. # zero vel at boundary
      bcLst = []
      xyLst = []

      if not sum(walls):
        if UTIL.PRINT_DEBUG: print('Warning frame %d: All domain walls open. Returning early' % frame)
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

      # Keep track of bubble border cells per frame
      self.nBcDomain.append(len(xyLst))
  
      if Util.IMG_DEBUG:
        debugGrid = np.zeros((self.size), dtype=int)
        for x, y in xyLst:
          debugGrid[x,y] = 1
        Util.save_image(src=debugGrid, subdir='extract', name='domainPts_extract', frame=frame, origin='upper')

    self.bcDomain = np.zeros((np.sum(self.nBcDomain), self.dim))
    self.xyDomain = np.zeros((np.sum(self.nBcDomain), self.dim + 1))

    s = 0
    for frame in range(self.nTotalFrames):
      Util.print_progress(frame, self.nTotalFrames)
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


  def extract_data_points(self, velEps=0.1):
    print('Extracting data points')

    bcFrameLst = []
    xyFrameLst = []
    xyFluidFrameLst = []
    sizeX, sizeY = self.size

    for frame in range(self.nTotalFrames):
      Util.print_progress(frame, self.nTotalFrames)

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
              flags[n[0], n[1]] = self.FLAG_FLUID
            else:
              intersection[n[0], n[1]] = 1 # Mark this cell in boundary condition mask

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

      if Util.IMG_DEBUG:
        Util.save_image(src=flags, subdir='extract', name='flags_extract', frame=frame)
        Util.save_image(src=intersection, subdir='extract', name='bubblePts_extract', frame=frame)
        Util.save_image(src=mag, subdir='extract', name='magnitude_extract', frame=frame)

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

    if Util.PRINT_DEBUG:
      print('Total number of bubble boundary samples: {}'.format(np.sum(self.nBcBubble)))

    # Allocate arrays for preprocessed data ...
    self.bc   = np.zeros((np.sum(self.nBcBubble), self.dim),     dtype=float)
    self.xyBc = np.zeros((np.sum(self.nBcBubble), self.dim + 1), dtype=float) # dim + 1 for t
    self.xyFluid = np.zeros((np.sum(self.nFluid), self.dim + 1), dtype=float)

    # ... and insert data from lists
    s, sFl = 0, 0
    for frame in range(self.nTotalFrames):
      Util.print_progress(frame, self.nTotalFrames)
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


  def extract_collocation_points(self):
    print('Extracting collocation points')
    rng = np.random.default_rng(2022)

    self.xyCol = np.zeros((self.nColPnt, self.dim + 1))
    nColPntPerFrame = self.nColPnt // self.nTotalFrames
    s = 0
    for frame in range(self.nTotalFrames):
      Util.print_progress(frame, self.nTotalFrames)

      # Get all fluid coords for the current frame
      xyFluidFrame = self.get_xy_fluid(frame)

      # Only use every other grid point (self.colRes == interval) as collocation point
      mask = np.logical_and(xyFluidFrame[:,0] % self.colRes == 0, xyFluidFrame[:,1] % self.colRes == 0)
      xyFluidFrameMasked = xyFluidFrame[mask]

      # Insert random selection of fluid coords into collocation array
      indices = np.arange(0, xyFluidFrameMasked.shape[0])
      randIndices = rng.choice(indices, size=nColPntPerFrame, replace=False)
      e = s + nColPntPerFrame
      self.xyCol[s:e, :] = xyFluidFrameMasked[randIndices, :]
      s = e


  def save(self, dir='../data/', filePrefix='bdata'):
    if not os.path.exists(dir):
      os.makedirs(dir)

    nSampleBc = np.sum(self.nBcBubble)
    nSampleFluid = np.sum(self.nFluid)
    nSampleDomain = np.sum(self.nBcDomain)
    nSampleCol = np.sum(self.nColPnt)

    fname = os.path.join(dir, filePrefix + '_{}_d{}_c{}_r{}.h5'.format(self.size[0], nSampleBc, nSampleCol, self.colRes))
    dFile = h5.File(fname, 'w')
    dFile.attrs['size']      = self.size
    dFile.attrs['frames']    = self.nTotalFrames
    dFile.attrs['nColPnt']   = self.nColPnt
    dFile.attrs['nBcBubble'] = np.asarray(self.nBcBubble)
    dFile.attrs['nFluid']    = np.asarray(self.nFluid)
    dFile.attrs['nBcDomain'] = np.asarray(self.nBcDomain)

    # Compression
    comp_type = 'gzip'
    comp_level = 9

    dFile.create_dataset('bc', (nSampleBc, self.dim), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.bc)
    dFile.create_dataset('xyBc', (nSampleBc, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyBc)
    dFile.create_dataset('xyFluid', (nSampleFluid, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyFluid)
    dFile.create_dataset('bcDomain', (nSampleDomain, self.dim), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.bcDomain)
    dFile.create_dataset('xyDomain', (nSampleDomain, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyDomain)
    dFile.create_dataset('xyCol', (nSampleCol, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xyCol)

    dFile.close()
    print('Saved dataset to file {}'.format(fname))


  def restore(self, fname):
    if not os.path.exists(fname):
      sys.exit('File {} does not exist'.format(fname))

    dFile = h5.File(fname, 'r')

    self.size         = dFile.attrs['size']
    self.nTotalFrames = dFile.attrs['frames']
    self.nColPnt      = dFile.attrs['nColPnt']
    self.nBcBubble    = dFile.attrs['nBcBubble']
    self.nFluid       = dFile.attrs['nFluid']
    self.nBcDomain    = dFile.attrs['nBcDomain']

    self.bc       = np.array(dFile.get('bc'))
    self.xyBc     = np.array(dFile.get('xyBc'))
    self.xyFluid  = np.array(dFile.get('xyFluid'))
    self.bcDomain = np.array(dFile.get('bcDomain'))
    self.xyDomain = np.array(dFile.get('xyDomain'))
    self.xyCol    = np.array(dFile.get('xyCol'))

    self.isLoaded = True
    dFile.close()
    print('Restored dataset from file {}'.format(fname))
    print('Dataset: size [{},{}], frames {}'.format(self.size[0], self.size[1], self.nTotalFrames))


  def get_num_wall_pts(self):
    return len(self.xyDomain)


  def get_num_data_pts(self):
    return len(self.xyBc)


  def get_num_col_pts(self):
    return len(self.xyCol)


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


