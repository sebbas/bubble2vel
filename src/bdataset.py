#!/usr/bin/env python3

import h5py as h5
import os.path
import sys
import copy
import numpy as np
import random as rn

import butils as UT

rn.seed(2022)
np.random.seed(2022)

class BubbleDataSet:

  FLAG_FLUID = 0
  FLAG_BUBBLE = 1
  FLAG_VISITED = 1

  N_IN_VAR = 3 # x, y, t
  N_OUT_VAR = 3 # u, v, p

  def __init__(self, fName='', startFrame=0, endFrame=399, dim=2, \
               wallPoints=-1, colPoints=-1, ifacePoints=-1, icondPoints=-1, walls=[1,1,1,1], \
               interface=1, source=UT.SRC_FLASHX):
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
    self.nBubble = []
    self.nFluid     = []
    self.nWalls     = []
    # Requested number of points per frame
    self.nColPnt   = colPoints
    self.nWallPnt  = wallPoints
    self.nIfacePnt = ifacePoints
    self.nIcondPnt = icondPoints
    # Array to store ground truth data (after processing .flo input)
    self.vel      = None # [frames, width, height, dim + 1]
    # Arrays to store processed data (after processing ground truth)
    self.uvpBubble = None # [nSamples, dim + 1]
    self.xytBubble = None # [nSamples, dim + 1]
    self.xytFluid  = None # [nSamples, dim + 1]
    self.uvpFluid  = None # [nSamples, dim + 1]
    self.uvpDomain = None # [nSamples, dim + 1]
    self.xytDomain = None # [nSamples, dim + 1]
    self.idDomain  = None # [nSamples]
    # Arrays to store selection of points
    self.xytCol   = None # [nColPnt,  dim + 1]
    self.uvpCol   = None # [nColPnt,  dim + 1]
    self.xytIface = None # [nIfacePnt, dim + 1]
    self.uvpIface = None # [nIfacePnt, dim + 1]
    self.xytWalls = None # [nWallPnt, dim + 1]
    self.uvpWalls = None # [nWallPnt, dim + 1]
    self.idWalls  = None # [nWallPnt]
    self.xytIcond = None # [nIcondPnt, dim + 1]
    self.uvpIcond = None # [nIcondPnt, dim + 1]
    # Arrays to store boundary condition per frame
    self.xytDataBc   = None # [nTotalFrames, max(nBubble), dim + 1]
    self.uvpDataBc   = None # [nTotalFrames, max(nBubble), dim + 1]
    self.validDataBc = None # [nTotalFrames, max(nWalls) + max(nBubble)]
    # Arrays to store (shuffled) mix of data and collocation points
    self.uvpBatch = None
    self.xytBatch = None
    self.idBatch  = None
    # Resolution of collocation points (only use every other points as col point)
    self.colRes = 1
    # FlashX
    self.pshape = [] # Num of points per block in each dimension
    self.bshape = [] # Num of blocks in each dimension


  def _load_flashx(self, n_channels=3, n_dimensions=2, mode=3):
    frame = self.startFrame
    fName = self.fName % frame
    file = h5.File(fName, 'r')

    attrs = ['velx', 'vely', 'dfun']#, 'temp', ]
    assert len(attrs) == n_channels, 'Number of attributes must match channel size'

    n_blocks, points_z, points_y, points_x = (file[attrs[0]].shape)

    print('n_blocks {}, points ({}, {}, {})'.format(n_blocks, points_z, points_y, points_x))
    print('Mode is {}'.format(mode))

    # mode = 384: 384 dataset, mode = 2: 2D dataset, mode = 3: tiny single bubble, mode = 4: single bubble
    if mode == 384:
      fac = [24, 24, 1]
      min_offset, max_offset = [12, 0, -1], [12, 0, 0]
    elif mode == 3: # tiny single bubble
      fac = [6, 9, 1]
      min_offset, max_offset = [2, 0, -1], [2, 0, 0]
    else:
      sys.exit('Invalid mode. Mode is {}'.format(mode))

    frame_z, frame_y, frame_x = points_z*fac[2], points_y*fac[1], points_x*fac[0]
    self.size = [frame_x, frame_y]

    print('size {}'.format(self.size))

    if n_dimensions == 2:
      self.vel = np.zeros((self.nTotalFrames, frame_y, frame_x, n_channels), dtype=float)
    elif n_dimensions == 3:
      self.vel = np.zeros((self.nTotalFrames, frame_z, frame_y, frame_x, n_channels), dtype=float)

    cnt = 0 # Frame counter

    while os.path.exists(fName) and frame <= self.endFrame:
      UT.print_progress(cnt, self.nTotalFrames)

      sz, sy, sx = 0, 0, 0
      ez, ey, ex = 0, 0, 0

      for n in range(n_blocks):
        bb = file['bounding box'][n]
        bs = file['block size'][n]

        print('bounding box [{}, {}, {}]'.format(bb[0], bb[1], bb[2]))
        print('block size [{}, {}, {}]'.format(bs[0], bs[1], bs[2]))

        xmin, xmax = bb[0][0] + min_offset[0], bb[0][1] + max_offset[0]
        ymin, ymax = bb[1][0] + min_offset[1], bb[1][1] + max_offset[1]
        zmin, zmax = bb[2][0] + min_offset[2], bb[2][1] + max_offset[2]

        # Start, end positions in image space
        if not bs[2]: bs[2] = 1.0
        sx, ex = int(np.rint(xmin/bs[0]) * points_x), int(np.rint(xmax/bs[0]) * points_x)
        sy, ey = int(np.rint(ymin/bs[1]) * points_y), int(np.rint(ymax/bs[1]) * points_y)
        sz, ez = int(np.rint(zmin/bs[2]) * points_z), int(np.rint(zmax/bs[2]) * points_z)

        if 1:
          print('x {}, {}'.format(xmin, xmax))
          print('y {}, {}'.format(ymin, ymax))
          print('z {}, {}'.format(zmin, zmax))
        if 1:
          print('{}, {}'.format(sx, ex))
          print('{}, {}'.format(sy, ey))
          print('{}, {}'.format(sz, ez))

        for i, attr_name in enumerate(attrs):
          field = file[attr_name]
          block = field[n]

          if n_dimensions == 2:
            self.vel[cnt, sy:ey, sx:ex, i] = block[0] # z has only 1 dim
          elif n_dimensions == 3:
            self.vel[cnt, sz:ez, sy:ey, sx:ex, i] = block

      # Mirror dataset vertically
      mirror = 1
      if mirror and mode == 3:
        mirrorData = copy.deepcopy(np.fliplr(self.vel[cnt, ...]))

        # Flip velx (2D) and velz (3D only) in mirrored data
        mirrorData[:,:,0] *= -1.0
        if n_dimensions == 3:
          mirrorData[:,:,2] *= -1.0

        self.vel[cnt, ...] += mirrorData

      if UT.IMG_DEBUG:
        UT.save_image(self.vel[cnt,:,:,0], '{}/raw'.format(self.sourceName), 'velu_raw', frame, cmap='hot')
        UT.save_image(self.vel[cnt,:,:,1], '{}/raw'.format(self.sourceName), 'velv_vely_raw', frame, cmap='hot')
        UT.save_image(self.vel[cnt,:,:,2], '{}/raw'.format(self.sourceName), 'dfun_raw', frame, cmap='jet', vmin=-0.5, vmax=0.5)
        UT.save_velocity(self.vel[cnt], '{}/raw'.format(self.sourceName), 'vel_stream', frame, size=self.size, type='stream', density=5.0)
        UT.save_velocity(self.vel[cnt], '{}/raw'.format(self.sourceName), 'vel_vector', frame, size=self.size, type='quiver', arrow_res=2)
        UT.save_velocity(self.vel[cnt], '{}/raw'.format(self.sourceName), 'vel_mag', frame, size=self.size, type='mag')
        UT.save_plot(self.vel[cnt,:,:,2], '{}/raw'.format(self.sourceName), 'dfun_plt', frame, size=self.size, cmin=-1.0, cmax=1.0, cmap='jet')

      cnt += 1 # Next array index
      frame += 1 # Next frame index
      fName = self.fName % frame
      file = h5.File(fName, 'r')

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


  def generate_predict_pts(self, begin, end, xyPred=[1,1,1], resetTime=True, zeroMean=False, batchSize=int(1.5e4)):
    print('Generating prediction points')

    for f in range(begin, end):
      uvpBubble = self.get_uvp_bubble(f)
      xytBubble = self.get_xyt_bubble(f)
      uvpFluid  = self.get_uvp_fluid(f)
      xytFluid  = self.get_xyt_fluid(f)
      uvpWalls  = self.get_uvp_walls(f)
      xytWalls  = self.get_xyt_walls(f)
      idsWalls  = self.get_ids_walls(f)

      # Get ground truth xyt and uvp of bubble, fluid, and / or domain points
      xytTarget, uvpTarget, ids = np.empty(shape=(0, self.dim + 1)), np.empty(shape=(0, self.dim + 1)), np.empty(shape=(0, 1))
      if xyPred[0]:
        # Only use points within boundaries
        mask = self.get_wall_mask(xytBubble)
        xytBubbleMasked = xytBubble[mask]
        uvpBubbleMasked = uvpBubble[mask]
        xytTarget = np.concatenate((xytTarget, xytBubbleMasked))
        uvpTarget = np.concatenate((uvpTarget, uvpBubbleMasked))
        bubbleIds = np.full((len(xytBubbleMasked), 1), 1)
        ids       = np.concatenate((ids, bubbleIds))
      if xyPred[1]:
        # Only use points within boundaries
        mask = self.get_wall_mask(xytFluid)
        xytFluidMasked = xytFluid[mask]
        uvpFluidMasked = uvpFluid[mask]
        xytTarget = np.concatenate((xytTarget, xytFluidMasked))
        uvpTarget = np.concatenate((uvpTarget, uvpFluidMasked))
        colIds  = np.full((len(xytFluidMasked), 1), 0)
        ids     = np.concatenate((ids, colIds))
      if xyPred[2]:
        xytTarget = np.concatenate((xytTarget, xytWalls))
        uvpTarget = np.concatenate((uvpTarget, uvpWalls))
        wallIds  = np.full((len(xytWalls), 1), 0)
        ids      = np.concatenate((ids, np.expand_dims(idsWalls, axis=-1)))

      nGridPnt = len(xytTarget)

      print('nGridPnt is {}'.format(nGridPnt))

      s, e = 0, 0
      while s < nGridPnt:
        e = min(nGridPnt-s, batchSize)

        # Arrays to store batches
        xy = np.zeros((e, self.dim),     dtype=float)
        t  = np.zeros((e, 1),            dtype=float)
        uv = np.zeros((e, self.dim + 1), dtype=float)
        id = np.zeros((e, 1),            dtype=float)

        # Fill batch arrays
        xy[0:e, :] = xytTarget[s:s+e, :self.dim]
        t[0:e, 0]  = xytTarget[s:s+e, self.dim]
        uv[0:e, :] = uvpTarget[s:s+e, :]
        id[0:e, 0] = ids[s:s+e, 0]

        # Shift time range to start at zero
        if resetTime:
          t[:,0] -= self.startFrame
          #if UT.PRINT_DEBUG:
          #  print('Start frame: {}'.format(self.startFrame))

        # Fetch the bc xy and bc value for every point in this batch
        idxT = np.concatenate(t.astype(int))
        xyDataBc = self.xytDataBc[idxT, :, :2]
        uvpDataBc = self.uvpDataBc[idxT, :, :]
        validDataBc = self.validDataBc[idxT, :]

        # Shift time and position to negative range (use -1 to account for 0 in center)
        if zeroMean:
          xy[:,:] -= (self.size-1) / 2
          xyDataBc[:,:] -= (self.size-1) / 2
          #t[:,0]  -= (self.nTotalFrames-1) / 2
          #xyDataBc[:,2]  -= (self.nTotalFrames-1) / 2

          if UT.PRINT_DEBUG:
            print('min, max xyt[0]: [{}, {}]'.format(np.min(t[:,0]), np.max(t[:,0])))
            print('min, max xyt[1]: [{}, {}]'.format(np.min(xy[:,0]), np.max(xy[:,0])))
            print('min, max xyt[2]: [{}, {}]'.format(np.min(xy[:,1]), np.max(xy[:,1])))

        '''
        # Convert from domain space to world space
        pos  = UT.pos_domain_to_world(xy, worldSize, imageSize)
        time = UT.time_domain_to_world(t, fps)
        xyDataBc = UT.pos_domain_to_world(xyDataBc, worldSize, imageSize)

        # Convert from world space to dimensionless quantities
        pos  = UT.pos_world_to_dimensionless(pos, L)
        time = UT.time_world_to_dimensionless(time, T)
        xyDataBc = UT.pos_world_to_dimensionless(xyDataBc, L)

        # Only non-dimensionalize velocities from flownet dataset
        vel = uv
        if self.source == UT.SRC_FLOWNET:
          vel = UT.vel_domain_to_world(uv, worldSize, fps)
          vel = UT.vel_world_to_dimensionless(vel, V)
        '''
        print('Feeding batch from {} to {}'.format(s, s+e))
        s += e

        xy = UT.get_xy_scaled(xy)
        xyDataBc = UT.get_xy_scaled(xyDataBc)
        t = UT.get_t_scaled(t)

        yield [xy, t, uv, xyDataBc, uvpDataBc, validDataBc, id], uv


  def prepare_batch_arrays(self, numFrames, resetTime=True, zeroMean=False):
    print('Preparing samples for batch generator')
    rng = np.random.default_rng(2022)

    if numFrames < 0: numFrames = self.nTotalFrames

    # Extract selection of interface points (otherwise use all)
    useAll = (self.nIfacePnt < 0)
    self.select_iface_points(numFrames, useAll)

    # Extract selection of collocation points
    useDefault = (self.nColPnt < 0)
    self.select_collocation_points(numFrames, useDefault)

    # Extract selection of wall points (otherwise use all)
    useAll = (self.nWallPnt < 0)
    self.select_wall_points(numFrames, useAll)

    # Extract selection of wall points (otherwise use all)
    useDefault = (self.nIcondPnt < 0)
    self.select_icond_points(numFrames, useDefault)

    uvpSamples = np.concatenate((self.uvpIface, self.uvpCol, self.uvpWalls, self.uvpIcond))
    xytSamples = np.concatenate((self.xytIface, self.xytCol, self.xytWalls, self.xytIcond))
    ifaceId    = np.full((len(self.xytIface), 1), 1)
    colId      = np.full((len(self.xytCol), 1), 0)
    wallId     = np.expand_dims(self.idWalls, axis=-1)
    icondId    = np.full((len(self.xytIcond), 1), 3)
    idSamples  = np.concatenate((ifaceId, colId, wallId, icondId))

    assert len(uvpSamples) == len(xytSamples)
    assert len(uvpSamples) == len(idSamples)

    # Shuffle the combined point arrays. Shuffle all arrays with same permutation
    perm = np.random.permutation(len(uvpSamples))
    self.uvpBatch = uvpSamples[perm]
    self.xytBatch = xytSamples[perm]
    self.idBatch  = idSamples[perm]

    # Shift time range to start at zero
    if resetTime:
      self.xytBatch[:,2] -= self.startFrame

    # Zero mean for pos range (use -1 to account for 0 in center)
    if zeroMean:
      self.xytBatch[:,:2] -= (self.size-1) / 2
      #self.xytBatch[:,2] -= (self.nTotalFrames-1) / 2

    normalizeXy = False
    if normalizeXy:
      self.xytBatch[:,0] /= 383.0
      self.xytBatch[:,1] /= 383.0
      self.uvpBatch[:,0] /= 383.0
      self.uvpBatch[:,1] /= 383.0

    print('min, max uvp: [{}, {}]'.format(np.min(self.uvpBatch, axis=0), np.max(self.uvpBatch, axis=0)))
    print('min, max xyt: [{}, {}]'.format(np.min(self.xytBatch, axis=0), np.max(self.xytBatch, axis=0)))

    assert len(self.uvpBatch) == len(self.xytBatch)
    assert len(self.uvpBatch) == len(self.idBatch)


  def prepare_hard_boundary_condition(self, zeroMean=False):
    print('Preparing hard boundary condition')

    # Use frame with largest interface cell count for array dimension
    maxBubble = np.max(self.nBubble)
    maxWalls = np.max(self.nWalls)

    interface = 0
    walls = 1

    maxCells = 0
    if interface: maxCells += maxBubble
    if walls: maxCells += maxWalls

    if not maxCells:
      print('Returning early, no boundary condition points present')
      return

    print('maxBubble, maxWalls: {}, {}'.format(maxBubble, maxWalls))

    # Bubble interface and domain boundary condition: [nFrames, maxCells, nDim + 1]
    self.xytDataBc = np.zeros((self.nTotalFrames, maxCells, self.dim + 1), dtype=float)
    self.uvpDataBc = np.zeros((self.nTotalFrames, maxCells, self.dim + 1), dtype=float)
    # Binary array that tracks valid entries (the ending entries might just be unused 0s)
    self.validDataBc = np.zeros((self.nTotalFrames, maxCells), dtype=float)

    # Zero mean for pos range (use -1 to account for 0 in center)
    if zeroMean:
      self.xytDataBc[:,:2] -= (self.size-1) / 2
      #self.xytDataBc[:,2] -= (self.nTotalFrames-1) / 2

    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      eData = self.nBubble[f]
      eWalls = self.nWalls[f]

      print('nBubbles at frame {}: {}'.format(f, eData))

      s = 0
      if interface:
        self.xytDataBc[f, s:eData, :] = self.get_xyt_bubble(f)
        self.uvpDataBc[f, s:eData, :] = self.get_uvp_bubble(f)
        self.validDataBc[f, s:eData] = 1
        s += maxBubble

      if walls:
        self.xytDataBc[f, s:s+eWalls, :] = self.get_xyt_walls(f)
        self.uvpDataBc[f, s:s+eWalls, :] = self.get_uvp_walls(f)
        self.validDataBc[f, s:s+eWalls] = self.get_ids_walls(f)

      print(self.xytDataBc[f, :, :].shape)

      print('min, max uvpDataBc[0]: [{}, {}]'.format(np.min(self.uvpDataBc[f,:,0]), np.max(self.uvpDataBc[f,:,0])))
      print('min, max uvpDataBc[1]: [{}, {}]'.format(np.min(self.uvpDataBc[f,:,1]), np.max(self.uvpDataBc[f,:,1])))
      print('min, max uvpDataBc[2]: [{}, {}]'.format(np.min(self.uvpDataBc[f,:,2]), np.max(self.uvpDataBc[f,:,2])))


  def generate_train_valid_batch(self, begin, end, batchSize=64, shuffle=True):
    generatorType = 'training' if begin == 0 else 'validation'
    UT.print_info('\nGenerating {} batches with size {}'.format(generatorType, batchSize))

    # Arrays to store iface + collocation + wall point batch
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
      uv[:, :self.dim] = self.uvpBatch[s:e, :self.dim]
      phi[:, 0]        = self.uvpBatch[s:e, self.dim] # phi for now
      xy[:, :]         = self.xytBatch[s:e, :self.dim]
      t[:, 0]          = self.xytBatch[s:e, self.dim]
      id[:, 0]         = self.idBatch[s:e, 0]

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

      idxT = np.concatenate(t.astype(int))
      xyDataBc = self.xytDataBc[idxT, :, :2]
      uvpDataBc = self.uvpDataBc[idxT, :, :]
      validDataBc = self.validDataBc[idxT, :]

      # Convert from domain space to world space
      '''
      pos  = UT.pos_domain_to_world(xy, worldSize, imageSize)
      time = UT.time_domain_to_world(t, fps)
      xyDataBc = UT.pos_domain_to_world(xyDataBc, worldSize, imageSize)

      # Convert from world space to dimensionless quantities
      pos  = UT.pos_world_to_dimensionless(pos, L)
      time = UT.time_world_to_dimensionless(time, T)
      xyDataBc = UT.pos_world_to_dimensionless(xyDataBc, L)

      # Only non-dimensionalize velocities from flownet dataset
      vel = uv
      if self.source == UT.SRC_FLOWNET:
        vel = UT.vel_domain_to_world(uv, worldSize, fps)
        vel = UT.vel_world_to_dimensionless(vel, V)
      '''

      xy = UT.get_xy_scaled(xy)
      xyDataBc = UT.get_xy_scaled(xyDataBc)
      t = UT.get_t_scaled(t)

      yield [xy, t, uv, xyDataBc, uvpDataBc, validDataBc, id, phi], uv


  # Define domain border locations + attach uvp at those locations
  def extract_wall_points(self, useDataBc=False):
    UT.print_info('Extracting domain wall points')

    rng = np.random.default_rng(2022)

    if not sum(self.walls):
      if UT.PRINT_DEBUG: print('No walls defined. Not extracting any wall points')
      return

    bcFrameLst = [] # Domain boundary condition for every frame
    xyFrameLst = [] # Domain boundary condition xy for every frame
    idFrameLst = [] # Domain boundary condition id for every frame

    idLeft, idTop, idRight, idBottom = 20, 21, 22, 23

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
      xyLst, idLst = [], []
      cellCnt = 0

      # Walls:
      # Left domain wall
      for i in range(self.walls[0]):
        numCells = sizeY - bWBottom - bWTop - 2 # -2 to exclude corners
        x = np.full((numCells,), bWLeft-i, dtype=float)
        y = np.linspace(bWBottom+1, sizeY-1-bWTop-1, num=numCells, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        d = np.full((numCells,), idLeft, dtype=int)
        xyLst.extend(list(zip(x,y,t)))
        idLst.extend(list(d))
        cellCnt += numCells

      # Top domain wall
      for i in range(self.walls[1]):
        numCells = sizeX - bWLeft - bWRight -2
        x = np.linspace(bWLeft+1, sizeX-1-bWRight-1, num=numCells, dtype=float)
        y = np.full((numCells,), sizeY-1-bWTop+i, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        d = np.full((numCells,), idTop, dtype=int)
        xyLst.extend(list(zip(x,y,t)))
        idLst.extend(list(d))
        cellCnt += numCells

      # Right domain wall
      for i in range(self.walls[2]):
        numCells = sizeY - bWTop - bWBottom - 2
        x = np.full((numCells,), sizeX-1-bWRight+i, dtype=float)
        y = np.linspace(sizeY-1-bWTop-1, bWBottom+1, num=numCells, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        d = np.full((numCells,), idRight, dtype=int)
        xyLst.extend(list(zip(x,y,t)))
        idLst.extend(list(d))
        cellCnt += numCells

      # Bottom domain wall
      for i in range(self.walls[3]):
        numCells = sizeX - bWRight - bWLeft - 2
        x = np.linspace(sizeX-1-bWRight-1, bWLeft+1, num=numCells, dtype=float)
        y = np.full((numCells,), bWBottom-i, dtype=float)
        t = np.full((numCells,), frame, dtype=float)
        d = np.full((numCells,), idBottom, dtype=int)
        xyLst.extend(list(zip(x,y,t)))
        idLst.extend(list(d))
        cellCnt += numCells

      # Corners:
      # Top left corner
      if hasLeftWall or hasTopWall:
        accXyt, accId = [], []
        for i in range(0, max(1, self.walls[0])): # left
          for j in range(min(sizeY-self.walls[1], sizeY-1), sizeY): # top
            accXyt.append((i,j,frame))
            accId.append(idLeft) # Define top left corner as part of left wall
            cellCnt += 1
        xyLst.extend(accXyt)
        idLst.extend(accId)

      # Top right corner
      if hasRightWall or hasTopWall:
        accXyt, accId = [], []
        for i in range(min(sizeX-self.walls[2], sizeX-1), sizeX): # right
          for j in range(min(sizeY-self.walls[1], sizeY-1), sizeY): # top
            accXyt.append((i,j,frame))
            accId.append(idRight) # Define top right corner as part of right wall
            cellCnt += 1
        xyLst.extend(accXyt)
        idLst.extend(accId)

      # Bottom left corner
      if hasLeftWall or hasBottomWall:
        accXyt, accId = [], []
        for i in range(0, max(1, self.walls[0])): # left
          for j in range(0, max(1, self.walls[3])): # bottom
            accXyt.append((i,j,frame))
            accId.append(idLeft) # Define bottom left corner as part of left wall
            cellCnt += 1
        xyLst.extend(accXyt)
        idLst.extend(accId)

      # Bottom right corner
      if hasRightWall or hasBottomWall:
        accXyt, accId = [], []
        for i in range(min(sizeX-self.walls[2], sizeX-1), sizeX): # right
          for j in range(0, max(1, self.walls[3])): # bottom
            accXyt.append((i,j,frame))
            accId.append(idRight) # Define bottom right corner as part of right wall
            cellCnt += 1
        xyLst.extend(accXyt)
        idLst.extend(accId)

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
      idFrameLst.append(idLst)

      if UT.IMG_DEBUG:
        UT.save_array(xyLst, '{}/all'.format(self.sourceName), 'wallpts_all', frame, self.size)

      # Keep track of number of wall cells per frame
      self.nWalls.append(len(xyLst))

    if UT.PRINT_DEBUG:
      print('Total number of wall samples: {}'.format(np.sum(self.nWalls)))

    self.uvpDomain = np.zeros((np.sum(self.nWalls), self.dim + 1))
    self.xytDomain = np.zeros((np.sum(self.nWalls), self.dim + 1))
    self.idDomain = np.zeros((np.sum(self.nWalls)), dtype=int)

    s = 0
    for f in range(self.nTotalFrames):
      assert len(bcFrameLst[f]) == len(xyFrameLst[f]), 'Number of velocity labels must match number of xy positions'
      assert len(bcFrameLst[f]) == len(xyFrameLst[f]), 'Number of velocity labels must match number of ids'

      n = len(bcFrameLst[f])
      e = s + n
      if n:
        uvp = np.asarray(bcFrameLst[f], dtype=float)
        xyt = np.asarray(xyFrameLst[f], dtype=float)
        ids = np.asarray(idFrameLst[f], dtype=int)
        self.uvpDomain[s:e, :] = uvp
        self.xytDomain[s:e, :] = xyt
        self.idDomain[s:e]    = ids
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
          if ni < 0 or ni >= sizeY or nj < 0 or nj >= sizeX:
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
        isBubble = phi[j,i] > 0
        isFluid = not isBubble

        # Safe grid indices
        iN, iP, jN, jP = max(i-1,0), min(i+1, sizeX-1), max(j-1,0), min(j+1, sizeY-1)

        isIntersection = ( phi[jN, i ] > 0 or phi[jN, i ] > 0 or \
                           phi[j,  iP] > 0 or phi[j,  iN] > 0 or \
                           phi[jP, iP] > 0 or phi[jN, iP] > 0 or \
                           phi[jP, iN] > 0 or phi[jN, iN] > 0) and phi[j,i] <= 0

        intersection[j, i] = isIntersection
        flags[j, i] = self.FLAG_BUBBLE if isBubble else self.FLAG_FLUID


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
      intersection = np.zeros((sizeY, sizeX), dtype=int)
      # Flags == grid with the type of the cell (fluid or bubble)
      flags = np.full((sizeY, sizeX), self.FLAG_BUBBLE)

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
        self._thicken_interface(self.interface, flags, intersection, phi=None, fromInside=False)

      if UT.IMG_DEBUG:
        UT.save_image(intersection, '{}/all'.format(self.sourceName), 'bubblepts_all', frame, i=self.interface)
        UT.save_image(mag, '{}/all'.format(self.sourceName), 'magnitude_all', frame)
        UT.save_image(curPhi, '{}/all'.format(self.sourceName), 'phi_all', frame, cmap='jet')
        if self.source == UT.SRC_FLOWNET:
          UT.save_plot(self.vel[f,:,:,2], '{}/plots'.format(self.sourceName), 'flownet_phi_plt', frame, size=self.size, cmin=np.min(curPhi), cmax=np.max(curPhi), cmap='jet')

      # Add bubble border positions to xyBubble list
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

      assert len(bcLst) == len(bubbleBorderIndices), 'Number of bubble point velocities must match number of indices'

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
      self.nBubble.append(len(bubbleBorderIndices))
      self.nFluid.append(len(fluidIndices))

    if UT.PRINT_DEBUG:
      print('Total number of bubble points: {}'.format(np.sum(self.nBubble)))
      print('Total number of fluid points: {}'.format(np.sum(self.nFluid)))

    # Allocate arrays for preprocessed data ...
    self.uvpBubble   = np.zeros((np.sum(self.nBubble), self.dim + 1), dtype=float) # dim + 1 for p
    self.xytBubble = np.zeros((np.sum(self.nBubble), self.dim + 1), dtype=float) # dim + 1 for t
    self.xytFluid = np.zeros((np.sum(self.nFluid), self.dim + 1), dtype=float)
    self.uvpFluid = np.zeros((np.sum(self.nFluid), self.dim + 1), dtype=float)

    # ... and insert data from lists
    s, sFl = 0, 0
    for f in range(self.nTotalFrames):
      assert len(bcFrameLst[f]) == len(xyFrameLst[f]), 'Number of labels must match number of positions'
      assert len(uvpFluidFrameLst[f]) == len(xyFluidFrameLst[f]), 'Number of labels must match number of positions'

      frame = f + self.startFrame

      # Insertion of bc and xyBubble lists
      n = len(bcFrameLst[f])
      t = np.full((n, 1), frame, dtype=float)
      e = s + n
      if n: # only insert if there is at least 1 cell
        uvp = np.asarray(bcFrameLst[f], dtype=float)
        xy  = np.asarray(xyFrameLst[f], dtype=float)
        self.uvpBubble[s:e, :] = uvp
        self.xytBubble[s:e, :] = np.hstack((xy, t))
        if UT.IMG_DEBUG:
          UT.save_array(self.xytBubble[s:e, :], '{}/positions'.format(self.sourceName), 'xyBubble_extract', frame, self.size)

      if UT.PRINT_DEBUG:
        print('Min / max of positions')
        print('  x ({},{}), y ({},{})'.format(np.min(self.xytBubble[s:e, 0]), np.max(self.xytBubble[s:e, 0]),np.min(self.xytBubble[s:e, 1]), np.max(self.xytBubble[s:e, 1])))
      s = e

      # Insertion of xytFluid list
      nFl = len(xyFluidFrameLst[f])
      tFl = np.full((nFl, 1), frame, dtype=float)
      eFl = sFl + nFl
      if nFl: # only insert if there is at least 1 cell
        uvp = np.asarray(uvpFluidFrameLst[f], dtype=float)
        xy = np.asarray(xyFluidFrameLst[f], dtype=float)
        self.uvpFluid[sFl:eFl, :] = uvp
        self.xytFluid[sFl:eFl, :] = np.hstack((xy, tFl))
        if UT.IMG_DEBUG:
          UT.save_array(self.xytFluid[sFl:eFl, :], '{}/positions'.format(self.sourceName), 'xyFluid_extract', frame, self.size)
      sFl = eFl

    if UT.IMG_DEBUG:
      UT.save_velocity_bins(self.uvpBubble[:, 0], '{}/histograms'.format(self.sourceName), 'bc_x_bins', frame, bmin=-3.0, bmax=3.0, bstep=0.1)
      UT.save_velocity_bins(self.uvpBubble[:, 1], '{}/histograms'.format(self.sourceName), 'bc_y_bins', frame, bmin=-3.0, bmax=3.0, bstep=0.1)


  def _fill_batch_arrays(self, xytTarget, uvpTarget, nFrames, nPoints, pType, idsTarget=None, replace=False):
    UT.print_info('Selecting {} points'.format(pType))
    rng = np.random.default_rng(2022)
    s = 0

    for f in range(nFrames):
      frame = f + self.startFrame
      UT.print_progress(f, nFrames)

      # Get all coords for the current frame
      if pType == 'icond':
        xyt = self.get_xyt_fluid(0)
        uvp = self.get_uvp_fluid(0)
      elif pType == 'colloc':
        xyt = self.get_xyt_fluid(f)
        uvp = self.get_uvp_fluid(f)
      elif pType == 'wall':
        xyt = self.get_xyt_walls(f)
        uvp = self.get_uvp_walls(f)
        ids = self.get_ids_walls(f)
      elif pType == 'iface':
        xyt = self.get_xyt_bubble(f)
        uvp = self.get_uvp_bubble(f)
      else:
        sys.exit('Invalid pType')

      # Only use every other grid point as data point
      mask = np.logical_and(xyt[:,0] % self.colRes == 0, xyt[:,1] % self.colRes == 0)
      xyt = xyt[mask]
      uvp = uvp[mask]

      # Only use points within boundaries
      if pType != 'wall':
        mask = self.get_wall_mask(xyt)
        xyt = xyt[mask]
        uvp = uvp[mask]

      # Insert random selection of fluid coords into data array
      if UT.PRINT_DEBUG: print('Taking {} {} points out of {} available'.format(nPoints, pType, xyt.shape[0]))
      indices = np.arange(0, xyt.shape[0])
      randIndices = rng.choice(indices, size=nPoints, replace=replace)

      e = s + nPoints
      xytTarget[s:e, :] = xyt[randIndices, :]
      uvpTarget[s:e, :] = uvp[randIndices, :]
      if idsTarget is not None:
        idsTarget[s:e] = ids[randIndices]
      s = e

      if UT.IMG_DEBUG:
        UT.save_array(xyt[randIndices, :], '{}/{}'.format(self.sourceName, pType), '{}_pts_select'.format(pType), frame, self.size)


  def select_iface_points(self, numFrames, useAll=False):
    numPoints = np.sum(self.nBubble[:numFrames])

    if useAll:
      self.xytIface = copy.deepcopy(self.xytBubble[:numPoints, :])
      self.uvpIface = copy.deepcopy(self.uvpBubble[:numPoints, :])
      mask = self.get_wall_mask(self.xytIface)
      self.xytIface = self.xytIface[mask]
      self.uvpIface = self.uvpIface[mask]
      print('Using all {} available interface points'.format(numPoints))
    else:
      nPts = self.nIfacePnt * numFrames
      self.xytIface = np.zeros((nPts, self.N_IN_VAR))
      self.uvpIface = np.zeros((nPts, self.N_OUT_VAR))
      replace = (nPts > numPoints)
      self._fill_batch_arrays(self.xytIface, self.uvpIface, nFrames=numFrames, nPoints=self.nIfacePnt, pType='iface', replace=replace)


  def select_wall_points(self, numFrames, useAll=False):
    numPoints = np.sum(self.nWalls[:numFrames])
    if useAll:
      self.xytWalls = copy.deepcopy(self.xytDomain[:numPoints, :])
      self.uvpWalls = copy.deepcopy(self.uvpDomain[:numPoints, :])
      self.idWalls = copy.deepcopy(self.idDomain)
      print('Using all {} available wall points'.format(numPoints))
    else:
      nPts = self.nWallPnt * numFrames
      self.xytWalls = np.zeros((nPts, self.N_IN_VAR))
      self.uvpWalls = np.zeros((nPts, self.N_OUT_VAR))
      self.idWalls = np.zeros((nPts))
      replace = (nPts > numPoints)
      self._fill_batch_arrays(self.xytWalls, self.uvpWalls, nFrames=numFrames, nPoints=self.nWallPnt, pType='wall', idsTarget=self.idWalls, replace=replace)


  def select_collocation_points(self, numFrames, default=False):
    if default:
      self.nColPnt = 1000
    nPts = self.nColPnt * numFrames
    self.xytCol = np.zeros((nPts, self.N_IN_VAR))
    self.uvpCol = np.zeros((nPts, self.N_OUT_VAR))
    self._fill_batch_arrays(self.xytCol, self.uvpCol, nFrames=numFrames, nPoints=self.nColPnt, pType='colloc')


  def select_icond_points(self, numFrames, default=False):
    if default:
      self.nIcondPnt = 1000
    nPts = self.nIcondPnt * numFrames
    self.xytIcond = np.zeros((nPts, self.N_IN_VAR))
    self.uvpIcond = np.zeros((nPts, self.N_OUT_VAR))
    self._fill_batch_arrays(self.xytIcond, self.uvpIcond, nFrames=numFrames, nPoints=self.nIcondPnt, pType='icond')


  def save(self, dir='../data/', filePrefix='bdata'):
    if not os.path.exists(dir):
      os.makedirs(dir)

    nSampleBubble = np.sum(self.nBubble)
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
    dFile.attrs['nBubble']      = np.asarray(self.nBubble)
    dFile.attrs['nFluid']       = np.asarray(self.nFluid)
    dFile.attrs['nWalls']       = np.asarray(self.nWalls)

    # Compression
    comp_type = 'gzip'
    comp_level = 9

    dFile.create_dataset('uvpBubble', (nSampleBubble, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.uvpBubble)
    dFile.create_dataset('xytBubble', (nSampleBubble, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xytBubble)
    dFile.create_dataset('uvpFluid', (nSampleFluid, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.uvpFluid)
    dFile.create_dataset('xytFluid', (nSampleFluid, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xytFluid)
    dFile.create_dataset('uvpDomain', (nSampleWalls, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.uvpDomain)
    dFile.create_dataset('xytDomain', (nSampleWalls, self.dim + 1), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xytDomain)
    dFile.create_dataset('idDomain', (nSampleWalls,), compression=comp_type,
                          compression_opts=comp_level, dtype='int32', chunks=True, data=self.idDomain)

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
    self.nBubble      = dFile.attrs['nBubble']
    self.nFluid       = dFile.attrs['nFluid']
    self.nWalls       = dFile.attrs['nWalls']

    self.uvpBubble = np.array(dFile.get('uvpBubble'))
    self.xytBubble = np.array(dFile.get('xytBubble'))
    self.uvpFluid  = np.array(dFile.get('uvpFluid'))
    self.xytFluid  = np.array(dFile.get('xytFluid'))
    self.uvpDomain = np.array(dFile.get('uvpDomain'))
    self.xytDomain = np.array(dFile.get('xytDomain'))
    self.idDomain  = np.array(dFile.get('idDomain'))

    self.isLoaded = True
    self.nTotalFrames = self.endFrame - self.startFrame + 1
    dFile.close()
    print('Restored dataset from file {}'.format(fname))
    print('Dataset: size [{},{}], frames {}'.format(self.size[0], self.size[1], self.nTotalFrames))


  def get_num_total_pts(self):
    nIfacePnt = self.nIfacePnt * self.nTotalFrames if self.nIfacePnt > -1 else self.xytIface.shape[0]
    nWallPnt = self.nWallPnt * self.nTotalFrames if self.nWallPnt > -1 else self.xytWalls.shape[0]
    nColPnt = self.nColPnt * self.nTotalFrames
    nIcondPnt = self.nIcondPnt * self.nTotalFrames
    return nIfacePnt + nWallPnt + nColPnt + nIcondPnt


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


  def get_start_frame(self):
    return self.startFrame


  def get_end_frame(self):
    return self.endFrame


  def get_source(self):
    return self.source


  def get_source_name(self):
    return self.sourceName


  def get_xyt_bubble(self, f):
    s = sum(self.nBubble[:f])
    e = s + self.nBubble[f]
    return self.xytBubble[s:e, ...]


  def get_xyt_fluid(self, f):
    s = sum(self.nFluid[:f])
    e = s + self.nFluid[f]
    return self.xytFluid[s:e, ...]


  def get_xyt_walls(self, f):
    s = sum(self.nWalls[:f])
    e = s + self.nWalls[f]
    return self.xytDomain[s:e, ...]


  def get_uvp_bubble(self, f):
    s = sum(self.nBubble[:f])
    e = s + self.nBubble[f]
    return self.uvpBubble[s:e, ...]


  def get_uvp_fluid(self, f):
    s = sum(self.nFluid[:f])
    e = s + self.nFluid[f]
    return self.uvpFluid[s:e, ...]


  def get_uvp_walls(self, f):
    s = sum(self.nWalls[:f])
    e = s + self.nWalls[f]
    return self.uvpDomain[s:e, ...]


  def get_ids_walls(self, f):
    s = sum(self.nWalls[:f])
    e = s + self.nWalls[f]
    return self.idDomain[s:e]



