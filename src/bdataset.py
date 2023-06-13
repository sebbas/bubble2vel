#!/usr/bin/env python3

import h5py as h5
import os.path
import sys
import copy
import numpy as np
import random as rn
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

  N_IN_VAR = 3 # x, y, t
  N_OUT_VAR = 4 # u, v, phi, scalar

  def __init__(self, fName='', startFrame=0, endFrame=399, dim=2, \
               wallPoints=-1, colPoints=-1, ifacePoints=-1, icondPoints=-1, dataPoints=-1, \
               walls=[1,1,1,1], interface=1, source=UT.SRC_FLASHX):
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
    # Requested number of points
    self.nColPnt   = colPoints
    self.nWallPnt  = wallPoints
    self.nIfacePnt = ifacePoints
    self.nIcondPnt = icondPoints
    self.nDataPnt  = dataPoints
    # Array to store ground truth data (after processing .flo input)
    self.rawData   = None # [frames, width, height, N_OUT_VAR]
    # Arrays to store processed data (after processing ground truth)
    self.xytBubble = None # [nSamples, N_IN_VAR]
    self.uvpBubble = None # [nSamples, N_OUT_VAR]
    self.xytFluid  = None # [nSamples, N_IN_VAR]
    self.uvpFluid  = None # [nSamples, N_OUT_VAR]
    self.xytDomain = None # [nSamples, N_IN_VAR]
    self.uvpDomain = None # [nSamples, N_OUT_VAR]
    self.idDomain  = None # [nSamples]
    # Arrays to store selection of points
    self.xytCol   = None # [nColPnt,  N_IN_VAR]
    self.uvpCol   = None # [nColPnt,  N_OUT_VAR]
    self.xytIface = None # [nIfacePnt, N_IN_VAR]
    self.uvpIface = None # [nIfacePnt, N_OUT_VAR]
    self.xytWalls = None # [nWallPnt, N_IN_VAR]
    self.uvpWalls = None # [nWallPnt, N_OUT_VAR]
    self.idWalls  = None # [nWallPnt]
    self.xytIcond = None # [nIcondPnt, N_IN_VAR]
    self.uvpIcond = None # [nIcondPnt, N_OUT_VAR]
    self.xytData  = None # [nDataPnt, N_IN_VAR]
    self.uvpData  = None # [nDataPnt, N_OUT_VAR]
    # Arrays to store velocity state
    self.stateVel = None # [nTotalFrames, latentDim]
    self.latentDim = 9216 # Adjust according to latent vector
    self.nTotalFramesState = 100 # Adjust according to latent vector
    # Arrays to store boundary condition per frame
    self.xytDataBc   = None # [nTotalFrames, max(nBubble), N_IN_VAR]
    self.uvpDataBc   = None # [nTotalFrames, max(nBubble), N_OUT_VAR]
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


  def _load_flashx(self, filter=True):
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
        pressure = np.zeros((self.nTotalFrames, self.size[1], self.size[0]), dtype=float)
        self.rawData = np.zeros((self.nTotalFrames, self.size[1], self.size[0], N_OUT_VAR), dtype=float)
        allocArrays = False
      else:
        assert pressure.shape[1] == self.size[1] and pressure.shape[2] == self.size[0]
        assert self.rawData.shape[1] == self.size[1] and self.rawData.shape[2] == self.size[0]

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
                pressure[cnt, sy, sx:ex] = pres[0:self.pshape[2]]
                self.rawData[cnt, sy, sx:ex, 0] = velx[0:self.pshape[2]]
                self.rawData[cnt, sy, sx:ex, 1] = vely[0:self.pshape[2]]
                self.rawData[cnt, sy, sx:ex, 2] = dfun[0:self.pshape[2]]
                self.rawData[cnt, sy, sx:ex, 3] = temp[0:self.pshape[2]]
            bcnt += 1

      # Filter out the outlier velocities above a certain velocity magnitude
      if filter:
        velMag = np.sqrt(np.square(self.rawData[:,:,:,0]) + np.square(self.rawData[:,:,:,1]))

        # Replace outliers (e.g. fixed value or quantile)
        maxMag = 10.0
        #maxMag = np.quantile(velMag[:,:,:], 0.99) # e.g. find the 0.99 quantile in all vels
        print('Capping velocites above magnitude of {}'.format(maxMag))

        # Scale velocity vectors in each dimension
        scaleVec = np.where(velMag[:,:,:] > maxMag, maxMag / (velMag[:,:,:] + 1.0e-10), 1)
        scaleVec = np.expand_dims(scaleVec, axis=-1)
        self.rawData[:,:,:,:self.dim] *= scaleVec

        # Scaling c
        thresMin = 5.0e-3
        self.rawData[:,:,:,3] = np.where(np.abs(self.rawData[:,:,:,3]) < thresMin, 0, self.rawData[:,:,:,3])
        #self.rawData[:,:,:,3] = np.expm1(self.rawData[:,:,:,3]) / 0.05
        #self.rawData[:,:,:,3] = np.sqrt(np.square(self.rawData[:,:,:,0]) + np.square(self.rawData[:,:,:,1])) / 10.0

      if 0 and UT.PRINT_DEBUG:
        tempMin, tempMax = self.rawData[cnt,:,:,3].min(), self.rawData[cnt,:,:,3].max()
        print('Frame {}: Min/Max temp [{},{}]'.format(frame, tempMin, tempMax))
        presMin, presMax = pressure[cnt].min(), pressure[cnt].max()
        print('Frame {}: Min/Max pressure [{},{}]'.format(frame, presMin, presMax))

        velx = abs(max(self.rawData[cnt,:,:,0].min(), self.rawData[cnt,:,:,0].max(), key = abs))
        vely = abs(max(self.rawData[cnt,:,:,1].min(), self.rawData[cnt,:,:,1].max(), key = abs))
        print('Frame {}: Maximum vel [{}, {}]'.format(frame, velx, vely))
        dfunMin, dfunMax = self.rawData[cnt,:,:,2].min(), self.rawData[cnt,:,:,2].max()
        print('Frame {}: Min/Max levelset [{},{}]'.format(frame, dfunMin, dfunMax))

      if UT.IMG_DEBUG:
        # Raw grids
        UT.save_image(pressure[cnt], '{}/raw'.format(self.sourceName), 'flashx_pres_raw', frame, cmap='jet')
        UT.save_image(self.rawData[cnt,:,:,0], '{}/raw'.format(self.sourceName), 'flashx_velx_raw', frame, cmap='hot')
        UT.save_image(self.rawData[cnt,:,:,1], '{}/raw'.format(self.sourceName), 'flashx_vely_raw', frame, cmap='hot')
        UT.save_image(self.rawData[cnt,:,:,2], '{}/raw'.format(self.sourceName), 'flashx_phi_raw', frame, cmap='jet', vmin=-0.5, vmax=0.5)
        UT.save_image(self.rawData[cnt,:,:,3], '{}/raw'.format(self.sourceName), 'flashx_temp_raw', frame, cmap='jet', vmin=0.0, vmax=5.0)
        # Velocity matplotlib plots
        UT.save_velocity(self.rawData[cnt], '{}/plots'.format(self.sourceName), 'flashx_vel_stream', frame, size=self.size, type='stream', density=5.0)
        UT.save_velocity(self.rawData[cnt], '{}/plots'.format(self.sourceName), 'flashx_vel_vector', frame, size=self.size, type='quiver', arrow_res=8, cmin=0.0, cmax=5.0)
        # Grid matplotlib plots
        UT.save_plot(pressure[cnt], '{}/plots'.format(self.sourceName), 'flashx_pres_plt', frame, size=self.size, cmin=0.0, cmax=1.0, cmap='jet')
        UT.save_plot(self.rawData[cnt,:,:,2], '{}/plots'.format(self.sourceName), 'flashx_phi_plt', frame, size=self.size, cmin=-1.0, cmax=1.0, cmap='jet')
        UT.save_plot(self.rawData[cnt,:,:,3], '{}/plots'.format(self.sourceName), 'flashx_temp_plt', frame, size=self.size, cmin=0.0, cmax=1.0, cmap='jet')
        # Distribution velocities in bins
        UT.save_velocity_bins(self.rawData[cnt,:,:,0], '{}/histograms'.format(self.sourceName), 'flashx_velx_bins', frame, bmin=-1.0, bmax=1.0, bstep=0.05)
        UT.save_velocity_bins(self.rawData[cnt,:,:,1], '{}/histograms'.format(self.sourceName), 'flashx_vely_bins', frame, bmin=-1.0, bmax=1.0, bstep=0.05)
        UT.save_velocity_bins(self.rawData[cnt,:,:,3], '{}/histograms'.format(self.sourceName), 'flashx_temp_bins', frame, bmin=0, bmax=1.0, bstep=0.05)

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
        self.rawData = np.zeros((self.nTotalFrames, sizeFromFile[1], sizeFromFile[0], N_OUT_VAR), dtype=float)
        # Initialize 3rd dimension with negative phi init value
        self.rawData[:,:,:,2] = -1.0 * self.phiInit
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

      self.rawData[cnt,:,:,:self.dim] = dataFromFile

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
    print('  u    {}, {}'.format(np.amin(self.rawData[:,:,:,0]), np.amax(self.rawData[:,:,:,0])))
    print('  v    {}, {}'.format(np.amin(self.rawData[:,:,:,1]), np.amax(self.rawData[:,:,:,1])))
    print('  phi  {}, {}'.format(np.amin(self.rawData[:,:,:,2]), np.amax(self.rawData[:,:,:,2])))
    print('  temp {}, {}'.format(np.amin(self.rawData[:,:,:,3]), np.amax(self.rawData[:,:,:,3])))
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


  def generate_predict_pts(self, begin, end, worldSize, imageSize, fps, L, T, xyPred=[1,1,1], resetTime=True, zeroMean=True, batchSize=int(1e4), hardBc=True):
    print('Generating prediction points')

    for f in range(begin, end):
      uvpBubble = self.get_uvp_bubble(f)
      xytBubble = self.get_xyt_bubble(f)
      uvpFluid  = self.get_uvp_fluid(f)
      xytFluid  = self.get_xyt_fluid(f)
      uvpWalls  = self.get_uvp_walls(f)
      xytWalls  = self.get_xyt_walls(f)

      # Get ground truth xyt and uvp of bubble, fluid, and / or domain points
      xytTarget, uvpTarget = np.empty(shape=(0, self.N_IN_VAR)), np.empty(shape=(0, self.N_OUT_VAR))
      if xyPred[0]:
        # Only use points within boundaries
        mask = self.get_wall_mask(xytBubble)
        xytBubbleMasked = xytBubble[mask]
        uvpBubbleMasked = uvpBubble[mask]
        xytTarget = np.concatenate((xytTarget, xytBubbleMasked))
        uvpTarget = np.concatenate((uvpTarget, uvpBubbleMasked))
      if xyPred[1]:
        # Only use points within boundaries
        mask = self.get_wall_mask(xytFluid)
        xytFluidMasked = xytFluid[mask]
        uvpFluidMasked = uvpFluid[mask]
        xytTarget = np.concatenate((xytTarget, xytFluidMasked))
        uvpTarget = np.concatenate((uvpTarget, uvpFluidMasked))
      if xyPred[2]:
        xytTarget = np.concatenate((xytTarget, xytWalls))
        uvpTarget = np.concatenate((uvpTarget, uvpWalls))

      nGridPnt = len(xytTarget)

      print('nGridPnt is {}'.format(nGridPnt))

      s, e = 0, 0
      while s < nGridPnt:
        e = min(nGridPnt-s, batchSize)

        # Arrays to store batches
        uv  = np.zeros((e, self.dim), dtype=float)
        xy  = np.zeros((e, self.dim), dtype=float)
        t   = np.zeros((e, 1),        dtype=float)
        id  = np.zeros((e, 1),        dtype=float) # TODO: must be filled, empty for now
        phi = np.zeros((e, 1),        dtype=float)
        c   = np.zeros((e, 1),        dtype=float)
        state = np.zeros((e, self.latentDim), dtype=float)

        # Fill batch arrays
        xy[0:e, :]  = xytTarget[s:s+e, :self.dim]
        t[0:e, 0]   = xytTarget[s:s+e, self.dim]
        uv[0:e, :]  = uvpTarget[s:s+e, :self.dim]
        phi[0:e, 0] = uvpTarget[s:s+e, 2]
        c[0:e, 0]   = uvpTarget[s:s+e, 3]

        # Shift time range to start at zero
        if resetTime:
          t[:,0] -= self.startFrame
          #if UT.PRINT_DEBUG:
          #  print('Start frame: {}'.format(self.startFrame))

        # Fetch the bc xy and bc value for every point in this batch
        idxT = np.concatenate(t.astype(int))
        xyDataBc = self.xytDataBc[idxT, :, :2]
        uvDataBc = self.uvpDataBc[idxT, :, :2]
        validDataBc = self.validDataBc[idxT, :]

        # Use t to select entry from state array
        state[:,:] = self.stateVel[idxT, :]

        # Shift time and position to negative range (use -1 to account for 0 in center)
        if zeroMean:
          print(self.nTotalFrames)
          xy[:,:] -= (self.size-1) / 2
          xyDataBc[:,:1] -= (self.size-1) / 2
          #t[:,0]  -= (self.nTotalFrames-1) / 2
          #xyDataBc[:,2]  -= (self.nTotalFrames-1) / 2

          if UT.PRINT_DEBUG:
            print('min, max xyt[0]: [{}, {}]'.format(np.min(t[:,0]), np.max(t[:,0])))
            print('min, max xyt[1]: [{}, {}]'.format(np.min(xy[:,0]), np.max(xy[:,0])))
            print('min, max xyt[2]: [{}, {}]'.format(np.min(xy[:,1]), np.max(xy[:,1])))

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

        print('Feeding batch from {} to {}'.format(s, s+e))
        s += e

        dummy = vel
        if hardBc:
          yield [pos, time, state, vel, xyDataBc, uvDataBc, validDataBc, id, phi, c], dummy
        else:
          yield [pos, time, state, vel], dummy


  def prepare_batch_arrays(self, resetTime=True, zeroMean=True):
    print('Preparing samples for batch generator')

    # Extract selection of data points (by default, will use all or default number)
    useAll = (self.nIfacePnt < 0)
    self.select_iface_points(useAll)
    useDefault = (self.nColPnt < 0)
    self.select_collocation_points(useDefault)
    useAll = (self.nWallPnt < 0)
    self.select_wall_points(useAll)
    useDefault = (self.nIcondPnt < 0)
    self.select_icond_points(useDefault)
    useDefault = (self.nDataPnt < 0)
    self.select_data_points(useDefault)

    uvpSamples = np.concatenate((self.uvpIface, self.uvpCol, self.uvpWalls, self.uvpIcond, self.uvpData))
    xytSamples = np.concatenate((self.xytIface, self.xytCol, self.xytWalls, self.xytIcond, self.xytData))
    ifaceId    = np.full((len(self.xytIface), 1), 1)
    colId      = np.full((len(self.xytCol), 1), 0)
    wallId     = np.expand_dims(self.idWalls, axis=-1)
    icondId    = np.full((len(self.xytIcond), 1), 3)
    dataId     = np.full((len(self.xytData), 1), 4)
    idSamples  = np.concatenate((ifaceId, colId, wallId, icondId, dataId))

    assert len(uvpSamples) == len(xytSamples)
    assert len(uvpSamples) == len(idSamples)

    # Shuffle the combined point arrays. Shuffle all arrays with same permutation
    perm = np.random.permutation(len(uvpSamples))
    self.uvpBatch = uvpSamples[perm]
    self.xytBatch = xytSamples[perm]
    self.idBatch  = idSamples[perm]

    print('min, max uvp[0]: [{}, {}]'.format(np.min(self.uvpBatch[:,0]), np.max(self.uvpBatch[:,0])))
    print('min, max uvp[1]: [{}, {}]'.format(np.min(self.uvpBatch[:,1]), np.max(self.uvpBatch[:,1])))
    print('min, max uvp[2]: [{}, {}]'.format(np.min(self.uvpBatch[:,2]), np.max(self.uvpBatch[:,2])))
    print('min, max uvp[3]: [{}, {}]'.format(np.min(self.uvpBatch[:,3]), np.max(self.uvpBatch[:,3])))

    # Shift time range to start at zero
    if resetTime:
      self.xytBatch[:,2] -= self.startFrame

    # Zero mean for pos range (use -1 to account for 0 in center)
    if zeroMean:
      self.xytBatch[:,:1] -= (self.size-1) / 2
      #self.xytBatch[:,2] -= (self.nTotalFrames-1) / 2

    print('min, max xyt[0]: [{}, {}]'.format(np.min(self.xytBatch[:,0]), np.max(self.xytBatch[:,0])))
    print('min, max xyt[1]: [{}, {}]'.format(np.min(self.xytBatch[:,1]), np.max(self.xytBatch[:,1])))
    print('min, max xyt[2]: [{}, {}]'.format(np.min(self.xytBatch[:,2]), np.max(self.xytBatch[:,2])))

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
    self.xytDataBc = np.zeros((self.nTotalFrames, maxCells, self.N_IN_VAR), dtype=float)
    self.uvpDataBc = np.zeros((self.nTotalFrames, maxCells, self.N_OUT_VAR), dtype=float)
    # Binary array that tracks valid entries (the ending entries might just be unused 0s)
    self.validDataBc = np.zeros((self.nTotalFrames, maxCells), dtype=float)

    # Zero mean for pos range (use -1 to account for 0 in center)
    if zeroMean:
      self.xytDataBc[:,:1] -= (self.size-1) / 2
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
        self.validDataBc[f, s:s+eWalls] = 1

      print(self.xytDataBc[f, :, :].shape)

      print('min, max uvpDataBc[0]: [{}, {}]'.format(np.min(self.uvpDataBc[f,:,0]), np.max(self.uvpDataBc[f,:,0])))
      print('min, max uvpDataBc[1]: [{}, {}]'.format(np.min(self.uvpDataBc[f,:,1]), np.max(self.uvpDataBc[f,:,1])))


  def generate_train_valid_batch(self, begin, end, worldSize, imageSize, fps, \
                                 V, L, T, batchSize=64, shuffle=True):
    generatorType = 'training' if begin == 0 else 'validation'
    UT.print_info('\nGenerating {} sample {} batches'.format(batchSize, generatorType))

    # Arrays to store iface + collocation + wall point batch
    uv    = np.zeros((batchSize, self.dim), dtype=float)
    xy    = np.zeros((batchSize, self.dim), dtype=float)
    t     = np.zeros((batchSize, 1),        dtype=float)
    id    = np.zeros((batchSize, 1),        dtype=float)
    phi   = np.zeros((batchSize, 1),        dtype=float)
    c     = np.zeros((batchSize, 1),        dtype=float)
    state = np.zeros((batchSize, self.latentDim), dtype=float)

    s = begin
    while True:
      if s + batchSize > end:
        s = begin
      e = s + batchSize

      # Fill batch arrays
      uv[:,:]    = self.uvpBatch[s:e, :self.dim]
      phi[:,0]   = self.uvpBatch[s:e, 2]
      c[:,0]     = self.uvpBatch[s:e, 3]
      xy[:,:]    = self.xytBatch[s:e, :self.dim]
      t[:,0]     = self.xytBatch[s:e, 2]
      id[:,0]    = self.idBatch[s:e, 0]

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
        c    = c[perm]

      idxT = np.concatenate(t.astype(int))
      xyDataBc = self.xytDataBc[idxT, :, :2]
      uvDataBc = self.uvpDataBc[idxT, :, :2]
      validDataBc = self.validDataBc[idxT, :]

      # Use t to select entry from state array
      state[:,:] = self.stateVel[idxT, :]

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

      dummy = vel
      yield [pos, time, state, vel, xyDataBc, uvDataBc, validDataBc, id, phi, c], dummy


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
        x = np.linspace(sizeY-1-bWRight-1, bWLeft+1, num=numCells, dtype=float)
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
          curUvp = self.rawData[f,yi,xi,:]
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

    self.uvpDomain = np.zeros((np.sum(self.nWalls), self.N_OUT_VAR))
    self.xytDomain = np.zeros((np.sum(self.nWalls), self.N_IN_VAR))
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
      U, V = self.rawData[f,:,:,0], self.rawData[f,:,:,1]
      mag = np.sqrt(np.square(U) + np.square(V))

      # Thicken interface and generate levelset if needed (flownet only)
      assert self.source in [UT.SRC_FLOWNET, UT.SRC_FLASHX], "Unknown fluid points source"
      curPhi = self.rawData[f,:,:,2]
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
          UT.save_plot(self.rawData[f,:,:,2], '{}/plots'.format(self.sourceName), 'flownet_phi_plt', frame, size=self.size, cmin=np.min(curPhi), cmax=np.max(curPhi), cmap='jet')

      # Add bubble border positions to xyBubble list
      nzIndices = np.nonzero(intersection)
      bubbleBorderIndices = list(zip(nzIndices[1], nzIndices[0])) # np.nonzero returns data in [rows, columns], ie [y,x]
      xyFrameLst.append(bubbleBorderIndices)

      # Add bubble border velocities to bc list, same order as indices list
      bcLst = []
      for idx in zip(nzIndices[0], nzIndices[1]):
        i, j = idx[0], idx[1]
        curUvp = self.rawData[f,i,j,:]
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
        curUvp = self.rawData[f,i,j,:]
        uvpLst.append(curUvp)
      uvpFluidFrameLst.append(uvpLst)

      # Keep track of number of bubble border cells and fluid cells per frame
      self.nBubble.append(len(bubbleBorderIndices))
      self.nFluid.append(len(fluidIndices))

    if UT.PRINT_DEBUG:
      print('Total number of bubble points: {}'.format(np.sum(self.nBubble)))
      print('Total number of fluid points: {}'.format(np.sum(self.nFluid)))

    if UT.VID_DEBUG:
      UT.save_video(subdir='extract', name='flags_extract', imgDir='../img/extract/', fps=15)
      UT.save_video(subdir='extract', name='bubblePts_extract_i{:02d}'.format(self.interface), imgDir='../img/extract/', fps=15)
      UT.save_video(subdir='extract', name='magnitude_extract', imgDir='../img/extract/', fps=15)

    # Allocate arrays for preprocessed data ...
    self.uvpBubble   = np.zeros((np.sum(self.nBubble), self.N_OUT_VAR), dtype=float)
    self.xytBubble = np.zeros((np.sum(self.nBubble), self.N_IN_VAR), dtype=float)
    self.uvpFluid = np.zeros((np.sum(self.nFluid), self.N_OUT_VAR), dtype=float)
    self.xytFluid = np.zeros((np.sum(self.nFluid), self.N_IN_VAR), dtype=float)

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


  def select_iface_points(self, useAll=False):
    UT.print_info('Selecting interface points')
    rng = np.random.default_rng(2022)

    # Use all points that are available
    if useAll:
      self.xytIface = copy.deepcopy(self.xytBubble)
      self.uvpIface = copy.deepcopy(self.uvpBubble)

      # Only use points within boundaries
      mask = self.get_wall_mask(self.xytIface)
      self.xytIface = self.xytIface[mask]
      self.uvpIface = self.uvpIface[mask]

      self.nIfacePnt = len(self.xytIface)
      print('Using {} interface points'.format(self.nIfacePnt))
      return

    # Else, use the exact number of points that was specified
    nIfacePntPerFrame = self.nIfacePnt // self.nTotalFrames
    # Update actual number of points
    self.nIfacePnt = nIfacePntPerFrame * self.nTotalFrames
    # Allocate arrays based on actual number of points
    self.xytIface = np.zeros((self.nIfacePnt, self.N_IN_VAR))
    self.uvpIface = np.zeros((self.nIfacePnt, self.N_OUT_VAR))

    print('Using {} interface points'.format(self.nIfacePnt))

    # Return early if no iface points requested
    if not self.nIfacePnt:
      return

    s = 0
    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      # Get all interface point coords and vels for the current frame
      xytIfaceFrame = self.get_xyt_bubble(f)
      uvpIfaceFrame = self.get_uvp_bubble(f)

      # Only use points within boundaries
      mask = self.get_wall_mask(xytIfaceFrame)
      xytIfaceFrameMasked = xytIfaceFrame[mask]
      uvpIfaceFrameMasked = uvpIfaceFrame[mask]

      # Insert random selection of interface point coords into interface point array
      if UT.PRINT_DEBUG:
        print('Taking {} interface points out of {} available'.format(nIfacePntPerFrame, xytIfaceFrame.shape[0]))
      indices = np.arange(0, xytIfaceFrameMasked.shape[0])
      randIndices = rng.choice(indices, size=nIfacePntPerFrame, replace=False)
      e = s + nIfacePntPerFrame
      self.xytIface[s:e, :] = xytIfaceFrameMasked[randIndices, :]
      self.uvpIface[s:e, :] = uvpIfaceFrameMasked[randIndices, :]
      if UT.IMG_DEBUG:
        UT.save_array(self.xytIface[s:e, :], '{}/interface'.format(self.sourceName), 'ifacepts_select', frame, self.size)
      s = e


  def _fill_select_arrays(self, xyt, uvp, nFrames, nPoints, pType):
    rng = np.random.default_rng(2022)
    s = 0
    for f in range(nFrames):
      frame = f + self.startFrame
      UT.print_progress(f, nFrames)

      # Get all fluid coords for the current frame
      xyFluidFrame = self.get_xyt_fluid(f)
      uvpFluidFrame = self.get_uvp_fluid(f)

      # Only use every other grid point as data point
      mask = np.logical_and(xyFluidFrame[:,0] % self.colRes == 0, xyFluidFrame[:,1] % self.colRes == 0)
      xyFluidFrameMasked = xyFluidFrame[mask]
      uvpFluidFrameMasked = uvpFluidFrame[mask]

      # Only use points within boundaries
      mask = self.get_wall_mask(xyFluidFrameMasked)
      xyFluidFrameMasked = xyFluidFrameMasked[mask]
      uvpFluidFrameMasked = uvpFluidFrameMasked[mask]

      # Insert random selection of fluid coords into data array
      if UT.PRINT_DEBUG:
        print('Taking {} {} points out of {} available'.format(nPoints, pType, xyFluidFrameMasked.shape[0]))
      indices = np.arange(0, xyFluidFrameMasked.shape[0])

      # Select collocation points adaptively based on scalar c (i.e. select more points with non-zero c value)
      if 0:
        indicesNonZero = np.nonzero(uvpFluidFrameMasked[:,3])
        indicesZero = np.nonzero(uvpFluidFrameMasked[:,3] == 0)
        fac = 0.85
        randIndices = rng.choice(indices[indicesNonZero], size=int(nPoints*fac), replace=False)
        randIndices = np.append(randIndices, rng.choice(indices[indicesZero], size=int(nPoints*(1.0-fac)), replace=False))
      else:
        randIndices = rng.choice(indices, size=nPoints, replace=False)

      e = s + nPoints
      xyt[s:e, :] = xyFluidFrameMasked[randIndices, :]
      uvp[s:e, :] = uvpFluidFrameMasked[randIndices, :]
      s = e

      if UT.IMG_DEBUG:
        UT.save_array(xyFluidFrameMasked[randIndices, :], '{}/{}'.format(self.sourceName, pType), '{}_pts_select'.format(pType), frame, self.size)


  def select_collocation_points(self, default=False):
    UT.print_info('Selecting collocation points')

    # No specific number of collocation points supplied in cmd-line args
    if default:
      self.nColPnt = self.nIfacePnt * 10

    nPntPerFrame = self.nColPnt // self.nTotalFrames
    # Update actual number of points
    self.nColPnt = nPntPerFrame * self.nTotalFrames
    # Allocate array based on actual number of points
    self.xytCol = np.zeros((self.nColPnt, self.N_IN_VAR))
    self.uvpCol = np.zeros((self.nColPnt, self.N_OUT_VAR))

    print('Using {} collocation points'.format(self.nColPnt))

    # Return early if no collocation points requested
    if not self.nColPnt:
      return

    self._fill_select_arrays(self.xytCol, self.uvpCol, nFrames=self.nTotalFrames, nPoints=nPntPerFrame, pType='collocation')


  def select_icond_points(self, default=False):
    UT.print_info('Selecting initial condition points')

    # No specific number of icond points supplied in cmd-line args
    if default:
      self.nIcondPnt = self.nIfacePnt * 2

    # Allocate array based on actual number of points
    self.xytIcond = np.zeros((self.nIcondPnt, self.N_IN_VAR))
    self.uvpIcond = np.zeros((self.nIcondPnt, self.N_OUT_VAR))

    print('Using {} icond points'.format(self.nIcondPnt))

    # Return early if no points requested
    if not self.nIcondPnt:
      return

    self._fill_select_arrays(self.xytIcond, self.uvpIcond, nFrames=1, nPoints=self.nIcondPnt, pType='icond')


  def select_data_points(self, default=False):
    UT.print_info('Selecting data points')

    # No specific number of collocation points supplied in cmd-line args
    if default:
      self.nDataPnt = self.nIfacePnt * 10

    nPntPerFrame = self.nDataPnt // self.nTotalFrames
    # Update actual number of points
    self.nDataPnt = nPntPerFrame * self.nTotalFrames
    # Allocate array based on actual number of points
    self.xytData = np.zeros((self.nDataPnt, self.N_IN_VAR))
    self.uvpData = np.zeros((self.nDataPnt, self.N_OUT_VAR))

    print('Using {} data points'.format(self.nDataPnt))

    # Return early if no collocation points requested
    if not self.nDataPnt:
      return

    self._fill_select_arrays(self.xytData, self.uvpData, nFrames=self.nTotalFrames, nPoints=nPntPerFrame, pType='data')


  def select_wall_points(self, useAll=False):
    UT.print_info('Selecting wall points')
    rng = np.random.default_rng(2022)

    # Use all points that are available
    if useAll:
      self.xytWalls = copy.deepcopy(self.xytDomain)
      self.uvpWalls = copy.deepcopy(self.uvpDomain)
      self.idWalls = copy.deepcopy(self.idDomain)
      self.nWallPnt = len(self.xytWalls)
      print('Using {} wall points'.format(self.nWallPnt))
      return

    # Else, use the exact number of points that was specified
    nWallsPntPerFrame = self.nWallPnt // self.nTotalFrames
    # Update actual number of points
    self.nWallPnt = nWallsPntPerFrame * self.nTotalFrames
    # Allocate arrays based on actual number of points
    self.xytWalls = np.zeros((self.nWallPnt, self.N_IN_VAR))
    self.uvpWalls = np.zeros((self.nWallPnt, self.N_OUT_VAR))
    self.idWalls = np.zeros((self.nWallPnt))

    print('Using {} wall points'.format(self.nWallPnt))

    # Return early if no points requested
    if not self.nWallPnt:
      return

    s = 0
    for f in range(self.nTotalFrames):
      frame = f + self.startFrame
      UT.print_progress(f, self.nTotalFrames)

      # Get all wall point coords and vels for the current frame
      xytWallsFrame = self.get_xyt_walls(f)
      uvpWallsFrame = self.get_uvp_walls(f)
      idWallsFrame = self.get_id_walls(f)

      # Insert random selection of wall point coords into wall point array
      if UT.PRINT_DEBUG:
        print('Taking {} wall points out of {} available'.format(nWallsPntPerFrame, xytWallsFrame.shape[0]))
      indices = np.arange(0, xytWallsFrame.shape[0])
      randIndices = rng.choice(indices, size=nWallsPntPerFrame, replace=False)
      e = s + nWallsPntPerFrame
      self.xytWalls[s:e, :] = xytWallsFrame[randIndices, :]
      self.uvpWalls[s:e, :] = uvpWallsFrame[randIndices, :]
      self.idWalls[s:e]    = idWallsFrame[randIndices]
      s = e

      if UT.IMG_DEBUG:
        UT.save_array(xytWallsFrame[randIndices, :], '{}/wall'.format(self.sourceName), 'wallpts_select', frame, self.size)


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

    dFile.create_dataset('uvpBubble', (nSampleBubble, self.N_OUT_VAR), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.uvpBubble)
    dFile.create_dataset('xytBubble', (nSampleBubble, self.N_IN_VAR), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xytBubble)
    dFile.create_dataset('uvpFluid', (nSampleFluid, self.N_OUT_VAR), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.uvpFluid)
    dFile.create_dataset('xytFluid', (nSampleFluid, self.N_IN_VAR), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.xytFluid)
    dFile.create_dataset('uvpDomain', (nSampleWalls, self.N_OUT_VAR), compression=comp_type,
                          compression_opts=comp_level, dtype='float64', chunks=True, data=self.uvpDomain)
    dFile.create_dataset('xytDomain', (nSampleWalls, self.N_IN_VAR), compression=comp_type,
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


  def restoreState(self, fname):
    if not os.path.exists(fname):
      sys.exit('File {} does not exist'.format(fname))

    self.stateVel = np.empty((self.nTotalFramesState, self.latentDim))
    dictRead = {'state_vel': self.stateVel}
    UT.read_array_hdf5(arrays=dictRead, fname=fname)

    print('Restored state from file {}'.format(fname))


  def get_num_icond_pts(self):
    return self.nIcondPnt


  def get_num_wall_pts(self):
    return self.nWallPnt


  def get_num_iface_pts(self):
    return self.nIfacePnt


  def get_num_col_pts(self):
    return self.nColPnt


  def get_num_data_pts(self):
    return self.nDataPnt


  def get_num_total_pts(self):
    return self.get_num_icond_pts() + self.get_num_wall_pts() + \
           self.get_num_iface_pts() + self.get_num_col_pts() + \
           self.get_num_data_pts()


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


  def get_id_walls(self, f):
    s = sum(self.nWalls[:f])
    e = s + self.nWalls[f]
    return self.idDomain[s:e]


  '''
  def get_xy_data(self, f):
    ptsPerFrame = (self.nIfacePnt // self.nTotalFrames)
    s = ptsPerFrame * f
    e = s + ptsPerFrame
    return self.xytIface[s:e, ...]


  def get_xy_col(self, f):
    ptsPerFrame = (self.nColPnt // self.nTotalFrames)
    s = ptsPerFrame * f
    e = s + ptsPerFrame
    return self.xytCol[s:e, ...]


  def get_xy_wall(self, f):
    ptsPerFrame = (self.nWallPnt // self.nTotalFrames)
    s = ptsPerFrame * f
    e = s + ptsPerFrame
    return self.xytDomain[s:e, ...]
  '''

