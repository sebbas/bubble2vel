#!/usr/bin/env python3

import argparse
import sys
import bdataset as BD
import butils as UT

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--startFrame', type=int, default=0,
                    help='first frame to load from dataset')
parser.add_argument('-e', '--endFrame', type=int, default=399,
                    help='last frame to load from dataset')
parser.add_argument('-w', '--walls', type=int, nargs=4, default=[1,1,1,1],
                    help='domain boundary condition sampling width [left, top, right, bottom]')
parser.add_argument('-i', '--interface', type=int, default=1,
                    help='thickness of bubble liquid interface in pixels')
parser.add_argument('-src', '--source', type=int, default=0,
                    help='type of training data source, either 1: experiment or 2: simulation')

args = parser.parse_args()

assert args.startFrame <= args.endFrame, 'Start frame must be smaller/equal than/to end frame'

# Set source file location based on source type
fName = None
if args.source == UT.SRC_FLOWNET:
  fName = '../data/PB_30W_RGB/512/30W%06d.flo'
elif args.source == UT.SRC_FLASHX:
  #fName = '../data/PB_simulation/384/INS_Pool_Boiling_hdf5_plt_cnt_%04d'
  #fName = '../data/PB_simulation/SingleBubble/INS_Pool_Boiling_hdf5_plt_cnt_%04d'
  fName = '../../Multiphase-Simulations/simulation/PoolBoiling/SingleBubble/INS_Pool_Boiling_hdf5_plt_cnt_%04d'

assert fName is not None, 'Invalid training data source'

# Create dataset object,
dataSet = BD.BubbleDataSet(fName=fName, startFrame=args.startFrame, \
                           endFrame=args.endFrame, walls=args.walls, \
                           interface=args.interface, source=args.source)

# Load training / validation data from files
isLoaded, filePrefix = dataSet.load_data(args.source)
if not isLoaded: sys.exit()

# Extract points from dataset and store in h5
dataSet.extract_wall_points(useDataBc=False)
dataSet.extract_fluid_points(velEps=1.0)
dataSet.save(filePrefix=filePrefix)
dataSet.summary()

