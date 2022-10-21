#!/usr/bin/env python3

import argparse
import bdataset as BD
import butils as Util

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--nBcDomainPoint', type=int, default=5000,
                    help='number of boundary points in training')
parser.add_argument('-c', '--nColPoint', type=int, default=100000,
                    help='number of collocation points in training')
parser.add_argument('-d', '--nDataPoint', type=int, default=100000,
                    help='number of data points in training')

parser.add_argument('-f', '--file', default='../data/PB_30W_RGB/512/30W%06d.flo',
                    help='the file(s) containing flow velocity training data')
parser.add_argument('-tf', '--totalFrames', type=int, default=400,
                    help='number of frames to load from dataset')

parser.add_argument('-w', '--walls', type=int, nargs=4, default=[0,0,1,0],
                    help='domain walls, [top, right, bottom, left]')

args = parser.parse_args()

assert args.file.endswith('.flo')

# Create dataset object,
dataSet = BD.BubbleDataSet(fName=args.file, totalframes=args.totalFrames, \
                           bcdomainPoints=args.nBcDomainPoint, \
                           colPoints=args.nColPoint, \
                           dataPoints=args.nDataPoint)

# Extract points from images and save in arrays from dataset
if not dataSet.load_data(normalize=False):
  sys.exit()
dataSet.extract_wall_points(walls=args.walls)
dataSet.extract_fluid_points(velEps=1.0)
dataSet.extract_data_points()
dataSet.extract_collocation_points()
dataSet.save(walls=args.walls)
dataSet.summary()

