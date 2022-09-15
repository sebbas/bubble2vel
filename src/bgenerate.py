#!/usr/bin/env python3

import argparse
import bdataset as BD
import butils as Util

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--nBcPoint', type=int, default=250,
                    help='number of boundary points in training')
parser.add_argument('-c', '--nColPoint', type=int, default=2000,
                    help='number of collocation points in training')

parser.add_argument('-f', '--file', default='../data/PB_30W_RGB/512/30W%06d.flo',
                    help='the file(s) containing flow velocity training data')
parser.add_argument('-tf', '--totalFrames', type=int, default=400,
                    help='number of frames to load from dataset')

args = parser.parse_args()

assert args.file.endswith('.flo')

# Create dataset object,
dataSet = BD.BubbleDataSet(fName=args.file, totalframes=args.totalFrames,
                           bcpoints=args.nBcPoint, colpoints=args.nColPoint)

# Extract points from images and save in arrays from dataset
if not dataSet.load_data():
  sys.exit()
dataSet.extract_wall_points(walls=[0,0,1,0])
dataSet.extract_data_points(velEps=0.1)
dataSet.extract_collocation_points()
dataSet.save()
dataSet.summary()

