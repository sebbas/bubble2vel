#!/usr/bin/env python3

import argparse
import bdataset as BD
import butils as Util

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', default='../data/PB_30W_RGB/512/30W%06d.flo',
                    help='the file(s) containing flow velocity training data')
parser.add_argument('-s', '--startFrame', type=int, default=0,
                    help='first frame to load from dataset')
parser.add_argument('-e', '--endFrame', type=int, default=399,
                    help='last frame to load from dataset')
parser.add_argument('-w', '--walls', type=int, nargs=4, default=[1,1,1,1],
                    help='domain walls, [top, right, bottom, left]')
parser.add_argument('-i', '--interface', type=int, default=1,
                    help='thickness of bubble liquid interface in pixels')

args = parser.parse_args()

assert args.file.endswith('.flo')

# Create dataset object,
dataSet = BD.BubbleDataSet(fName=args.file, startFrame=args.startFrame, \
                           endFrame=args.endFrame, walls=args.walls, \
                           interface=args.interface)

# Extract points from images and save in arrays from dataset
if not dataSet.load_data(normalize=False):
  sys.exit()
dataSet.extract_wall_points()
dataSet.extract_fluid_points(velEps=1.0)
dataSet.save(walls=args.walls)
dataSet.summary()

