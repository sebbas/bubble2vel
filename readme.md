# bubble2vel

Tensorflow implementation of a PINN that predicts fluid velocities around
bubbles whose optical flow was obtained with
[`flownet2-pytorch`](https://github.com/NVIDIA/flownet2-pytorch).

<figure>
<p float="left">
<img src="img/docs/raw_0000.png" width="300" />
<img src="img/docs/magnitude_0000.png" width="300" />
</p>
<figcaption align = "left">
  <b>Fig 1: </b>Left: Original data. Right: Preprocessed input data for model
</figcaption>
</figure>

## Usage

Data generation, training and prediction in one go. Navigate to `src/` and execute:

`python3 bmain.py -c 50000 -e 2 -tf 50 -p 10 -bs 64 -ev -ci -f ../data/bdata_512_56389.h5`

This extracts data for 50 frames from the `.h5` dataset in `../data`. Then it
trains the model with all available fluid/vapor data points and 50000 uniformly
and randomly sampled collocation points placed in fluid areas (1000 per frame).
Finally it makes a prediction of the fluid velocity field of the first 10 frames.

<figure>
<p float="left">
<img width="300" alt="Original velocities" src="img/docs/origvels_0000.png"/>
  <img width="300" alt="Predicted velocities" src="img/docs/predvels_stream_0000.png"/>
</p>
<figcaption align = "left">
  <b>Fig 2: </b>Left: Original vapor/fluid boundary velocities
    (`matplotlib quiver`). Right: Predicted fluid flow (`matplotlib streamplot`)
</figcaption>
</figure>

### Generating a dataset

With option `-ci` the script skips preprocessing and loads data directly from a
`.h5` file. See above command.

To generate a custom `.h5` dataset from `.flo` files, simply omit option `-ci`
and set the file option to point to the flownet files:

`python3 bmain.py -c 50000 -e 2 -tf 50 -p 10 -bs 64 -ev -f ../data/PB_30W_RGB/512/30W%06d.flo`

A new `bdata_<sizeX>_<numDataPoints>.h5` dataset will be saved in `data/`.

## Command line arguments

* `-c 10000` use 10000 collocation points during training, 10 per frame when using together with `-tf 10`
* `-e 20` specifies the number of epochs used during training
* `-tf 200` the number of frames to extract data from
* `-p 10` the number of frames to predict using the model after training
* `-ci` load training data from `.h5` instead of processing files
from the `data/` directory.
* `-pd` predict velocities of entire domain and do not skip bubble areas
* `-bs 64` use a 64 sample batchsize for training
* `-l 50 50 50` hidden layers of the densely connected PINN. Here with 3 hidden layers and 50 nodes per layer (use more than this!)
* `-f ../data/PB_30W_RGB/512/30W%06d.flo`  specifies the location and format of the flownet input files
* `-ev` export videos of images that were generated 

## Project structure
 
* `src/` includes PINN and util functions that load and prepare external data
* `data/` location of optical flow data for training (`.flo` format). Also
stores compressed `.h5` data files here.

A full run of the model creates the following additional directories:

* `img/` contains plots of input data, predictions as well as loss plots
* `vid/` visualization of `img/` contents with `ffmpeg`


