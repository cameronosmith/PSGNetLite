# PSGNet Lite

This is a reimplementation of the core principles from the
[PSGNet](https://arxiv.org/abs/2006.12373) architecture in PyTorch. You can find
their official TensorFlow implementation 
[here](https://github.com/neuroailab/PSGNets). 

### Features missing/TODO that make this implementation PSGNet*Lite*:

* Using [Residual Dense Network](https://arxiv.org/abs/1802.08797) as a
  multi-scale convolutional feature extractor to map each pixel to a feature 
  vector instead of the authors' [ConvRNN](https://arxiv.org/abs/1807.00053), as
  the ConvRNN currently has no PyTorch implementation.
* Computing and aggregating geometric information: The PSGNet aggregates
  geometric information (such as explicitly computing statistics of the four
  quadrants of child nodes and their 1D boundaries), whereas this implementation
  does not yet.
* Rendering: Only constant-texture rendering is implemented thus far (not
  quadratic texture rendering or quadratic shape rendering). These will be
  implemented soon, but the main intention of this project is to extend PSGNet's
  with a more expressive decoder (will be conducting such experiments in a
  separate branch)
* Instead of computing nearest neighbors to perform label propagation with, this
  implementation uses an initial grid adjacency matrix and aggregates edges when
  pooling nodes.
* Only the static RGB architecture is implemented but will implement movie
  inputs and the motion based affinity principles. 

### High-Level structure: 
* Affinities.py implements the affinity principle modules.
* LabelProp.py implements sparse-matrix multiplication label propagation.
* PSGNetLite.py implements the main model that reconstructs an image.
* VAE.py a simple VAE module.
* RDN.py an external file that implements a Residual Dense Network.
* Train.py a training script. Currently this project is still in the testing
  phase so the training script is hackish but eventually will transition to
  using a config file.

### Data:
Currently testing with DeepMind's Objects Room dataset, but you can substitute
with any data loader. You can download the dataset from their [Google Cloud
Storage](https://console.cloud.google.com/storage/browser/multi-object-datasets)
with 'wget https://storage.googleapis.com/multi-object-datasets/objects_room/objects_room_train.tfrecords'.
