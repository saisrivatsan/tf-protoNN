# tf-protoNN
The repository contains **ProtoNN** (a KNN based algorithm) implemented in Tensorflow for large-scale multi-label learning. The repository also has a script to run the training on multiple GPUs


## Extreme multi-label (XML) algorithms

Unlike multi-class or binary classification, extreme multi-label (XML) algorithms tag data points with a subset of labels (rather than
just a single label), usually from an extremely large label-set. XML problems usually deal with a large number of labels (10^3 - 10^6labels) and a large number of dimensions and training points. 

For datasets, check: [XML-repository](http://manikvarma.org/downloads/XC/XMLRepository.html)

## Required packages

1. Tensorflow and
2. [FAISS](https://github.com/facebookresearch/faiss)
3. Numpy
4. Scipy
5. Easydict
 
## Usage

Check the [ipython notebook](https://github.com/saisrivatsan/tf-protoNN/blob/master/run_eurlex_with_preprocessing.ipynb) to run the code on Eurlex-4k dataset. To change the parameters, modify the [config file](https://github.com/saisrivatsan/tf-protoNN/blob/master/cfgs/config_eurlex_with_preprocessing.py).

To run on a new dataset:

1. Create a new folder with the directory name. Place two separate files train\_data.mat and test\_data.mat in that directory. Note that each of these files must have two variables: X with shape:  (num instances, num features) and Y with shape (num instances, num labels)

2. Create a config file in [cfgs folder](https://github.com/saisrivatsan/tf-protoNN/tree/master/cfgs) with the required parameters.
 
3. Modify eurlex_train.py -> train.py (import the correct config file) and run
