from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# Directory to load data from
__C.dir_name = "datasets/amazoncat"

# Directory to load param-initialization. Uses random
__C.use_pre_init = 'datasets/amazoncat/init_weights_200d.mat'


# Data-related, Must match with properties of data loaded from __C.dir_name
# N, Nt - train and test instances. D - Number of features. L - Number of Labels
__C.N = 1186239
__C.Nt = 306782
__C.D = 203882
__C.L = 13330

# Params-related, Must match with properties of data loaded from _use_pre_init_fname
# d - projection dimension, m - number of prototypes 
__C.d = 200
__C.m = 12922

__C.num_pts_per_cluster = 10

# Gamma 
__C.gamma = 2.5

# Input type
__C.is_sparse_input = True

# Sparsity of params
__C.sW = 0.8
__C.sB = 0.8
__C.sZ = 0.2

# Batch Size
__C.train_batch_size = 256
__C.test_batch_size = 256

# Train-Val split
__C.train_val_split_ratio = 0.9

__C.N_tr = int(__C.N * __C.train_val_split_ratio)
__C.N_v = __C.N - __C.N_tr
__C.train_num_batches = int(np.ceil(__C.N_tr/__C.train_batch_size))
__C.val_num_batches = int(np.ceil(__C.N_v/__C.train_batch_size))
__C.test_num_batches = int(np.ceil(__C.Nt/__C.test_batch_size))


# Number of epochs
__C.max_iter = 1000

# Learning Rate
__C.learning_rate = 1e-3

# Saver [ Set 1 to save every epoch, 2 to save every other epoch ... ]
__C.save_iter = 100
__C.max_to_keep = 5

# Restore training from Epoch
__C.train_restore_iter = 0

# Hard-threshold at iter
__C.HT_iter = 1

# Summaries to tensorboard [ Set 1 to summarize every epoch, 2 to save every other epoch ... ]
__C.summaries_iter = 1

# Print iter [ Set 1 to print at every iter, 2 to print at every other iter ... ]
__C.train_print_iter = 1

# Validation Epoch [ Set 1 to perform validation at every iter, 2 for validation every other iter ... ]
__C.val_print_iter = 10
__C.test_print_iter = 50

__C.k = 5
__C.train_seed = 0
__C.test_seed = 0


__C.wd = None
__C.initW = None
__C.initB = None
__C.initZ = None

__C.num_gpus = 1
__C.normalize = False
__C.summaries = False