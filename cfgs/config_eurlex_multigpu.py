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
__C.dir_name = "datasets/eurlex"

# Directory to load param-initialization. Uses random
__C.use_pre_init = 'datasets/eurlex/init_params_faiss.mat'

# Directory for output
__C.out_dir_name = "experiments/eurlex"
if not os.path.exists(__C.out_dir_name):
    os.mkdir(__C.out_dir_name)

# Data-related, Must match with properties of data loaded from __C.dir_name
# N, Nt - train and test instances. D - Number of features. L - Number of Labels
__C.N = 15539
__C.Nt = 3809
__C.D = 5000
__C.L = 3993
__C.is_sparse_input = True
__C.normalize = False

# Params-related, Must match with properties of data loaded from _use_pre_init_fname
# d - projection dimension, m - number of prototypes 
__C.d = 250
__C.m = 1000
__C.normalize_D = False
__C.normalize_output = True
__C.gamma = 2.5
__C.sW = 0.9
__C.sB = 0.9
__C.sZ = 0.5

# Preprocess related
__C.num_pts_per_cluster = 15

# Training
__C.train_val_split_ratio = 0.8
__C.train_batch_size = 2048
__C.test_batch_size = 256
__C.learning_rate = 1e-3
__C.wd = 1e-1
__C.summaries = False
__C.train_restore_iter = 0
__C.max_iter = 1000
__C.HT_iter = 1

# Print and Summary
__C.train_print_iter = 100
__C.val_print_iter = 100
__C.test_print_iter = __C.max_iter
__C.summaries_iter = 1

# Saver
__C.save_iter = 100

# Don't change these params
# k - prec@1..k, seeds and max models to save to disk
__C.k = 5
__C.train_seed = 0
__C.test_seed = 1
__C.max_to_keep = 2

# Other dependent params
__C.N_tr = int(__C.N * __C.train_val_split_ratio)
__C.N_v = __C.N - __C.N_tr
__C.train_num_batches = int(np.ceil(__C.N_tr/__C.train_batch_size))
__C.val_num_batches = int(np.ceil(__C.N_v/__C.train_batch_size))
__C.test_num_batches = int(np.ceil(__C.Nt/__C.test_batch_size))
__C.initW = None
__C.initB = None
__C.initZ = None
