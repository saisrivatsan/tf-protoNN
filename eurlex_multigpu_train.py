
# --------------------------------------------------------
# Tensorflow ProtoNN for Multi-label learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Sai Srivatsa Ravindranath
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import scipy.io as sio

from model import *
from preprocess import *

from trainer.multi_gpu_train import Trainer
import cfgs.config_eurlex_multigpu as config

# Set number of GPU's to use
config.cfg.num_gpus = 2

# Check if batch-size is a multiple of num-gpus
assert( config.cfg.train_batch_size % config.cfg.num_gpus == 0), "train batch_size must be a multiple of num_gpus"
assert( config.cfg.test_batch_size % config.cfg.num_gpus == 0), "test batch_size must be a multiple of num_gpus"

# We load the file saved by the single-gpu version. 
# If the single gpu version hasn't been run, set this to True
run_preprocess = False

path = os.path.join(config.cfg.dir_name, "train_data")
X, Y = data_loader.data_loader(path, config.cfg)
print("Data-Stats:")
print("Num instances = ", X.shape[0])
print("Feature dimensionality = ", X.shape[-1])
print("Label dimensionality = ", Y.shape[-1])
print("Mean pts per label = ", Y.nnz/Y.shape[1])
print("Mean labels per pt = ", Y.nnz/Y.shape[0])

print("Param-stats:")
print("Projection-Dim: %d"%config.cfg.d)
print("Num Projections: %d"%config.cfg.m)

tic = time.time()
D = X.shape[-1]
d = config.cfg.d
m = config.cfg.m
W0, Wx = pca.train_pca(X, d)
B0 = clustering.train_kmeans(Wx, m, ngpu = 1).T
Z0 = prototypes.get_prototypes(Y, Wx, B0, num_pts_per_cluster=config.cfg.num_pts_per_cluster)
t_elapsed = time.time() - tic;
print("Time-taken for pre-training: %.4f"%(t_elapsed))

path = os.path.join(config.cfg.dir_name, "init_params_faiss.mat")
sio.savemat(path, {'W':W0, 'B':B0, 'Z':Z0})

m = Trainer(config.cfg)
m.train()
print("Logged at %s"%(m.log_fname))
