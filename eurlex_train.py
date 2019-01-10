from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import scipy.io as sio

from model import *
from preprocess import *

from trainer.single_gpu_train import Trainer
import cfgs.config_eurlex_with_preprocessing as config

# Set number of GPU's to use
config.cfg.num_gpus = 1

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