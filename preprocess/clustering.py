from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import faiss
import numpy as np
from scipy.sparse import issparse

def train_kmeans(x, k, ngpu = 1):
    """ Runs PCA and return projection matrix W and embeddings Wx
    Args:
        x: numpy array, (N, d)
        k: number of clusters, scalar
        ngpu: number of GPUs to be used, scalar
    Returns:
        centroids: cluster centres, (k, d)
    """
    
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    #clus.verbose = True
    clus.niter = 20

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpu)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return centroids.reshape(k, d)