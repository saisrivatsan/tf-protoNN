from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import faiss
import numpy as np
from scipy.sparse import csr_matrix

def get_prototypes(Y, Wx, B, num_pts_per_cluster):
    """ Gets Z from embeddings and centroids 
    Args:
        Y: sparse label matrix, (N, L)
        Wx: embeddings, (N, d)
        B: centroids, (d, m)
        num_pts_per_cluster: scalar
    Retuns:
        Z0: sparse matrix, (m, L)
    """
    L = Y.shape[-1]
    d = Wx.shape[-1] 
    m = B.shape[-1]
    index = faiss.IndexFlatL2(d)
    index.add(Wx)
    D, I = index.search (B.T, num_pts_per_cluster)
    
    Z0 = np.zeros((m, L))
    for idx,(dd,ii) in enumerate(zip(D,I)): 
        z = Y[ii].T.dot(dd)
        Z0[idx] = z/z.max()
    Z0 = csr_matrix(Z0)
    return Z0