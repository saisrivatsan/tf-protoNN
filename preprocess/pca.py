from __future__ import absolute_import
from __future__ import division
from __future__ import print_functio

import faiss
from scipy.sparse import issparse

def train_pca(X, d):
    """ Runs PCA and return projection matrix W and embeddings Wx
    Args:
        X: numpy array or scipy matrix, (N, D)
        d: projection dimension
    Returns:
        W: projection matrix, (D, d)
        Wx: embeddings, (N, d)
    """
        
    
    if issparse(X):
        x = X.todense().astype(np.float32)
    else:
        x = X
        
    D = x.shape[1]
    pca = faiss.PCAMatrix(D, d)
    pca.verbose = True
    pca.have_bias = False
        
    pca.train(x)
    assert pca.is_trained
    
    W = faiss.vector_float_to_array(pca.A)
    Wx = pca.apply_py(x)
    
    return W.reshape(d, D).T, Wx   