# --------------------------------------------------------
# Tensorflow ProtoNN for Multi-label learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Sai Srivatsa Ravindranath
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def mse_loss(y_pred, y_true):
    """
    Computes MSE Loss
    Args:
        y_pred: (N, L)
        y_true: (N, L)
    Returns:
        loss: (1,)
    """
    error = tf.sparse_add(y_true, -y_pred)
    loss = tf.reduce_mean(tf.reduce_sum(error**2, -1))
    return loss

def huber_loss(y_pred, y_true, d = 1.0):
    """
    Computes Huber Loss
    Args:
        y_pred: (N, L)
        y_true: (N, L)
    Returns:
        loss: (1,)
    """   

    error = tf.abs(tf.sparse_add(y_true, -y_pred))
    squared_loss = 0.5 * (error**2)
    linear_loss  = d * (error - 0.5 * d) 
    cond  = tf.less_equal(error, d)
    loss = tf.reduce_mean(tf.reduce_sum(tf.where(cond, squared_loss, linear_loss)))
    tf.add_to_collection('losses', loss)
    return loss
    
def eval_precision_k(Y_pred_labels, Y_true):
    """
    Computes precision @ 1...k
    Args:
        Y_pred_labels: (N, k)
        Y_true: (N, L)
    Returns:
        p: (K,)
    """
    num_instances, k = Y_pred_labels.shape
    idx = np.tile(np.expand_dims(np.arange(num_instances),1),[1, k])
    p = np.array(np.mean(Y_true[idx, Y_pred_labels], 0)).flatten()
    p = np.cumsum(p)
    p = p / (1.0 + np.arange(k))
    return p
