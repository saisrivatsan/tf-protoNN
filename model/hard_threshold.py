# --------------------------------------------------------
# Tensorflow ProtoNN for Multi-label learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Sai Srivatsa Ravindranath
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def HT(x, s):
    """
    Hard Threshold Function, output vector has sparsity s
    Args:
        x: weight tensor of any shape
        s: sparsity
    Returns:
        out: weight tensor with sparsity s
    """
    th = tf.contrib.distributions.percentile(tf.abs(x), (1-s)*100.0, interpolation='higher')
    cond = tf.less(abs(x), th)
    return tf.where(cond, tf.zeros(tf.shape(x)), x)