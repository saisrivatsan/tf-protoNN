# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
import copy
import os.path
from datetime import datetime

import numpy as np
import scipy.io as sio
import tensorflow as tf

from Net_2 import Net
from data import Generator

TOWER_NAME = 'tower'

def tower_loss(protonn, scope, x_batch, y_batch):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
        scope: unique prefix string identifying the tower, e.g. 'tower_0'
        x_batch: features. 2D tensor of shape [batch_size, D].
        y_batch: Labels. 2D tensor of shape [batch_size, L].
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    y_pred = protonn.forward(x_batch)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = protonn.loss(y_pred, y_batch)
    

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        print(l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss, y_pred


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        
    return average_grads


def hard_threshold(x, s):
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


def eval_precision_k(Y_pred_labels, Y_true):
    """
    Y_pred_labels: [num_instances, k]
    Y_true: [num_instances, L]
    """
    num_instances, k = Y_pred_labels.shape
    idx = np.tile(np.expand_dims(np.arange(num_instances),1),[1, k])
    p = np.array(np.mean(Y_true[idx, Y_pred_labels], 0)).flatten()
    p = np.cumsum(p)
    p = p / (1.0 + np.arange(k))
    return p

    
def train(config):
      
    train_gen = Generator(config, 'train')
        
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    
        protonn = Net(config)
        
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    
        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(config.learning_rate)

        # Get images and labels for CIFAR-10.
        x = tf.sparse_placeholder(tf.float32, shape=[None, config.D], name='x')
        y = tf.sparse_placeholder(tf.float32, shape=[None, config.L], name='y')
            
        X_batches = tf.sparse_split(sp_input = x, axis = 0, num_split = config.num_gpus)
        Y_batches = tf.sparse_split(sp_input = y, axis = 0, num_split = config.num_gpus)
    
        # Calculate the gradients for each model tower.
        tower_grads = []
        top_labels = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(config.num_gpus):        
                with tf.device('/gpu:%d' % i):               
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss, y_pred = tower_loss(protonn, scope, X_batches[i], Y_batches[i])
                        
                        _, top_labels_batch = tf.nn.top_k(y_pred, k = config.k, sorted = True)
                        top_labels.append(top_labels_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)

        # Stack all top_labels
        top_labels = tf.concat(top_labels, axis = 0)
        
        # Add summaries for trainable variables.
        for var in tf.trainable_variables():
            
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                        
            summaries.append(tf.summary.scalar('mean', mean))            
            summaries.append(tf.summary.scalar('stddev', stddev))
            summaries.append(tf.summary.scalar('max', tf.reduce_max(var)))
            summaries.append(tf.summary.scalar('min', tf.reduce_min(var)))
            summaries.append(tf.summary.histogram(var.op.name, var))


        # Group all updates to into a single train op.
        train_op = apply_gradient_op
        
        # Hard threshold op to ensure sparsity
        clip_W = protonn.W.assign(hard_threshold(protonn.W, config.sW))
        clip_B = protonn.B.assign(hard_threshold(protonn.B, config.sB))
        clip_Z = protonn.Z.assign(hard_threshold(protonn.Z, config.sZ))
        clip_op = tf.group(clip_W, clip_B, clip_Z)
        
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        sess.run(init)

        summary_writer = tf.summary.FileWriter(config.dir_name, sess.graph)
        
        
        # Restore model if not starting from 0
        step = config.train_restore_iter
        if step > 0:
            model_path = os.path.join(config.dir_name, 'model-' + str(iter))
            saver.restore(sess, model_path)
        else:
            if config.use_pre_init_fname is not None:
                print("Using pre-init")
                params = sio.loadmat(config.use_pre_init_fname)
                W0, B0, Z0 = params['W'], params['B'], params['Z']
                Z0 = np.asarray(Z0.todense())
                sess.run(protonn.W.assign(W0))
                sess.run(protonn.B.assign(B0))
                sess.run(protonn.Z.assign(Z0))
                
                
        while step  < config.max_iter:
        
            start_time = time.time()
            X, Y, Y_true = next(train_gen.gen_func)
            _, loss_value, TOP_labels = sess.run([apply_gradient_op, loss, top_labels], feed_dict = {x: X, y: Y})
            print(TOP_labels.shape)
            duration = time.time() - start_time
            
            step += 1
            
            if ((step % config.hard_threshold_iter) == 0) or (step == config.max_iter):
                sess.run(clip_op)

            if ((step % config.train_print_iter) == 0) or (step == config.max_iter):
                format_str = ('%s: step %d, loss = %.2f')
                p = eval_precision_k(TOP_labels, Y_true)
                print(p)
                print (format_str % (datetime.now(), step, loss_value))
                
            if ((step % config.val_print_iter) == 0) or (step == config.max_iter):
                pass

            if ((step % config.summaries_iter) == 0) or (step == config.max_iter) or (step == 1):
                summary_str = sess.run(summary_op, feed_dict = {x: X, y: Y})
                summary_writer.add_summary(summary_str, step)

            if ((step % config.save_iter) == 0) or (step == config.max_iter): 
                checkpoint_path = os.path.join(config.dir_name, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)