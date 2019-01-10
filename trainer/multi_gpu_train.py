# --------------------------------------------------------
# Tensorflow ProtoNN for Multi-label learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Sai Srivatsa Ravindranath
# --------------------------------------------------------

# Multi-GPU train code based on <>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time
import logging
import numpy as np
import scipy.io as sio
import tensorflow as tf
from datetime import datetime

from model import *



TOWER_NAME = 'tower'

def tower_loss(graph, scope, x_batch, y_batch):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
        scope: unique prefix string identifying the tower, e.g. 'tower_0'
        x_batch: features. 2D tensor of shape [batch_size, D].
        y_batch: Labels. 2D tensor of shape [batch_size, L].
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    D, y_pred = graph.forward(x_batch)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = eval_utils.huber_loss(y_pred, y_batch)
    
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


class Trainer:
    
    def __init__(self, config):
        
        # Add config as attribute
        self.config = config
              
        # Build ProtoNN Graph
        self.graph = protoNN.Graph(self.config)
        
    def get_HT_op(self):
        clip_W = self.graph.W.assign(hard_threshold.HT(self.graph.W, self.config.sW))
        clip_B = self.graph.B.assign(hard_threshold.HT(self.graph.B, self.config.sB))
        clip_Z = self.graph.Z.assign(hard_threshold.HT(self.graph.Z, self.config.sZ))
        HT_op = tf.group(clip_W, clip_B, clip_Z)
        return HT_op
    
    
    def train(self):
                
        # Set Seeds
        np.random.seed(self.config.train_seed)

        # Set log_fname
        log_suffix = datetime.now().strftime("%d_%m_%Y_%H_%M")
        self.log_fname = os.path.join(self.config.out_dir_name, 'train' + log_suffix + '.txt')
        self.logger = logger.get_logger(self.log_fname)
        
        with tf.device('/cpu:0'):
        
            # Create an optimizer and a global_step variable
            opt = tf.train.AdamOptimizer(self.config.learning_rate)
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            # Input placeholders
            inp_shape = [None, self.config.D]
            if self.config.is_sparse_input:
                x = tf.sparse_placeholder(tf.float32, shape = inp_shape, name = 'x')
            else:
                x = tf.placeholder(tf.float32, shape = inp_shape, name='x')

            # Output placeholders
            out_shape = [None, self.config.L]
            y_true = tf.sparse_placeholder(tf.float32, shape = out_shape, name = 'y_true')

            if self.config.is_sparse_input:
                x_batches = tf.sparse_split(sp_input = x, axis = 0, num_split = self.config.num_gpus)
            else:
                x_batches = ""

            y_batches = tf.sparse_split(sp_input = y_true, axis = 0, num_split = self.config.num_gpus)

            # Calculate the gradients for each model tower.
            tower_grads = []
            y_pred_labels = []
            loss = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.config.num_gpus):        
                    with tf.device('/gpu:%d' % i):               
                        with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            loss_batch, y_pred_batch = tower_loss(self.graph, scope, x_batches[i], y_batches[i])

                            _, y_pred_labels_batch = tf.nn.top_k(y_pred_batch, k = self.config.k, sorted = True)
                            y_pred_labels.append(y_pred_labels_batch)
                            
                            reg_losses = tf.get_collection('reg_losses')
                            if len(reg_losses) > 0:
                                reg_loss_mean = tf.reduce_mean(reg_losses)/self.config.num_gpus
                                loss_batch += reg_loss_mean
                            
                            loss.append(loss_batch)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = opt.compute_gradients(loss_batch)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = average_gradients(tower_grads)

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
            
            # Group all updates to into a single train op.
            train_step = apply_gradient_op

            # Stack all top_labels
            y_pred_labels = tf.concat(y_pred_labels, axis = 0)
            loss =  tf.add_n(loss)

            # Hard Threshold Ops
            HT_op = self.get_HT_op()

            #Summary
            merged = tf.summary.merge_all()
        
            init = tf.global_variables_initializer()
            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
            
            sess.run(init)
                
            # Saver
            saver = tf.train.Saver(max_to_keep = self.config.max_to_keep)
            
            # Data generators
            data = data_loader.TrainGenerator(self.config)
            data_t = data_loader.TestGenerator(self.config)
        
        
            # Restore model if not starting from 0
            iter = self.config.train_restore_iter
            if iter > 0:
                model_path = os.path.join(self.config.out_dir_name, 'model-' + str(iter))
                saver.restore(sess, model_path)
            else:            
                # If starting from iter-0, check if pre-init is needed
                if self.config.use_pre_init is not None:               
                    params = sio.loadmat(self.config.use_pre_init)
                    W0, B0, Z0 = params['W'], params['B'], params['Z']
                    Z0 = np.asarray(Z0.todense()) 
                    sess.run(self.graph.W.assign(W0))
                    sess.run(self.graph.B.assign(B0))
                    sess.run(self.graph.Z.assign(Z0))
                    
                    
            time_elapsed_train = 0.0        
            while iter < (self.config.max_iter):

                # Get feed_dict val
                X, Y, Y_true = next(data.train_gen)

                tic = time.time()    

                # Train Step
                sess.run(train_step, feed_dict = {x: X, y_true: Y}) 
                iter += 1

                # HT Step
                if (iter % self.config.HT_iter) == 0:
                    sess.run(HT_op)

                toc = time.time()
                time_elapsed_train += (toc - tic)

                # Save Model
                if ((iter % self.config.save_iter) == 0) or (iter == self.config.max_iter):              
                    saver.save(sess, os.path.join(self.config.out_dir_name,'model'), global_step = iter) 

                # Print Train Stats
                if (iter % self.config.train_print_iter) == 0:                
                    # Train Set Stats
                    Y_pred_labels, Loss = sess.run([y_pred_labels, loss], feed_dict = {x: X, y_true: Y})
                    P = eval_utils.eval_precision_k(Y_pred_labels, Y_true)
                    vals = (iter, time_elapsed_train, Loss, P[0], P[2], P[4])
                    fmt_str = "TRAIN-BATCH Iter = %d, t = %.2f, Loss = %.2f, Prec@1: %.4f, Prec@3: %.4f, Prec@5: %.4f"%vals
                    self.logger.info(fmt_str)

                # Print Validation Stats
                if (iter % self.config.val_print_iter) == 0:                
                    #Validation Set Stats
                    Loss_tot = 0.0
                    P_tot = np.zeros(self.config.k)
                    tic = time.time()               
                    for _ in range(self.config.val_num_batches):
                        X, Y, Y_true = next(data.val_gen)
                        Y_pred_labels, Loss = sess.run([y_pred_labels, loss], feed_dict = {x: X, y_true: Y})
                        P = eval_utils.eval_precision_k(Y_pred_labels, Y_true)                    
                        Loss_tot += (Loss * Y_true.shape[0])
                        P_tot += (P * Y_true.shape[0])
                    toc = time.time()
                    time_elapsed_val = (toc - tic)    
                    P_tot = P_tot/self.config.N_v
                    Loss_tot = Loss_tot/self.config.N_v
                    vals = (iter, time_elapsed_val, Loss_tot, P_tot[0], P_tot[2], P_tot[4])
                    fmt_str = "VAL-ALL Iter = %d, t = %.2f, Loss = %.2f, Prec@1: %.4f, Prec@3: %.4f, Prec@5: %.4f"%vals 
                    self.logger.info(fmt_str)

                # Print Test Stats
                if (iter % self.config.test_print_iter) == 0:                
                    #Validation Set Stats
                    Loss_tot = 0.0
                    P_tot = np.zeros(self.config.k)
                    tic = time.time()               
                    for _ in range(self.config.test_num_batches):
                        X, Y, Y_true = next(data_t.test_gen)
                        Y_pred_labels, Loss = sess.run([y_pred_labels, loss], feed_dict = {x: X, y_true: Y})
                        P = eval_utils.eval_precision_k(Y_pred_labels, Y_true)                    
                        Loss_tot += (Loss * Y_true.shape[0])
                        P_tot += (P * Y_true.shape[0])
                    toc = time.time()
                    time_elapsed_test = (toc - tic)    
                    P_tot = P_tot/self.config.Nt
                    Loss_tot = Loss_tot/self.config.Nt
                    vals = (iter, time_elapsed_test, Loss_tot, P_tot[0], P_tot[2], P_tot[4])
                    fmt_str = "TEST-ALL Iter = %d, t = %.2f, Loss = %.2f, Prec@1: %.4f, Prec@3: %.4f, Prec@5: %.4f"%vals
                    self.logger.info(fmt_str)