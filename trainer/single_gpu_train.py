# --------------------------------------------------------
# Tensorflow ProtoNN for Multi-label learning
# Licensed under The MIT License [see LICENSE for details]
# Written by Sai Srivatsa Ravindranath
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import scipy.io as sio
import tensorflow as tf
from datetime import datetime

from model import *



class Trainer:
    def __init__(self, config):
        
        # Add config as attribute
        self.config = config
              
        # Build ProtoNN Graph
        self.graph = protoNN.Graph(config)
        
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
        log_fname = os.path.join(self.config.out_dir_name, 'train' + log_suffix + '.txt')
        self.logger = logger.get_logger(log_fname)
                
        # Input placeholders
        inp_shape = [None, self.config.D]
        if self.config.is_sparse_input:
            x = tf.sparse_placeholder(tf.float32, shape = inp_shape, name = 'x')
        else:
            x = tf.placeholder(tf.float32, shape = inp_shape, name='x')
        
        # Output placeholders
        out_shape = [None, self.config.L]
        y_true = tf.sparse_placeholder(tf.float32, shape = out_shape, name = 'y_true')
        
        
        # Forward pass: Compute predictions and top-k labels
        D, y_pred = self.graph.forward(x)
        _, y_pred_labels = tf.nn.top_k(y_pred, k = self.config.k, sorted = True)
        
        # Define Loss        
        #loss = loss_utils.mse_loss(y_pred, y_true)
        loss = eval_utils.huber_loss(y_pred, y_true)
        tf.summary.scalar('loss', loss)
        
        reg_losses = tf.get_collection('reg_losses')
        if len(reg_losses) > 0:
            reg_loss_mean = tf.reduce_mean(reg_losses)
            tf.summary.scalar('reg_loss', reg_loss_mean)
            total_loss = loss + reg_loss_mean
        else:
            total_loss = loss
            
        tf.summary.scalar('total_loss', total_loss)
               
        # Optimizer       
        learning_rate = tf.Variable(self.config.learning_rate, trainable=False)       
        opt = tf.train.AdamOptimizer(learning_rate)
        train_step = opt.minimize(loss)
        
        # Hard Threshold Ops
        HT_op = self.get_HT_op()
                
        #Summary
        merged = tf.summary.merge_all()
        
        # Saver
        saver = tf.train.Saver(max_to_keep = self.config.max_to_keep)
        
        # Start Session and init variables
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(self.config.out_dir_name, sess.graph)
        
        # Data generators
        data = data_loader.TrainGenerator(self.config)
        data_t = data_loader.TestGenerator(self.config)
        
        # Restore model if not starting from 0
        iter = self.config.train_restore_iter
        if iter > 0:
            model_path = os.path.join(config.out_dir_name, 'model-' + str(iter))
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
                fmt_str = "TRAIN-BATCH Iter = %d, t = %.2f, Loss = %.2f, Prec@1: %.2f, Prec@3: %.2f, Prec@5: %.2f"%vals
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
                fmt_str = "VAL-ALL Iter = %d, t = %.2f, Loss = %.2f, Prec@1: %.2f, Prec@3: %.2f, Prec@5: %.2f"%vals 
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
                fmt_str = "TEST-ALL Iter = %d, t = %.2f, Loss = %.2f, Prec@1: %.2f, Prec@3: %.2f, Prec@5: %.2f"%vals
                self.logger.info(fmt_str)


                    
                
                    
                    
                
                
                

    