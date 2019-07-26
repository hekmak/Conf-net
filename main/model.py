#!/usr/bin/env python
# Code implemented by H.H Aug 2018

import sys
import os
sys.path.append('..')
from depth_completion import DepthCompletion
from conv_deconv import normal_conv, normal_deconv
import tensorflow as tf
  
class CompletionWithError(DepthCompletion):
    def __init__(self):
        super(CompletionWithError, self).__init__()

        # for multi GPU
        ##self.parameters.optimizer = tf.contrib.estimator.TowerOptimizer(self.parameters.optimizer)

    def res_block(self,x1,filters=64, kernel_size=(3, 3), padding='same', strides = 1):
            x = normal_conv(x1 , filters=filters, kernel_size=kernel_size, padding='same', strides = 2)
            x = tf.nn.relu(x)
            x = normal_conv(x, filters=filters, kernel_size=kernel_size, padding='same', strides = 1)
            x_r = normal_conv(x1 , filters=filters, kernel_size=kernel_size, padding='same', strides = 2)
            return tf.nn.relu(x_r+x)
        
    def forground(self,features,pool_size=(2,2),strides=(1,1)):
        inv_x = tf.where(tf.equal(features, 0),
                         tf.zeros_like(features),
                         tf.ones_like(features)*65535-features)

        x_for = tf.layers.max_pooling2d(inv_x,pool_size=pool_size,strides=strides,padding='same')
        x_for = tf.where(tf.equal(x_for, 0),
                             tf.zeros_like(x_for),
                             tf.ones_like(x_for)*65535-x_for)
        return x_for
    
    def background(self,features,pool_size=(2,2),strides=(1,1)):
        x_back = tf.layers.max_pooling2d(features,pool_size=pool_size,strides=strides,padding='same')
        return x_back
    
    def hourglass(self,features,f=32):
        x2 = features
        #  convs
        x2_1 = self.res_block(x2 , filters=f, kernel_size=(3, 3), padding='same', strides = 2)
        x2_2 = self.res_block(x2_1 , filters=f*2, kernel_size=(3, 3), padding='same', strides = 2)
        x2_3 = self.res_block(x2_2 , filters=f*4, kernel_size=(3, 3), padding='same', strides = 2)
        x2_4 = self.res_block(x2_3 , filters=f*8, kernel_size=(3, 3), padding='same', strides = 2)
        #x2_4 = tf.nn.dropout(x2_4,0.5)

        #  transpose convs
        xd  = normal_deconv(x2_4, filters=f*8, strides=2, kernel_size=(3, 3), padding='same') 
        xd = tf.nn.relu(xd)
        xd  = tf.concat([x2_3,xd],3)

        xd  = normal_deconv(xd, filters=f*4, strides=2, kernel_size=(3, 3), padding='same') 
        xd = tf.nn.relu(xd)
        xd  = tf.concat([x2_2,xd],3)

        xd  = normal_deconv(xd, filters=f*2, strides=2, kernel_size=(3, 3), padding='same') 
        xd = tf.nn.relu(xd)
        xd  = tf.concat([x2_1,xd],3)

        xd  = normal_deconv(xd, filters=f, strides=2, kernel_size=(3, 3), padding='same')
        xd = tf.nn.relu(xd)
        xd  = tf.concat([x2,xd],3)
         
        return xd
    
    def pred_var(self,features,n_errors=1):
        xd = features
        xd_mu0 = normal_conv(xd, filters=16, kernel_size=(3,3), padding='same')
        xd_mu0 = tf.nn.relu(xd_mu0)
        xd_mu0 = normal_conv(xd_mu0, filters=16, kernel_size=(3,3), padding='same')
        xd_mu0_ = normal_conv(xd, filters=16, kernel_size=(3,3), padding='same')
        xd_mu0 = tf.nn.relu(xd_mu0_+xd_mu0)
        xd_mu = normal_conv(xd_mu0, filters=1, strides=1, kernel_size=(1, 1), padding='same')
        
        if n_errors>0:
            xd_va0 = normal_conv(xd, filters=16, kernel_size=(3,3), padding='same')
            xd_va0 = tf.nn.relu(xd_va0)
            xd_va0 = normal_conv(xd_va0, filters=16, kernel_size=(3,3), padding='same')
            xd_va0_ = normal_conv(xd, filters=16, kernel_size=(3,3), padding='same')
            xd_va0 = tf.nn.relu(xd_va0_+xd_va0)
            xd_va = normal_conv(xd_va0, filters=n_errors, strides=1, kernel_size=(1, 1), padding='same')

            return xd_mu,xd_va
        else:
            return xd_mu,1
    
    def network(self, tf_input, **kwargs):

        reuse = kwargs.get('reuse', False)
        tf_input = tf.identity(tf_input,name='model_input')
        with tf.variable_scope('DepthCompletion', reuse=reuse):
            l = tf_input
  
            bkground = self.background(l,pool_size=(15,15))
            frground = self.forground(l,pool_size=(15,15))
        
            x2  = tf.concat([frground,bkground,l],3)
            x2 = self.hourglass(x2, f=32)
            pred, var = self.pred_var(x2,n_errors=1)
            
            return tf.identity(pred,name='pred_depth'),tf.identity(var,name='pred_error')

if __name__ == '__main__':
    exp = CompletionWithError()
    exp.run()



