# H.H Aug 2018

import tensorflow as tf

#BN_EPSILON = 0.001
_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-3

def batch_normalization_layer(inputs,training):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def normal_conv(tensor,
                filters=32,
                kernel_size=3,
                strides=1,
                l2_scale=0.0,#5e-5,#0.0, 
                padding='same',
                dilation_rate =(1,1),
                kernel_initializer=tf.contrib.layers.xavier_initializer(), non_lin = tf.nn.relu, bn=False, training=True):

    features = tf.layers.conv2d(tensor, 
                                filters=filters, 
                                kernel_size=kernel_size, 
                                kernel_initializer=kernel_initializer,
                                strides=(strides, strides), 
                                trainable=True, 
                                use_bias=True, 
                                padding=padding,
                                dilation_rate = dilation_rate,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))
    if bn == False:
        return features
    else:
        bn_layer = batch_normalization_layer(features, training)
        return bn_layer

def normal_deconv(tensor,
                filters=32, trainable = True,
                kernel_size=3,
                strides=1,
                l2_scale=0.0,#5e-5,#0.0, 
                padding='same', 
                kernel_initializer=tf.contrib.layers.xavier_initializer(), bn=False, training=True):

    features = tf.layers.conv2d_transpose(
                                inputs=tensor,
                                filters=filters,
                                kernel_size= kernel_size,
                                strides=(strides, strides),
                                padding=padding,
                                data_format='channels_last',
                                activation=None,
                                use_bias=True,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=tf.zeros_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale),
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                bias_constraint=None,
                                trainable=trainable,
                                name=None,
                                reuse=None)

    if bn == False:
        return features
    else:
        bn_layer = batch_normalization_layer(features, training)
        return bn_layer
