import sys
sys.path.append('/home/ffy/ffyPackage')
import os
import time

import tensorflow as tf
import numpy as np
import cv2

from tfNet import tf_net as net
import dataWash as dw


def densenet_fc(inp, _shape, training, add_b=False):
    """In densenet, fc mean to BN-ReLU-Conv"""
    bn_inp = net.batch_normalization(inp, training)
    act_bn_inp = net.actv_net(bn_inp)
    fc_act_bn_inp = net.dense_net(act_bn_inp, _shape, add_b=add_b)
    return fc_act_bn_inp


def densenet_conv(inp, _filter, strides, training):
    """In densenet, conv mean to BN-ReLU-Conv"""
    bn_inp = net.batch_normalization(inp, training)
    act_bn_inp = net.actv_net(bn_inp)
    conv_act_bn_inp = net.conv_net(act_bn_inp, _filter, strides)
    return conv_act_bn_inp


def bottleneck_layer(inp, inchannel, outchannel, strides, training):
    """bottleneck: [conv 1*1, conv 3*3]
        we let each 1*1 convolution pruduce 4k feature-maps 
    """
    inp_11 = densenet_conv(inp, (1, 1, inchannel, 4 * outchannel), (1, 1, 1, 1), training)
    inp_11_33 = densenet_conv(inp_11, (3, 3, 4 * outchannel, outchannel), strides, training)
    return inp_11_33


def dense_block(inp, bottleneck_num, inchannel, growth_rate_k, training):
    """Dense Block: [conv 1*1, conv 3*3] * bottleneck_num
        dense_block will concat feature map
        dense_block's conv's strides is 1
        
        For every bottleneck, it output feature map which have size of k. k is the growth rate   
    """
    precede_fmap = inp
    for _ in range(bottleneck_num):
        #inchannel = tf.shape(precede_fmap)[-1]
        fmap = bottleneck_layer(precede_fmap, inchannel, growth_rate_k, (1, 1, 1, 1), training)
        inchannel += growth_rate_k
        precede_fmap = tf.concat([precede_fmap, fmap], -1)
    return precede_fmap, inchannel


def dense_net(data_x, training, bottleneck_nums=(6, 12, 24, 16), growth_rate_k=12):
    """DenseNet
        bottleneck_nums = (6, 12, 64, 48) -> DenseNet-264
    """
    inp = data_x
    training = training

    with tf.variable_scope('DenseNet'):
        bottleneck_num_1, bottleneck_num_2, bottleneck_num_3, bottleneck_num_4 = bottleneck_nums
        logits = densenet_conv(inp, (7, 7, 3, 2 * growth_rate_k), (1, 2, 2, 1), training)
        logits = net.max_pool(logits, (1, 3, 3, 1), (1, 2, 2, 1))

        logits, inchannel = dense_block(logits, bottleneck_num_1, 2 * growth_rate_k, growth_rate_k, training)
        logits = densenet_conv(logits, (1, 1, inchannel, 4 * growth_rate_k), (1, 1, 1, 1), training)
        logits = net.avg_pool(logits, (1, 2, 2, 1), (1, 2, 2, 1))

        logits, inchannel = dense_block(logits, bottleneck_num_2, 4 * growth_rate_k, growth_rate_k, training)
        logits = densenet_conv(logits, (1, 1, inchannel, 4 * growth_rate_k), (1, 1, 1, 1), training)
        logits = net.avg_pool(logits, (1, 2, 2, 1), (1, 2, 2, 1))

        logits, inchannel = dense_block(logits, bottleneck_num_3, 4 * growth_rate_k, growth_rate_k, training)
        logits = densenet_conv(logits, (1, 1, inchannel, 4 * growth_rate_k), (1, 1, 1, 1), training)
        logits = net.avg_pool(logits, (1, 2, 2, 1), (1, 2, 2, 1))

        logits, inchannel = dense_block(logits, bottleneck_num_4, 4 * growth_rate_k, growth_rate_k, training)
    
    return logits, inchannel