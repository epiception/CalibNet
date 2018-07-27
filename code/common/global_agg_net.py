import numpy as np
import tensorflow as tf
import scipy.misc as smc
import matplotlib.pyplot as plt

import config_res as config
from cnn_utils_res import *

# import resnet_rgb_model as model
# import resnet_depth_model as model_depth

import resnet_rgb_v1 as model
import resnet_depth_v1 as model_depth

batch_size = config.net_params['batch_size']
current_epoch = config.net_params['load_epoch']

def End_Net_weights_init():

    """
    Initialize Aggregation Network Weights and Summaries
    """

    W_ext1 = weight_variable([3,3,768,384], "_8")
    W_ext2 = weight_variable([3,3,384,192], "_9")
    W_ext3 = weight_variable([1,2,192,96], "_10")

    W_ext4_rot = weight_variable([1,1,96,96], "_11")
    W_fc_rot = weight_variable_fc([2880,3], "_12")

    W_ext4_tr = weight_variable([1,1,96,96], "_13")
    W_fc_tr = weight_variable_fc([2880,3], "_14")

    end_weights = [W_ext1, W_ext2, W_ext3, W_ext4_rot, W_fc_rot, W_ext4_tr, W_fc_tr]

    weight_summaries = []

    for weight_index in range(len(end_weights)):
        with tf.name_scope('weight_%d'%weight_index):
            weight_summaries += variable_summaries(end_weights[weight_index])

    return end_weights, weight_summaries

def End_Net(input_x, phase_depth, keep_prob):

    """
    Define Aggregation Network
    """

    weights, summaries = End_Net_weights_init()

    print input_x.shape
    
    layer8 = conv2d_batchnorm_init(input_x, weights[0], name="conv_9", phase= phase_depth, stride=[1,2,2,1])
    layer9 = conv2d_batchnorm_init(layer8, weights[1], name="conv_10", phase= phase_depth, stride=[1,2,2,1])
    layer10 = conv2d_batchnorm_init(layer9, weights[2], name="conv_11", phase= phase_depth, stride=[1,2,2,1])

    print layer8.shape
    print layer9.shape
    print layer10.shape

    layer11_rot = conv2d_batchnorm_init(layer10, weights[3], name="conv_12", phase= phase_depth, stride=[1,1,1,1])
    layer11_m_rot = tf.reshape(layer11_rot, [batch_size, 2880])
    layer11_drop_rot = tf.nn.dropout(layer11_m_rot, keep_prob)
    layer11_vec_rot = (tf.matmul(layer11_drop_rot, weights[4]))

    layer11_tr = conv2d_batchnorm_init(layer10, weights[5], name="conv_13", phase= phase_depth, stride=[1,1,1,1])
    layer11_m_tr = tf.reshape(layer11_tr, [batch_size, 2880])
    layer11_drop_tr = tf.nn.dropout(layer11_m_tr, keep_prob)
    layer11_vec_tr = (tf.matmul(layer11_drop_tr, weights[6]))

    output_vectors = tf.concat([layer11_vec_tr, layer11_vec_rot], 1)
    return output_vectors, summaries


def End_Net_Out(X1, phase_rgb, pooled_input2, phase, keep_prob):

    """
    Computation Graph
    """

    RGB_Net_obj = model.ResNet(X1, phase_rgb)
    Depth_Net_obj = model_depth.DepthNet(pooled_input2, phase)


    output_rgb = RGB_Net_obj.Net()
    output_depth = Depth_Net_obj.Net()
    # with tf.variable_scope('ResNet'):
    #     with tf.device('/device:GPU:0'):
    #         output_rgb = RGB_Net_obj.Net()
    # with tf.variable_scope('DepthNet'):
    #     with tf.device('/device:GPU:1'):
    #         output_depth = Depth_Net_obj.Net()

    layer_next = tf.concat([output_rgb, output_depth], 3)

    end_net_op = End_Net(layer_next, phase, keep_prob)

    return end_net_op
