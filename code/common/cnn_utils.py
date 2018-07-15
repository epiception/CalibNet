import numpy as np
import tensorflow as tf

def weight_variable(shape, to_train, name):
    '''
    Helper function to create a weight variable initialized with
    a normal distribution (truncated to two standard devs)
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''

    W = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=to_train)
    return W

def weight_variable_fc(shape, to_train, name):
    '''
    Helper function to create a weight variable initialized with
    a normal distribution (truncated to two standard devs)
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''

    W = 0.01*tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=to_train)
    return W

def bias_variable(shape, to_train, name):

    B = tf.Variable(tf.constant(0.0, shape= shape, dtype=tf.float32), name=name, trainable=to_train)
    return B

def init_weights(W, to_train, name):
    init = tf.constant_initializer(W)
    weight = tf.get_variable(name, shape=W.shape, dtype=tf.float32, initializer= init, trainable = to_train)
    return weight

def init_bias(B, to_train, str):
    init = tf.constant_initializer(B)
    bias = tf.get_variable(name, shape=B.shape, dtype=tf.float32, initializer= init, trainable = to_train)
    return bias

def conv2d_init(x, shape, name, bias = None, stride = [1,1,1,1], padding = "SAME", to_train = True):

    with tf.name_scope("conv"):

        W = weight_variable(shape, to_train, name)

        if(bias == None):
            return tf.nn.conv2d(x, W, strides = stride, padding = padding)
        else:
            bias = bias_variable(shape, to_train, "bias_" + name)
            return tf.nn.conv2d(x, W, strides = stride, padding = padding) + bias

def batchnorm_init(x, phase, name, decay = 0.9):
    
    with tf.name_scope("bn_"+name):
        
        return tf.contrib.layers.batch_norm(x, is_training = phase, updates_collections = None)


def conv2d(x, W, name, to_train, bias = None, stride = [1,1,1,1], padding = "SAME"):
    
    with tf.name_scope("conv"):
    
        W_init = init_weights(W, to_train, name)

        if(bias == None):
            return tf.nn.conv2d(x, W_init, strides = stride, padding = padding)
        else:
            bias_init = init_bias(bias, to_train, "bias_" + name)
            return tf.nn.conv2d(x, W_init, strides = stride, padding = padding) + bias_init


def batchnorm(x, phase, name, beta_r, gamma_r, mean_r, variance_r, decay = 0.9):
    
    with tf.name_scope("bn_"+name):

        beta = tf.constant_initializer(beta_r)
        gamma = tf.constant_initializer(gamma_r)
        moving_mean = tf.constant_initializer(mean_r)
        moving_variance = tf.constant_initializer(variance_r)


        return tf.contrib.layers.batch_norm(x, param_initializers={'beta': beta, 'gamma': gamma, 'moving_mean': moving_mean,'moving_variance': moving_variance,}, is_training = phase, updates_collections = None, scale = True, decay = decay)    



def max_pool(x, name):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name=name)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    sum_mean = tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    sum_stddev = tf.summary.scalar('stddev', stddev)
    #tf.summary.scalar('max', tf.reduce_max(var))
    #tf.summary.scalar('min', tf.reduce_min(var))
    sum_hist = tf.summary.histogram('histogram', var)
    return [sum_mean, sum_hist, sum_stddev]
