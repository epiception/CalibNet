import numpy as np
import tensorflow as tf

def weight_variable(shape, str):
    '''
    Helper function to create a weight variable initialized with
    a normal distribution (truncated to two standard devs)
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''

    W = tf.get_variable("weight" + str, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return W

def weight_variable_fc(shape, str):
    '''
    Helper function to create a weight variable initialized with
    a normal distribution (truncated to two standard devs)
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''

    W = 0.01*tf.get_variable("weight" + str, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return W

def bias_variable(shape, str):

    B = tf.Variable(tf.constant(0.0, shape= shape, dtype=tf.float32), name="bias" + str)
    return B


def init_weights(W, str, to_train):
    init = tf.constant_initializer(W)
    weight = tf.get_variable('weight'+str, shape=W.shape, dtype=tf.float32, initializer= init, trainable = to_train)
    return weight

def init_bias(B, layerno, to_train):
    init = tf.constant_initializer(B)
    bias = tf.get_variable('bias_%d'%layerno, shape=B.shape, dtype=tf.float32, initializer= init, trainable = to_train)
    return bias

def conv2d_batchnorm(x, W, name, phase, beta_r, gamma_r, mean_r, variance_r, stride = [1,1,1,1], relu = True):

    beta = tf.constant_initializer(beta_r)
    gamma = tf.constant_initializer(gamma_r)
    moving_mean = tf.constant_initializer(mean_r)
    moving_variance = tf.constant_initializer(variance_r)

    with tf.name_scope(name):
        mid1 =  tf.nn.conv2d(x, W, strides = stride, padding = "SAME")
        with tf.name_scope('batch_norm'):
            mid2 = tf.contrib.layers.batch_norm(mid1, param_initializers={'beta': beta, 'gamma': gamma, 'moving_mean': moving_mean,'moving_variance': moving_variance,}, is_training = phase, updates_collections = None, scale = True, decay = 0.9)
            if(relu == True):
                return tf.nn.relu(mid2)
            else:
                return mid2

def conv2d_batchnorm_init(x, W, name, phase, stride = [1,1,1,1], relu = True):

    with tf.name_scope(name):
        mid1 =  tf.nn.conv2d(x, W, strides = stride, padding ="SAME")
        with tf.name_scope('batch_norm'):
            mid2 = tf.contrib.layers.batch_norm(mid1, is_training = phase, updates_collections = None)
            if(relu == True):
                return tf.nn.relu(mid2)
            else:
                return mid2

def conv2d_init(x, W, name, phase, stride, padding):
    with tf.name_scope(name):
        mid1 =  tf.nn.conv2d(x, W, strides = stride, padding = padding)
        mid2 = tf.nn.relu(mid1)

        return mid2


def conv2d_bias_init(x, W, b, name):
    with tf.name_scope(name):
        mid1 =  tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME") + b
        mid2 = tf.nn.relu(mid1)

        return mid2

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
