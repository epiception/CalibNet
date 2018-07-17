"""
tf.data Loader
"""

import tensorflow as tf

import numpy as np
import glob, os, argparse
import scipy.misc as smc
from tqdm import tqdm
import matplotlib.pyplot as plt

import config_res as config

plt.ion()

# ResNet mean and std_dev
img_mean = tf.constant(np.array([0.485, 0.456, 0.406]), dtype = tf.float32)
img_std = tf.constant(np.array([0.229, 0.224, 0.225]), dtype = tf.float32)

def load_single(path_depth_source, path_depth_target, path_color_source, transform, mask):
    
    # same steps as nw_loader_color
    # but applied as tensor operations directly

    source_img = tf.read_file(path_depth_source)
    source_img_decoded = tf.cast(tf.image.decode_png(source_img, channels=1), 'float32')[:,:,0]*mask[:,:,0]
    source_img_decoded = (source_img_decoded - 40.0)/40.0

    target_img = tf.read_file(path_depth_target)
    target_img_decoded = tf.cast(tf.image.decode_png(target_img, channels=1), 'float32')[:,:,0]*mask[:,:,0]
    target_img_decoded = (target_img_decoded - 40.0)/40.0

    source_img_color = tf.read_file(path_color_source)
    source_img_color_decoded = tf.cast(tf.image.decode_png(source_img_color, channels=3), 'float32')*mask
    source_img_color_decoded = source_img_color_decoded/255.0
    source_img_color_decoded = (source_img_color_decoded - img_mean)/img_std

    transform_out = tf.matrix_inverse(tf.reshape(transform, (4,4)))

    return source_img_decoded[0], target_img_decoded[0], source_img_color_decoded, transform_out


total = config.net_params['total_frames']
total_train = config.net_params['total_frames_train']
total_validation = config.net_params['total_frames_validation']
partition_limit = config.net_params['partition_limit']

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = config.net_params['batch_size']

dataset = np.loadtxt("./parsed_set.txt", dtype = str)

# Creating a 3 channel mask to pad zeros to incoming images and depth maps (Needed for spatial transformer, done in earlier loader as well)
mask = tf.ones((IMG_HT - 10, IMG_WDT - 10), dtype = tf.float32)
padding = tf.constant([[5,5], [5,5]])
mask = tf.tile(tf.expand_dims(tf.pad(mask, padding, "CONSTANT"), 2), [1, 1, 3])

print mask.shape
print mask.dtype

#new partition limit, cause smaller txt size
partition_limit = 100


# Think of it as a series of steps to create tf.data.Dataset 

# structure of data (in this case list of paths or transform matrices): 3 path lists, 1 float32 array of transformation matrices
tensor_paths = (tf.constant(list(dataset[:,0])), tf.constant(list(dataset[:,1])), tf.constant(list(dataset[:,2])), tf.constant(dataset[:,4:], dtype = tf.float32))

# Lots of API, but core idea is, that this is all lambda map functions
# from_tensor_slices takes all data having same first dimension (so it can be easily converted to batches)
# subsequently, all steps

tr_data = (tf.data.Dataset.from_tensor_slices(tensor_paths).map(lambda x,y,z,w: load_single(x, y, z, w, mask), num_parallel_calls = 12) # parallel load_single function called
                                                           .prefetch(partition_limit) # buffer
                                                           .shuffle(partition_limit, reshuffle_each_iteration=True) # shuffle the buffer
                                                           .batch(batch_size)) # make batches

# Iterator can be thought of as a tf_op that needs to be started every session
iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)

next_element = iterator.get_next()# fetch next batch

# initialize the operation
iteration_op = iterator.make_initializer(tr_data)

with tf.Session() as sess:
    
    for epoch in range(5):
        
        # run iterator initialization tf_op
        sess.run(iteration_op)    
        
        for iter in range(partition_limit/batch_size):
            try:
                out = sess.run(next_element)

                # to display images

                # plt.imshow(np.hstack((out[0][0,:,:,0], out[1][0,:,:,0])))
                # plt.pause(0.1)
                
                plt.imshow(out[2][0])
                plt.pause(0.2)

                print (out[3][0])
            
            except tf.errors.OutOfRangeError:
                # Handle training number errors easily
                print("End of training dataset.")
                
