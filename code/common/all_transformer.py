import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smc

import config_res_half as config

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = 1

shape = (IMG_HT, IMG_WDT)

def _simple_transformer(depth_map, t_mat, k_mat, small_transform):

    batch_grids, transformed_depth_map, sparse_cloud  = _3D_meshgrid_batchwise_diff(IMG_HT, IMG_WDT, depth_map, batch_size, t_mat, k_mat, small_transform)

    x_all = tf.reshape(batch_grids[:,0], (IMG_HT, IMG_WDT))
    y_all = tf.reshape(batch_grids[:,1], (IMG_HT, IMG_WDT))

    return _bilinear_sampling(transformed_depth_map, x_all, y_all), sparse_cloud


def sparsify_cloud(S):

    """
    Cluster centers of point clouds used to sparsify cloud for Earth Mover's Distance. Using 4096 centroids
    """

    with tf.device('/cpu:0'):

        point_limit = 4096
        no_points = tf.shape(S)[0]
        no_partitions = no_points/tf.constant(point_limit, dtype=tf.int32)
        saved_points = tf.gather_nd(S, [tf.expand_dims(tf.range(0, no_partitions*point_limit), 1)])
        saved_points = tf.reshape(saved_points, [point_limit, no_partitions, 3])
        saved_points_sparse = tf.reduce_mean(saved_points, 1)

        return saved_points_sparse

def _3D_meshgrid_batchwise_diff(height, width, depth_img, num_batch, transformation_matrix, tf_K_mat, small_transform):

    """
    Creates 3d sampling meshgrid
    """

    x_index = tf.linspace(-1.0, 1.0, width)
    y_index = tf.linspace(-1.0, 1.0, height)
    z_index = tf.range(0, width*height)

    x_t, y_t = tf.meshgrid(x_index, y_index)

    # flatten
    x_t_flat = tf.reshape(x_t, [1,-1])
    y_t_flat = tf.reshape(y_t, [1,-1])
    ZZ = tf.reshape(depth_img, [-1])

    zeros_target = tf.zeros_like(ZZ)
    mask = tf.not_equal(ZZ, zeros_target)
    ones = tf.ones_like(x_t_flat)

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)
    sampling_grid_2d_sparse = tf.transpose(tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))
    ZZ_saved = tf.boolean_mask(ZZ, mask)
    ones_saved = tf.expand_dims(tf.ones_like(ZZ_saved), 0)

    projection_grid_3d = tf.matmul(tf.matrix_inverse(tf_K_mat), sampling_grid_2d_sparse*ZZ_saved)

    homog_points_3d = tf.concat([projection_grid_3d, ones_saved], 0)

    final_transformation_matrix = tf.matmul(transformation_matrix, small_transform)[:3,:]
    warped_sampling_grid = tf.matmul(final_transformation_matrix, homog_points_3d)

    points_2d = tf.matmul(tf_K_mat, warped_sampling_grid[:3,:])

    Z = points_2d[2,:]

    x_dash_pred = points_2d[0,:]
    y_dash_pred = points_2d[1,:]
    point_cloud = tf.stack([x_dash_pred, y_dash_pred, Z], 1)

    sparse_point_cloud = sparsify_cloud(point_cloud)

    x = tf.transpose(points_2d[0,:]/Z)
    y = tf.transpose(points_2d[1,:]/Z)

    mask_int = tf.cast(mask, 'int32')

    updated_indices = tf.expand_dims(tf.boolean_mask(mask_int*z_index, mask), 1)

    updated_Z = tf.scatter_nd(updated_indices, Z, tf.constant([width*height]))
    updated_x = tf.scatter_nd(updated_indices, x, tf.constant([width*height]))
    neg_ones = tf.ones_like(updated_x)*-1.0
    updated_x_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_x)

    updated_y = tf.scatter_nd(updated_indices, y, tf.constant([width*height]))
    updated_y_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_y)

    reprojected_grid = tf.stack([updated_x_fin, updated_y_fin], 1)

    transformed_depth = tf.reshape(updated_Z, (IMG_HT, IMG_WDT))

    return reprojected_grid, transformed_depth, sparse_point_cloud

def reverse_all(z):

    """Reversing from cantor function indices to correct indices"""

    z = tf.cast(z, 'float32')
    w = tf.floor((tf.sqrt(8.*z + 1.) - 1.)/2.0)
    t = (w**2 + w)/2.0
    y = tf.clip_by_value(tf.expand_dims(z - t, 1), 0.0, IMG_HT - 1)
    x = tf.clip_by_value(tf.expand_dims(w - y[:,0], 1), 0.0, IMG_WDT - 1)

    return tf.concat([y,x], 1)

def get_pixel_value(img, x, y):

    """Cantor pairing for removing non-unique updates and indices. At the time of implementation, unfixed issue with scatter_nd causes problems with int32 update values. Till resolution, implemented on cpu """

    with tf.device('/cpu:0'):
        indices = tf.stack([y, x], 2)
        indices = tf.reshape(indices, (IMG_HT*IMG_WDT, 2))
        values = tf.reshape(img, [-1])

        Y = indices[:,0]
        X = indices[:,1]
        Z = (X + Y)*(X + Y + 1)/2 + Y

        filtered, idx = tf.unique(tf.squeeze(Z))
        updated_values  = tf.unsorted_segment_max(values, idx, tf.shape(filtered)[0])

        # updated_indices = tf.map_fn(fn=lambda i: reverse(i), elems=filtered, dtype=tf.float32)
        updated_indices = reverse_all(filtered)
        updated_indices = tf.cast(updated_indices, 'int32')
        resolved_map = tf.scatter_nd(updated_indices, updated_values, img.shape)

        return resolved_map

def _bilinear_sampling(img, x_func, y_func):

    """
    Sampling from input image and performing bilinear interpolation
    """

    max_y = tf.constant(IMG_HT - 1, dtype=tf.int32)
    max_x = tf.constant(IMG_WDT - 1, dtype=tf.int32)

    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W/H/D]
    x = 0.5 * ((x_func + 1.0) * tf.cast(IMG_WDT - 1, 'float32'))
    y = 0.5 * ((y_func + 1.0) * tf.cast(IMG_HT - 1, 'float32'))

    x = tf.clip_by_value(x, 0.0, tf.cast(max_x, 'float32'))
    y = tf.clip_by_value(y, 0.0, tf.cast(max_y, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    # find Ia, Ib, Ic, Id

    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    loc = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return loc
