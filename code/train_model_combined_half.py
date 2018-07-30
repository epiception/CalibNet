import numpy as np
import tensorflow as tf
import scipy.misc as smc

import config_res_half as config

from common.cnn_utils_res import *
from common import all_transformer as at3
from common import global_agg_net_half as global_agg_net
from common.Lie_functions import exponential_map_single

import nw_loader_color_half as ldr
import model_utils


_BETA_CONST = 1.0
_ALPHA_CONST = 1.0
IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = config.net_params['batch_size']
learning_rate = config.net_params['learning_rate']
n_epochs = config.net_params['epochs']
current_epoch = config.net_params['load_epoch']

tf.reset_default_graph()

X1 = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 3), name = "X1")
X2 = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 1), name = "X2")
depth_maps_target = tf.placeholder(tf.float32, shape = (batch_size, IMG_HT, IMG_WDT, 1), name = "depth_maps_target")
expected_transforms = tf.placeholder(tf.float32, shape = (batch_size, 4, 4), name = "expected_transforms")

phase = tf.placeholder(tf.bool, [], name = "phase")
phase_rgb = tf.placeholder(tf.bool, [], name = "phase_rgb")
keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

fx = config.camera_params['fx']
fy = config.camera_params['fy']
cx = config.camera_params['cx']
cy = config.camera_params['cy']

fx_scaled = 2*(fx)/np.float32(IMG_WDT)              # focal length x scaled for -1 to 1 range
fy_scaled = 2*(fy)/np.float32(IMG_HT)               # focal length y scaled for -1 to 1 range
cx_scaled = -1 + 2*(cx - 1.0)/np.float32(IMG_WDT)   # optical center x scaled for -1 to 1 range
cy_scaled = -1 + 2*(cy - 1.0)/np.float32(IMG_HT)    # optical center y scaled for -1 to 1 range

K_mat_scaled = np.array([[fx_scaled,  0.0, cx_scaled],
                         [0.0, fy_scaled,  cy_scaled],
                         [0.0, 0.0, 1.0]], dtype = np.float32)

K_final = tf.constant(K_mat_scaled, dtype = tf.float32)
small_transform = tf.constant(config.camera_params['cam_transform_02_inv'], dtype = tf.float32)


X2_pooled = tf.nn.max_pool(X2, ksize=[1,1,2,1], strides=[1,1,1,1], padding="SAME")
depth_maps_target_pooled = tf.nn.max_pool(depth_maps_target, ksize=[1,1,2,1], strides=[1,1,1,1], padding="SAME")

output_vectors, weight_summaries = global_agg_net.End_Net_Out(X1, phase_rgb, X2_pooled, phase, keep_prob)

# se(3) -> SE(3) for the whole batch
predicted_transforms = tf.map_fn(lambda x:exponential_map_single(output_vectors[x]), elems=tf.range(0, batch_size, 1), dtype=tf.float32)

# transforms depth maps by the predicted transformation
depth_maps_predicted, cloud_pred = tf.map_fn(lambda x:at3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, predicted_transforms[x], K_final, small_transform), elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32))

# transforms depth maps by the expected transformation
depth_maps_expected, cloud_exp = tf.map_fn(lambda x:at3._simple_transformer(X2_pooled[x,:,:,0]*40.0 + 40.0, expected_transforms[x], K_final, small_transform), elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32))

# photometric loss between predicted and expected transformation
photometric_loss = tf.nn.l2_loss(tf.subtract((depth_maps_expected[:,10:-10,10:-10] - 40.0)/40.0, (depth_maps_predicted[:,10:-10,10:-10] - 40.0)/40.0))

# earth mover's distance between point clouds
cloud_loss = model_utils.get_emd_loss(cloud_pred, cloud_exp)

# final loss term
predicted_loss_train = _ALPHA_CONST*photometric_loss + _BETA_CONST*cloud_loss

tf.add_to_collection('losses1', predicted_loss_train)
loss1 = tf.add_n(tf.get_collection('losses1'))

train_step = tf.train.AdamOptimizer(learning_rate = config.net_params['learning_rate'],
                                    beta1 = config.net_params['beta1']).minimize(predicted_loss_train)

predicted_loss_validation = tf.nn.l2_loss(tf.subtract((depth_maps_expected[:,10:-10,10:-10] - 40.0)/40.0, (depth_maps_predicted[:,10:-10,10:-10] - 40.0)/40.0))

cloud_loss_validation = model_utils.get_emd_loss(cloud_pred, cloud_exp)

training_summary_1 = tf.summary.scalar('cloud_loss', _BETA_CONST*cloud_loss)
training_summary_2 = tf.summary.scalar('photometric_loss', photometric_loss)
validation_summary_1 = tf.summary.scalar('Validation_loss', predicted_loss_validation)
validation_summary_2 = tf.summary.scalar('Validation_cloud_loss', cloud_loss_validation)

merge_train = tf.summary.merge([training_summary_1] + [training_summary_2] + weight_summaries)
merge_val = tf.summary.merge([validation_summary_1] + [validation_summary_2])

saver = tf.train.Saver()

# tensorflow gpu configuration. Not to be confused with network configuration file

config_tf = tf.ConfigProto(allow_soft_placement=True)
config_tf.gpu_options.allow_growth=True

with tf.Session(config = config_tf) as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./logs_simple_transformer/")

    total_iterations_train = 0
    total_iterations_validate = 0

    # if(current_epoch == 0):
    #    writer.add_graph(sess.graph)

    checkpoint_path = config.paths['checkpoint_path']

    if(current_epoch > 0):
        print("Restoring Checkpoint")

        saver.restore(sess, checkpoint_path + "/model-%d"%current_epoch)
        current_epoch+=1
        total_iterations_train = current_epoch*config.net_params['total_frames_train']/batch_size
        total_iterations_validate = current_epoch*config.net_params['total_frames_validation']/batch_size

    for epoch in range(current_epoch, n_epochs):
	print current_epoch
	print n_epochs
        total_partitions_train = config.net_params['total_frames_train']/config.net_params['partition_limit']
        total_partitions_validation = config.net_params['total_frames_validation']/config.net_params['partition_limit']
        ldr.shuffle()

	print total_partitions_train
        for part in range(total_partitions_train):
	    print part
            source_container, target_container, source_img_container, target_img_container, transforms_container = ldr.load(part, mode = "train")

            for source_b, target_b, source_img_b, target_img_b, transforms_b in zip(source_container, target_container, source_img_container, target_img_container, transforms_container):

                outputs= sess.run([depth_maps_predicted, depth_maps_expected, predicted_loss_train, X2_pooled, train_step, merge_train, predicted_transforms, cloud_loss, photometric_loss, loss1], feed_dict={X1: source_img_b, X2: source_b, depth_maps_target: target_b, expected_transforms: transforms_b ,phase:True, keep_prob:0.5, phase_rgb: False})

                dmaps_pred = outputs[0]
                dmaps_exp = outputs[1]
                loss = outputs[2]
                source = outputs[3]

                if(total_iterations_train%10 == 0):
                    writer.add_summary(outputs[5], total_iterations_train/10)

                print(outputs[8], _ALPHA_CONST*outputs[8], outputs[7], _BETA_CONST*outputs[7], outputs[9],total_iterations_train)

                random_disp = np.random.randint(batch_size)
                print(outputs[6][random_disp])
                print(transforms_b[random_disp])

                if(total_iterations_train%125 == 0):

                    smc.imsave(config.paths['training_imgs_path'] + "/training_save_%d.png"%total_iterations_train, np.vstack((source[random_disp,:,:,0]*40.0 + 40.0, dmaps_pred[random_disp], dmaps_exp[random_disp])))

                total_iterations_train+=1

        if (epoch%1 == 0):
            print("Saving after epoch %d"%epoch)
            saver.save(sess, checkpoint_path + "/model-%d"%epoch)

        for part in range(total_partitions_validation):
            source_container, target_container, source_img_container, target_img_container, transforms_container = ldr.load(part, mode = "validation")

            for source_b, target_b, source_img_b, target_img_b, transforms_b in zip(source_container, target_container, source_img_container, target_img_container, transforms_container):

                outputs= sess.run([depth_maps_predicted, depth_maps_expected, predicted_loss_validation, X2_pooled, merge_val, cloud_loss_validation], feed_dict={X1: source_img_b, X2: source_b, depth_maps_target: target_b, expected_transforms: transforms_b ,phase:False, keep_prob:1.0, phase_rgb: False})

                dmaps_pred = outputs[0]
                dmaps_exp = outputs[1]
                loss = outputs[2]
                source = outputs[3]

                writer.add_summary(outputs[4], total_iterations_validate)
                total_iterations_validate+=1

                print(loss, total_iterations_validate, outputs[5])

                if(total_iterations_validate%25 == 0):

                    random_disp = np.random.randint(batch_size)

                    smc.imsave(config.paths['validation_imgs_path'] + "/validation_save_%d.png"%total_iterations_validate, np.vstack((source[random_disp,:,:,0]*40.0 + 40.0, dmaps_pred[random_disp], dmaps_exp[random_disp])))
