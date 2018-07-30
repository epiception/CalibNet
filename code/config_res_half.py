import numpy as np

# Important paths

# resnet_params_path: path containing .pkl file of RESNET18_BN weights
# dataset_path_full: path to parser file
# checkpoint_path: dir where checkpoints are saved and loaded from
# training_imgs_path: dir to save training progress frames (spatial transformer outputs during training)
# validation_imgs_path: dir to save validation progress frames (spatial transformer outputs during validation)
paths = dict(
	# resnet_params_path = "../../../resnet_exp/parameters_resnet18.json",
	# dataset_path_full = "/u/jatavalk/code/CalibNet/cache/parsed_set.txt",
	# checkpoint_path = "/u/jatavalk/saved_weights/CalibNet/checkpoint_simple_transformer",
	# training_imgs_path = "/u/jatavalk/saved_weights/CalibNet/training_imgs",
	# validation_imgs_path = "/u/jatavalk/saved_weights/CalibNet/validation_imgs"
        
        resnet_params_path = "/home/ganeshiyer/Extrinsic_Calibration_2/parameters.json",
	dataset_path_full = "./dataset_files/parsed_set.txt",
	checkpoint_path = "/scratch/ganesh/saved_Weights/checkpoint_simple_transformer",
	training_imgs_path = "/scratch/ganesh/saved_Weights/training",
	validation_imgs_path = "/scratch/ganesh/saved_Weights/validation"
)

# Depth Map parameters

# IMG_HT: input image height
# IMG_WDT: input image width
depth_img_params = dict(
	IMG_HT = int(375)/2,
	IMG_WDT = int(1242)/2
)

# Camera parameters
# The current parameters are for the training set created for the all raw sequences: http://www.cvlibs.net/datasets/kitti/raw_data.php on  2011_09_26. Camera parameters change based on the date of sequence, hence used in config

# fx: focal length x
# fy: focal length y
# cx: camera center cx
# cy: camera center cy
# cam_transform_02: Transform to color camera 02 (applied after transform)
# cam_transform_02_inv: Inverse translation of above
camera_params = dict(
	fx = (7.215377e+02)/2.0,
    fy = (7.215377e+02)/2.0,
    cx = (6.095593e+02)/2.0,
    cy = (1.728540e+02)/2.0,

    cam_transform_02 =  np.array([1.0, 0.0, 0.0, (-4.485728e+01)/7.215377e+02,
	                              0.0, 1.0, 0.0, (-2.163791e-01)/7.215377e+02,
	                              0.0, 0.0, 1.0, -2.745884e-03,
	                              0.0, 0.0, 0.0, 1.0]).reshape(4,4),

	cam_transform_02_inv =  np.array([1.0, 0.0, 0.0, (4.485728e+01)/7.215377e+02,
	                                  0.0, 1.0, 0.0, (2.163791e-01)/7.215377e+02,
	                                  0.0, 0.0, 1.0, 2.745884e-03,
	                                  0.0, 0.0, 0.0, 1.0]).reshape(4,4)
)


# Network and Training Parameters

# batch_size: batch_size taken during training
# total_frames: total instances (check parsed_set.txt for total number of lines)
# total_frames_train: total training instances
# total_frames_validation: total validation instances
# partition_limit: partition size of the total dataset loaded into memory during training
# epochs: total number of epochs
# learning_rate
# beta1: momentum term for Adam Optimizer
# load_epoch: Load checkpoint no. 0 at the start of training (can be changed to resume training)
net_params = dict(
	batch_size = 12,
	total_frames = 480,
	total_frames_train = 360,
	total_frames_validation = 120,
	partition_limit = 240,
	epochs = 10,
	learning_rate = 5e-4,
	beta1 = 0.9,
	load_epoch = 0
)
