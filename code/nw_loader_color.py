import numpy as np
import glob, os, argparse
import scipy.misc as smc
from tqdm import tqdm
import matplotlib.pyplot as plt

import config_res as config

total = config.net_params['total_frames']
total_train = config.net_params['total_frames_train']
total_validation = config.net_params['total_frames_validation']
partition_limit = config.net_params['partition_limit']

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = config.net_params['batch_size']

dataset = np.loadtxt(config.paths['dataset_path_full'], dtype = str)

dataset_train = dataset[:total_train]
dataset_validation = dataset[total_train:total]

img_mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
img_std = np.array([0.229, 0.224, 0.225], dtype = np.float32)

def shuffle():
    np.random.shuffle(dataset_train)
    np.random.shuffle(dataset_validation)

def load(p_no, mode):

    if(mode == "train"):
        dataset_part = dataset_train[p_no*partition_limit:(p_no + 1)*partition_limit]
    elif(mode == "validation"):
        dataset_part = dataset_validation[p_no*partition_limit:(p_no + 1)*partition_limit]
    source_file_names = dataset_part[:,0]
    target_file_names = dataset_part[:,1]
    source_image_names = dataset_part[:,2]
    target_image_names = dataset_part[:,3]
    transforms = np.float32(dataset_part[:,4:])

    target_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 1), dtype = np.float32)
    source_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 1), dtype = np.float32)
    source_img_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 3), dtype = np.float32)
    target_img_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 3), dtype = np.float32)
    transforms_container = np.zeros((partition_limit, 4, 4), dtype = np.float32)

    c_idx = 0
    for s_name, t_name, img_source_name, img_target_name, transform in tqdm(zip(source_file_names, target_file_names, source_image_names, target_image_names, transforms)):

        warped_ip = np.float32(smc.imread(s_name, True))
        warped_ip[0:5,:] = 0.0 ; warped_ip[:,0:5] = 0.0 ; warped_ip[IMG_HT - 5:,:] = 0.0 ; warped_ip[:,IMG_WDT-5:] = 0.0 ;
        warped_ip = (warped_ip - 40.0)/40.0
        source_container[c_idx, :, :, 0] = warped_ip

        target_ip = np.float32(smc.imread(t_name, True))
        target_ip[0:5,:] = 0.0 ; target_ip[:,0:5] = 0.0 ; target_ip[IMG_HT - 5:,:] = 0.0 ; target_ip[:,IMG_WDT-5:] = 0.0 ;
        target_ip = (target_ip - 40.0)/40.0
        target_container[c_idx, :, :, 0] = target_ip

        source_img = np.float32(smc.imread(img_source_name))
        source_img[0:5,:,:] = 0.0 ; source_img[:,0:5,:] = 0.0 ; source_img[IMG_HT - 5:,:,:] = 0.0 ; source_img[:,IMG_WDT-5:,:] = 0.0 ;
        # source_img = (source_img - 127.5)/127.5
        # source_img_container[c_idx, :, :, :] = source_img

        source_img = source_img/255.0
        source_img = (source_img - img_mean)/img_std
        source_img_container[c_idx, :, :, :] = source_img

        target_img = np.float32(smc.imread(img_target_name))
        target_img[0:5,:,:] = 0.0 ; target_img[:,0:5,:] = 0.0 ; target_img[IMG_HT - 5:,:,:] = 0.0 ; target_img[:,IMG_WDT-5:,:] = 0.0 ;
        # target_img = (target_img - 127.5)/127.5
        # target_img_container[c_idx, :, :, :] = target_img

        target_img = target_img/255.0
        target_img = (target_img - img_mean)/img_std
        target_img_container[c_idx, :, :, :] = target_img

        transforms_container[c_idx, :, :] = np.linalg.inv(transform.reshape(4,4))
        c_idx+=1

    source_container = source_container.reshape(partition_limit/batch_size, batch_size, IMG_HT, IMG_WDT , 1)
    target_container = target_container.reshape(partition_limit/batch_size, batch_size, IMG_HT, IMG_WDT , 1)
    source_img_container = source_img_container.reshape(partition_limit/batch_size, batch_size, IMG_HT, IMG_WDT, 3)
    target_img_container = target_img_container.reshape(partition_limit/batch_size, batch_size, IMG_HT, IMG_WDT, 3)
    transforms_container = transforms_container.reshape(partition_limit/batch_size, batch_size, 4, 4)

    return source_container, target_container, source_img_container, target_img_container, transforms_container
