import numpy as np
import scipy.misc as smc
from natsort import natsorted as ns
import glob, os
import argparse

parser = argparse.ArgumentParser(description="Create Lidar Dataset Parser file")
parser.add_argument("path", help = "path_to_folder", type = str)
args = parser.parse_args()

dataset_path = args.path

#Picking up all sync folders
folder_names = ns(glob.glob(dataset_path +"*_sync" + os.path.sep))

dataset_array = np.zeros(dtype = str, shape = (1,20))
dataset_array_2 = np.zeros(dtype = str, shape = (1,20))

for fn in folder_names:
    print(fn)
    print(len(fn))
    if fn == '/data/milatmp1/jatavalk/KITTI/2011_09_26/2011_09_26_drive_0009_sync/':
        continue
    if fn == '/data/milatmp1/jatavalk/KITTI/2011_09_26/2011_09_26_drive_0119_sync/':
        continue
    file_names_source = ns(glob.glob(fn + "depth_maps_transformed/*.png"))
    file_names_target = ns(glob.glob(fn + "depth_maps/*.png"))
    img_source = ns(glob.glob(fn + "image_02/data/*.png"))
    img_target = ns(glob.glob(fn + "image_03/data/*.png"))
    transforms_list = np.loadtxt(fn + "angle_list.txt", dtype = str)

    file_names_source = np.array(file_names_source, dtype=str).reshape(-1,1)
    file_names_target = np.array(file_names_target, dtype=str).reshape(-1,1)
    img_source = np.array(img_source, dtype=str).reshape(-1,1)
    img_target = np.array(img_target, dtype=str).reshape(-1,1)

    print(file_names_source.shape, file_names_target.shape, img_source.shape, img_target.shape, \
        transforms_list.shape)
    dataset = np.hstack((file_names_source, file_names_target, img_source, img_target, transforms_list))
    print(dataset.shape)

    dataset_array = np.vstack((dataset_array, dataset))

    # #######################################################################################

    # file_names_source_2 = ns(glob.glob(fn + "depth_maps_transformed_2/*.png"))
    # file_names_target_2 = ns(glob.glob(fn + "depth_maps_2/*.png"))

    # transforms_list_2 = np.loadtxt(fn + "angle_list_2.txt", dtype = str)

    # file_names_source_2 = np.array(file_names_source_2, dtype=str).reshape(-1,1)
    # file_names_target_2 = np.array(file_names_target_2, dtype=str).reshape(-1,1)

    # dataset_2 = np.hstack((file_names_source_2, file_names_target_2, img_source, img_target, transforms_list_2))
    # print(dataset_2.shape)

    # dataset_array_2 = np.vstack((dataset_array_2, dataset_2))




dataset_array = dataset_array[1:]
# dataset_array_2 = dataset_array_2[1:]

# final_array = np.vstack((dataset_array, dataset_array_2))
final_array = np.vstack((dataset_array))

np.random.shuffle(final_array)
np.savetxt("parsed_set.txt", final_array, fmt = "%s", delimiter=' ')
