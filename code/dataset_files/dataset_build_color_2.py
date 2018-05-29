"""

For sequence: 2011_09_26

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
from natsort import natsorted as ns
from skimage import io

import scipy.misc as smc
plt.ion()


IMG_HT = 375
IMG_WDT = 1242

fx = 7.215377e+02
fy = 7.215377e+02
cx = 6.095593e+02
cy = 1.728540e+02

K = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3)

velo_to_cam_R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02]).reshape(3,3)
velo_to_cam_T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3,1)

velo_to_cam = np.vstack((np.hstack((velo_to_cam_R, velo_to_cam_T)), np.array([[0,0,0,1]])))

R_rect_00 =  np.array([9.999239e-01, 9.837760e-03, -7.445048e-03, 0.0,
                      -9.869795e-03, 9.999421e-01, -4.278459e-03, 0.0,
                       7.402527e-03, 4.351614e-03, 9.999631e-01,  0.0,
                       0.0,          0.0,          0.0,           1.0]).reshape(4,4)

cam_02_transform = np.array([1.0, 0.0, 0.0, 4.485728e+01/fx,
                             0.0, 1.0, 0.0, 2.163791e-01/fy,
                             0.0, 0.0, 1.0, 2.745884e-03,
                             0.0, 0.0, 0.0, 1.0]).reshape(4,4)


parser = argparse.ArgumentParser(description="Create Lidar Dataset")
parser.add_argument("path", help = "path_to_folder, end with number", type = str)
args = parser.parse_args()

#
# main_path = "/tmp/lidar_calibration_dataset/2011_09_26/2011_09_26_drive_0104"

main_path = args.path

def timestamp_sync(path):
    txt1 = np.loadtxt(path + "_extract/velodyne_points/timestamps.txt", dtype = str)
    txt2 = np.loadtxt(path + "_sync/velodyne_points/timestamps.txt", dtype = str)
    file_list = ns(glob.glob(path + "_extract/velodyne_points/data/*.txt"))

    times1 = txt1[:,1]
    times2 = txt2[:,1]

    for idx in range(times1.shape[0]):
        times1[idx] = times1[idx].split(":")[2]

    for idx in range(times2.shape[0]):
        times2[idx] = times2[idx].split(":")[2]

    start_pt = times2[0]
    end_pt = times2[-1]

    index_start = np.where(times1 == start_pt)[0][0]
    index_end = np.where(times1 == end_pt)[0][0]

    return file_list[index_start:index_end+1]

if not os.path.exists(main_path + "_sync/depth_maps_2"):
    os.makedirs(main_path + "_sync/depth_maps_2")

if not os.path.exists(main_path + "_sync/target_imgs_2"):
    os.makedirs(main_path + "_sync/target_imgs_2")

if not os.path.exists(main_path + "_sync/depth_maps_transformed_2"):
    os.makedirs(main_path + "_sync/depth_maps_transformed_2")

depth_maps_folder = main_path + "_sync/depth_maps_2"
target_img_folder = main_path + "_sync/target_imgs_2"
depth_maps_transformed_folder = main_path + "_sync/depth_maps_transformed_2"

point_files = timestamp_sync(main_path)
imgs_files = ns(glob.glob(main_path + "_sync/image_02/data/*.png"))

angle_limit = 0.34722965035593395/2.50
tr_limit = 0.34722965035593395*1.5

angle_list = np.zeros((1,16), dtype = np.float32)

for img_name, cloud_name in zip(imgs_files, point_files):

    print(img_name, cloud_name)

    omega_x = angle_limit*np.random.random_sample() - (angle_limit/2.0)
    omega_y = angle_limit*np.random.random_sample() - (angle_limit/2.0)
    omega_z = angle_limit*np.random.random_sample() - (angle_limit/2.0)
    tr_x = tr_limit*np.random.random_sample() - (tr_limit/2.0)
    tr_y = tr_limit*np.random.random_sample() - (tr_limit/2.0)
    tr_z = tr_limit*np.random.random_sample() - (tr_limit/2.0)

    theta = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    omega_cross = np.array([0.0, -omega_z, omega_y, omega_z, 0.0, -omega_x, -omega_y, omega_x, 0.0]).reshape(3,3)

    A = np.sin(theta)/theta
    B = (1.0 - np.cos(theta))/(theta**2)

    R = np.eye(3,3) + A*omega_cross + B*np.matmul(omega_cross, omega_cross)

    T = np.array([tr_x, tr_y, tr_z]).reshape(3,1)

    random_transform = np.vstack((np.hstack((R, T)), np.array([[0.0, 0.0, 0.0, 1.0]])))

    to_write_tr = np.expand_dims(np.ndarray.flatten(random_transform), 0)
    angle_list = np.vstack((angle_list, to_write_tr))

    points = np.loadtxt(cloud_name)
    points = points[:,:3]
    ones_col = np.ones(shape=(points.shape[0],1))
    points = np.hstack((points,ones_col))
    current_img = smc.imread(img_name)
    # current_img_color = smc.imread(color_name)

    img = smc.imread(img_name)
    img_ht = img.shape[0]
    img_wdt = img.shape[1]

    points_in_cam_axis = np.matmul(R_rect_00, (np.matmul(velo_to_cam, points.T)))
    transformed_points = np.matmul(random_transform, points_in_cam_axis)
    # transformed_points = transformed_points[:-1,:]

    points_2d = np.matmul(K, np.matmul(cam_02_transform, transformed_points)[:-1,:])

    # points_2d = np.matmul(K, points_in_cam_axis[:-1,:])

    Z = points_2d[2,:]
    x = (points_2d[0,:]/Z).T
    y = (points_2d[1,:]/Z).T

    x = np.clip(x, 0.0, img_wdt - 1)
    y = np.clip(y, 0.0, img_ht - 1)

    reprojected_img = np.zeros_like(img)
    for x_idx, y_idx,z_idx in zip(x,y,Z):
        if(z_idx>0):
            reprojected_img[int(y_idx), int(x_idx)] = z_idx

    smc.imsave(depth_maps_transformed_folder + "/" + img_name[-14:], reprojected_img)

    points_2d = np.matmul(K, np.matmul(cam_02_transform, points_in_cam_axis)[:-1,:])

    Z = points_2d[2,:]
    x = (points_2d[0,:]/Z).T
    y = (points_2d[1,:]/Z).T

    x = np.clip(x, 0.0, img_wdt - 1)
    y = np.clip(y, 0.0, img_ht - 1)

    reprojected_img = np.zeros_like(img)
    for x_idx, y_idx,z_idx in zip(x,y,Z):
        if(z_idx>0):
            reprojected_img[int(y_idx), int(x_idx)] = z_idx
    pooled_img = reprojected_img

    print(img_name[-14:])
    print(np.max(pooled_img), np.min(pooled_img))

    reconstructed_img = current_img*(pooled_img>0.)
    smc.imsave(depth_maps_folder + "/" + img_name[-14:], pooled_img)
    smc.imsave(target_img_folder + "/" + img_name[-14:], reconstructed_img)

np.savetxt(depth_maps_transformed_folder + "/../angle_list_2.txt", angle_list[1:], fmt = "%.4f")
