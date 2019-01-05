import os
import random
import glob
import numpy as np
from scipy.misc import imread, imresize
import numpy as np
import h5py
import scipy.io as sio
import pickle
import tensorflow as tf

# if args input of size changes, the size should also be changed here.
resize_len = [256, 512]

def read_data_v6(data_dir):
    def _parse_function(image_name, corr_image_name, depth_name):
        image_string = tf.read_file(image_name)
        corr_image_string = tf.read_file(corr_image_name)
        depth_string = tf.read_file(depth_name)

        image_decoded = tf.image.decode_jpeg(image_string, 3)
        corr_image_decoded = tf.image.decode_jpeg(corr_image_string, 3)
        depth_decoded = tf.image.decode_png(depth_string, 1)

        image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        corr_image_decoded = tf.image.convert_image_dtype(corr_image_decoded, dtype=tf.float32)
        depth_decoded = tf.image.convert_image_dtype(depth_decoded, dtype=tf.float32)

        image_resized = tf.image.resize_images(image_decoded, resize_len)
        corr_image_resized = tf.image.resize_images(corr_image_decoded, resize_len)
        depth_resized = tf.image.resize_images(depth_decoded, resize_len)

        return {'image': image_resized, 'image_corr': corr_image_resized, 'annotation': depth_resized}

    image_cat_path = "/your/path/to/KITTI"

    image_left_train_list = []
    image_right_train_list = []
    depth_train_list = []

    image_left_val_list = []
    image_right_val_list = []
    depth_val_list = []

    image_date_path_list = os.listdir(image_cat_path)
    list.sort(image_date_path_list)

    count = 0

    for item in image_date_path_list:
        if os.path.isdir(os.path.join(image_cat_path, item)):
            image_path = os.path.join(image_cat_path, item)
            image_path_list = os.listdir(image_path)
            list.sort(image_path_list)
            for item2 in image_path_list:
                left_image_path = os.path.join(image_path, item2, 'image_02', 'data', '*.png')
                right_image_path = os.path.join(image_path, item2, 'image_03', 'data', '*.png')
                depth_name = item.split('_')[-2] + item.split('_')[-1] + '_' + item2.split('_')[-2]
                #depth_path = os.path.join("/media/zhoukeyang/软件/kitti_depth/data", depth_name, '*.png')
                depth_path = os.path.join("/your/path/to/kitti-depth", depth_name, '*.png')

                left_image_path_list = []
                right_image_path_list = []
                depth_path_list = []

                left_image_path_list.extend(glob.glob(left_image_path))
                right_image_path_list.extend(glob.glob(right_image_path))
                depth_path_list.extend(glob.glob(depth_path))

                list.sort(left_image_path_list)
                list.sort(right_image_path_list)
                list.sort(depth_path_list)

                if count <= 2:
                    image_left_val_list.extend(left_image_path_list)
                    image_right_val_list.extend(right_image_path_list)
                    depth_val_list.extend(depth_path_list)
                
                else:
                    image_left_train_list.extend(left_image_path_list)
                    image_right_train_list.extend(right_image_path_list)
                    depth_train_list.extend(depth_path_list)

                count += 1

    image_left_tensor_2 = tf.convert_to_tensor(np.array(image_left_train_list))
    image_right_tensor_2 = tf.convert_to_tensor(np.array(image_right_train_list))
    depth_tensor_2 = tf.convert_to_tensor(np.array(depth_train_list))

    dataset_train = tf.data.Dataset.from_tensor_slices((image_left_tensor_2, image_right_tensor_2, depth_tensor_2))
    dataset_train = dataset_train.map(_parse_function)

    image_left_tensor_3 = tf.convert_to_tensor(np.array(image_left_val_list))
    image_right_tensor_3 = tf.convert_to_tensor(np.array(image_right_val_list))
    depth_tensor_3 = tf.convert_to_tensor(np.array(depth_val_list))

    dataset_val = tf.data.Dataset.from_tensor_slices((image_left_tensor_3, image_right_tensor_3, depth_tensor_3))
    dataset_val = dataset_val.map(_parse_function)

    return dataset_train, dataset_val, len(depth_train_list), len(depth_val_list)