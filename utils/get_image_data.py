#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 08:45:09 2018
@author: jehill
"""
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow as tf


def get_image_filenames(datadir):
    mask_files = [];
    train_files = [];

    for dirName, subdirList, fileList in os.walk(datadir):

        for filename in fileList:

            name = os.path.join(dirName, filename)
            if ".tif" in filename.lower() and "mask" in filename:
                mask_files.append(name)
            else:
                train_files.append(name)

    return mask_files, train_files;


def read_tiff(filename):
    im = Image.open(filename)
    im_array = np.array(im)

    return im_array;


def get_batch_data(filelist, batch_size, dim_x, dim_y):
    x = np.zeros([batch_size, dim_x, dim_y])
    y = np.zeros([batch_size, dim_x, dim_y])

    count = 0;
    for i in range(len(filelist)):
        file1 = filelist[i];
        file2 = file1.replace('.tif', '_mask.tif', 1);
        image1 = read_tiff(file1)
        image2 = read_tiff(file2)

        x[count, :, :] = image1;
        y[count, :, :] = image2;

        count = count + 1;

    print(x.shape)
    print(y.shape)
    return x, y


if __name__ == "__main__":
    """"   
    currentdir=os.getcwd()
    datadir1=os.path.join(currentdir, 'train')

    mask_files,image_files=get_image_filenames(datadir1)

    size=len(image_files)


    # these are these are the input file names        
    #x = tf.placeholder([420,580], tf.float32) # suppose your image is 18*18 with 3 channels
    #y = tf.placeholder([420,580], tf.float32) # suppose your image is 18*18 with 3 channels

    batch_size=10
    index=0


    while (index<100):        
            filelist=image_files[index:index+batch_size];         
            image_x,image_y=get_batch_data(filelist, batch_size, 420,580);
            index=index+batch_size+1;



            #image_content = read_tiff_file(file_name)
            #sess.run(train_step, feed_dict={x:image, y:mask})

    """
    # Creates a graph.
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
print(sess.run(c))