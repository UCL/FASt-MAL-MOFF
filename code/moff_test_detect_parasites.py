#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:11:21 2020

@author: pmanescu
"""

import tensorflow as tf

import vgg19_fastmal as vgg19
import utils

import argparse 
import os
import csv
import numpy as np
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import skimage

def crop_roi(complete_image, centroid, size=64, crop_size=64):
    """ Crops and saves a region from an image """
    
    nx_0 = max(int(centroid[0] - size/2),0)
    ny_0 = max(int(centroid[1] - size/2),0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    #roi_type = shape['type'].split(':')[-1]
    cropped_image = complete_image[ny_0:ny_1, nx_0: nx_1,:]
    
    return skimage.transform.resize(cropped_image, (crop_size, crop_size,3))



    
    
def parasitesonly(rgb_image): 
    #path = str(path)
    gray_f = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    #b = cv2.GaussianBlur(gray_f,(9,9),0)
    b= cv2.medianBlur(gray_f,9)#.astype('uint8')
    #cv2.imwrite('b.png', b)
    b_min=np.amin(b)
    b_max=np.amax(b)
    #b1= np.clip(b, b_min, int(0.9*b_max), out=None)
    b1=np.amax(b)-b
    t = threshold_otsu(b1)
    
    
    #t = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    
    
    #mask = 1-(1*(b1>t).astype(np.uint8))
    mask=(b1>t).astype(np.uint8)
    cv2.imwrite('tbf_threshold.png', 255*mask)
    kernel = np.ones((10,10),np.uint8)
    closing = 255*cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #closing=255*cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('tbf_closing.png', closing)
    #closing1 = cv2.imread('closing.png')
    from skimage import morphology
    small_artefacts = 255*morphology.remove_small_objects(closing.astype(bool), min_size=50, connectivity=4).astype(np.uint8)
    #cv2.imwrite('small_artefacts.png', small_artefacts)
    parasites_gone = 255*morphology.remove_small_objects(small_artefacts.astype(bool), min_size=800, connectivity=4).astype(np.uint8)
    #cv2.imwrite('parasites_gone.png', parasites_gone)
    #parasites_gone1 = cv2.imread('parasites_gone.ong')
    #print(closing.shape, parasites_gone.shape)
    parasites_only = cv2.subtract(small_artefacts, parasites_gone) #how to make them the same size?
    cv2.imwrite('parasites_only.png', parasites_only)
    #cv2.imshow('gaussian', parasites_only)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return parasites_only





parser = argparse.ArgumentParser(description='FastMal Classification')
parser.add_argument('--fov', dest='fov', default='../test/FieldPos009_EDOF_RGB.tiff', help='path to the test image')
parser.add_argument('--trained_model', dest='trained_model', default='/home/pmanescu/SkyNet/VGG/january_fastmal_models/malaria_classifier1301_max_segmentation_ce20000_vgg19_model.npy', help='path to slide images')
parser.add_argument('--output_dir', dest='output_dir', default='../output_test', help='path to the test folder')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] ='0'#str(args.gpu)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
    
IMSIZE = 64
num_labels=2
num_steps = 0
batch_size=1
rpt_interval=100
min_nb_images = 1

labels_to_names = {0: 'distractor', 1:'parasite'}
label_colors = [
    (200  , 0   , 50) ,
    (60   , 0 , 150) ]


test_img_path=args.fov# '/home/pmanescu/SkyNet/RawSplitDetectionData/test/398/FieldPos009_EDOF_RGB.tiff'
#
img = cv2.imread(test_img_path)

csize=44

with tf.Session() as sess:
    mp_masks=parasitesonly(img)
    cv2.imwrite(os.path.join(args.output_dir,'test_parasite_mask.png'), mp_masks)
    #saver = tf.train.Saver()
    images = tf.placeholder(tf.float32, [None, IMSIZE, IMSIZE, 3])
    true_out = tf.placeholder(tf.float32, [None, 2])
    train_mode = tf.placeholder(tf.bool)

    #vgg = vgg19.Vgg19('../models/vgg19.npy', imsize=64)
    vgg = vgg19.Vgg19(args.trained_model, imsize=64)
    #vgg = vgg19.Vgg19('../january_fastmal_models/malaria_classifier1301_mean_segmentation_ce20000_vgg19_model.npy', imsize=64)
    vgg.build_avg_pool2(images, train_mode=train_mode)
    sess.run(tf.global_variables_initializer())
    

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(mp_masks)
    plt.show()   
    output  = cv2.connectedComponentsWithStats(mp_masks, connectivity=8)
    centroids = output[3]             
    all_stats=output[2]
    #print(stats)
    ii=0
    for c in centroids:
        mp_roi = crop_roi(img, c)
        mp_roi= np.reshape(mp_roi, (1, IMSIZE, IMSIZE, 3))
        prob = sess.run(vgg.new_prob, feed_dict={images: mp_roi, train_mode: False})
        #region_x, region_y, region_width, region_height = stats[2], stats[3]
        prediction=np.argmax(prob)  
        stats=all_stats[ii]
        b = np.array([stats[0],stats[1], stats[0]+stats[2], stats[1]+stats[3]]).astype(int)
        nx_0 = max(int(c[0] - csize/2),0)
        ny_0 = max(int(c[1] - csize/2),0)
        nx_1 = min(nx_0 + csize, img.shape[1])
        ny_1 = min(ny_0 + csize, img.shape[0])
        #b=np.array()
        #cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), label_colors[prediction], 2, cv2.LINE_AA)
        cv2.rectangle(img, (nx_0, ny_0), (nx_1, ny_1), label_colors[prediction], 2, cv2.LINE_AA)
        txtsize = cv2.getTextSize(labels_to_names[prediction], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(img, (nx_0, ny_0 - 20), (nx_0+ txtsize[0],ny_0 ), label_colors[prediction], cv2.FILLED)
        
        #cv2.putText(img, labels_to_names[prediction], (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(img, labels_to_names[prediction], (nx_0, ny_0 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        ii+=1
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(img)
    plt.show() 
    cv2.imwrite(os.path.join(args.output_dir, 'test_parasite_detection.png'), img)     
    
