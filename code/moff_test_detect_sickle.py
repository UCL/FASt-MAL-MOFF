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
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, h_maxima
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt
import skimage
from skimage import morphology
from scipy.ndimage.morphology import distance_transform_edt


def crop_roi(complete_image, centroid, size=128, crop_size=128):
    """ Crops and saves a region from an image """
    
    nx_0 = max(int(centroid[0] - size/2),0)
    ny_0 = max(int(centroid[1] - size/2),0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    #roi_type = shape['type'].split(':')[-1]
    cropped_image = complete_image[ny_0:ny_1, nx_0: nx_1,:]
    
    return skimage.transform.resize(cropped_image, (crop_size, crop_size,3))

def crop_roi2(complete_image, bbox, size=128, crop_size=128):
    """ Crops and saves a region from an image """
    
    x_0, y_0 = bbox[0], bbox[1]
    max_length=max(bbox[2], bbox[3])
    x_1=x_0+max_length 
    y_1 =y_0+max_length 
    #print(x_0,y_0,x_1,y_1)
    nx_0 = max(x_0,0)
    ny_0 = max(y_0,0)
    nx_1 = min(x_1 , complete_image.shape[1])
    ny_1 = min(y_1, complete_image.shape[0])
    #roi_type = shape['type'].split(':')[-1]
    cropped_image = complete_image[ny_0:ny_1, nx_0: nx_1,:]

    return skimage.transform.resize(cropped_image, (crop_size, crop_size,3))

    
    
def rbc_segmentation(img, outputdir=None):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    sure_bg = cv2.dilate(opening,kernel,iterations=2)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    h_max = h_maxima(dist_transform, 5)
    #h_max = cv2.dilate(h_max,kernel, iterations=3)
    #h_max = np.ma.masked_less(h_max,1)
    cv2.imwrite('bfs_threshold.png', thresh)
    cv2.imwrite('bfs_opening.png', opening)
    #plt.figure()
    #plt.imshow(img)
    #plt.imshow(h_max)
    #plt.show()
    #ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    ret, sure_fg = cv2.threshold(h_max,0.3*h_max.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    bw=(markers>1).astype(int)
    bw=binary_fill_holes(bw).astype(np.uint8)
    
    #img[markers == -1] = [255,0,0]
    
    bw_clean=morphology.remove_small_objects(bw.astype(bool), min_size=1500, connectivity=4).astype(np.uint8)
    cv2.imwrite('bfs_cleaning.png', 255*bw_clean)
    rbc_gone = 255*remove_small_objects(bw_clean.astype(bool), min_size=19000, connectivity=4).astype(np.uint8)
    rbc_only = cv2.subtract(255*bw_clean, rbc_gone)
    #cv2.imwrite(outputdir +'.jpg', img)
    return distance_transform_edt((rbc_only>0).astype(np.uint8)),rbc_only




parser = argparse.ArgumentParser(description='FastMal Classification')
parser.add_argument('--fov', dest='fov', default='../test/pos008_EDOF_RGB.tiff', help='path to the test image')
parser.add_argument('--trained_model', dest='trained_model', default='/home/pmanescu/SkyNet/VGG/sickle_fastmal_models/sickle_classifier0902_max_segmentation_ce35000_vgg19_model.npy', help='path to slide images')
parser.add_argument('--output_dir', dest='output_dir', default='../output_test', help='path to the test folder')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] ='0'#str(args.gpu)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
    
IMSIZE = 128
num_labels=2
num_steps = 0
batch_size=1
rpt_interval=100
min_nb_images = 1

labels_to_names = {0: 'non-sickle', 1:'sickle'}
#label_colors = [
#    (31  , 0   , 255) ,
#    (0   , 159 , 255) ]


label_colors = [
    (150  , 0   , 50) ,
    (50   , 0 , 150) ]

#test_img_path='/home/pmanescu/shares/zdrive/thin/sickle-edofed/101017-07-S-A-S1-20190510112330/FieldPos091_EDOF_RGB.tiff'
test_img_path=args.fov #'/home/pmanescu/Data/thin/sickle_test/Annotated/Sickle/1080/pos006_EDOF_RGB.tiff'

#
img = cv2.imread(test_img_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

csize=100

with tf.Session() as sess:

    #saver = tf.train.Saver()
    images = tf.placeholder(tf.float32, [None, IMSIZE, IMSIZE, 3])
    true_out = tf.placeholder(tf.float32, [None, 2])
    train_mode = tf.placeholder(tf.bool)

    #vgg = vgg19.Vgg19('../models/vgg19.npy', imsize=64)
    #vgg = vgg19.Vgg19('../sickle_fastmal_models/sickle_classifier0902_max_segmentation_ce35000_vgg19_model.npy', imsize=128)
    vgg = vgg19.Vgg19(args.trained_model, imsize=128)
    #vgg = vgg19.Vgg19('../january_fastmal_models/malaria_classifier1301_mean_segmentation_ce20000_vgg19_model.npy', imsize=64)
    vgg.build_avg_pool(images, train_mode=train_mode)
    sess.run(tf.global_variables_initializer())
    
    distnace_rbc, mp_masks=rbc_segmentation(img)
    
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(mp_masks)
    plt.show()   
    cv2.imwrite(os.path.join(args.output_dir,'sickle_segmentation.png'), mp_masks)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(distnace_rbc)
    plt.show() 
    output  = cv2.connectedComponentsWithStats(mp_masks, connectivity=8)
    centroids = output[3]             
    all_stats=output[2]
    #print(stats)
    ii=0
    for c in centroids:
        stats=all_stats[ii]
        mp_roi = crop_roi2(img, stats)
        mp_roi= np.reshape(mp_roi, (1, IMSIZE, IMSIZE, 3))
        prob = sess.run(vgg.new_prob, feed_dict={images: mp_roi, train_mode: False})
        #region_x, region_y, region_width, region_height = stats[2], stats[3]
        prediction=np.argmax(prob)  
        #print(prob)
        
        b = np.array([stats[0],stats[1], stats[0]+stats[2], stats[1]+stats[3]]).astype(int)
        nx_0 = max(int(c[0] - csize/2),0)
        ny_0 = max(int(c[1] - csize/2),0)
        nx_1 = min(nx_0 + csize, img.shape[1])
        ny_1 = min(ny_0 + csize, img.shape[0])
        #b=np.array()
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), label_colors[prediction], 2, cv2.LINE_AA)
        #cv2.rectangle(img, (nx_0, ny_0), (nx_1, ny_1), label_colors[prediction], 2, cv2.LINE_AA)
        txtsize = cv2.getTextSize(labels_to_names[prediction], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]
        cv2.rectangle(img, (b[0], b[1] - 20), (b[0]+ txtsize[0],b[1] ), label_colors[prediction], cv2.FILLED)
        #cv2.putText(img, labels_to_names[prediction], (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(img, labels_to_names[prediction], (b[0], b[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        ii+=1
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(img)
    plt.show()   
    cv2.imwrite(os.path.join(args.output_dir,'sickle_detection_test.png'), img)
    
