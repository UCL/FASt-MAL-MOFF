#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:23:49 2020

@author: pmanescu
"""
import argparse
import os
import sys
import numpy as np
import csv
import cv2
import argparse 
import imageio
from skimage.filters import threshold_otsu, threshold_mean, threshold_local, threshold_isodata
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, h_maxima
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import square
from skimage import morphology
from skimage.segmentation import clear_border
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


parser = argparse.ArgumentParser(description='FastMal parasite segmentation')
parser.add_argument('--dataset_path', dest='dataset', default='.', help='Folder with slide images')
parser.add_argument('--dataset_path2', dest='dataset2', default='.', help='Folder with slide images')
parser.add_argument('--output_folder', dest='output_path', default='.', help='path to save the evaluations in csv format')
parser.add_argument('--slide_list', dest='slide_list', default='/home/pmanescu/Data/Leukemias/ALL_vs_Normal2/slides_labels.csv', help='list of slides to process')
opt = parser.parse_args()


def crop_and_save2(complete_image, bbox, shapeid, output_dir, size=64):
    """ Crops and saves a region from an image """
    x_0, y_0 = bbox[0], bbox[1]
    x_1, y_1 = bbox[2], bbox[3]
    
    nx_0 = max(int(x_0+(x_1-x_0)/2 - size/2),0)
    ny_0 = max(int(y_0+(y_1-y_0)/2 - size/2),0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    #roi_type = shape['type'].split(':')[-1]
    roi_file = os.path.join(output_dir, str(shapeid)+'.png')
    cropped_image = complete_image[ny_0:ny_1, nx_0: nx_1,:]
    imageio.imwrite(roi_file, cropped_image)
    
    
def crop_and_save3(complete_image, centroid, shapeid, output_dir, size=200):
    """ Crops and saves a region from an image """
    
    nx_0 = max(int(centroid[0] - size/2),0)
    ny_0 = max(int(centroid[1] - size/2),0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    #roi_type = shape['type'].split(':')[-1]
    roi_file = os.path.join(output_dir, str(shapeid)+'.png')
    cropped_image = complete_image[ny_0:ny_1, nx_0: nx_1,:]
    cv2.imwrite(roi_file, cropped_image)    
    

def crop_roi_mask(complete_image, bbox, size=128, crop_size=128):
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
    
    return (complete_image[ny_0:ny_1, nx_0: nx_1]).astype(np.uint8)

    
    
def wbc_segmentation(img, outputdir=None):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    sure_bg = cv2.erode(opening,kernel,iterations=2)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    h_max = h_maxima(dist_transform, 1)
    h_max = cv2.dilate(h_max,kernel, iterations=3)
    #h_max = np.ma.masked_less(h_max,1)
    #cv2.imwrite('bfs_threshold.png', thresh)
    #cv2.imwrite('bfs_opening.png', opening)

    #ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    ret, sure_fg = cv2.threshold(h_max,0.8*h_max.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    bw=(markers>1).astype(int)
    #bw=binary_fill_holes(bw).astype(np.uint8)
    
    #plt.figure()
    #plt.imshow(h_max)
    #plt.imshow(h_max)
    #plt.show()
    
    #img[markers == -1] = [255,0,0]
    
    bw_clean=morphology.remove_small_objects(bw.astype(bool), min_size=2450, connectivity=4).astype(np.uint8)
    #cv2.imwrite('bfs_cleaning.png', 255*bw_clean)
    #rbc_gone = 255*remove_small_objects(bw_clean.astype(bool), min_size=30000, connectivity=4).astype(np.uint8)
    #rbc_only = cv2.subtract(255*bw_clean, rbc_gone)
    #cv2.imwrite(outputdir +'.jpg', img)
    return bw_clean




def wbc_segmentation_hsv(img, outputdir=None):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray=255-hsv[:,:,1]

    gray = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY_INV)
    

    
    thresh_clean=255*morphology.remove_small_objects(thresh.astype(bool), min_size=2500, connectivity=4).astype(np.uint8)

 
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh_clean,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.erode(opening,kernel,iterations=2)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    h_max = h_maxima(dist_transform, 1)
    h_max = cv2.dilate(h_max,kernel, iterations=3)
    #h_max = np.ma.masked_less(h_max,1)
    #cv2.imwrite('bfs_threshold.png', thresh)
    #cv2.imwrite('bfs_opening.png', opening)

    #ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    ret, sure_fg = cv2.threshold(h_max,0.3*h_max.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    bw=(markers>1).astype(int)
    #bw=binary_fill_holes(bw).astype(np.uint8)
    

    
    #img[markers == -1] = [255,0,0]
    
    #bw_clean=255*morphology.remove_small_objects(bw.astype(bool), min_size=1000, connectivity=4).astype(np.uint8)
    #cv2.imwrite('bfs_cleaning.png', 255*bw_clean)
    #rbc_gone = 255*remove_small_objects(bw_clean.astype(bool), min_size=30000, connectivity=4).astype(np.uint8)
    #rbc_only = cv2.subtract(255*bw_clean, rbc_gone)
    #cv2.imwrite(outputdir +'.jpg', img)
    return 255*bw.astype(np.uint8)#bw_clean

def chop_thumbnails(image, output_dir, current_shapeid=0):
    shapeid = current_shapeid


    mp_masks=wbc_segmentation_hsv(image)
    
    output  = cv2.connectedComponentsWithStats(mp_masks, connectivity=8)
#    num_labels = output[0]
#    labels = output[1]
#    stats = output[2]
    centroids = output[3]    
    #all_stats=output[2]  
    ii=0             
    for c in centroids:
        #stats=all_stats[ii]
        #mask_mp_roi =crop_roi_mask(mp_masks, stats)
        #rp=regionprops(mask_mp_roi)
        
        #if rp[0].solidity<0.92:
        crop_and_save3(image, c, shapeid, output_dir)
        shapeid=shapeid+1
        ii=ii+1        
               
    return shapeid

#
#slides_to_read=opt.slide_list
#
#
#with open(slides_to_read, newline='') as csvfile:
#    data = np.array(list(csv.reader(csvfile)))
#
#fmal_list = list(data[:,0])
#print(fmal_list)


if not os.path.exists(opt.output_path):
    os.makedirs(opt.output_path)
    
train_path=os.path.join(opt.output_path,'Train')
if not os.path.exists(train_path):
    os.makedirs(train_path)
test_path=os.path.join(opt.output_path,'Test')
if not os.path.exists(test_path):
    os.makedirs(test_path)
    
    
    
#subdirs = [x[0] for x in os.walk(opt.dataset)]  
#if opt.dataset2 is not None:
#    subdirs2 = [x[0] for x in os.walk(opt.dataset2)]
#    subdirs=subdirs+subdirs2
    
image_files=[] 
for file in os.listdir(opt.dataset):
    print(file)
    if file.endswith(".jpg"):
        image_files.append(os.path.join(opt.dataset, file))
        
predictionFile= open(os.path.join(opt.slide_list),'w')  
wr = csv.writer(predictionFile, dialect='excel')

for img_file in image_files: 
    predicted_rois = []
    #'/home/petre/mount_point/validation_data/355/'
    #print (dataset_path.split(os.path.sep))
    #dataset_id = int(dataset_path.split(os.path.sep)[-1])
    fname = img_file.split(os.path.sep)[-1]
    dataset_id = fname.split('_')[0]
    ext_imlabel=fname.split('_')[1]
    imlabel=ext_imlabel.split('.')[0]
    wr.writerow([dataset_id, imlabel]) 
    #print (dataset_path)
#    if dataset_id in fmal_list:
        #print(dataset_id, fmal_list[10])
    if image_files.index(img_file) % 3 == 0:
            output_folder=os.path.join(test_path, dataset_id)
    else:
            output_folder=os.path.join(train_path, dataset_id)    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)      
    
        

          
        print (image_files)
        image_files.sort()
        start_time=time.time()
        current_shapeid=0
        #for img_file in image_files:#s[:3]: #while True:
    
        
        img = cv2.imread(img_file)
        
        if img is not None:
            current_shapeid = chop_thumbnails(img, output_folder, current_shapeid)
            if current_shapeid>5000:
                break
                #sess.close()
                                    
       
    
                print("--------------- %s seconds ------------- " % (time.time()-start_time))   




#python segment_parasites.py --dataset_path /home/pmanescu/OMERO/testGit/FASt-Mal-IMS/omero/vagrant/TestOmeroScripts/trainFovSlides/ --dataset_path2 /home/pmanescu/Data/Dataset4/ --slide_list /home/pmanescu/SkyNet/VGG/mp_initialData/slides_labelsd2d4.csv --output_folder /home/pmanescu/SkyNet/VGG/mp_segmented/Train
                