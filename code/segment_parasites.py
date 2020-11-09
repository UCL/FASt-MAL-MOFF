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
from skimage.morphology import remove_small_objects
from skimage.morphology import square
from skimage.segmentation import clear_border
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

parser = argparse.ArgumentParser(description='FastMal parasite segmentation')
parser.add_argument('--dataset_path', dest='dataset', default='.', help='Folder with slide images')
parser.add_argument('--dataset_path2', dest='dataset2', default='.', help='Folder with slide images')
parser.add_argument('--output_folder', dest='output_path', default='.', help='path to save the evaluations in csv format')
parser.add_argument('--slide_list', dest='slide_list', default='/home/pmanescu/SkyNet/VGG/mp_initialData/slides_labels.csv', help='list of slides to process')
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
    
    
def crop_and_save3(complete_image, centroid, shapeid, output_dir, size=64):
    """ Crops and saves a region from an image """
    
    nx_0 = max(int(centroid[0] - size/2),0)
    ny_0 = max(int(centroid[1] - size/2),0)
    nx_1 = min(nx_0 + size, complete_image.shape[1])
    ny_1 = min(ny_0 + size, complete_image.shape[0])
    #roi_type = shape['type'].split(':')[-1]
    roi_file = os.path.join(output_dir, str(shapeid)+'.png')
    cropped_image = complete_image[ny_0:ny_1, nx_0: nx_1,:]
    imageio.imwrite(roi_file, cropped_image)    
    



    
    
def parasitesonly(rgb_image): 
    #path = str(path)
    gray_f = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    #b = cv2.GaussianBlur(gray_f,(9,9),0)
    b= cv2.medianBlur(gray_f,9)#.astype('uint8')
    #cv2.imwrite('b.png', b)
    b_min=np.amin(b)
    b_max=np.amax(b)
    b1= np.clip(b, b_min, int(0.7*b_max), out=None)
    b1=np.amax(b1)-b1
    t = threshold_otsu(b1)
    
    
    #t = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    
    
    #mask = 1-(1*(b1>t).astype(np.uint8))
    mask=(b1>t).astype(np.uint8)
    kernel = np.ones((10,10),np.uint8)
    closing = 255*cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    from skimage import morphology
    small_artefacts = 255*morphology.remove_small_objects(closing.astype(bool), min_size=100, connectivity=4).astype(np.uint8)
    #cv2.imwrite('small_artefacts.png', small_artefacts)
    parasites_gone = 255*morphology.remove_small_objects(small_artefacts.astype(bool), min_size=900, connectivity=4).astype(np.uint8)
    #cv2.imwrite('parasites_gone.png', parasites_gone)
    #parasites_gone1 = cv2.imread('parasites_gone.ong')
    #print(closing.shape, parasites_gone.shape)
    parasites_only = cv2.subtract(small_artefacts, parasites_gone) #how to make them the same size?
    #cv2.imwrite('parasites_only.png', parasites_only)
    #cv2.imshow('gaussian', parasites_only)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return parasites_only


def chop_thumbnails(image, output_dir, current_shapeid=0):
    shapeid = current_shapeid


    mp_masks=parasitesonly(image)
    
    output  = cv2.connectedComponentsWithStats(mp_masks, connectivity=8)
#    num_labels = output[0]
#    labels = output[1]
#    stats = output[2]
    centroids = output[3]    
#                
    for c in centroids:
        crop_and_save3(image, c, shapeid, output_dir)
        shapeid=shapeid+1
                
               
    return shapeid


slides_to_read=opt.slide_list


with open(slides_to_read, newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)))

fmal_list = list(data[:,0])
#print(fmal_list)


if not os.path.exists(opt.output_path):
    os.makedirs(opt.output_path)
subdirs = [x[0] for x in os.walk(opt.dataset)]  
if opt.dataset2 is not None:
    subdirs2 = [x[0] for x in os.walk(opt.dataset2)]
    subdirs=subdirs+subdirs2
#print (subdirs)  
for subdir in subdirs[1:]: 
    predicted_rois = []
    dataset_path = subdir#'/home/petre/mount_point/validation_data/355/'
    #print (dataset_path.split(os.path.sep))
    #dataset_id = int(dataset_path.split(os.path.sep)[-1])
    dataset_id = dataset_path.split(os.path.sep)[-1]
    print (dataset_path)
    if dataset_id in fmal_list:
        #print(dataset_id, fmal_list[10])
        output_folder = os.path.join(opt.output_path, dataset_id)
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)             
                image_files=[] 
                for file in os.listdir(dataset_path):
                    print(file)
                    if file.endswith(".tiff"):
                        image_files.append(os.path.join(dataset_path, file))
                  
                print (image_files)
                
                start_time=time.time()
                current_shapeid=0
                for img_file in image_files:#s[:3]: #while True:
                    img = cv2.imread(img_file)
                    
                    if img is not None:
                        current_shapeid = chop_thumbnails(img, output_folder, current_shapeid)
                        if current_shapeid>5000:
                            break
                    #sess.close()
                                    
       
    
                print("--------------- %s seconds ------------- " % (time.time()-start_time))   




#python segment_parasites.py --dataset_path /home/pmanescu/OMERO/testGit/FASt-Mal-IMS/omero/vagrant/TestOmeroScripts/trainFovSlides/ --dataset_path2 /home/pmanescu/Data/Dataset4/ --slide_list /home/pmanescu/SkyNet/VGG/mp_initialData/slides_labelsd2d4.csv --output_folder /home/pmanescu/SkyNet/VGG/mp_segmented/Train
                