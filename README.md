
# Multiple Instance Learning for Blood Film Microscopy Morphological Analysis 



High throughput automated blood film analysis under a brightfield microscope can be a rapid and unified solution to identify blood related disorders, especially in resource constrained settings.The major bottleneck in successfully analyzing blood films with deep learning vision techniquesis a lack of object-level annotations of disease markers such as parasites or abnormal red blood cells. This work proposes a deep learning supervised approach that leverages weak labels readily available from routine clinical microscopy to detect malaria and sickle cells in digital blood film microscopy. This approach is based on aggregating the convolutional features of multiple objects from multiple high resolution image fields.


![The idea](/figs/sickle_moff.png)  
<p align="center">
 MOFF for sickle cell detection in BFS</center>
</p>


## **Getting the data**

You can download the images used to train the models here: 

- Blood Film Smears (BFS) for sickle cell detection: https://doi.org/10.5522/04/12407567
- Thick Blood Films (TBF) malaria parasite detection: https://doi.org/10.5522/04/12173568
- Blood Film Smears (BFS) from normal and ALL patients for lymphoblast cell detection: http://homes.di.unimi.it/scotti/all/

Each folder in each dataset represents a sample (xxyyzz-tt-...). Each folder in the TBF dataset contains 100 image fields acquired with a 100x/1.4 NA oil magnification objective. Each folder in the BFS dataset contains 10 to 20 image fields acquired with a 100x/1.4 NA oil magnification objective.   
Each dataset contains a file named 'slides_labels.csv' that stores the weak sample level labels. Each row of the csv file looks like: <sample-id>, <label> (0 or 1).   


## **Using the code**


### *White Blood Cell Segmentation* 

    python segment_wbcs.py --dataset_path /path/to/your/dataset/or/downloaded/dataset/ --output_folder /path/to/an/output/destination/ 

This will create a subfolder in the --output_folder corresponding to each of the images existing in the --dataset_path. In each of these subfolders (corresponding to a BFS sample) this script will write images of individual WBCs cropped from the initial image fields. 




### *Red Blood Cell Segmentation* 

![The idea](/figs/moff_rbc_segmentation.png)  
<p align="center">
 RBC segmentation</center>
</p>

    python segment_rbcs.py --dataset_path /path/to/your/dataset/or/downloaded/dataset/ --output_folder /path/to/an/output/destination/ 

This will create a subfolder in the --output_folder corresponding to each of the folders existing in the --dataset_path. In each of these subfolders (corresponding to a BFS sample) this script will write images of individual RBC cropped from the initial image fields. 


### *Malaria Parasite Object Candidate Segmentation* 

![The idea](/figs/moff_parasite_segmentation.png)  
<p align="center">
 RBC segmentation</center>
</p>

    python segment_parasites.py --dataset_path /path/to/your/dataset/or/downloaded/dataset/ --output_folder /path/to/an/output/destination/ 

This will create a subfolder in the --output_folder corresponding to each of the folders existing in the --dataset_path. In each of these subfolders (corresponding to a RBF sample) this script will write images of parasite-like objects cropped from the initial image fields. 


### *Training*

The model architecture and code is based on [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16). We modified a vgg architecture with the convolutional layers pre-trained on the ImageNet dataset. 

For Sickle Cell Detection:

    python moff_vgg19_sickle.py --dataset /path/to/your/training/segmented/dataset/ --csv_labels /path/to/the/file/containing/the/weak/sample/level/labels --save_dir /path/to/save/trained/model --test_dir /path/to/your/test/segmented/dataset --test_csv_labels /path/to/your/test/weak/labels --output_dir /path/where/you/want/to/save/the/predictions 
    
For Malaria Detection: 

    python moff_vgg19_malaria.py --dataset /path/to/your/training/segmented/dataset/ --csv_labels /path/to/the/file/containing/the/weak/sample/level/labels --save_dir /path/to/save/trained/model --test_dir /path/to/your/test/segmented/dataset --test_csv_labels /path/to/your/test/weak/labels --output_dir /path/where/you/want/to/save/the/predictions 

There is not a significant difference between the two scripts. There are some specific parameters hard coded for each case.



### *Trained models*

The malaria MOFF trained model: https://drive.google.com/file/d/1f8v0-EX0xVwiGnKhvD0d-UJ8Z6SUjkjv/view?usp=share_link


### *Lymphoblast Detection Test*
Once a model is trained on the ALL vs Normal weak labels, it can be used to identify individual blast cells in image fields: 

    python moff_test_detect_blast.py --fov ../test/Im024_1.jpg --trained_model /path/to/your/trained/model --output_dir ../output_test

![WBC detection](/figs/blast_detection_test.png)  
<p align="center">
 Lymphoblast detection in BFS using MILCA</center>
</p>



### *Sickle Cell Detection Test*
Once a model is trained on the SCD weak labels, it can be used to identify individual abnormal sickle cells in image fields: 

    python moff_test_detect_sickle.py --fov ../test/pos008_EDOF_RGB.tiff --trained_model /path/to/your/trained/model --output_dir ../output_test

Output the test image fields:

![RBC detection](/figs/sickle_detection_test.png)  
<p align="center">
 Individual abdnormal Sickle Cell detection in BFS using MILCA</center>
</p>

### *Malaria Parasite Detection Test*

Same approach works for Malaria Parasites detection.

    python moff_test_detect_parasites.py --fov ../test/FieldPos009_EDOF_RGB.tiff --trained_model /path/to/your/trained/model --output_dir ../output_test

Output the test image fields:

![RBC detection](/figs/test_parasite_detection.png)  
<p align="center">
 Individual P.Falciparum parasite detection in TBF with MILCA</center>
</p>



## **System Requirements**

- Python3.6
- Numpy
- OpenCV
- scikit-image
- tensorflow 1.3


