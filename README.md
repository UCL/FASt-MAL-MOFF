# FASt-MAL-MOFF

## **Multiple Object Feature Fusion (MOFF) for weakly supervised deep learning**



High throughput automated blood film analysis under a brightfield microscope can be a rapid and unified solution to identify blood related disorders, especially in resource constrained settings.The major bottleneck in successfully analyzing blood films with deep learning vision techniquesis a lack of object-level annotations of disease markers such as parasites or abnormal red blood cells. This work proposes a deep learning supervised approach that leverages weak labels readily available from routine clinical microscopy to detect malaria and sickle cells in digital blood film microscopy. This approach is based on aggregating the convolutional features of multiple objects from multiple high resolution image fields.


![The idea](/figs/sickle_moff.png)  
<p align="center">
 MOFF for sickle cell detection in BFS</center>
</p>


## **Getting the data**

You can download the images used to train the models here: 

- Blood Film Smears (BFS) for sickle cell detection: https://doi.org/10.5522/04/12407567
- Thick Blood Films (TBF) malaria parasite detection: https://doi.org/10.5522/04/12173568

Each folder in each dataset represents a sample (xxyyzz-tt-...). Each folder in the TBF dataset contains 100 image fields acquired with a 100x/1.4 NA oil magnification objective. Each folder in the BFS dataset contains 10 to 20 image fields acquired with a 100x/1.4 NA oil magnification objective.   
Each dataset contains a file named 'slides_labels.csv' that stores the weak sample level labels. Each row of the csv file looks like: <sample-id>, <label> (0 or 1).   


## **Using the code**

# * Red Blood Cell Segmentation ** 

![The idea](/figs/moff_rbc_segmentation.png)  
<p align="center">
 RBC segmentation</center>
</p>

    python segment_rbcs.py --dataset_path /path/to/your/dataset/or/downloaded/dataset/ --output_folder /path/to/an/output/destination/ 

This will create a subfolder in the --output_folder corresponding to each of the folders existing in the --dataset_path. In each of these subfolders (corresponding to a BFS sample) this script will write images of individual RBC cropped from the initial image fields. 


# * Malaria Parasite Object Candidate Segmentation * 

![The idea](/figs/moff_parasite_segmentation.png)  
<p align="center">
 RBC segmentation</center>
</p>

    python segment_parasites.py --dataset_path /path/to/your/dataset/or/downloaded/dataset/ --output_folder /path/to/an/output/destination/ 

This will create a subfolder in the --output_folder corresponding to each of the folders existing in the --dataset_path. In each of these subfolders (corresponding to a RBF sample) this script will write images of parasite-like objects cropped from the initial image fields. 


