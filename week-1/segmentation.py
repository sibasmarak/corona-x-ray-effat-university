# this is an optional file for segmenting the data available in images-1 and images-2 folder
# the following code would segregate into three folders - 'corona', 'normal', 'pneumonia'
# modify the code according to the need (i.e. if there is a different directory structure)

# import the required libraries
import shutil 
import os
import pandas as pd
import numpy as np
from PIL import Image 
from os import listdir
import matplotlib.pyplot as plt

# read the metadata files (available in the root directory)
metadata_1 = pd.read_csv('metadata_1.csv', header=0)
metadata_2 = pd.read_csv('metadata_2.csv', header=0)

# function to load images from the input path
def loadImages(path):
    '''
        parameters
        ----------
        path : input path of the images (ensure no .DS_Store file is there - delete if exists)
        
        returns
        -------
        loadedImages : list of loaded images in form of a tuple (image, image_name)
    '''
    
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = Image.open(path + image)
        loadedImages.append((img, image))

    return loadedImages
    
# read the images in folder /images-1 (should be there in the current location)
path1 = 'images-1/'
images1 = loadImages(path1)

refined_images = []
for img_tuple in images1:
    filename = img_tuple[1]
    label = metadata_1[metadata_1.filename == filename].finding.all()
    if label == 'No Finding':
        label = 0
    elif label == 'COVID-19':
        label = 1
    else:
        label = 2 # other type of Pneumonia
        
    refined_images.append((img_tuple[0], img_tuple[1], label))

for img_tuple in refined_images:
    if img_tuple[2] == 0:
        shutil.copyfile('images-1/' + img_tuple[1], 'normal/' + img_tuple[1])
    elif img_tuple[2] == 1:
        shutil.copyfile('images-1/' + img_tuple[1], 'corona/' + img_tuple[1])
    else:
        shutil.copyfile('images-1/' + img_tuple[1], 'pneumonia/' + img_tuple[1])
        
# read the images in folder /images-2 (should be there in the current location)
path2_test_normal = 'images-2/test/NORMAL/'
path2_test_pneumonia = 'images-2/test/PNEUMONIA/'
path2_train_normal = 'images-2/train/NORMAL/'
path2_train_pneumonia = 'images-2/train/PNEUMONIA/'

images2_test_normal = loadImages(path2_test_normal)
images2_test_pneumonia = loadImages(path2_test_pneumonia)
images2_train_normal = loadImages(path2_train_normal)
images2_train_pneumonia = loadImages(path2_train_pneumonia)

# for normal
for img_tuple in images2_test_normal:
        shutil.copyfile('images-2/test/NORMAL/' + img_tuple[1], 'normal/' + img_tuple[1]);

for img_tuple in images2_train_normal:
        shutil.copyfile('images-2/train/NORMAL/' + img_tuple[1], 'normal/' + img_tuple[1]);
        
# for pneumonia - there are some COVID 19 x-ray images in pneumonia folder also      
for img_tuple in images2_train_pneumonia:
    if metadata_2[metadata_2.X_ray_image_name == img_tuple[1]].Label_2_Virus_category.all() == 'COVID-19': 
        shutil.copyfile('images-2/train/PNEUMONIA/' + img_tuple[1], 'corona/' + img_tuple[1]);
    else:
        shutil.copyfile('images-2/train/PNEUMONIA/' + img_tuple[1], 'pneumonia/' + img_tuple[1]);  
        
for img_tuple in images2_test_pneumonia:
    if metadata_2[metadata_2.X_ray_image_name == img_tuple[1]].Label_2_Virus_category.all() == 'COVID-19': 
        shutil.copyfile('images-2/test/PNEUMONIA/' + img_tuple[1], 'corona/' + img_tuple[1]);
    else:
        shutil.copyfile('images-2/test/PNEUMONIA/' + img_tuple[1], 'pneumonia/' + img_tuple[1]);
