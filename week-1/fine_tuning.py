# dataset is available in the Google Drive link shared in the root README.md
# the below two lines of code is to be used while using Google Colab to train 
# else comment the two lines
from google.colab import drive
drive.mount('/content/gdrive')

# import libraries not related to Keras / TensorFlow/ scikit-learn
import cv2
import os
import skimage
import numpy as np
import pandas as pd
from os import listdir
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize

# import scikit-learn related libraries
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# import Keras/ Tensorflow related libraries
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input

# define the file path 
# the below definition is used if the dataset is in a directory structure: My Drive/covid-19/
fp = '/content/gdrive/My Drive/covid-19/'

# define a function to obtain all the data from a folder
# here pass the file path fp as folder 
def get_data(folder, X=[], y=[]):
  """
    parameters
    ----------
    folder : input folder name to obtain all the images - should have three sub-folders ('normal', 'corona' and 'pneumonia')
    X : an empty array to store the images
    Y : an empty aray to store the labels
    returns
    -------
    X : a list (numy.ndarray) of all the images
    y : a list (numy.ndarray) of all the labels (not one-hot encoded) {0:normal 1: corona 2:pneumonia}

  """

  for folderName in ['normal','corona','pneumonia']:
    if not folderName.startswith('.'): # to remove .DS_Store
      if folderName in ['normal']:
        label = 0
      elif folderName in ['corona']:
        label = 1
      elif folderName in ['pneumonia']:
        label = 2 # for all other types of pnuemonia
      for image_filename in tqdm(listdir(folder + folderName)):
        img_file = cv2.imread(folder + folderName + '/' + image_filename)
        if img_file is not None:
          img_file = skimage.transform.resize(img_file, (224, 224, 3))
          img_arr = np.asarray(img_file)
          X.append(img_arr)
          y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

# obtain the data
X, y = get_data('/content/gdrive/My Drive/covid-19/', X, y)

# convert the labels to one_hot encoded format
yoh = to_categorical(y, num_classes=3)

# shuffle the data and labels
Xshuffle, yshuffle_oh = shuffle(X, yoh)

# split the shuffled dataset and labels 
X_train, X_test, y_train, y_test = train_test_split(Xshuffle, yshuffle_oh, test_size=0.05, random_state=0)

# begin the model training and fine-tuning
# change the epochs and bach_size wherever required for better result
input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
# if required remove one metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

# train the model on the new data for a few epochs
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# we chose to train the top 1 inception blocks, i.e. we will freeze
# the first 250 layers and unfreeze the rest:
for layer in model.layers[:250]:
   layer.trainable = False
for layer in model.layers[250:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
# if required remove one metric
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

# we train our model again (this time fine-tuning the top 1 inception blocks)
# alongside the top Dense layers
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

# Use the following code for training in a mini-batch basis.
# When limited resources are available.
# 287 -> number of COVID 19 photos
# 1595 -> number of Normal photos
# 4371 -> number of Pneumonia photos
# Used a mini-batch of maximum 200 photos in one go from each class
for i in range(23):
  # 23 would be sufficient to cover all the images in the largest category here (pneumonia)
  overall = []
  c = listdir(fp + 'corona')
  startc = (i*200) % 287
  endc = ((i*200) + 200) % 287
  if endc - startc <= 0:
    endc = 287
  overall.append(fp + 'corona/' + c for c in c[startc:endc])
  n = listdir(fp + 'normal')
  startn = (i*200) % 1595
  endn = ((i*200) + 200) % 1595
  if endn - startn <= 0:
    endn = 1595
  overall.append(fp + 'normal/' + n for n in n[startn:endn])
  p = listdir(fp + 'pneumonia')
  startp = (i*200) % 4371
  endp = ((i*200) + 200) % 4371
  if endp - startp <= 0:
    endp = 4371
  overall.append(fp + 'pneumonia/' + p for p in p[startp:endp])
  
  res = [0] * (endc - startc) + [1] * (endn - startn) + [2] * (endp - startp)
  res = to_categorical(res, num_classes=3)
  X = []
  for path in overall[0]:
    # print(path)
    img_file = cv2.imread(path)
    if img_file is not None:
      img_file = skimage.transform.resize(img_file, (224, 224, 3))
      img_arr = np.asarray(img_file)
      X.append(img_arr)
  for path in overall[1]:
    img_file = cv2.imread(path)
    if img_file is not None:
      img_file = skimage.transform.resize(img_file, (224, 224, 3))
      img_arr = np.asarray(img_file)
      X.append(img_arr)
  for path in overall[2]:
    img_file = cv2.imread(path)
    if img_file is not None:
      img_file = skimage.transform.resize(img_file, (224, 224, 3))
      img_arr = np.asarray(img_file)
      X.append(img_arr)  
  y = res
  print(len(X), len(y))
  X = np.asarray(X)
  y = np.asarray(y)
  X, y = shuffle(X, y)
  # train the model on the new data for a few epochs
  model.fit(X, y, epochs=10)

# evaluate for the test set
# use the last set of X and y
# Or pick X and y at random for evaluation purposes
scores = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
