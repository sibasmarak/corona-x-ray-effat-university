# -*- coding: utf-8 -*-

# connecting and mounting gdrive
from google.colab import drive
drive.mount('/content/gdrive')

fp = '/content/gdrive/My Drive/covid-19/' # root file path

import cv2
from PIL import Image
from os import listdir
import skimage
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input

"""- Obtain the data (input size (224, 224, 3))"""

def get_data(folder, X=[], y=[]):
  """
    parameters
    ----------
    folder : input folder name to obtain all the images - should have three sub-folders ('normal', 'corona' and 'pneumonia')

    returns
    -------
    X : a list of all the images - resized into (224, 224, 3)
    y : a list of all the labels (not one-hot encoded) {0:normal 1: corona 2:pneumonia}

  """
  for folderName in listdir(folder)[:2]:
    if not folderName.startswith('.') and not folderName.endswith('.ipynb'): # to not consider .DS_Store and .ipynb files
      if folderName in ['normal']:
        label = 0
      elif folderName in ['corona']:
        label = 1
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

X,y = get_data(fp)

print(len(X), len(y))

y = to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(*shuffle(X, y), test_size=0.05, random_state=0)

"""InceptionV3"""

from tensorflow.keras.applications.inception_v3 import InceptionV3
input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

## Define the model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
  layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit data to model
history_inceptionV3 = model.fit(X_train, y_train, epochs=15, verbose=1, validation_split=0.1)

score_inceptionV3 = model.evaluate(x=X_test, y=y_test)

met = list(zip(model.metrics_names, score_inceptionV3))
print('Scores for test set:')
print('{}: {:.4f}'.format(met[0][0], met[0][1]))
print('{}: {:.4f}'.format(met[1][0], met[1][1]))

print('Average scores for all epochs:')
print('accuracy: {:.4f} (+- {:.4f})'.format(np.mean(history_inceptionV3.history['accuracy']), np.std(history_inceptionV3.history['accuracy'])))
print('loss: {:.4f}'.format(np.mean(history_inceptionV3.history['loss'])))

y_pred_test = model.predict(X_test)
confmat = confusion_matrix(y_test.argmax(axis=1),y_pred_test.argmax(axis=1))

#Plot Confusion Matrix 
plt.figure(figsize=(4,4))
sns.heatmap(confmat, cmap='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='d')
plt.title('Confusion Matrix (InceptionV3)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
fig = plt.gcf()
fig.savefig(fp + 'Confusion-Matrix-(InceptionV3).png');
plt.show();
# Plot Train Accuracy vs Epochs 
plt.plot(range(len(history_inceptionV3.history['accuracy'])), history_inceptionV3.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Train Accuracy vs Epochs (InceptionV3)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Accuracy-vs-Epochs-(InceptionV3).png');
# Plot Train Loss vs Epochs 
plt.plot(range(len(history_inceptionV3.history['loss'])), history_inceptionV3.history['loss'], label='loss', c='g')
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Train Loss vs Epochs (InceptionV3)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Loss-vs-Epochs-(InceptionV3).png');

"""ResNet50"""

from tensorflow.keras.applications.resnet50 import ResNet50
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Define the model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
  layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit data to model
history_resnet50 = model.fit(X_train, y_train, epochs=15, verbose=1, validation_split=0.1)

score_resnet50 = model.evaluate(x=X_test, y=y_test)

met = list(zip(model.metrics_names, score_resnet50))
print('Scores for test set:')
print('{}: {:.4f}'.format(met[0][0], met[0][1]))
print('{}: {:.4f}'.format(met[1][0], met[1][1]))

print('Average scores for all epochs:')
print('accuracy: {:.4f} (+- {:.4f})'.format(np.mean(history_resnet50.history['accuracy']), np.std(history_resnet50.history['accuracy'])))
print('loss: {:.4f}'.format(np.mean(history_resnet50.history['loss'])))

y_pred_test = model.predict(X_test)
confmat = confusion_matrix(y_test.argmax(axis=1),y_pred_test.argmax(axis=1))

#Plot Confusion Matrix 
plt.figure(figsize=(4,4))
sns.heatmap(confmat, cmap='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='d')
plt.title('Confusion Matrix (ResNet50)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
fig = plt.gcf()
fig.savefig(fp + 'Confusion-Matrix-(ResNet50).png');
plt.show();
# Plot Train Accuracy vs Epochs 
plt.plot(range(len(history_resnet50.history['accuracy'])), history_resnet50.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Train Accuracy vs Epochs (ResNet50)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Accuracy-vs-Epochs-(ResNet50).png');
# Plot Train Loss vs Epochs 
plt.plot(range(len(history_resnet50.history['loss'])), history_resnet50.history['loss'], label='loss', c='g')
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Train Loss vs Epochs (ResNet50)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Loss-vs-Epochs-(ResNet50).png');

"""VGG19"""

from tensorflow.keras.applications.vgg19 import VGG19
input_tensor = Input(shape=(224, 224, 3))
base_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Define the model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
  layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit data to model
history_vgg19 = model.fit(X_train, y_train, epochs=15, verbose=1, validation_split=0.1)

score_vgg19 = model.evaluate(x=X_test, y=y_test)

met = list(zip(model.metrics_names, score_vgg19))
print('Scores for test set:')
print('{}: {:.4f}'.format(met[1][0], met[1][1]))
print('{}: {:.4f}'.format(met[0][0], met[0][1]))

print('Average scores for all epochs:')
print('accuracy: {:.4f} (+- {:.4f})'.format(np.mean(history_vgg19.history['accuracy']), np.std(history_vgg19.history['accuracy'])))
print('loss: {:.4f}'.format(np.mean(history_vgg19.history['loss'])))

y_pred_test = model.predict(X_test)
confmat = confusion_matrix(y_test.argmax(axis=1),y_pred_test.argmax(axis=1))

#Plot Confusion Matrix 
plt.figure(figsize=(4,4))
sns.heatmap(confmat, cmap='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='d')
plt.title('Confusion Matrix (VGG19)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
fig = plt.gcf()
fig.savefig(fp + 'Confusion-Matrix-(VGG19).png');
plt.show();
# Plot Train Accuracy vs Epochs 
plt.plot(range(len(history_vgg19.history['accuracy'])), history_vgg19.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Train Accuracy vs Epochs (VGG19)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Accuracy-vs-Epochs-(VGG19).png');
# Plot Train Loss vs Epochs 
plt.plot(range(len(history_vgg19.history['loss'])), history_vgg19.history['loss'], label='loss', c='g')
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Train Loss vs Epochs (VGG19)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Loss-vs-Epochs-(VGG19).png');

"""MobileNetV2"""

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
input_tensor = Input(shape=(224, 224, 3))
base_model = MobileNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Define the model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
  layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit data to model
history_mobilenetV2 = model.fit(X_train, y_train, epochs=15, verbose=1, validation_split=0.1)

score_mobilenetV2 = model.evaluate(x=X_test, y=y_test)

met = list(zip(model.metrics_names, score_mobilenetV2))
print('Scores for test set:')
print('{}: {:.4f}'.format(met[1][0], met[1][1]))
print('{}: {:.4f}'.format(met[0][0], met[0][1]))

print('Average scores for all epochs:')
print('accuracy: {:.4f} (+- {:.4f})'.format(np.mean(history_mobilenetV2.history['accuracy']), np.std(history_mobilenetV2.history['accuracy'])))
print('loss: {:.4f}'.format(np.mean(history_mobilenetV2.history['loss'])))

y_pred_test = model.predict(X_test)
confmat = confusion_matrix(y_test.argmax(axis=1),y_pred_test.argmax(axis=1))

#Plot Confusion Matrix 
plt.figure(figsize=(4,4))
sns.heatmap(confmat, cmap='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='d')
plt.title('Confusion Matrix (MobileNetV2)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
fig = plt.gcf()
fig.savefig(fp + 'Confusion-Matrix-(MobileNetV2).png');
plt.show();
# Plot Train Accuracy vs Epochs 
plt.plot(range(len(history_mobilenetV2.history['accuracy'])), history_mobilenetV2.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Train Accuracy vs Epochs (MobileNetV2)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Accuracy-vs-Epochs-(MobileNetV2).png');
# Plot Train Loss vs Epochs 
plt.plot(range(len(history_mobilenetV2.history['loss'])), history_mobilenetV2.history['loss'], label='loss', c='g')
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Train Loss vs Epochs (MobileNetV2)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Loss-vs-Epochs-(MobileNetV2).png');

"""Xception"""

from tensorflow.keras.applications.xception import Xception
input_tensor = Input(shape=(224, 224, 3))
base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Define the model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
# freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
  layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit data to model
history_xception = model.fit(X_train, y_train, epochs=15, verbose=1, validation_split=0.1)

score_xception = model.evaluate(x=X_test, y=y_test)

met = list(zip(model.metrics_names, score_xception))
print('Scores for test set:')
print('{}: {:.4f}'.format(met[1][0], met[1][1]))
print('{}: {:.4f}'.format(met[0][0], met[0][1]))

print('Average scores for all epochs:')
print('accuracy: {:.4f} (+- {:.4f})'.format(np.mean(history_xception.history['accuracy']), np.std(history_xception.history['accuracy'])))
print('loss: {:.4f}'.format(np.mean(history_xception.history['loss'])))

y_pred_test = model.predict(X_test)
confmat = confusion_matrix(y_test.argmax(axis=1),y_pred_test.argmax(axis=1))

#Plot Confusion Matrix 
plt.figure(figsize=(4,4))
sns.heatmap(confmat, cmap='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='d')
plt.title('Confusion Matrix (Xception)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
fig = plt.gcf()
fig.savefig(fp + 'Confusion-Matrix-(Xception).png');
plt.show();
# Plot Train Accuracy vs Epochs 
plt.plot(range(len(history_xception.history['accuracy'])), history_xception.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('train accuracy')
plt.title('Train Accuracy vs Epochs (Xception)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Accuracy-vs-Epochs-(Xception).png');
# Plot Train Loss vs Epochs 
plt.plot(range(len(history_xception.history['loss'])), history_xception.history['loss'], label='loss', c='g')
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.title('Train Loss vs Epochs (Xception)')
plt.legend()
fig = plt.gcf()
fig.savefig(fp + 'Train-Loss-vs-Epochs-(Xception).png');