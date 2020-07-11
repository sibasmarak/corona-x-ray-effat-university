# =============================================================================
# Define Utility Functions
# =============================================================================
def plot_model_accuracy_loss(history, fp=fp, name=''):
    """
        parameters
        ----------
        history : the output of a Keras model's fit method (history = model.fit(...))

        returns
        -------
        plots the accuracy, val_accuracy, loss and val_loss for the input history
    
    """
    plt.figure(figsize=(4, 4))
    plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    fig = plt.gcf()
    fig.savefig(fp + 'History'+ str(name) +'.png')
    plt.show()

def print_performance_metrics(y_test, max_y_pred_test):
    """
        parameters
        ----------
        y_test : actual label (must be one hot encoded form)
        y_pred_test : predicted labels (must be in non-one hot encoded form, common output of predict methods of classifers)

        returns
        -------
        prints the accuracy, precision, recall, F1 score, ROC AUC score, Cohen Kappa Score, Matthews Corrcoef and classification report   
    
    """
    print('Accuracy:', np.round(metrics.accuracy_score(y_test.argmax(axis=1), max_y_pred_test),4))
    print('Precision:', np.round(metrics.precision_score(y_test.argmax(axis=1), max_y_pred_test, average='weighted'),4))
    print('Recall:', np.round(metrics.recall_score(y_test.argmax(axis=1), max_y_pred_test, average='weighted'),4))
    print('F1 Score:', np.round(metrics.f1_score(y_test.argmax(axis=1), max_y_pred_test, average='weighted'),4))
    print('ROC AUC Score:', np.round(metrics.roc_auc_score(y_test, max_y_pred_test.reshape(-1,1), multi_class='ovr'),4))
    print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test.argmax(axis=1), max_y_pred_test),4))
    print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test.argmax(axis=1), max_y_pred_test),4)) 
    print('\t\tClassification Report:\n', metrics.classification_report(y_test.argmax(axis=1), max_y_pred_test))

def print_confusion_matrix_and_save(y_test, y_pred_test, name='', fp=fp):
    """
        parameters
        ----------
        fp : file path to store the confusion matrix
        y_test : actual label (should be one hot encoded form)
        y_pred_test : predicted labels (should be in non-one hot encoded form, common output of predict methods of classifers)
        name : optional, string to add at the end of the file saved corresponding to the confusion matrix

        returns
        -------
        plots the confusion matrix as a heatmap and saves the confusion matrix in '.png' format 
        at location "fp + 'Confusion'+ name +'.png'"   
    
    """
    mat = confusion_matrix(y_test.argmax(axis=1), y_pred_test)
    plt.figure(figsize=(4, 4))
    sns.heatmap(mat, cmap='coolwarm', square=True, annot=True, fmt='d', cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    fig = plt.gcf()
    fig.savefig(fp + 'Confusion'+ str(name) +'.png')
    plt.show();

def get_data(folder, X=[], y=[]):
  """
    parameters
    ----------
    folder : input folder name to obtain all the images - should have three sub-folders ('normal', 'corona' and 'pneumonia')
    X : optional, the input images in a list format
    y : optional, the corresponding labels (not one-hot encoded) {0:normal 1: corona 2:pneumonia}

    returns
    -------
    X : a list of all the images - resized into (224, 224, 3)
    y : a list of all the labels (not one-hot encoded) {0:normal 1: corona 2:pneumonia}

  """
  for folderName in listdir(folder)[0:3]:
    if not folderName.startswith('.') and not folderName.endswith('.ipynb'): # to not consider .DS_Store and .ipynb files
      if folderName in ['corona']:
        label = 0
      elif folderName in ['pneumonia']:
        label = 1
      elif folderName in ['normal']:
        label = 2 
      val = 0
      for image_filename in tqdm(listdir(folder + folderName)):
        img_file = cv2.imread(folder + folderName + '/' + image_filename)
        if img_file is not None:
          if label == 0 or label == 1 or label == 2:
            val = val + 1
            if val > 500:
              break
          img_file = skimage.transform.resize(img_file, (224, 224, 3))
          img_arr = np.asarray(img_file)
          X.append(img_arr)
          y.append(label)
  X = np.asarray(X)
  y = np.asarray(y)
  return X,y
# =============================================================================
# obtain X_train, y_train, X_dev, y_dev, X_test, y_test
# =============================================================================
X,y = get_data(fp)
print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(*shuffle(X, y), test_size=0.05, random_state=0)
X_train, X_dev, y_train, y_dev = train_test_split(*shuffle(X_train, y_train), test_size=0.05, random_state=42)
del X,y

# convert the actual labels to categorical variables
y_train = to_categorical(y_train,2)
y_dev = to_categorical(y_dev, 2)
y_test = to_categorical(y_test,2)

# =============================================================================
# obtain the features using transfer learning
# then fit the required ML Algorithms
# =============================================================================
base_model = NASNetMobile(include_top=False, weights='imagenet', input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='tanh')(x)
predictions = Dense(8, activation='tanh')(x)
model_feat = Model(inputs=base_model.input,outputs=predictions)
feat_train = model_feat.predict(X_train)
