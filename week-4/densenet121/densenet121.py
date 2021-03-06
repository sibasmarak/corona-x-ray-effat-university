# DenseNet121

input_tensor = Input(shape=(224, 224, 3))
base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False)

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
history_densenet121 = model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.1)


print('cv accuracy:' , np.average(history_densenet121.history['val_accuracy'])) # Cross Validation

print('------------------------ train set ------------------------')
# for train set
y_pred_train = model.predict(X_train)
print_confusion_matrix_and_save(y_train, np.argmax(y_pred_train, axis=1), name='-densenet121-train', fp=fp)
print_performance_metrics(y_train, np.argmax(y_pred_train, axis=1))

print('------------------------ validation set ------------------------')
# for dev set
y_pred_dev = model.predict(X_dev)
print_confusion_matrix_and_save(y_dev, np.argmax(y_pred_dev, axis=1), name='-densenet121-dev', fp=fp)
print_performance_metrics(y_dev, np.argmax(y_pred_dev, axis=1))

print('------------------------ test set ------------------------')
# for test set 
y_pred_test = model.predict(X_test)
print_confusion_matrix_and_save(y_test, np.argmax(y_pred_test, axis=1),name='-densenet121-test', fp=fp)
print_performance_metrics(y_test, np.argmax(y_pred_test, axis=1))

plot_model_accuracy_loss(history_densenet121)