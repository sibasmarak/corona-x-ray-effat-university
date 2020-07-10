# 2 CNN

model = Sequential()

model.add(Conv2D(16,(5,5),padding='valid',input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.4))

model.add(Conv2D(32,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model = Model(inputs=model.input, outputs=model.output)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)

print(np.average(history.history['val_accuracy'])) # Cross Validation

print('------------------------ train set ------------------------')
# for train set
y_pred_train = model.predict(X_train)
print('Score train: ', model.evaluate(X_train, y_train))
print_confusion_matrix_and_save(y_train, np.argmax(y_pred_train, axis=1), name='-2CNN-train', fp=fp)
print_performance_metrics(y_train, np.argmax(y_pred_train, axis=1))

print('------------------------ validation set ------------------------')
# for dev set
y_pred_dev = model.predict(X_dev)
print('Score validation: ', model.evaluate(X_dev, y_dev))
print_confusion_matrix_and_save(y_dev, np.argmax(y_pred_dev, axis=1), name='-2CNN-dev', fp=fp)
print_performance_metrics(y_dev, np.argmax(y_pred_dev, axis=1))

print('------------------------ test set ------------------------')
# for test set
y_pred_test = model.predict(X_test)
print('Score test: ', model.evaluate(X_test, y_test))
print_confusion_matrix_and_save(y_test, np.argmax(y_pred_test, axis=1),name='-2CNN-test', fp=fp)
print_performance_metrics(y_test, np.argmax(y_pred_test, axis=1))

plot_model_accuracy_loss(history)
