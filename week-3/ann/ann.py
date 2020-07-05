# ANN

model = Sequential()
model.add(Dense(units=8, activation='tanh', input_shape=(8,)))
model.add(Dense(units=4, activation='tanh'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

model = Model(inputs=model.input, outputs=model.output)

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(feat_train, y_train, epochs=30, validation_split=0.1)
print('cv accuracy:' , np.average(history.history['val_accuracy'])) # Cross Validation

print('------------------------ train set ------------------------')
# for train set
y_pred_train = model.predict(feat_train)
print_confusion_matrix_and_save(y_train, np.argmax(y_pred_train, axis=1), name='-ann-train', fp=fp)
print_performance_metrics(y_train, np.argmax(y_pred_train, axis=1))

print('------------------------ validation set ------------------------')
# for dev set
y_pred_dev = model.predict(model_feat.predict(X_dev))
print_confusion_matrix_and_save(y_dev, np.argmax(y_pred_dev, axis=1), name='-ann-dev', fp=fp)
print_performance_metrics(y_dev, np.argmax(y_pred_dev, axis=1))

print('------------------------ test set ------------------------')
# for test set
y_pred_test = model.predict(model_feat.predict(X_test))
print_confusion_matrix_and_save(y_test, np.argmax(y_pred_test, axis=1),name='-ann-test', fp=fp)
print_performance_metrics(y_test, np.argmax(y_pred_test, axis=1))

plot_model_accuracy_loss(history)