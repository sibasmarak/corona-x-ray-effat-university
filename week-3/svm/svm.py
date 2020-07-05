# SVM
svm = SVC(kernel='rbf', random_state=0) 
svm.fit(feat_train, np.argmax(y_train, axis=1))

print('------------------------ train set ------------------------')
# for train set
y_pred_train = svm.predict(feat_train)
print('Score train: ', np.round(svm.score(feat_train, np.argmax(y_train, axis=1)),4))
print_confusion_matrix_and_save(y_train, y_pred_train, name='-svm-train', fp=fp)
print_performance_metrics(y_train, y_pred_train)

print('------------------------ validation set ------------------------')
# for dev set
feat_dev = model_feat.predict(X_dev)
y_pred_dev = svm.predict(feat_dev)
print('Score validation: ', np.round(svm.score(feat_dev,np.argmax(y_dev, axis=1)),4))
print_confusion_matrix_and_save(y_dev, y_pred_dev, name='-svm-dev', fp=fp)
print_performance_metrics(y_dev, y_pred_dev)

print('------------------------ test set ------------------------')
# for test set
feat_test = model_feat.predict(X_test)
y_pred_test = svm.predict(feat_test)
print('Score test: ', np.round(svm.score(feat_test,np.argmax(y_test, axis=1)),4))
print_confusion_matrix_and_save(y_test, y_pred_test,name='-svm-test', fp=fp)
print_performance_metrics(y_test, y_pred_test)