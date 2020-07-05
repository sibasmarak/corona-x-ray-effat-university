# RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(feat_train, np.argmax(y_train, axis=1))

print('------------------------ train set ------------------------')
# for train set
y_pred_train = rf.predict(feat_train)
print('Score train: ', np.round(rf.score(feat_train, np.argmax(y_train, axis=1)),4))
print_confusion_matrix_and_save(y_train, y_pred_train, name='-rf-train', fp=fp)
print_performance_metrics(y_train, y_pred_train)

print('------------------------ validation set ------------------------')
# for dev set
feat_dev = model_feat.predict(X_dev)
y_pred_dev = rf.predict(feat_dev)
print('Score validation: ', np.round(rf.score(feat_dev,np.argmax(y_dev, axis=1)),4))
print_confusion_matrix_and_save(y_dev, y_pred_dev, name='-rf-dev', fp=fp)
print_performance_metrics(y_dev, y_pred_dev)

print('------------------------ test set ------------------------')
# for test set
feat_test = model_feat.predict(X_test)
y_pred_test = rf.predict(feat_test)
print('Score test: ', np.round(rf.score(feat_test,np.argmax(y_test, axis=1)),4))
print_confusion_matrix_and_save(y_test, y_pred_test,name='-rf-test', fp=fp)
print_performance_metrics(y_test, y_pred_test)