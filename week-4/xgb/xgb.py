# use XGBClassifier 
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(feat_train, np.argmax(y_train, axis=1))

print('------------------------ train set ------------------------')
# fit in the train set
# feat_train = model_feat.predict(X_train)
model.fit(feat_train, np.argmax(y_train, axis=1))
y_pred_train = model.predict(feat_train)
print('Score train: ', np.round(model.score(feat_train, np.argmax(y_train, axis=1)),4))
print_confusion_matrix_and_save(y_train, y_pred_train, name='-xgb-train', fp=fp)
print_performance_metrics(y_train, y_pred_train)

print('------------------------ validation set ------------------------')
# for dev set
feat_dev = model_feat.predict(X_dev)
y_pred_dev = model.predict(feat_dev)
print('Score validation: ', np.round(model.score(feat_dev,np.argmax(y_dev, axis=1)),4))
print_confusion_matrix_and_save(y_dev, y_pred_dev, name='-xgb-dev', fp=fp)
print_performance_metrics(y_dev, y_pred_dev)

print('------------------------ test set ------------------------')
# for test set
feat_test = model_feat.predict(X_test)
y_pred_test = model.predict(feat_test)
print('Score test: ', np.round(model.score(feat_test,np.argmax(y_test, axis=1)),4))
print_confusion_matrix_and_save(y_test, y_pred_test,name='-xgb-test', fp=fp)
print_performance_metrics(y_test, y_pred_test)