"""
Using XGBoost
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split, cross_val_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV


# Call train and test data processing from before
execfile('processing.py')
# returns train_df and test_df

###### Create Cross-Validation Dataset ######
x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Survived',axis=1),train_df.Survived,
                                                     test_size=.3, random_state=0)

##### Run Gradient Boosting using XGBoost #####
# First see how it performs without any tuning
xgb1 = xgb.XGBClassifier(learning_rate=.05,
                        max_depth=3,
                        n_estimators=1000,
                        min_child_weight=1,
                        gamma=0,
                        subsample=.8,
                        colsample_bytree=.8,
                        objective='binary:logistic'
                        )

xgb1.fit(x_train, y_train)

cross_val_score(xgb1, x_train, y_train, cv=10).mean()
# 0.788110439068
cross_val_score(xgb1, x_test, y_test, cv=10).mean()
# 0.779772079772

feat_imp = pd.Series(xgb1.booster().get_fscore())
feat_imp
# Age              1189
# Fare             1484
# Parch             125
# PassengerId      2045
# Pclass            195
# SibSp             171
# cabin_boolean      53
# embarked_C         88
# embarked_Q         59
# embarked_S         92
# sex_boolean       218

submission = test_df[['PassengerId']]
submission['Survived'] = xgb1.predict(test_df)
submission.to_csv('submissions/model7_xgboost1.csv', index=False)
# Kaggle score of 0.73206

###### Tuning XGBoost ######
# start by tuning the max depth of a tree and the number of boosted trees to fit
xgb2 = xgb.XGBClassifier(learning_rate=.05,
                         subsample=.8,
                         colsample_bytree=.8)
clf = GridSearchCV(xgb2,
                  {'max_depth': range(2,10,2),
                   'n_estimators': range(50,200,50)},
                   verbose=1)
clf.fit(x_train, y_train)
print clf.best_score_, clf.best_params_
# Fitting 3 folds for each of 12 candidates, totalling 36 fits
# [Parallel(n_jobs=1)]: Done  36 out of  36 | elapsed:    1.1s finished
# 0.818619582665 {'n_estimators': 150, 'max_depth': 2}

xgb2 = xgb.XGBClassifier(learning_rate=.05,
                         max_depth=2,
                         n_estimators=150,
                         subsample=.8,
                         colsample_bytree=.8)
clf = GridSearchCV(xgb2, {'gamma': np.linspace(0, 0.5, 6)}, verbose=1)
clf.fit(x_train, y_train)
print clf.best_score_, clf.best_params_
# Fitting 3 folds for each of 6 candidates, totalling 18 fits
# [Parallel(n_jobs=1)]: Done  18 out of  18 | elapsed:    0.4s finished
# 0.818619582665 {'gamma': 0.0}

xgb2 = xgb.XGBClassifier(learning_rate=.05,
                         max_depth=2,
                         n_estimators=150,
                         gamma=0)
clf = GridSearchCV(xgb2,
                  {'subsample': np.linspace(.6, 1, 5),
                   'colsample_bytree': np.linspace(.6, 1, 5)},
                  verbose=1)
clf.fit(x_train, y_train)
print clf.best_score_, clf.best_params_
# Fitting 3 folds for each of 25 candidates, totalling 75 fits
# [Parallel(n_jobs=1)]: Done  49 tasks       | elapsed:    1.0s
# [Parallel(n_jobs=1)]: Done  75 out of  75 | elapsed:    1.5s finished
# 0.828250401284 {'subsample': 0.59999999999999998, 'colsample_bytree': 1.0}

xgb2 = xgb.XGBClassifier(learning_rate=.05,
                         max_depth=2,
                         n_estimators=150,
                         gamma=0,
                         subsample=.6,
                         colsample_bytree=1)
clf = GridSearchCV(xgb2,
                  {'reg_alpha': np.logspace(-5, 3, 9)},
                  verbose=1)
clf.fit(x_train, y_train)
print clf.best_score_, clf.best_params_
# Fitting 3 folds for each of 9 candidates, totalling 27 fits
# [Parallel(n_jobs=1)]: Done  27 out of  27 | elapsed:    0.6s finished
# 0.828250401284 {'reg_alpha': 1.0000000000000001e-05}

# let's give it a more detailed learning rate
xgb2 = xgb.XGBClassifier(learning_rate=.01,
                         max_depth=2,
                         n_estimators=150,
                         gamma=0,
                         subsample=.6,
                         colsample_bytree=1,
                         reg_alpha=.00001)
xgb2.fit(x_train, y_train)

print cross_val_score(xgb2, x_train, y_train, cv=10).mean()
# 0.805901977727
print cross_val_score(xgb2, x_test, y_test, cv=10).mean()
# 0.738888888889

submission2 = test_df[['PassengerId']]
submission2['Survived'] = xgb2.predict(test_df)
submission2.to_csv('submissions/model7_xgboost2.csv', index=False)
# Kaggle score of 0.77512

# How many observations changed between predicitons?
print len(test_df) - sum(xgb1.predict(test_df) == xgb2.predict(test_df))
# 77
# Tuning made quite a few changes to the predictions between the first and 
# second submissions. 18% of the predictions changed and the Kaggle score
# improved about 4%.
