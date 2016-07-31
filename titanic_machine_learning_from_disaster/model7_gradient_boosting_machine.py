
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

cross_val_score(xgb1, x_train, y_train, cv=10)
# [ 0.796875    0.77777778  0.83870968  0.75806452  0.82258065  0.77419355
#   0.75806452  0.80645161  0.79032258  0.75806452]
cross_val_score(xgb1, x_train, y_train, cv=10).mean()
# 0.788110439068
cross_val_score(xgb1, x_test, y_test, cv=10)
# [ 0.77777778  0.92592593  0.77777778  0.77777778  0.81481481  0.74074074
#   0.81481481  0.62962963  0.80769231  0.73076923]
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
