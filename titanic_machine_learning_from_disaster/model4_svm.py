"""
Using Support Vector Machines on the titanic data
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.grid_search import GridSearchCV

# Calling my training and test data processing from my previous models, 
# returning train_df and test_df
execfile("processing.py")

##### Setting up train test split on training data frame #####

x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Survived', axis=1),train_df.Survived,
                                                    test_size=0.2, random_state=0)

clf = svm.SVC()
clf.fit(x_train, y_train)

cross_val_score(clf, x_train, y_train, cv=10).mean()
# 0.609590319696

cross_val_score(clf, x_test, y_test, cv=10).mean()
# 0.614705882353

# Try various kernels (without tuning of C or gamma parameters for now)
results = []
for kernel in ['linear', 'poly']:
    clf = svm.SVC(kernel=kernel, degree=2) # Polynomial kernels can take longer to run, so start with d=2
    clf.fit(x_train, y_train)
    # Since a polynomial kernel takes longer to run, I will only run a cross validation on others
    if kernel == 'poly':
        results.append([kernel, clf.score(x_test, y_test)])
    else:
        results.append([kernel, cross_val_score(clf, x_test, y_test, cv=5).mean()])

    # create Kaggle submission file 
    submission = test_df[['PassengerId']]
    submission['Survived'] = clf.predict(test_df)
    submission.to_csv('submissions/model4_svm_'+kernel+'.csv', index=False)

print results
#[['linear', 0.75412698412698409], ['poly', 0.79329608938547491]]  

# linear SVC gets 0.77033 accuracy on Kaggle
# polynomial of degree 2 gets 0.74163 accuracy on Kaggle 

# A polynomial kernel of degree 2 performs worse than a linear kernel, even though its accuracy 
# on the training set was 0.793 compared to the lack of kernel's accuracy of 0.754. This means
# that the polynomial kernel was overfitting the training data. It is therefore unlikely an 
# even higher-degree polynomial kernel would do better. 

# So far, the Support Vector Classifier with a linear kernel and default parameters did best (0.770) 
# on the leaderboard, beating the Random Forest Classifier's accuracy (0.751).
