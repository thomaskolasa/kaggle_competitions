"""
Creating my first Titanic submission. In this model, I predict everybody dies. Turns out, 63% of people died.
    ___o .--.
   /___| |OO|
  /'   |_|  |_
       (_    _)
       | |   \
       | |oo_/
"""

import pandas as pd

# Load the test data
df = pd.read_csv('test.csv')

# Remove all columns except for PassengerId
df = df[['PassengerId']]

# Predict that everybody dies
df['Survived'] = 0

# Save the file for submission
df.to_csv('submissions/model1_all_died.csv', index=False)
