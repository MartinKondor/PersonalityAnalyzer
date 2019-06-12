import gc
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.path.append('..')
import utils


df = pd.read_csv('../data/mbti_1.csv')


# Spliting posts
X, y = [], []

for xi, yi in zip(df['posts'].values, df['type'].values):
    posts = xi.split('|||')
    X.extend(posts)
    
    for _ in range(len(posts)):
        y.append(yi)
    
del df
gc.collect()


# Count classes
classes = {}

for c in y:
    if c in classes.keys():
        classes[c] += 1
    else:
        classes[c] = 0

        
# Set a limit for each class
class_limit = 2000
class_counter = {k: 0 for k in classes.keys()}

X_new, y_new = [], []

for xi, yi in zip(X, y):
    if class_counter[yi] > class_limit:
        continue

    class_counter[yi] += 1
    X_new.append(xi)
    y_new.append(yi)
    
# Swap variables
del X, y
X, y = np.array(X_new), np.array(y_new)
del X_new, y_new
gc.collect()


# Encode y values
class_encoder = OneHotEncoder()
y_encoded = class_encoder.fit_transform(y.reshape(-1, 1)).toarray()

del y
gc.collect()


# Preprocess X values
vectorizer = utils.StemmedTfidfVectorizer(min_df=1, stop_words='english')
X_vectorized = vectorizer.fit_transform(X).toarray()

del X
gc.collect()


# Build model
# Train model
# Test model
# Save model and preprocessors
