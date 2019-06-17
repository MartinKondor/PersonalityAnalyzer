import gc
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

sys.path.append('..')  # Adding the upper directory to the python path
import utils


df = pd.read_csv('../data/mbti_1.csv')
X, y = [], []

print('Spliting posts')
for xi, yi in zip(df['posts'].values, df['type'].values):
    posts = xi.split('|||')
    X.extend(posts)
    
    for _ in range(len(posts)):
        y.append(yi)

del df
classes = {}

print('Count classes')
for c in y:
    if c in classes.keys():
        classes[c] += 1
    else:
        classes[c] = 0

        
print('Set a limit for each class')
class_limit = 1000
class_counter = {k: 0 for k in classes.keys()}
del classes

X_new, y_new = [], []

for xi, yi in zip(X, y):
    if class_counter[yi] > class_limit:
        continue

    class_counter[yi] += 1
    X_new.append(xi)
    y_new.append(yi)
    
    
print('Swap variables')
del X, y, class_limit, class_counter
X_new, y_new = np.array(X_new), np.array(y_new)

print('Encode y values')
class_encoder = OneHotEncoder()
y_encoded = class_encoder.fit_transform(y_new.reshape(-1, 1)).toarray()
del y_new

print('Data selection')
gc.collect()
X_train, X_test, y_train, y_test = train_test_split(X_new, y_encoded, test_size=.1)
del X_new, y_encoded

print('Preprocess X values')
vectorizer = utils.StemmedTfidfVectorizer(min_df=1, stop_words='english')
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

del np, pd, OneHotEncoder, train_test_split, utils
gc.collect()

# Variables in memory here: X_train, X_test, y_train, y_test, vectorizer, class_encoder
print('Build model')
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam')

print('Train model')
history = model.fit(x=X_train, y=y_train, verbose=1, epochs=1, shuffle=True)

print('Save model and preprocessors')
from sklearn.externals import joblib
model.save('../trained/new/model.h5')
joblib.dump(class_encoder, '../trained/new/class_encoder.pkl')
joblib.dump(vectorizer, '../trained/new/vectorizer.pkl')

print('Test model')
train_score = model.evaluate(X_train, y_train, verbose=1)
print('Train score', train_score)
test_score = model.evaluate(X_test, y_test, verbose=1)
print('Test score', test_score)


print('Done.')
