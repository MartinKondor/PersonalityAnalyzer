import gc
import sys

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

sys.path.append('..')  # Adding the upper directory to the python path
import tools as utils


CLASS_LIMIT = 250  # The number of measurements (posts) to choose from each classes
class_counter = {}
df = pd.read_csv('../data/mbti_1.csv')
X, y = [], []


for xi, yi in tqdm.tqdm(zip(df['posts'].values, df['type'].values)):
    posts = xi.split('|||')
    
    if yi in class_counter:
        if class_counter[yi] > CLASS_LIMIT:
            continue
        else:
            class_counter[yi] += len(posts)
    else:
        class_counter[yi] = len(posts)
        
    posts = [utils.prepare(p) for p in posts]
    X.extend(posts)

    for _ in range(len(posts)):
        y.append(yi)


del xi, yi, df, posts, _
X, y = np.array(X), np.array(y)

print(len(X), 'post')
print(len(np.unique(y)), 'classes')
print('Number of posts for each class:')

for class_name in class_counter:
    print(class_name, '\t', class_counter[class_name])

del class_name, class_counter, CLASS_LIMIT

print()
print('Example:')
print(X[-1])
print('Is:', y[-1])
print()
    
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.1)
del X, y

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
# del vectorizer

encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train[:, np.newaxis]).toarray()
y_test = encoder.transform(y_test[:, np.newaxis]).toarray()
# del encoder

del pd, tqdm, train_test_split, TfidfVectorizer, OneHotEncoder, utils
gc.collect()

print('Building and training the model ...')
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dense(16, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(x=X_train, y=y_train, verbose=1, epochs=20, shuffle=True)
print()

train_score = model.evaluate(X_train, y_train, verbose=1)
print('Train score:', train_score)
test_score = model.evaluate(X_test, y_test, verbose=1)
print('Test score:', test_score)

pred = model.predict(X_test[:1])[0]
pred[np.argmax(pred)] = 1
pred[pred != 1] = 0
pred = np.array(pred, dtype=np.uint8)

print()
print('Testing:')
print('Post:\n', vectorizer.inverse_transform(X_test[:1]))
print('Prediction:\n', encoder.inverse_transform([pred])[0][0])
print('Real:\n', encoder.inverse_transform(y_test[:1])[0][0])
