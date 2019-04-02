import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from nltk.stem import SnowballStemmer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tools import *


# loading the dataset
N = 3500  # number of elements to use from the dataset, because of high ram usage  
df = pd.read_csv('datasets/mbti-myers-briggs-personality-type-dataset/mbti_1.csv')[:N]


# preprocessing
print('Preprocessing ...', end=' ')
type_encoder = OneHotEncoder()
y = type_encoder.fit_transform( np.array([df['type'].values]).T ).toarray()

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english')
X = vectorizer.fit_transform(df['posts'].values).toarray()

joblib.dump(type_encoder, 'trained/type_encoder.pkl')
joblib.dump(vectorizer, 'trained/vectorizer.pkl')
print('finished')

# model selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# building the model
model = Sequential()
model.add(Dense(250, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adagrad')

history = model.fit(x=X_train, y=y_train, verbose=1, epochs=50, shuffle=True)
print()

train_score = model.evaluate(X_train, y_train, verbose=1)
print('Train score', train_score)
test_score = model.evaluate(X_test, y_test, verbose=1)
print('Test score', test_score)

model.save('trained/model.temp.h5')
