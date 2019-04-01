import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten


# loading the dataset
N = 1500  # number of elements to use from the dataset, because of high ram usage  
df = shuffle( pd.read_csv('../data/mbti-myers-briggs-personality-types.csv') )[:N]


# preprocessing
type_encoder = OneHotEncoder()
y = type_encoder.fit_transform( np.array([df['type'].values]).T ).toarray()

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english')
X = vectorizer.fit_transform(df['posts'].values).toarray()


# model selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# building the model
model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adagrad')

history = model.fit(x=X_train, y=y_train, verbose=1, epochs=22, shuffle=True)

train_score = model.evaluate(X_train, y_train, verbose=0)
print('Train score', train_score)
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Test score', test_score)

model.save('../trained/temp.h5')
