from sklearn.externals.joblib import load as load_prep
from keras.models import load_model
from tools import *


class MBType(object):

    def __init__(self):
        self.model = load_model('../trained/model.h5')
        self.vectorizer = load_prep('../trained/vectorizer.pkl')
        self.type_encoder = load_prep('../trained/type_encoder.pkl')


    def predict(self, X):
        """
        From an unprocessed string predict the class.
        """
        return self.type_encoder.inverse_transform(
                   self.model.predict(
                     self.vectorizer.transform([X]).toarray() 
                    )
                )[0][0]



if __name__ == '__main__':
    from pandas import read_csv

    predictor = MBType()
    df = read_csv('../data/mbti-myers-briggs-personality-types.csv')
    sample = df['posts'].iloc[-1].split('|||')[-1]

    print('Target is', df['type'].iloc[-1])
    print('Predicted is', predictor.predict(sample))
    print('Sample:\n', sample)

