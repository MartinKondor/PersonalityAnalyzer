import sys
import joblib
from joblib import load as load_prep
from keras.models import load_model
from tools import StemmedTfidfVectorizer, tensorflow_shutup
sys.modules['sklearn.externals.joblib'] = joblib  # needed for saved models


class MBType(object):

    def __init__(self):
        self.model = load_model('trained/model.h5')
        self.vectorizer = load_prep('trained/vectorizer.pkl')
        self.type_encoder = load_prep('trained/type_encoder.pkl')


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
    def test_from_data():
        from pandas import read_csv

        predictor = MBType()
        df = read_csv('data/mbti-myers-briggs-personality-types.csv')
        sample = df['posts'].iloc[-1].split('|||')[-1]

        print('Target is', df['type'].iloc[-1])
        print('Predicted is', predictor.predict(sample))
        print('Sample:\n', sample)


    print("Loading the model...", end="")
    predictor = MBType()
    print("done!")

    sample = input("Write something about yourself: ")
    print("Predicted MBType is:", predictor.predict(sample))
