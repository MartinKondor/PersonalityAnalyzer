from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):
    
    def build_analyzer(self, stemmer=None):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        
        if stemmer is None:
            stemmer = SnowballStemmer('english')
        
        return lambda text: (stemmer.stem(w) for w in analyzer(text))
    