from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):
    
    def build_analyzer(self, stemmer=None):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        
        if stemmer is None:
            stemmer = SnowballStemmer('english')
        
        return lambda text: (stemmer.stem(w) for w in analyzer(text))


def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass


if __name__ != '__main__':
    import warnings
    
    warnings.filterwarnings('ignore')
    tensorflow_shutup()

