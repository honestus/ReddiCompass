from ReddiCompass import default_config
_spacy_nlp = None
_word2vec_model = None
_stopwords = None
_tbwd = None
_vader_analyzer = None
_engspacysentiment_nlp = None
_detoxify = None
_frameaxis = None
_vad_extractor = None
_socialness_extractor = None
_nrc_extractor = None

def get_stopwords():
    from nltk.corpus import stopwords
    global _stopwords
    if _stopwords is None:
        _stopwords = stopwords.words('english')
    return _stopwords

def get_detokenizer():
    from nltk.tokenize import TreebankWordDetokenizer
    global _tbwd
    if _tbwd is None:
        _tbwd = TreebankWordDetokenizer()
    return _tbwd


def get_spacy_nlp():
    import spacy
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

def __load_word2vec_model__(model_path):
    from gensim.models import KeyedVectors
    from pathlib import Path
    from ReddiCompass.download_utils import download_from_huggingface
        
    model_path = Path(model_path)
    if not model_path.exists():
        download_from_huggingface(huggingface_repo='honestus/twitter_word2vec', filename='word2vec_twitter_model.bin' , saving_path=model_path)
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
    return word2vec_model
        


def get_frameaxis_instance():
    from ReddiCompass.Moral_Foundations_Frameaxis.frameAxis import FrameAxis
    global _word2vec_model
    global _frameaxis
    if _frameaxis is None:
        if _word2vec_model is None:
            _word2vec_model = __load_word2vec_model__(model_path = default_config.word2vec_model_filename)        
        _frameaxis = FrameAxis(mfd="emfd",w2v_model=_word2vec_model)
    return _frameaxis

def get_detoxify_instance():
    from detoxify import Detoxify
    global _detoxify
    if _detoxify is None:
        _detoxify=Detoxify('unbiased')
    return _detoxify


def get_engspacysentiment_nlp():
    import eng_spacysentiment
    global _engspacysentiment_nlp
    if _engspacysentiment_nlp is None:
        _engspacysentiment_nlp = eng_spacysentiment.load()
    return _engspacysentiment_nlp

def get_vader_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer
    
    
def get_socialness_extractor():
    from ReddiCompass.text_processing.vad_socialness_scoring import SocialnessCalculator
    global _socialness_extractor
    if _socialness_extractor is None:
        _socialness_extractor = SocialnessCalculator(str(default_config.social_lexicon_filename), min_max=(0,1), expand_lexicon=True, limit_expansion=True)
    return _socialness_extractor
    
    
    
def get_vad_extractor():
    from ReddiCompass.text_processing.vad_socialness_scoring import NRCVad
    global _vad_extractor
    if _vad_extractor is None:
        _vad_extractor = NRCVad(str(default_config.vad_lexicon_filename), expand_lexicon=True, limit_expansion=True, min_max=(-1,1))
    return _vad_extractor
    
    
def get_nrclex_extractor():
    from nrclex import NRCLex
    global _nrc_extractor
    if _nrc_extractor is None:
        _nrc_extractor = NRCLex(str(default_config.nrc_lexicon_filename))
    return _nrc_extractor