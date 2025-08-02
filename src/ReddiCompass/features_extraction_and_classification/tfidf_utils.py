from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted as sklearn_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline



DEFAULT_NGRAM_RANGE = (1, 1)
DEFAULT_TOP_K = 20

def do_nothing(tokens: list[str]) -> list[str]:
    """ Returns the same tokens in input, useful to avoid that tokenizer splits an already tokenized text """
    return tokens

def custom_tokenizer_ngram_extractor(text: str or list[str], stopwords: list[str] =[], ngram_range: tuple[int,int] = DEFAULT_NGRAM_RANGE) -> list[str] :
    """ Only builds ngrams from consecutive tokens, by avoiding ngrams that come from following tokens after stopword removal
        E.G.: suppose text_tokens = ['tok1', 'tok_stopword', tok3'].
        After stopword removal we have ['tok1', 'tok3'] but when we build ngrams (n>1) we should avoid 'tok1 tok3', 
        as they never appear consecutively in the text.
    """
    if not isinstance(text, (list, np.ndarray, pd.Series)):
        words = text.split()
    else:
        words = text

    #words = [word for word in words if word not in stop_words]

    ngrams = []
    for i in range(len(words)):
        if (curr_w:=words[i]) not in stopwords:
            ngrams.append(curr_w)  # Unigram
            for j in range(*ngram_range[::-1],-1):
                if j+i>len(words):
                    continue
                curr_potential_ngram_words = words[i:j+i]
                #print(curr_w, ' ... ', curr_potential_ngram_words)
                if any([w in stopwords for w in curr_potential_ngram_words]):
                    continue
                else:
                    ngrams.append(' '.join(curr_potential_ngram_words) )  # Ngrams
    return ngrams


class Chi2ScorerWrapper(BaseEstimator):
    """Fake estimator, based on chi2, uses the avg of the top_k scores from chi2 as scoring method.
        Useful for cross validation of chi2 without using a classifier, to evaluate different tfidf combinations with cv.
        For example it might be used on different ngrams combinations and decide which is the most discriminative (ngram combination) 
        by the top20 ngrams scores."""
    def __init__(self, k=DEFAULT_TOP_K, score_func=chi2):
        self.k = k
        self.selector_ = SelectKBest(score_func=score_func, k=k)

    def fit(self, X, y):
        self.selector_.fit(X, y)
        return self

    def score(self, X, y):
        # La score è la media dei chi2 delle top k feature
        try:
            sklearn_is_fitted(self.selector_)
        except NotFittedError:
            self.fit(X, y)
        finally:
            scores = self.selector_.scores_
            support = self.selector_.get_support()
            return np.mean(scores[support])

    def __getattr__(self, attr):
        return getattr(self.selector_, attr)
        
        
        
def get_default_tfidf_extractor(ngram_range: tuple[int,int]=DEFAULT_NGRAM_RANGE, top_k: int=DEFAULT_TOP_K):
    """Builds pipeline made by CountVectorizer, TfidfTransformer and chi-2 for top tokens selection.
    If you want to build whole matrix without tokens selection, just use top_k=False.
    PLEASE CONSIDER THAT IF YOU BUILD THE WHOLE MATRIX, THE DIMENSION OF TF-IDF MATRIX WILL DOMINATE ALL THE OTHER FEATURES! USE IT WITH CAUTION
    """
    if ngram_range is None: ##mapping to default values
        ngram_range=DEFAULT_NGRAM_RANGE
    elif (not isinstance(ngram_range, tuple) or len(ngram_range)!= 2 or any(not isinstance(x, int) for x in ngram_range) or ngram_range[0]>ngram_range[1]):
        raise ValueError('Wrong ngram_range parameter')
        
    if top_k is None:
        top_k=DEFAULT_TOP_K
        
    if top_k is False: ###Pipeline without tokens selection
        #raise NotImplementedError('Pipeline without chi-square, might be interesting for different purposes')
        tfidf_extractor = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=do_nothing, preprocessor=do_nothing, ngram_range=ngram_range)),  # frequencies
        ('tfidf', TfidfTransformer()),  # tfidf
    ])
        return tfidf_extractor
    elif isinstance(top_k, int) and top_k>0:
        tfidf_extractor = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=do_nothing, preprocessor=do_nothing, ngram_range=ngram_range)),  # frequencies
        ('tfidf', TfidfTransformer()),  # tfidf
        ('kbest', SelectKBest(score_func=chi2, k=top_k)), #most discriminative tokens' selection
    ])
        return tfidf_extractor
        
    raise ValueError('wrong "top_k" in input. It must be a positive integer')
    
    
def get_ngram_topk_from_tfidf_extractor(tfidf_extractor):
    if isinstance(tfidf_extractor, Pipeline):
        ngram_range = tfidf_extractor['vectorizer'].ngram_range
        top_k = tfidf_extractor['kbest'].k if 'kbest' in tfidf_extractor.named_steps else False
    else:
        ngram_range = tfidf_extractor.ngram_range
        top_k=False
    return ngram_range, top_k