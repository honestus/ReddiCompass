import pytest, sys, random
sys.path.append('../')
sys.path.append('../features_extraction_and_classification/')

import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from features_extraction_and_classification.feature_extraction import extract_features
import features_extraction_and_classification.resume_utils as resume_utils
#from text_processing.textractor import TexTractor

new_train = pd.read_parquet('../../data/rundata/train.parquet').sample(50)
sys.path.append('../../../../tesi/code/Moral_Foundation_FrameAxis/')
def do_nothing(tokens: list[str]) -> list[str]:
    """ Returns the same tokens in input, useful to avoid that tokenizer splits an already tokenized text """
    return tokens
@pytest.fixture
def sample_texts():
    curr_int = random.randrange(3)
    sample_texts = new_train.sample(5).text
    return sample_texts if curr_int==0 else sample_texts.values if curr_int==1 else sample_texts.tolist()

@pytest.fixture
def tfidf_pipeline():
    return Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), tokenizer=do_nothing, preprocessor=do_nothing))
    ])

def test_extract_features_minimal(sample_texts):
    features = extract_features(sample_texts, extract_tfidf=False)
    assert isinstance(features, pd.DataFrame)
    assert not features.empty


def test_extract_features_single_string():
    features = extract_features("sample text", extract_tfidf=False)
    assert isinstance(features, pd.DataFrame)
    assert not features.empty

def test_extract_features_invalid_text_type():
    with pytest.raises(TypeError):
        extract_features(12345, extract_tfidf=False)

def test_extract_features_invalid_tfidf_extractor(sample_texts):
    with pytest.raises(TypeError):
        extract_features(sample_texts, extract_tfidf=True, tfidf_extractor="not_a_model")

def test_extract_features_with_tfidf(sample_texts, tfidf_pipeline):
    features = extract_features(sample_texts, extract_tfidf=True, tfidf_extractor=tfidf_pipeline, fit_tfidf=True)
    assert isinstance(features, pd.DataFrame)

def test_extract_features_with_kwargs(sample_texts):
    features = extract_features(
        sample_texts,
        extract_tfidf=True,
        tfidf_extractor=None,
        fit_tfidf=True,
        categories=["A", "B", "A", "A", "B"],
        ngram_range=(1, 2),
        top_k=10
    )
    assert isinstance(features, pd.DataFrame)

def test_extract_features_fit_tfidf_no_categories(sample_texts):
    with pytest.raises(ValueError):
        extract_features(sample_texts, extract_tfidf=True, fit_tfidf=True)


import tempfile

def test_extract_features_with_saving(sample_texts):
    tmpdir = '../../data/rundata/features/tmp/'
    features = extract_features(
        sample_texts,
        extract_tfidf=True,
        fit_tfidf=True,
        categories=["x", "y", "x", "a", "a"],
        save=True,
        saving_directory=tmpdir
    )
    assert (Path(tmpdir) / "features.parquet").exists()

def test_extract_features_invalid_ngram_range(sample_texts):
    with pytest.raises(ValueError):
        extract_features(sample_texts, extract_tfidf=True, ngram_range="bad_range")


def test_extract_features_resume_dir_behavior(monkeypatch, sample_texts):
    def mock_resume(*args, **kwargs):
        return pd.DataFrame({"feat": [1, 2]})
    monkeypatch.setattr("resume_utils.resume_extract_features", mock_resume)
    features = extract_features(sample_texts, extract_tfidf=True, resume_dir='../../data/rundata/features/tmp/')
    assert isinstance(features, pd.DataFrame)