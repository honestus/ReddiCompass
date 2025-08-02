import pytest, sys, random
sys.path.append('../')
sys.path.append('../features_extraction_and_classification/')
sys.path.append('../../../../tesi/code/Moral_Foundation_FrameAxis/')

import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
import features_extraction_and_classification.resume_utils as resume_utils
#from text_processing.textractor import TexTractor
from features_extraction_and_classification.main import train  # Replace with actual import
from sklearn.svm import LinearSVC

def do_nothing(tokens: list[str]) -> list[str]:
    """ Returns the same tokens in input, useful to avoid that tokenizer splits an already tokenized text """
    return tokens
def sample_data():
    texts = ["This is a sentence", "Another one here", "Yet another"]
    categories = [0, 1, 0]
    return texts, categories


# 1. Minimal call
def test_train_minimal():
    texts, cats = sample_data()
    output = train(texts=texts, categories=cats, save=False)
    assert isinstance(output, LinearSVC)



# 2. extract_tfidf=False should ignore tfidf-related kwargs
def test_train_extract_tfidf_false_ignores_kwargs():
    texts, cats = sample_data()
    output = train(
        texts=texts,
        categories=cats,
        extract_tfidf=False,
        ngram_range=(1, 3),
        top_k=50,
        save=False
    )
    assert isinstance(output, LinearSVC)


# 3. If tfidf_extractor is passed, warn if ngram_range/top_k are also set
def test_train_custom_tfidf_extractor_with_conflicting_kwargs(monkeypatch):
    texts, cats = sample_data()
    tfidf = TfidfVectorizer(tokenizer=do_nothing, preprocessor=do_nothing)

    with pytest.warns(UserWarning, match="ignored"):
        output = train(
            texts=texts,
            categories=cats,
            tfidf_extractor=tfidf,
            ngram_range=(1, 2),
            top_k=30,
            save=False
        )
    assert isinstance(output, LinearSVC)


# 4. Handle top_k=False
def test_train_top_k_false_valid():
    texts, cats = sample_data()
    output = train(
        texts=texts,
        categories=cats,
        extract_tfidf=True,
        top_k=False,
        save=False
    )
    assert isinstance(output, LinearSVC)


# 5. Invalid input: mismatched lengths
def test_train_mismatched_lengths():
    texts = ["doc1", "doc2"]
    cats = [0]  # mismatched

    with pytest.raises(ValueError):
        train(texts=texts, categories=cats, save=False)


# 6. extract_tfidf=True and no categories + top_k → should raise error
def test_train_top_k_requires_categories():
    texts, cats = sample_data()
    
    with pytest.raises(ValueError):
        output = train(
            texts=texts,
            categories=None,
            extract_tfidf=True,
            top_k=20,
            save=False
        )

tmp_path = '../../data/rundata/model/tmp/'
# 7. Saving test with mocked saving_directory
def test_train_with_saving_directory(tmp_path=tmp_path):
    texts, cats = sample_data()
    saving_dir = Path(tmp_path)
    #saving_dir.mkdir()

    output = train(
        texts=texts,
        categories=cats,
        save=True,
        saving_directory=str(saving_dir)
    )
    assert saving_dir.exists()
    # Optional: assert saved files exist

def test_train_with_saving_directory_no_save(tmp_path=Path(tmp_path)):
    texts, cats = sample_data()
    saving_dir = tmp_path / "/not_exist/"
    #saving_dir.mkdir()

    output = train(
        texts=texts,
        categories=cats,
        save=False,
        saving_directory=str(saving_dir)
    )
    assert not saving_dir.exists()
    # Optional: assert saved files exist

# 8. Resume_dir should skip training if model already exists
def test_train_with_resume(monkeypatch, tmp_path=Path(tmp_path)):
    texts, cats = sample_data()
    resume_dir = tmp_path / "resume_train"
    resume_dir.mkdir()
    # Simulate model.joblib as if training was done
    (resume_dir / "model.joblib").touch()

    monkeypatch.setattr("resume_utils.is_model_train_finished", lambda dir: True)

    output = train(
        texts=texts,
        categories=cats,
        resume_dir=str(resume_dir),
        save=True
    )
    assert isinstance(output, LinearSVC)


# 9. Invalid ngram_range
def test_train_invalid_ngram_range():
    texts, cats = sample_data()
    with pytest.raises(ValueError):
        train(
            texts=texts,
            categories=cats,
            extract_tfidf=True,
            ngram_range=(1, "a"),
            save=False
        )


# 10. extract_tfidf with default extractor
def test_train_with_default_tfidf():
    texts, cats = sample_data()
    output = train(
        texts=texts,
        categories=cats,
        extract_tfidf=True,
        save=False
    )
    assert isinstance(output, LinearSVC)
