import pytest

def test_train_pipeline_basic_fit():
    texts = ["This is a test", "Another test"]
    categories = [0, 1]
    pipeline = train_pipeline(texts, categories, extract_tfidf=True)
    
    assert isinstance(pipeline, SavingPipeline)
    assert hasattr(pipeline, "clf")
    assert pipeline.clf.__class__.__name__ == "LinearSVC"


def test_train_pipeline_basic_fit():
    texts = ["This is a test", "Another test"]
    categories = [0, 1]
    pipeline = train_pipeline(texts, categories, extract_tfidf=True)
    
    assert isinstance(pipeline, SavingPipeline)
    assert hasattr(pipeline, "clf")
    assert pipeline.clf.__class__.__name__ == "LinearSVC"


def test_train_pipeline_top_k_effect():
    texts = ["apple banana", "banana orange", "apple orange banana"]
    categories = [0, 1, 0]
    
    pipeline = train_pipeline(texts, categories, top_k=5)
    tfidf_pipeline = pipeline.named_steps['FeatureExtractor'].tfidf_extractor
    
    select_k = tfidf_pipeline.named_steps['kbest']
    assert select_k.k == 5



def test_train_pipeline_mismatched_lengths():
    texts = ["text one", "text two"]
    categories = [0]  # Mismatched length
    
    with pytest.raises(ValueError):
        train_pipeline(texts, categories)


def test_predict_pipeline_output_shape():
    texts = ["sample text", "another sample"]
    categories = [0, 1]
    
    pipeline = train_pipeline(texts, categories)
    predictions = predict_pipeline(texts, pipeline)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == len(texts)


import pytest

def test_predict_pipeline_invalid_type():
    texts = 12345  # Invalid type
    categories = [0, 1]
    
    pipeline = train_pipeline(["valid text", "another"], categories)
    
    with pytest.raises(TypeError):
        predict_pipeline(texts, pipeline)


