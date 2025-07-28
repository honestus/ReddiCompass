from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from features_extraction_and_classification.tfidf_utils import get_default_tfidf_extractor, get_ngram_topk_from_tfidf_extractor
from features_extraction_and_classification.feature_extraction import extract_features, _extract_features_in_batches
from features_extraction_and_classification.validate_utils import to_resume_flag
from features_extraction_and_classification.resume_utils import validate_meta_file, resume_extract_features
from text_processing.text_filtering_utils import get_nonstopwords_tokens
from features_extraction_and_classification import io_utils as io_utils

# Custom est
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, extract_tfidf=True, top_k=None, ngram_range=None):
        self.extract_tfidf = extract_tfidf
        tfidf_extractor = None 
        if self.extract_tfidf:
            tfidf_extractor = get_default_tfidf_extractor(ngram_range=ngram_range, top_k=top_k)
            ngram_range, top_k = get_ngram_topk_from_tfidf_extractor(tfidf_extractor)
        object.__setattr__(self, 'tfidf_extractor', tfidf_extractor)
        object.__setattr__(self, 'ngram_range', ngram_range)
        object.__setattr__(self, 'top_k', top_k)
        
    def fit(self, X, y):
        if self.extract_tfidf:
            X_tokens = [get_nonstopwords_tokens(x) for x in list(X)]
            self.tfidf_extractor.fit(X_tokens, y)
        self.fitted_ = True
        return self
        
        if not (to_resume:=to_resume_flag(resume_dir)):
        ###handling saving parameters
            if X is None or y is None:
                raise ValueError('Must set x and y to fit a new Feature Extractor')
            save, saving_directory = self.validate_saving_params(saving_directory=saving_directory)
        else:
            meta_resume_dict = validate_meta_file(resume_dir)
            object.__setattr__(self, 'extract_tfidf', meta_resume_dict[resume_utils.TFIDF_BOOL_ATTRIBUTE])
            object.__setattr__(self, 'tfidf_extractor', meta_resume_dict[resume_utils.TFIDF_EXTRACTOR_ATTRIBUTE])
            object.__setattr__(self, 'ngram_range', meta_resume_dict[resume_utils.TFIDF_BOOL_ATTRIBUTE])
            object.__setattr__(self, 'top_k', meta_resume_dict[resume_utils.TFIDF_BOOL_ATTRIBUTE])
            save, saving_directory = True, resume_dir
            
        object.__setattr__(self, '_curr_saving_directory', saving_directory)
        object.__setattr__(self, '_cached_features', None) ##parameter to pass to transform
        
        ##just extracting features if extracting tfidf, otherwise delegating it to transform
        if self.extract_tfidf:
            top_k = get_ngram_topk_from_tfidf_extractor(self.tfidf_extractor)[1]
            if not top_k:
                y = None ###y only needed if we are using top-k features selection for tfidf, otherwise setting it to None to avoid any error 
        ### feature extraction   
        object.__setattr__(self, '_cached_features', 
                           extract_features(texts=X, categories=y, 
                                            extract_tfidf=self.extract_tfidf, fit_tfidf=self.extract_tfidf, 
                                            tfidf_extractor=self.tfidf_extractor, batch_size=batch_size,
                                            save=save, saving_directory=saving_directory, resume_dir=resume_dir).values
                          )                
        
        return self

    def transform(self, X, saving_directory=None, batch_size=-1, resume_dir=None, **kwargs):
        

        if to_resume_flag(resume_dir): 
            #if 'feature_extractor' not in kwargs:
            #    raise TypeError("'feature_extractor' parameter is needed to properly restore features")
            save=True
            curr_features = resume_extract_features(resume_dir)

        else:
            if batch_size==-1:
                batch_size=len(X)            
            # Extracting features from texts by using current FeatureExtractor parameters
            curr_features = _extract_features_in_batches(
                texts=X, 
                extract_tfidf=self.extract_tfidf, 
                tfidf_extractor=self.tfidf_extractor, 
                fit_tfidf=False, 
                batch_size=batch_size, 
                saving_directory=saving_directory)

        return curr_features#.values

    def __setattr__(self, name, value):
        if name in ('tfidf_extractor', 'ngram_range', 'top_k', '_cached_features', '_saving_directory'):
            raise AttributeError(f"Cannot manually set '{name}'. It is read-only.")
        if name in ('ngram_range', 'top_k'):
            warnings.warn(f"Setting different {name} doesn't change the tfidf parameters of this current FeatureExtractor. Initialize a new Feature Extractor to use different {name}")
        super().__setattr__(name, value)

    def save(self, saving_directory):
        return io_utils.save_model(obj=self, filename_path=saving_directory+f'/{io_utils.FEATURE_EXTRACTOR_FILENAME}', validate_input=True)


class SavingPipeline(Pipeline):
    def __init__(self, extractor: FeatureExtractor, scaler: BaseEstimator, clf: BaseEstimator):
        steps = [
        ('features_extractor', extractor),
        ('features_scaler', scaler),
        ('classifier', clf)
    ]
        super().__init__(steps=steps)
        self.extractor = None
        self.scaler = None
        self.clf = None
        object.__setattr__(self, 'saving_directory', None) 
        
    def predict(self, X):
        X_feats = self['features_extractor'].transform(X)
        X_feats_scaled = self['features_scaler'].transform(X_feats.reindex(self['features_scaler'].get_feature_names_out(), axis=1))
        predictions = self['classifier'].predict(X_feats_scaled)
        return predictions
    
    def __setattr__(self, name, value):
        if name=='saving_directory':
            raise AttributeError(f"Cannot manually set '{name}'. It is read-only.")
        super().__setattr__(name, value)
        
    def _handle_saving_directory(self, save, saving_directory, funct):
        if not save:
            saving_directory = None
            return saving_directory
        
        if saving_directory in [True,1, False, 0 , None]:
            if funct=='fit':
                def_path = io_utils.DEFAULT_MODELS_PATH
            elif funct=='predict':
                saving_directory = curr_sav if (curr_sav:=self.saving_directory) is not None else io_utils.DEFAULT_DATA_PATH # using self.saving_directory IF current saving_directory is not chosen
                def_path = str(saving_directory)+'/'+io_utils.PREDICTIONS_DIR
            else: 
                raise NotImplementedError(f'Cannot handle function: {funct}')
            
            saving_directory = io_utils.__get_new_directory__(parent_dir=def_path)  
        else:
            saving_directory = io_utils.__validate_new_dir__(new_dir=saving_directory, force_to_default_path=False, funct='train')

        return str(saving_directory)