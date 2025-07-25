import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator


def __check_x_y_shapes__(x, y):
    if not isinstance(y, (list, np.ndarray, pd.Series)):
        raise TypeError('texts must be a collection')
    if not isinstance(y, (list, np.ndarray, pd.Series)):
        raise TypeError('categories must be a collection')
    if len(x)!=len(y):
        raise ValueError('Texts and categories must be of the same size to train the models')
    return True

def __check_equals_index_values__(x_index, y_index, check_order=True):
    if check_order:
        return x_index.equals(y_index)
    return sorted(x_index)==sorted(y_index)
    

def validate_x_y_inputs(x, y, check_order=True, to_pandas_series = True, raise_errors_on_wrong_indexes=False):
    """
    Verifies if two variables (x and y) have the same index, 
    handling both Pandas collections (DataFrame, Series) and non-Pandas collections (lists, tuples).
    
    The function follows these rules:
    
    - If neither has an index (both are lists, tuples, etc.), return True.
    - If only one of them has an index (Pandas object), raise a warning and reset the other to the default index.
    - If both have indices, but the index names are different but values are the same, raise a warning and return True.
    - If both have indices with different values, raise an error.
    
    Args:
        x: Can be a Pandas DataFrame, Series, or any non-Pandas collection like a list or tuple.
        y: Can be a Pandas DataFrame, Series, or any non-Pandas collection like a list or tuple.
        
    Returns:
        bool: True if indices are the same (following the rules above), False otherwise.
    
    Raises:
        ValueError: if the indices have different values.
    """
    
    
    
    
    if not isinstance(x, (list, pd.Series, np.ndarray)) or not isinstance(y, (list, pd.Series, np.ndarray)):
            raise ValueError('Both texts and categories must be collections (list, pandas Series, numpy array)')
    
    if not __check_x_y_shapes__(x,y):
        raise ValueError('Texts and categories have different shapes.')
    # Case when neither x nor y has an index (i.e., both are not Pandas collections)
    if not isinstance(x, (pd.Series, pd.DataFrame)) and not isinstance(y, (pd.Series, pd.DataFrame)):
        if not to_pandas_series:
            return x, y  
        else:
            return pd.Series(x), pd.Series(y)
    
    # Case when only one of them is a Pandas object with an index
    if isinstance(x, (pd.Series, pd.DataFrame)) and not isinstance(y, (pd.Series, pd.DataFrame)):
        warnings.warn("Only the x collection has an index. Resetting both to pandas series with default index to avoid misbehaviours.", UserWarning)
        y = pd.Series(y)
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        return x, y
    
    if not isinstance(x, (pd.Series, pd.DataFrame)) and isinstance(y, (pd.Series, pd.DataFrame)):
        warnings.warn("Only the y collection has an index. Resetting both to pandas series with default index to avoid misbehaviours.", UserWarning)
        x = pd.Series(x)
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        return x, y
    
    # Case when both x and y are Pandas objects with indices
    if isinstance(x, (pd.Series, pd.DataFrame)) and isinstance(y, (pd.Series, pd.DataFrame)):
        if __check_equals_index_values__(x.index, y.index, check_order=check_order):
            if x.index.names != y.index.names:
                warn_msg = "Different index names for the x and y collections. Please ensure you are passing proper texts and categories!"
                if raise_errors_on_wrong_indexes:
                    raise ValueError(warn_msg)
                warnings.warn(warn_msg, UserWarning) 
        
        # If the indices are different, raise a ValueError
        if not x.index.equals(y.index):
            warn_msg = "The indices of x and y are different in values. Please ensure you are passing proper texts and categories!!!"
            if raise_errors_on_wrong_indexes:
                raise ValueError(warn_msg)
               
            warn_msg += "\n If you want to avoid any potential misbehaviours use 'raise_errors_on_wrong_indexes'=True"
            warnings.warn(warn_msg, UserWarning)
            x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        
    return x,y
    
  
def validate_text_input(text):
    from text_processing.textractor import TexTractor
    if isinstance(text, (str, TexTractor)):
        return [text]
    if isinstance(text, (list, np.ndarray, pd.Series)):
        if any(not isinstance(txt, (str, TexTractor)) for txt in text):
            raise TypeError("Input text must be either a string, a TexTractor instance, or a collection (list, array, pandas.Series) of such types")
        return text
    raise TypeError( "Input text must be either a string, a TexTractor instance, or a collection (list, array, pandas.Series) of such types")

  
    
def to_resume_flag(resume_dir):
    if resume_dir in [None, False, 0]:
        return False

    warn_msg = f"Resuming from '{resume_dir}': all other input parameters will be ignored."
    warnings.warn(warn_msg, UserWarning)
    return True



def validate_tfidf_user_inputs(extract_tfidf: bool, fit_tfidf: bool, tfidf_extractor: BaseEstimator, y: list|np.ndarray|pd.Series = None, ngram_range: tuple[int,int] = False, top_k: int = False, raise_errors_on_wrong_inputs: bool = False):
    """ This function should be called with the original input parameters passed by the user to the outer method (e.g., extract_features), before any default resolution or internal modification!
    It checks for potentially conflicting inputs (e.g., providing both a custom tfidf_extractor and ngram_range/top_k) and issues warnings if needed.
    """    
    if extract_tfidf is False:
        if fit_tfidf or isinstance(tfidf_extractor, BaseEstimator) or ngram_range or top_k or (y is not None):
            warn_msg = "extract_tfidf is False. All other tfidf params will be ignored."
            if raise_errors_on_wrong_inputs:
                raise ValueError(warn_msg)
            warnings.warn(warn_msg) 
            return
    if tfidf_extractor is not None and (ngram_range is not None or top_k is not None):
        warnings.warn("`ngram_range` and `top_k` are ignored because a custom tfidf_extractor was provided.")
        #ngram_range, top_k = False, False 

def validate_tfidf_parameters(extract_tfidf: bool, fit_tfidf: bool, tfidf_extractor: BaseEstimator, y: list|np.ndarray|pd.Series = None):
    from features_extraction_and_classification.tfidf_utils import get_ngram_topk_from_tfidf_extractor
    from sklearn.utils.validation import check_is_fitted

    if not extract_tfidf:
        return True
    if not isinstance(tfidf_extractor, BaseEstimator):
        raise TypeError('Wrong tfidf_extractor type in input. Must be a sklearn estimator')
    if not fit_tfidf:
        try:
            check_is_fitted(tfidf_extractor)
            return True
        except:
            raise ValueError("Must pass an already fitted tfidf_extractor if fit_tfidf is False")
    
    top_k = get_ngram_topk_from_tfidf_extractor(tfidf_extractor)[1]
    if y is None and (top_k is not False):
        raise ValueError("Fitting a tfidf extractor with supervised selection requires 'y' labels.")
    
def validate_batch_size(batch_size):
    if type(batch_size) not in [int, float] or round(batch_size,0) !=round(batch_size,1):
        warnings.warn(f'Wrong batch_size input: "{batch_size}". Must be a positive integer. No batches will be extracted and all calculations will be done in a single batch.')
        return -1
    if batch_size==-1 or batch_size>0:
        return batch_size
    
    warnings.warn(f'Wrong batch_size input: "{batch_size}". Must be a positive integer. No batches will be extracted and all calculations will be done in a single batch.')
    return -1