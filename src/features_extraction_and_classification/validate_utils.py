import pandas as pd
import numpy as np
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
    import warnings
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
    
    
    
    
    
    if not __check_x_y_shapes__(x,y):
        raise ValueError('Wronge shapes')
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
                warn_message = "Different index names for the x and y collections. Please ensure you are passing proper texts and categories!"
                if raise_errors_on_wrong_indexes:
                    raise ValueError(warn_message)
                warnings.warn(warn_message, UserWarning) 
        
        # If the indices are different, raise a ValueError
        if not x.index.equals(y.index):
            warn_message = "The indices of x and y are different in values. Please ensure you are passing proper texts and categories!!! \n\
            If you want to avoid any potential misbehaviours use 'raise_errors_on_wrong_indexes'=True"
            if raise_errors_on_wrong_indexes:
                raise ValueError(warn_message)
            warnings.warn(warn_message, UserWarning)
            x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        
    return x,y