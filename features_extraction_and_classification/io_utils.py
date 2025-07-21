import os, glob, joblib, shutil
from pathlib import Path
from utils import get_package_root


    
DEFAULT_ROOT = get_package_root(os.getcwd())

DATA_DIR = 'rundata'
MODELS_DIR = DATA_DIR+'/models'
FEATURES_DIR = DATA_DIR+'/features'
PREDICTIONS_DIR = '/predictions'

TEXTS_FILENAME = 'texts.parquet'
FEATURES_FILENAME = 'features.parquet'
TFIDF_TOKENS_FILENAME = 'tfidf_tokens.parquet'
PREDICTIONS_FILENAME = 'predictions.parquet'
SCALER_FILENAME = 'scaler.joblib'
TFIDF_EXTRACTOR_FILENAME = 'tfidf_extractor.joblib'
MODEL_FILENAME = 'model.joblib'
PIPELINE_FILENAME = 'pipeline'


def get_filename(filepath, include_extension=True):
    filepath = Path(filepath)
    if include_extension:
        curr_filename = filepath.name
    else:
        curr_filename = filepath.stem
    return curr_filename
    
def get_file_extension(filepath):
    filepath = Path(filepath)
    return filepath.suffix


def __build_paths__(root_path: Path):
    """Builds default saving paths from the input root."""
    root_path = Path(root_path).resolve()
    paths = {}
    paths['root_path'] = root_path
    paths['data_path'] = root_path.joinpath(DATA_DIR)
    paths['models_path'] = root_path.joinpath(MODELS_DIR)
    paths['features_path'] = root_path.joinpath(FEATURES_DIR)
    return paths
    

__def_paths__ = __build_paths__(DEFAULT_ROOT)
DEFAULT_DATA_PATH = __def_paths__["data_path"]
DEFAULT_MODELS_PATH = __def_paths__["models_path"]
DEFAULT_FEATURES_PATH = __def_paths__["features_path"]
STANDARD_MODEL_PATH = DEFAULT_MODELS_PATH.joinpath('default')

__TYPE_FILENAME_MAP__ = None
def get_whole_filelist(path_str, file_format='', filename_string='') -> list[str]:
    path_str = str(Path(path_str)) + '/'
    file_format = file_format.strip()
    if file_format:
        if not file_format.startswith('.'):
            file_format = '.'+file_format
    searching_pattern = path_str+'*{}*{}'.format(filename_string, file_format)
    file_list = glob.glob(searching_pattern)
    return file_list
    
    
def get_stored_features_files(dirpath, return_batches_files=True, return_whole_file=True):
    features_filename = Path(FEATURES_FILENAME)
    curr_features_files = get_whole_filelist(dirpath, filename_string=features_filename.stem, file_format=features_filename.suffix)
    if not curr_features_files:
        return []
    if return_batches_files and return_whole_file:
        return curr_features_files
    
    if not return_whole_file:
        curr_features_files = [file for file in curr_features_files if file!=str(Path(dirpath).joinpath(FEATURES_FILENAME))]
    else:
        curr_features_files = [file for file in curr_features_files if file==str(Path(dirpath).joinpath(FEATURES_FILENAME))]
    return curr_features_files
    
    
def get_stored_tokens_files(dirpath, return_batches_files=True, return_whole_file=True):
    features_filename = Path(TFIDF_TOKENS_FILENAME)
    curr_features_files = get_whole_filelist(dirpath, filename_string=features_filename.stem, file_format=features_filename.suffix)
    if not curr_features_files:
        return []
    if return_batches_files and return_whole_file:
        return curr_features_files
    
    if not return_whole_file:
        curr_features_files = [file for file in curr_features_files if file!=str(Path(dirpath).joinpath(TFIDF_TOKENS_FILENAME))]
    else:
        curr_features_files = [file for file in curr_features_files if file==str(Path(dirpath).joinpath(TFIDF_TOKENS_FILENAME))]
    return curr_features_files
    
def get_saving_paths(root_dir=None):
    """Returns default saving paths from root.
    Useful to avoid building path each time if root==DEFAULT_ROOT
    """
    paths = {}
    if root_dir is None or Path(root_dir)==DEFAULT_ROOT:
        paths = __def_paths__
    else:
        root_dir = Path(root_dir)
        paths = __build_paths__(root_dir)
    return paths

def __check_is_valid_model_dir__(model_dir):
    """
    Checks if a (model_dir) directory contains all of the models files (SCALER_FILENAME, TFIDF_EXTRACTOR_FILENAME, MODEL_FILENAME).
    Otherwise returns False.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return False
    dir_files = [filepath.name for filepath in model_path.iterdir() if filepath.is_file()]
    all_model_files_present = all(model_file in dir_files for model_file in [SCALER_FILENAME, TFIDF_EXTRACTOR_FILENAME, MODEL_FILENAME]) 
    return all_model_files_present

def validate_existing_model_dir(model_dir: str) -> str:
    """
    Given the input directory, checks if it points to a valid model directory (see __check_is_a_valid_model_dir__), otherwise 
    joins it to the default data directory and repeats the check. If it still fails, it raises an error.
    """
    model_dirpath = Path(model_dir).resolve()
    if not __check_is_valid_model_dir__(model_dirpath):
        model_dirpath = Path(DEFAULT_MODELS_PATH).joinpath(model_dir)
        if not __check_is_valid_model_dir__(model_dirpath):
            raise ValueError('Cannot find model at the specified directory')
    return str(model_dirpath)#.absolute()


def __is_subdir__(subdir: str, parent_dir: str) -> bool:
    """ 
    Checks if the input subdir is within the subdirectories of parent dir.
    """
    subdir_path = Path(subdir).resolve()
    base_path = Path(parent_dir).resolve()
    return base_path in subdir_path.parents

def validate_new_dir(new_dir: str, force_to_default_path: bool=True, funct=None) -> str:
    """
    Given the input directory, checks if it already exists, otherwise raises an error.
    If force_to_default_dir is True, checks if it belongs to the DEFAULT_MODELS_PATH subdirectories, otherwise raises an error.
    """
    new_dir = Path(new_dir)
    if force_to_default_path:
        if funct is None:
            raise ValueError("Must specify method through the 'funct' parameter to properly store as a subtree of its root path")
        elif funct not in ['train','predict','extract_features']:
            raise ValueError("'funct' must be either 'train', 'predict' or 'extract_features'")
        if funct in ['train','predict']:
            default_path = DEFAULT_MODELS_PATH
        else:
            default_path = DEFAULT_FEATURES_PATH
        if not __is_subdir__(new_dir, default_path):
            raise ValueError(f"Directory '{new_dir}' must be inside '{default_path}'. \
                            Set force_to_default_path to False to create a directory outside of the default data directory.")    
            
    if new_dir.exists():
        raise FileExistsError('Already existing directory')
    return str(new_dir)#.absolute()
    
def load_model(filename_path: str, validate_input: bool=True) -> object:
    if not validate_input:
        return joblib.load(filename_path)
    filename_path = Path(filename_path)
    filename, file_extension = filename_path.stem, filename_path.suffix
    if not file_extension:
         file_extension = '.joblib'
         filename_path = Path(str(filename_path)+file_extension)
    if not __map_modelname_to_type__(filename+file_extension):
        raise ValueError(f'No default filename used. Use one among [{SCALER_FILENAME}, {TFIDF_EXTRACTOR_FILENAME}, {MODEL_FILENAME}] or set validate_input to False to load from non-default filenames')
        
    return joblib.load(filename_path)

def save_model(obj, filename_path: str, validate_input: bool=True) -> None:
    default_filename = __map_modeltype_to_name__(obj)
    if not default_filename:
        raise TypeError('Unknown object to store')
    filename_path = Path(filename_path)
    filename, file_extension = filename_path.stem, filename_path.suffix
    if not file_extension:
         file_extension = '.joblib'
         filename_path = Path(str(filename_path)+file_extension)
    if default_filename!=filename+file_extension:
        if validate_input:
            raise ValueError(f"Invalid filename '{filename}' to store the {type(obj)} object. Use {default_filename} or set validate_input to False to store it anyway")
        else:
            warnings.warnings(f"You're saving a {type(obj).__name__} object with a non-default filename: '{filename}'. \n"\
            "Storing it anyway, but this may lead to unexpected behaviors in future loadings.")
    if not file_extension:
        file_extension = '.joblib'
        filename_path = Path(str(filename_path)+file_extension)
    return joblib.dump(obj, filename_path)

def __get_new_directory__(parent_dir: str) -> str:
    """ The new subdir will have an integer name: i.e. max(subdirs)+1, where subdirs are the subdirectories of parent_dir with an integer name.
    """
    curr_existing_dirs = [int(dir_name) for d in get_whole_filelist(parent_dir) if Path(d).is_dir() and (dir_name:=Path(d).name).isdigit()]      
    new_dir = 1+max(curr_existing_dirs, default=-1)
    new_dir = Path(parent_dir).joinpath(str(new_dir))
    
    return new_dir

def validate_new_dir(new_dir: str, force_to_default_path: bool = True, funct = None, overwrite: bool = False) -> str:
    """
    Given the input directory, checks if it already exists.
    If force_to_default_path is True, checks if it belongs to the DEFAULT subdirectory, otherwise raises an error.
    For extract_features: if overwrite=True, deletes only features files.
    """
    new_dir = Path(new_dir)

    # Validate funct
    if funct is None:
        raise ValueError("Must specify 'funct' to properly validate path (train, predict, extract_features).")
    elif funct not in ['train', 'predict', 'extract_features']:
        raise ValueError("'funct' must be either 'train', 'predict' or 'extract_features'")

    # Force directory inside correct default path
    if force_to_default_path:
        if funct in ['train', 'predict']:
            default_path = DEFAULT_MODELS_PATH
        else:
            default_path = DEFAULT_FEATURES_PATH
        if not __is_subdir__(new_dir, default_path):
            raise ValueError(f"Directory '{new_dir}' must be inside '{default_path}'. "
                             f"Set force_to_default_path=False to allow custom paths.")

    if new_dir.exists():
        if funct == 'extract_features' and overwrite:
            # Remove only features*.parquet
            all_features_filepaths = [Path(f) for f in get_stored_features_files(new_dir)] + [Path(f) for f in get_stored_tokens_files(new_dir)]
            for f in all_features_filepaths:
                f.unlink()
        else:
            # For train/predict OR extract_features without overwrite
            raise FileExistsError(f"Directory '{new_dir}' already exists. "
                                  f"Use overwrite=True if you intend to clear features files (extract_features only).")

    return str(new_dir)


def prepare_new_directory(base_dir=None, parent_dir=None, force_to_default_path=False, funct=None, overwrite=False):
    """
    Creates and returns a new dir or validates an existing one.
    If base_dir: use it, otherwise creates a new numbered subdir inside parent_dir.
    """
    if base_dir is None and parent_dir is None:
        raise ValueError('Must specify one between base_dir and parent_dir')

    # Determine target dir
    curr_dir = Path(base_dir) if base_dir else __get_new_directory__(parent_dir)

    # Validate (includes overwrite handling)
    curr_dir = Path(validate_new_dir(curr_dir, force_to_default_path=force_to_default_path, funct=funct, overwrite=overwrite))

    # Create directory if not exists
    if not curr_dir.exists():
        curr_dir.mkdir(parents=True)

    return str(curr_dir)


def __get_model_type_name_mapping__():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.base import BaseEstimator
    
    global __TYPE_FILENAME_MAP__ 
    if __TYPE_FILENAME_MAP__ is None:
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.base import BaseEstimator
        def is_tfidf(obj):
            return (
                isinstance(obj, (TfidfVectorizer, TfidfTransformer)) or
                (isinstance(obj, Pipeline) and any(isinstance(step, (TfidfVectorizer, TfidfTransformer)) for _, step in obj.steps))
            )
            
        __TYPE_FILENAME_MAP__ = [
            (lambda obj: isinstance(obj, MinMaxScaler), SCALER_FILENAME),
            (lambda obj: is_tfidf(obj), TFIDF_EXTRACTOR_FILENAME),
            (lambda obj: isinstance(obj, BaseEstimator), MODEL_FILENAME)
        ]
    return __TYPE_FILENAME_MAP__

def __map_modeltype_to_name__(model_obj) -> str:
    for check_fn, filename in __get_model_type_name_mapping__():
        if check_fn(model_obj):
            return filename
    return False  # oppure solleva eccezione, se preferisci

def __map_modelname_to_type__(model_name) -> object:
    return {filename: check_fn for check_fn, filename in __get_model_type_name_mapping__()}.get(model_name, False)