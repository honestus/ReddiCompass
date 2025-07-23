from pathlib import Path
import json
import pandas as pd
from utils import get_columns_from_parquet_file
from features_extraction_and_classification.io_utils import TEXTS_FILENAME, TFIDF_TOKENS_FILENAME, FEATURES_FILENAME, get_stored_features_files, get_stored_tokens_files, get_whole_filelist
from default_config import ALL_FEATURES_COLUMNS, TFIDF_FEATURES_NAMES_LIKE

META_FILENAME = 'meta.json'

NUMBER_OF_TEXTS_ATTRIBUTE = 'n_of_texts'
MODEL_DIR_ATTRIBUTE = 'model_dir'
BATCH_SIZE_ATTRIBUTE = 'batch_size'
NGRAM_RANGE_ATTRIBUTE = 'ngram_range'
FEATURES_ATTRIBUTE = 'features'
SCALER_ATTRIBUTE = 'scaler'
MODEL_ATTRIBUTE = 'model'
PREDICTIONS_ATTRIBUTE = 'predictions'
FUNCTION_ATTRIBUTE = 'funct'
TFIDF_BOOL_ATTRIBUTE = 'extract_tfidf'
TFIDF_EXTRACTOR_ATTRIBUTE = 'tfidf_extractor'
FIT_TFIDF_ATTRIBUTE = 'fit_tfidf'

__all_possible_funct_resume__ = ["train", 'predict', "extract_features"]

def __validate_funct__(funct):
    if funct in __all_possible_funct_resume__:
        return funct
    raise ValueError(f"Funct must be be one of: {__all_possible_funct_resume__}")


def __load_meta_file__(meta_filepath):
    with open(meta_filepath, 'r') as f:
        return json.load(f)
        
def store_meta_file(dict_obj, saving_dir, filename=META_FILENAME):
    meta_filepath = Path(saving_dir).joinpath(filename)
    with open(meta_filepath, 'w') as f:
        json.dump(dict_obj, f)
        return True
    return False

def get_required_meta_params(funct):
    funct = __validate_funct__(funct)
    required = [NUMBER_OF_TEXTS_ATTRIBUTE, BATCH_SIZE_ATTRIBUTE, FUNCTION_ATTRIBUTE]
    if funct=='train':
        return required + [NGRAM_RANGE_ATTRIBUTE]
    if funct=='predict':
        return required + [MODEL_DIR_ATTRIBUTE]
    if funct=='extract_features':
        return required + [TFIDF_BOOL_ATTRIBUTE]
    return required

def _is_valid_meta_dict_(meta_dict) -> bool:    
    funct = meta_dict.get(FUNCTION_ATTRIBUTE, -1)
    if funct==-1:
        raise ValueError(f"'{FUNCTION_ATTRIBUTE}' not defined")
    
    required = get_required_meta_params(funct=funct)
    missing_in_meta = [k for k in required if k not in meta_dict]
    if any(missing_in_meta):
        raise ValueError(f'Meta file does not contain all the needed info. Missing: {missing_in_meta}')
    return True

def validate_meta_file(meta_filepath: str | Path) -> bool:
    meta_filepath = Path(meta_filepath)
    if not meta_filepath.exists():
        raise FileNotFoundError(f"Cannot find meta.json file at the specified resume directory. Impossible to resume.")
    meta = __load_meta_file__(meta_filepath)
    if _is_valid_meta_dict_(meta):
        if NGRAM_RANGE_ATTRIBUTE in meta:
            meta[NGRAM_RANGE_ATTRIBUTE] = tuple(meta[NGRAM_RANGE_ATTRIBUTE])
        return meta
    raise ValueError('wrong meta file')

def is_valid_resume_dir(resume_dir: str|Path) -> bool:
    if not isinstance(resume_dir, (str, Path)):
        raise ValueError('Please specify a valid directory to resume from')
    resume_path = Path(resume_dir)
    meta_filepath = resume_path.joinpath(META_FILENAME)
    texts_filepath = resume_path.joinpath(TEXTS_FILENAME)
    
    return meta_filepath.exists() and texts_filepath.exists()
   
   



def get_processed_features(resume_dir, index_only=False):
    #import numpy as np
    
    resume_dir = Path(resume_dir)
    features_filename = Path(FEATURES_FILENAME)
    
    curr_features_files = get_stored_features_files(resume_dir)
    if not curr_features_files:
        return pd.Index([]) if index_only else pd.DataFrame([])
        """
    whole_features_file = np.where(np.array(curr_features_files)==str(Path(resume_dir).joinpath(FEATURES_FILENAME)))[0]
    whole_features_file = curr_features_files.pop(whole_features_file[0]) if whole_features_file.size else False
    if whole_features_file:
    """
    whole_features_file = Path(resume_dir).joinpath(FEATURES_FILENAME)
    if whole_features_file.exists():
        processed_features = pd.read_parquet(whole_features_file, columns=[] if index_only else None)
        return processed_features.index if index_only else processed_features
    processed_features = pd.concat([pd.read_parquet(f, columns=[] if index_only else None) for f in curr_features_files if Path(f)!=whole_features_file])
    return processed_features.index if index_only else processed_features
 
 
def get_processed_tfidf_tokens(resume_dir, index_only=False):
    resume_dir = Path(resume_dir)
    
    whole_features_file = resume_dir.joinpath(FEATURES_FILENAME)
    if whole_features_file.exists():
        curr_stored_features = get_columns_from_parquet_file(whole_features_file)
        if all(feat in curr_stored_features for feat in ALL_FEATURES_COLUMNS):
            return True
    curr_tokens_files = get_stored_tokens_files(dirpath=resume_dir)
    if curr_tokens_files:
        processed_tokens = pd.concat([pd.read_parquet(f, columns=[] if index_only else None) for f in curr_tokens_files])
        return processed_tokens.index if index_only else processed_tokens
    return pd.Series([])
        
        
def is_features_extraction_finished(processed_features: pd.DataFrame, expected_all_features_index: pd.Index, columns: list|pd.Series=ALL_FEATURES_COLUMNS) -> bool:
    if any(feat not in processed_features.columns for feat in columns):
        return False
    if len(expected_all_features_index)!=len(processed_features):
        return False
    return expected_all_features_index.sort_values().equals(processed_features.index.sort_values())
       
    
def is_model_train_finished(resume_dir):
    import io_utils
    resume_dir = Path(resume_dir)
    curr_files = list(map(lambda x: Path(x).name, io_utils.get_whole_filelist(Path(resume_dir))))
    return io_utils.MODEL_FILENAME in curr_files
    
    
def is_predictions_finished(curr_predictions: pd.Series, expected_all_predictions_index: pd.Index) -> bool:
    if len(curr_predictions)==len(expected_all_predictions):
        if sorted(curr_predictions.index)==sorted(expected_all_predictions_index):
            return True
    return False
    
def is_scaler_finished(resume_dir):
    import io_utils
    resume_dir = Path(resume_dir)
    curr_files = list(map(lambda x: Path(x).name, io_utils.get_whole_filelist(Path(resume_dir))))
    return io_utils.SCALER_FILENAME in curr_files
    
   
    
def resume_extract_features(resume_dir):
    import features_extraction_and_classification.io_utils as io_utils
    from features_extraction_and_classification.feature_extraction import _extract_features_in_batches, get_default_tfidf_extractor
    if not is_valid_resume_dir(resume_dir):
        raise ValueError('Cannot resume from chosen resume directory: missing meta file or original texts file.')
    resume_dir = Path(resume_dir)
    orig_texts = pd.read_parquet(resume_dir.joinpath(TEXTS_FILENAME))
    processed_features = get_processed_features(resume_dir)
    processed_tokens = get_processed_tfidf_tokens(resume_dir)
    
    meta = validate_meta_file(Path(resume_dir).joinpath(META_FILENAME)) ##checking all of the current needed attributes are stored in the json file
    funct = meta[FUNCTION_ATTRIBUTE]
    batch_size = meta[BATCH_SIZE_ATTRIBUTE]
    total_texts = meta[NUMBER_OF_TEXTS_ATTRIBUTE]
    if funct=='extract_features':
        extract_tfidf_bool = meta[TFIDF_BOOL_ATTRIBUTE]
    else:
        extract_tfidf_bool = True ###TO CHANGE IF WE WANT TO ADD IT AS OPTIONAL DURING TRAIN/TEST
    
    columns_to_check_in_stored_features_dataframe = ALL_FEATURES_COLUMNS + ([] if not extract_tfidf_bool else [TFIDF_FEATURES_NAMES_LIKE.format('0')])
    if is_features_extraction_finished(processed_features, expected_all_features_index=orig_texts.index, columns=columns_to_check_in_stored_features_dataframe): 
        return processed_features ###if all the features have already been extracted and store, just returns them as they are
    
    ### Validating needed attributes for the current resuming function (either 'train', 'predict' or 'extract_features')
    if funct in ['train', 'predict']:
        model_dir = meta[MODEL_DIR_ATTRIBUTE]
        if funct=='train':
            ngram_range = meta[NGRAM_RANGE_ATTRIBUTE]
            tfidf_extractor=get_default_tfidf_extractor(ngram_range=ngram_range)
        elif funct=='predict':
            tfidf_extractor = io_utils.load_model(model_dir + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}')
    elif funct=='extract_features':
        if not extract_tfidf_bool: ##if no need to build tf-idf matrix, no need for any tfidf_tokens, then just recovering current unprocessed features and resuming them by appending them to the already processed ones
            unprocessed_texts = orig_texts.loc[~orig_texts.index.isin(processed_features.index)]
            saved_indexes_filenames = [int(Path(f).stem.split(Path(FEATURES_FILENAME).stem+'_batch')[-1]) for f in get_stored_features_files(resume_dir, return_whole_file=False)] ##appending indexes to properly store new features from max+1
            print(f"Resuming features extraction --- Last saved index: {max(saved_indexes_filenames)}")
            return _extract_features_in_batches(texts=unprocessed_texts[io_utils.TEXT_NAME_IN_STORED_DF],  
                                                extract_tfidf=False,
                                                saving_directory=str(resume_dir), 
                                                batch_size=batch_size,
                                                saved_indexes=saved_indexes_filenames, 
                                                already_processed_features_batches = processed_features)
        else:
            #raise NotImplementedError()
            tfidf_extractor = io_utils.load_model(filename_path=meta[TFIDF_EXTRACTOR_ATTRIBUTE], validate_input=False)
            fit_tfidf = meta[FIT_TFIDF_ATTRIBUTE]
        
    else:
        raise NotImplementedError('Invalid resume_dir, it must be either a previously stored directory from train,  predict, or extract_features')
    
    if processed_tokens.empty or processed_features.empty:        
        if funct=='train':
            return _extract_features_in_batches(texts=orig_texts[io_utils.TEXT_NAME_IN_STORED_DF], y=orig_texts[io_utils.CATEGORY_NAME_IN_STORED_DF], 
            batch_size=batch_size, 
            tfidf_extractor=tfidf_extractor, 
            fit_tfidf=True, 
            ngram_range=ngram_range,
            saving_directory=str(resume_dir))
        elif funct=='predict':
            return _extract_features_in_batches(texts=orig_texts[io_utils.TEXT_NAME_IN_STORED_DF], 
            tfidf_extractor=tfidf_extractor, 
            fit_tfidf=False, 
            batch_size=batch_size, 
            saving_directory=str(resume_dir))
        elif funct=='extract_features':
            return _extract_features_in_batches(texts=orig_texts[io_utils.TEXT_NAME_IN_STORED_DF], y=orig_texts[io_utils.CATEGORY_NAME_IN_STORED_DF], 
            batch_size=batch_size, 
            tfidf_extractor=tfidf_extractor, 
            fit_tfidf=fit_tfidf, 
            saving_directory=str(resume_dir))
                
        
    ###Recovering previously processed features filenames in order to store new files with proper indexes
    curr_features_files =  get_stored_features_files(resume_dir, return_whole_file=False)
    curr_tokens_files =  get_stored_tokens_files(resume_dir, return_whole_file=False)
    features_files_indexes = [int(Path(f).stem.split(Path(FEATURES_FILENAME).stem+'_batch')[-1]) for f in curr_features_files]
    tokens_files_indexes = [int(Path(f).stem.split(Path(TFIDF_TOKENS_FILENAME).stem+'_batch')[-1]) for f in curr_tokens_files]
    
    ###Checking that previously processed features and tokens are from the same imput texts - otherwise just keeping the texts processed for both features and tokens and re-processing all of the remaining ones 
    if sorted(processed_features.index) != sorted(processed_tokens.index): 
        common_processed_texts_idx = processed_features.index.intersection(processed_tokens.index) ##getting processed texts for both features and tfidftokens
        processed_features = processed_features.loc[common_processed_texts_idx]
        processed_tokens = processed_tokens.loc[common_processed_texts_idx]

        # Removing old files and only saving the common_processed_features into new files ('features_batch0.parquet' and 'tfidf_tokens_batch0.parquet')
        for f in curr_features_files + curr_tokens_files:
            Path(f).unlink()
        processed_features.to_parquet(resume_dir.joinpath(f'{Path(FEATURES_FILENAME).stem}_batch0{Path(FEATURES_FILENAME).suffix}'), index=True)
        processed_tokens.to_parquet(resume_dir.joinpath(f'{Path(TFIDF_TOKENS_FILENAME).stem}_batch0{Path(TFIDF_TOKENS_FILENAME).suffix}'), index=True)
        
        saved_indexes_filenames = [0]
        
    else:
        saved_indexes_filenames = features_files_indexes
    
    ### extracting features for the previously unprocessed texts only
    unprocessed_texts = orig_texts.loc[~orig_texts.index.isin(processed_features.index)]

    print(f"Resuming features extraction --- Last saved index: {max(saved_indexes_filenames)}")
    if funct=='train':
        return _extract_features_in_batches(texts=unprocessed_texts[io_utils.TEXT_NAME_IN_STORED_DF], y=orig_texts[io_utils.CATEGORY_NAME_IN_STORED_DF], 
        extract_tfidf=True,
        tfidf_extractor=tfidf_extractor, 
        fit_tfidf=True, 
        ngram_range=ngram_range,
        saving_directory=str(resume_dir), 
        batch_size=batch_size,
        saved_indexes=saved_indexes_filenames, 
        already_processed_features_batches = processed_features, 
        already_processed_tokens_batches = processed_tokens[processed_tokens.columns[0]],
        )        
        
    elif funct=='predict':
        return _extract_features_in_batches(
        texts=unprocessed_texts[io_utils.TEXT_NAME_IN_STORED_DF],
        extract_tfidf=True,
        tfidf_extractor = tfidf_extractor,
        fit_tfidf=False,
        saving_directory=str(resume_dir),
        batch_size=batch_size,
        saved_indexes=saved_indexes_filenames,
        already_processed_features_batches = processed_features,
        already_processed_tokens_batches = processed_tokens[processed_tokens.columns[0]],
        )
        
    elif funct=='extract_features':
        if extract_tfidf_bool:
            return _extract_features_in_batches(texts=unprocessed_texts[io_utils.TEXT_NAME_IN_STORED_DF], y=orig_texts[io_utils.CATEGORY_NAME_IN_STORED_DF], 
            extract_tfidf=True,
            tfidf_extractor=tfidf_extractor, 
            fit_tfidf=fit_tfidf, 
            saving_directory=str(resume_dir), 
            batch_size=batch_size,
            saved_indexes=saved_indexes_filenames, 
            already_processed_features_batches = processed_features, 
            already_processed_tokens_batches = processed_tokens[processed_tokens.columns[0]],
            )
        else:
            raise ValueError('Should never happen')
    raise NotImplementedError('')
   
    
    

    
def resume_scaler(resume_dir, meta_filename):
    raise NotImplementedError()


def resume_model(resume_dir, meta_filename):    
    raise NotImplementedError()
