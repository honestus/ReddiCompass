import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from features_extraction_and_classification.feature_extraction import _extract_features_in_batches
import features_extraction_and_classification.io_utils as io_utils
import features_extraction_and_classification.resume_utils as resume_utils
from features_extraction_and_classification.resume_utils import validate_meta_file, store_meta_file, is_valid_resume_dir, _is_features_extraction_finished_, is_model_train_finished, is_normalization_finished, get_processed_features, resume_extract_features
from features_extraction_and_classification.validate_utils import validate_x_y_inputs, to_resume_flag, validate_tfidf_user_inputs, validate_tfidf_parameters, validate_text_input, validate_batch_size
from features_extraction_and_classification.tfidf_utils import get_default_tfidf_extractor, get_ngram_topk_from_tfidf_extractor



import shutil
"""
 TO DO:
 1. APPLY LOGIC TO RESUME FROM PROPER PROCESSING CHECKPOINT
 IF MODEL NOT IN META...
        IF SCALER NOT IN META...
            CHECK IF FEATURES IN META... IF THAT, FEATURES=LOAD_FEATURES...
             OTHERWISE LOAD CURRENT EXTRACTED FEATURES(IF ANY), AND START PROCESSING FROM NEXT TEXTS... BY USING TEXTS INDEXES (IN DF OR LIST... AS WE RE PROCESSING THEM IN SEQUENCE ORDER)...
             KEEP SAVING, BY USING SAME BATCH_SIZE, FROM I=K+1...
        ELSE:
        SCALER = LOAD_MODEL(SCALER)
    ELSE:
        MODEL = LOAD_MODEL(SCALER)... AND WE'RE DONE IF TRAIN, OTHERWISE CHECK FOR PREDICTIONS IF FUNC=PREDICT
 
 2. APPLY SAME LOGIC WHEN SAVING (CREATE META FILE, SAVE IT, AND ADD THE 'FEATURES', 'SCALER', 'MODEL' ATTRIBUTES WHEN THAT PHASE CORRECTLY ENDS...
 
 3. CHECK THAT WHEN WE USE BATCH_SIZE, WE KEEP SAVING FROM K+1 AND NOT FROM 0... OTHERWISE WE'LL STORE ON ALREADY EXISTING FILES AND THEY'RE FOREVER GONE
 
"""

def predict(texts:  list|pd.Series = None , model_dir: str = None, save: bool = False, batch_size: int = False, resume_dir: str = None, ):
        
    """
    if not is_valid_resume_dir(resume_dir):
        raise ValueError('Cannot resume from chosen resume directory')
    
    orig_texts = pd.read_parquet(resume_dir.joinpath(io_utils.TEXTS_FILENAME))
    processed_features = __get_processed_features__(resume_dir)
    if not __is_features_finished__(processed_features, orig_texts):
        meta = validate_meta_file(Path(resume_dir).joinpath(resume_utils.META_FILENAME), funct='predict')
        model_dir = meta[resume_utils.MODEL_DIR_ATTRIBUTE]
        batch_size = meta[resume_utils.BATCH_SIZE_ATTRIBUTE]
        #processed_batches = meta["processed_batches"]
        total_texts = meta[resume_utils.NUMBER_OF_TEXTS_ATTRIBUTE]
    
    if False:
        return
    """   
    if to_resume_flag(resume_dir):
        if not is_valid_resume_dir(resume_dir):
            raise ValueError('Cannot resume from chosen resume directory')
        if is_predictions_finished(resume_dir):
            print('Prediction was already succesfully completed')
            predictions = pd.read_parquet(resume_dir+f'/{io_utils.PREDICTIONS_FILENAME}')
            return predictions
        
        save=True
        saving_directory = resume_dir
        meta_obj = validate_meta_file(str(resume_dir)+'/'+io_utils.META_FILENAME)
        model_dir = meta_obj[resume_utils.MODEL_DIR_ATTRIBUTE]
        
        #batch_size = validate_meta_file(resume_dir+f'/{io_utils.META_FILENAME}')[resume_utils.BATCH_SIZE_ATTRIBUTE] ##No need to handle it, as it is already handled in resume_extract_features.. But can be useful for future updates that handle batch sizes in other parts such as training.
        #texts, categories = (texts:=pd.read_parquet(resume_dir + f'/{io_utils.TEXTS_FILENAME}'))[io_utils.TEXT_NAME_IN_STORED_DF], texts[io_utils.CATEGORY_NAME_IN_STORED_DF]
        categories, texts = pd.read_parquet(resume_dir + f'/{io_utils.TEXTS_FILENAME}')[io_utils.CATEGORY_NAME_IN_STORED_DF], None
        
        if _is_features_extraction_finished_(resume_dir=resume_dir, check_tfidf_features=True):   
            features = get_processed_features(resume_dir)
        else:
            features = resume_extract_features(resume_dir=resume_dir)
            
        tfidf_extractor = io_utils.load_model(filename_path=resume_dir + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}')
            
    else:
        texts = pd.Series(validate_text_input(texts))
        if (batch_size:=validate_batch_size(batch_size))==-1:
            batch_size = len(texts)
        if model_dir is None:
            model_dir = str(io_utils.STANDARD_MODEL_PATH)
        else:
            model_dir = io_utils.validate_existing_model_dir(model_dir)
        print(f'Current model directory of the model used for predicting: {model_dir}')
        tfidf_extractor = io_utils.load_model(model_dir + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}')
        
        
        if save:# or (batch_size and batch_size<len(texts)):
            saving_directory = io_utils.prepare_new_directory(parent_dir=model_dir + io_utils.PREDICTIONS_DIR + ('/tmp' if not save else ''), funct='predict')
            print(f'Storing directory: {saving_directory}')
            pd.DataFrame(texts.rename(io_utils.TEXT_NAME_IN_STORED_DF)).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True)
            if not resume_dir:
                meta_obj = {}
                meta_obj[resume_utils.FUNCTION_ATTRIBUTE] = 'predict'
                meta_obj[resume_utils.NUMBER_OF_TEXTS_ATTRIBUTE] = len(texts)
                meta_obj[resume_utils.BATCH_SIZE_ATTRIBUTE] = batch_size
                meta_obj[resume_utils.MODEL_DIR_ATTRIBUTE] = model_dir
                meta_obj[resume_utils.TFIDF_BOOL_ATTRIBUTE] = True
                store_meta_file(meta_obj, saving_dir=saving_directory)
        else:
            saving_directory=False
            
        features = _extract_features_in_batches(texts, extract_tfidf=True, tfidf_extractor=tfidf_extractor, fit_tfidf=False, batch_size=batch_size, saving_directory=saving_directory)
        print('Features extracted correctly')
    
    if save:
        meta_obj[resume_utils.FEATURES_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    scaler = io_utils.load_model(model_dir + f'/{io_utils.SCALER_FILENAME}')
    x_test = scaler.transform(features.reindex(scaler.get_feature_names_out(), axis=1)) ### normalization of features
    if save:
        meta_obj[resume_utils.SCALER_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    model = io_utils.load_model(model_dir + f'/{io_utils.MODEL_FILENAME}')
    predictions = model.predict(x_test)
    if saving_directory and not resume_dir:
        meta_obj[resume_utils.PREDICTIONS_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    if save:
        pd.DataFrame(predictions, columns=['predictions']).set_index(pd.DataFrame(texts).index).to_parquet(saving_directory +f'/{io_utils.PREDICTIONS_FILENAME}', index=True)
        #features.to_parquet(saving_directory +f'/{io_utils.FEATURES_FILENAME}'), index=True)
    elif saving_directory:
        shutil.rmtree(saving_directory)
    return predictions
    
def train(texts: list|pd.Series = None, categories: list|pd.Series = None, save: bool = True, batch_size: int = False, saving_directory: str = None,  resume_dir: str = None, raise_errors_on_wrong_indexes: bool = None, model_type=None, **kwargs):
    """ Trains and stores a new model, given the inputs.
    Also stores all of the extracted features, the input texts and the preprocessor objects (tfidfvectorizer, scaler).
    To avoid storing, use saving_directory=False
    """
    
    if to_resume_flag(resume_dir): ##RESUMING
        if not is_valid_resume_dir(resume_dir):
            raise ValueError('Cannot resume from chosen resume directory')
        
        if is_model_train_finished(resume_dir, funct='train'):
            print('Model was already fitted.')
            return io_utils.load_model(resume_dir + f'/{io_utils.MODEL_FILENAME}')
        
        save=True
        saving_directory = resume_dir
        tfidf_extractor, scaler, model, features = None, None, None, None
        #batch_size = validate_meta_file(resume_dir+f'/{io_utils.META_FILENAME}')[resume_utils.BATCH_SIZE_ATTRIBUTE] ##No need to handle it, as it is already handled in resume_extract_features.. But can be useful for future updates that handle batch sizes in other parts such as training.
        #texts, categories = (texts:=pd.read_parquet(resume_dir + f'/{io_utils.TEXTS_FILENAME}'))[io_utils.TEXT_NAME_IN_STORED_DF], texts[io_utils.CATEGORY_NAME_IN_STORED_DF]
        categories, texts = pd.read_parquet(resume_dir + f'/{io_utils.TEXTS_FILENAME}')[io_utils.CATEGORY_NAME_IN_STORED_DF], None
        if is_normalization_finished(resume_dir, funct='train'):
            scaler = io_utils.load_model(resume_dir + f'/{io_utils.SCALER_FILENAME}')
            features = get_processed_features(resume_dir)
        else:
            if _is_features_extraction_finished_(resume_dir=resume_dir, check_tfidf_features=True):
                
                features = get_processed_features(resume_dir)
            else:
                features = resume_extract_features(resume_dir=resume_dir)
            
            tfidf_extractor = io_utils.load_model(filename_path=resume_dir + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}')
            scaler = get_default_scaler()
            model = get_default_model()
        
        
    else: ##NEW MODEL - FRESH RUN
        texts = validate_text_input(texts)
        texts, categories = validate_x_y_inputs(x=texts, y=categories) if raise_errors_on_wrong_indexes is None else validate_x_y_inputs(x=texts, y=categories, raise_errors_on_wrong_indexes=raise_errors_on_wrong_indexes)
        if (batch_size:=validate_batch_size(batch_size))==-1:
            batch_size = len(texts)

        extract_tfidf_bool = True #kwargs.get('extract_tfidf', True) ###UNUSED SO FAR... MAYBE CAN BE INCLUDED TO FUTURE UPDATES IF WE DONT WANT TO INCLUDE TFIDF AMONG FEATURES.
        tfidf_extractor=kwargs.get('tfidf_extractor', None)
        ngram_range = kwargs.get('ngram_range', None)
        top_k = kwargs.get('top_k', None)
        validate_tfidf_user_inputs(extract_tfidf=extract_tfidf_bool, fit_tfidf=True, tfidf_extractor=tfidf_extractor, y=categories, ngram_range=ngram_range, top_k=top_k)
        if extract_tfidf_bool:
            if tfidf_extractor is None:
                tfidf_extractor=get_default_tfidf_extractor(ngram_range=kwargs.get('ngram_range', None), top_k=kwargs.get('top_k', None))
                                
            validate_tfidf_parameters(extract_tfidf=extract_tfidf_bool, tfidf_extractor=tfidf_extractor, fit_tfidf=True, y=categories)
            ngram_range, top_k = get_ngram_topk_from_tfidf_extractor(tfidf_extractor)
        if not save:
            if saving_directory not in [None, False]:
                warnings.warn('Save is False. Saving directory is ignored')
                saving_directory=False
        else:
            if saving_directory in [None,False,0, True, 1]:
                saving_directory = io_utils.prepare_new_directory(parent_dir=io_utils.DEFAULT_MODELS_PATH, force_to_default_path=True, funct='train')    
            else:
                saving_directory = io_utils.prepare_new_directory(base_dir=saving_directory, force_to_default_path=False, funct='train')
            
            
            print(f'Storing directory: {saving_directory}')
            pd.DataFrame({io_utils.TEXT_NAME_IN_STORED_DF : texts, io_utils.CATEGORY_NAME_IN_STORED_DF : categories}).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True)
            io_utils.save_model(obj=tfidf_extractor, filename_path=saving_directory + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}')
            
            meta_obj = {} ##Meta parameters for any future resumes
            meta_obj[resume_utils.FUNCTION_ATTRIBUTE] = 'train'
            meta_obj[resume_utils.NUMBER_OF_TEXTS_ATTRIBUTE] = len(texts)
            meta_obj[resume_utils.BATCH_SIZE_ATTRIBUTE] = batch_size
            meta_obj[resume_utils.MODEL_DIR_ATTRIBUTE] = saving_directory
            meta_obj[resume_utils.NGRAM_RANGE_ATTRIBUTE] = ngram_range
            meta_obj[resume_utils.TOP_K_ATTRIBUTE] = top_k
            meta_obj[resume_utils.TFIDF_BOOL_ATTRIBUTE] = extract_tfidf_bool
            #meta_obj[resume_utils.FIT_TFIDF_ATTRIBUTE] = True
            store_meta_file(meta_obj, saving_dir=saving_directory)

        
        
        features = _extract_features_in_batches(texts=texts, extract_tfidf=True, tfidf_extractor=tfidf_extractor, fit_tfidf=True, categories=categories, batch_size=batch_size, 
        saving_directory=saving_directory)
        print('Features extracted correctly')
        if save and not resume_dir:
            meta_obj[resume_utils.FEATURES_ATTRIBUTE] = True
            #store_meta_file(meta_obj, saving_dir=saving_directory)
        scaler = get_default_scaler()
        
        
    x_train = scaler.fit_transform(features) ### normalization of features
    if save and not resume_dir:
        meta_obj[resume_utils.SCALER_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    model = get_default_model()
    model.fit(x_train, categories)
    if save and not resume_dir:
        meta_obj[resume_utils.MODEL_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    print('Model trained correctly')
    if save:
        print(f'Saving to {saving_directory}')
        io_utils.save_model(obj=scaler, filename_path=saving_directory + f'/{io_utils.SCALER_FILENAME}')
        io_utils.save_model(obj=model, filename_path=saving_directory + f'/{io_utils.MODEL_FILENAME}')
    return model
    
    
    
def get_default_scaler():
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler()
    
    
def get_default_model():
    from sklearn.svm import LinearSVC
    return LinearSVC()