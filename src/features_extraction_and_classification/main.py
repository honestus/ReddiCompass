import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import LinearSVC
from features_extraction_and_classification.feature_extraction import __extract_features_in_batches__, get_default_tfidf_extractor, __validate_text_input__
import features_extraction_and_classification.io_utils as io_utils
import features_extraction_and_classification.resume_utils as resume_utils
from features_extraction_and_classification.resume_utils import validate_meta_file, store_meta_file

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

def predict(texts=None , model_dir=None, resume_dir=False, save=False, batch_size=False):
    if texts is None and not resume_dir:
        raise ValueError('Either texts must be a valid collection or resume must point to an existing stored directory')
    if resume_dir:
        raise NotImplementedError('to do')
        if texts or model_dir or batch_size:
            warnings.warnings('Texts and model_dir will be ignored as resuming from previously processed texts. If you want to predict new texts or with a different model, set resume to False')
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
    texts = pd.Series(__validate_text_input__(texts), name=io_utils.TEXT_NAME_IN_STORED_DF)
    
    if model_dir is None:
        model_dir = str(io_utils.STANDARD_MODEL_PATH)
    else:
        model_dir = io_utils.validate_existing_model_dir(model_dir)
    print(f'Current model directory of the model used for predicting: {model_dir}')
    tfidf_extractor = io_utils.load_model(model_dir + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}')
    
    
    if save:# or (batch_size and batch_size<len(texts)):
        saving_directory = io_utils.prepare_new_directory(parent_dir=model_dir + io_utils.PREDICTIONS_DIR + ('/tmp' if not save else ''), funct='predict')
        print(f'Storing directory: {saving_directory}')
        pd.DataFrame(texts).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True)
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
        
    features = __extract_features_in_batches__(texts, extract_tfidf=True, tfidf_extractor=tfidf_extractor, fit_tfidf=False, batch_size=batch_size, saving_directory=saving_directory, overwrite=True)
    print('Features extracted correctly')
    if saving_directory and not resume_dir:
        meta_obj[resume_utils.FEATURES_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    scaler = io_utils.load_model(model_dir + f'/{io_utils.SCALER_FILENAME}')
    x_test = scaler.transform(features.reindex(scaler.get_feature_names_out(), axis=1)) ### normalization of features
    if saving_directory and not resume_dir:
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
    
def train(texts, categories, resume_dir=False, batch_size=False, saving_directory=None, model_type='svc', ngram_range=(1,1), raise_errors_on_wrong_indexes=None):
    """ Trains and stores a new model, given the inputs.
    Also stores all of the extracted features, the input texts and the preprocessor objects (tfidfvectorizer, scaler).
    To avoid storing, use saving_directory=False
    """
    from features_extraction_and_classification.validate_utils import validate_x_y_inputs

    if resume_dir:
        raise NotImplementedError('to do')
        if texts or model_dir or batch_size:
            warnings.warnings('Texts and model_dir will be ignored as resuming from previously processed texts. If you want to predict new texts or with a different model, set resume to False')
        
    texts = __validate_text_input__(texts)
    texts, categories = validate_x_y_inputs(x=texts, y=categories) if raise_errors_on_wrong_indexes is None else validate_x_y_inputs(x=texts, y=categories, raise_errors_on_wrong_indexes=raise_errors_on_wrong_indexes)
   
    
    if saving_directory is None:
        saving_directory = io_utils.prepare_new_directory(parent_dir=io_utils.DEFAULT_MODELS_PATH, force_to_default_path=True, funct='train')    
    elif saving_directory:
        saving_directory = io_utils.prepare_new_directory(base_dir=saving_directory, force_to_default_path=False, funct='train')
        
    if saving_directory:
        print(f'Storing directory: {saving_directory}')
        pd.DataFrame({io_utils.TEXT_NAME_IN_STORED_DF:texts, io_utils.CATEGORY_NAME_IN_STORED_DF:categories}).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True)
        if not resume_dir:
            meta_obj = {}
            meta_obj[resume_utils.FUNCTION_ATTRIBUTE] = 'train'
            meta_obj[resume_utils.NUMBER_OF_TEXTS_ATTRIBUTE] = len(texts)
            meta_obj[resume_utils.BATCH_SIZE_ATTRIBUTE] = batch_size
            meta_obj[resume_utils.MODEL_DIR_ATTRIBUTE] = saving_directory
            meta_obj[resume_utils.NGRAM_RANGE_ATTRIBUTE] = ngram_range
            meta_obj[resume_utils.TFIDF_BOOL_ATTRIBUTE] = True
            store_meta_file(meta_obj, saving_dir=saving_directory)
   
    tfidf_extractor=get_default_tfidf_extractor(ngram_range=ngram_range)   
    features = __extract_features_in_batches__(texts, extract_tfidf=True, tfidf_extractor=tfidf_extractor, fit_tfidf=True, ngram_range=ngram_range, y=categories, batch_size=batch_size, 
    saving_directory=saving_directory, overwrite=True)
    print('Features extracted correctly')
    if saving_directory and not resume_dir:
        meta_obj[resume_utils.FEATURES_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(features) ### normalization of features
    if saving_directory and not resume_dir:
        meta_obj[resume_utils.SCALER_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    model = LinearSVC()
    model.fit(x_train, categories)
    if saving_directory and not resume_dir:
        meta_obj[resume_utils.MODEL_ATTRIBUTE] = True
        #store_meta_file(meta_obj, saving_dir=saving_directory)
    print('model trained correctly')
    if saving_directory:
        print(f'Saving to {saving_directory}')
        io_utils.save_model(obj=scaler, filename_path=saving_directory + f'/{io_utils.SCALER_FILENAME}')
        io_utils.save_model(obj=tfidf_extractor, filename_path=saving_directory+ f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}')
        io_utils.save_model(obj=model, filename_path=saving_directory + f'/{io_utils.MODEL_FILENAME}')
    return model
    
    
    
