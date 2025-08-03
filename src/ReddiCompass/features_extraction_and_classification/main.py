import pandas as pd
import numpy as np
from ReddiCompass.features_extraction_and_classification.feature_extraction import _extract_features_in_batches
import ReddiCompass.features_extraction_and_classification.io_utils as io_utils
import ReddiCompass.features_extraction_and_classification.resume_utils as resume_utils
from ReddiCompass.features_extraction_and_classification.resume_utils import validate_meta_file, store_meta_file, is_valid_resume_dir, _is_features_extraction_finished_, is_model_train_finished, is_normalization_finished, is_predictions_finished, is_pipeline_stored, get_processed_features, resume_extract_features
from ReddiCompass.features_extraction_and_classification.validate_utils import validate_x_y_inputs, to_resume_flag, validate_tfidf_user_inputs, validate_tfidf_parameters, validate_text_input, validate_batch_size
from ReddiCompass.features_extraction_and_classification.tfidf_utils import get_default_tfidf_extractor, get_ngram_topk_from_tfidf_extractor
from ReddiCompass.features_extraction_and_classification.model_pipeline import SavingPipeline, FeatureExtractor
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


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


def train_pipeline(texts: str | list[str] | pd.Series = None, categories: np.ndarray | pd.Series | list = None, extract_tfidf: bool = True, ngram_range: tuple[int,int] = None, top_k: int = None, scaler: BaseEstimator = MinMaxScaler(), clf: BaseEstimator = LinearSVC(), batch_size: int = -1, save: bool = True, saving_directory: str = None, resume_dir: str = None, raise_errors_on_wrong_indexes: bool = False):
    if to_resume_flag(resume_dir):
        if is_pipeline_stored(resume_dir):
            print('Pipeline was already successfully stored')
            return io_utils.load_model(resume_dir+f'/{io_utils.PIPELINE_FILENAME}', validate_input=False)
        if not resume_utils.is_valid_resume_dir(resume_dir):
            raise ValueError(f'Cannot resume from {resume_dir}')

        save = True
        saving_directory = resume_dir
        scaler, clf, extract_tfidf, ngram_range, top_k, texts, categories, batch_size = None, None, None, None, None, None, None, None
        if is_model_train_finished(resume_dir, funct='train'): ##if model was already successfully trained, just init the pipeline (made by the previously stored components) and return it
            model = io_utils.load_model(resume_dir+f'/{io_utils.MODEL_FILENAME}')
            scaler = io_utils.load_model(resume_dir+f'/{io_utils.SCALER_FILENAME}')
            extractor = io_utils.load_model(resume_dir+f'/{io_utils.FEATURE_EXTRACTOR_FILENAME}', validate_input=False)
            pipeline = SavingPipeline(clf=model, scaler=scaler, extractor=extractor)
            object.__setattr__(pipeline, 'saving_directory', resume_dir) ##setting saving directory to resume_dir for future runs
            io_utils.save_model(pipeline, filename_path=str(pipeline.saving_directory)+f'/{io_utils.PIPELINE_FILENAME}', validate_input=False)
            return pipeline

        meta = validate_meta_file(str(resume_dir)+f'/{io_utils.META_FILENAME}')
        model = eval(meta[resume_utils.MODEL_TYPE_ATTRIBUTE])
        categories = pd.read_parquet(resume_dir + f'/{io_utils.TEXTS_FILENAME}', columns=[io_utils.CATEGORY_NAME_IN_STORED_DF])[io_utils.CATEGORY_NAME_IN_STORED_DF]
        if is_normalization_finished(resume_dir, funct='train'):
            X_feats = get_processed_features(resume_dir)
            extractor = io_utils.load_model(resume_dir+f'/{io_utils.FEATURE_EXTRACTOR_FILENAME}', validate_input=False)
            scaler = io_utils.load_model(resume_dir + f'/{io_utils.SCALER_FILENAME}')
        else:
            scaler = eval(meta[resume_utils.SCALER_TYPE_ATTRIBUTE])
            try:
                extractor = io_utils.load_model(resume_dir+f'/{io_utils.FEATURE_EXTRACTOR_FILENAME}', validate_input=False)
                
                if _is_features_extraction_finished_(resume_dir=resume_dir, check_tfidf_features=extract_tfidf):
                    X_feats = get_processed_features(resume_dir)
                else:
                    X_feats = extractor.transform(resume_dir=resume_dir, X=None)
            except:                
                orig_texts, orig_categories = (texts:=pd.read_parquet(resume_dir + f'/{io_utils.TEXTS_FILENAME}'))[io_utils.TEXT_NAME_IN_STORED_DF], texts[io_utils.CATEGORY_NAME_IN_STORED_DF]
                extract_tfidf = meta[resume_utils.TFIDF_BOOL_ATTRIBUTE]
                ngram_range = meta[resume_utils.NGRAM_RANGE_ATTRIBUTE]
                top_k = meta[resume_utils.TOP_K_ATTRIBUTE]
                batch_size = meta[resume_utils.BATCH_SIZE_ATTRIBUTE]
                return train_pipeline(texts=orig_texts, categories=orig_categories, extract_tfidf=extract_tfidf,
                                      ngram_range=ngram_range, top_k=top_k,
                                      batch_size=batch_size, save=True, saving_directory=resume_dir)

        pipeline = SavingPipeline(clf=model, scaler=scaler, extractor=extractor) ##CREATING PIPELINE TO FIT SCALER AND MODEL ON CURRENT RESUMED X_FEATS
        object.__setattr__(pipeline, 'saving_directory', resume_dir) ##setting saving directory to resume_dir for future runs


    else:     ##FRESH NEW RUN, NO RESUMING
        if texts is None or categories is None:
            raise ValueError("No resume directory provided. To train a new model from scratch, please provide valid 'texts' and 'categories' inputs.")
        if any(not isinstance(obj, BaseEstimator) for obj in [scaler, clf]):
            raise TypeError('Wrong model types in input. They must be sklearn estimators')
        texts = validate_text_input(texts)
        texts, categories = validate_x_y_inputs(x=texts, y=categories) if raise_errors_on_wrong_indexes is None else validate_x_y_inputs(x=texts, y=categories, raise_errors_on_wrong_indexes=raise_errors_on_wrong_indexes)
        validate_tfidf_user_inputs(extract_tfidf=extract_tfidf, tfidf_extractor=None, ngram_range=ngram_range, top_k=top_k, fit_tfidf=False)
        if (batch_size:=validate_batch_size(batch_size))==-1:
            batch_size = len(texts)
        pipeline = SavingPipeline(extractor=FeatureExtractor(extract_tfidf=extract_tfidf, ngram_range=ngram_range, top_k=top_k),
                                  scaler=scaler, clf=clf)
    
        if not save:    
            pipeline.fit(X=texts, y=categories)
            return pipeline
    
        saving_directory = pipeline._handle_saving_directory(save=save, saving_directory=saving_directory, funct='fit')
        object.__setattr__(pipeline, 'saving_directory', saving_directory) #setting saving_directory as attribute in order to use it for storing predictions

        #creating new dir if not already existing
        saving_directory = io_utils.prepare_new_directory(base_dir=saving_directory, force_to_default_path=False, funct='train')
        print(f'Storing directory: {saving_directory}')
        #STORING TEXTS AND CATEGORIES FOR POTENTIAL FUTURE RESUMES
        pd.DataFrame({io_utils.TEXT_NAME_IN_STORED_DF : texts, io_utils.CATEGORY_NAME_IN_STORED_DF : categories}).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True)

        if extract_tfidf:
            ngram_range, top_k = get_ngram_topk_from_tfidf_extractor(pipeline[pipeline.steps[0][0]].tfidf_extractor)
        meta_obj = {} ##Meta parameters for any future resumes
        meta_obj[resume_utils.FUNCTION_ATTRIBUTE] = 'train'
        meta_obj[resume_utils.NUMBER_OF_TEXTS_ATTRIBUTE] = len(texts)
        meta_obj[resume_utils.BATCH_SIZE_ATTRIBUTE] = batch_size
        meta_obj[resume_utils.MODEL_DIR_ATTRIBUTE] = saving_directory
        meta_obj[resume_utils.MODEL_TYPE_ATTRIBUTE] = str(pipeline.steps[2][1])
        meta_obj[resume_utils.SCALER_TYPE_ATTRIBUTE] = str(pipeline.steps[1][1])
        meta_obj[resume_utils.NGRAM_RANGE_ATTRIBUTE] = ngram_range
        meta_obj[resume_utils.TOP_K_ATTRIBUTE] = top_k
        meta_obj[resume_utils.TFIDF_BOOL_ATTRIBUTE] = extract_tfidf
        meta_obj[resume_utils.FEATURE_EXTRACTOR_ATTRIBUTE] = str(pipeline.steps[0][1])
        store_meta_file(meta_obj, saving_dir=saving_directory) ##Storing meta parameters
        
        pipeline['features_extractor'].fit(X=texts, y=categories)
        io_utils.save_model(obj=pipeline['features_extractor'].tfidf_extractor, filename_path=pipeline.saving_directory+f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}', validate_input=False)
        io_utils.save_model(pipeline['features_extractor'], filename_path=pipeline.saving_directory+f'/{io_utils.FEATURE_EXTRACTOR_FILENAME}', validate_input=False)

        X_feats = pipeline['features_extractor'].transform(X=texts, saving_directory=pipeline.saving_directory, batch_size=batch_size)
        
    X_feats_scaled = pipeline['features_scaler'].fit_transform(X=X_feats, y=categories)
    io_utils.save_model(pipeline['features_scaler'], filename_path=pipeline.saving_directory+f'/{io_utils.SCALER_FILENAME}')
    pipeline['classifier'].fit(X=X_feats_scaled, y=categories)
    io_utils.save_model(pipeline['classifier'], filename_path=pipeline.saving_directory+f'/{io_utils.MODEL_FILENAME}')
        
    io_utils.save_model(pipeline, filename_path=pipeline.saving_directory+f'/{io_utils.PIPELINE_FILENAME}', validate_input=False)
    return pipeline

def predict_pipeline(texts: str | list[str] | pd.Series = None, pipeline: SavingPipeline = None, save: bool = False, saving_directory: str = None, resume_dir: str = None, batch_size: int = -1):
    import warnings
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    
    if to_resume_flag(resume_dir):
        if is_predictions_finished(resume_dir):
            print('Prediction was already successfully completed')
            predictions = pd.read_parquet(resume_dir+f'/{io_utils.PREDICTIONS_FILENAME}')
            return predictions
        if not is_valid_resume_dir(resume_dir):
            raise ValueError('Cannot resume from chosen resume directory - missing meta/texts files.')
        
        save=True
        saving_directory = resume_dir
        meta_obj = validate_meta_file(str(resume_dir)+'/'+io_utils.META_FILENAME)
        model_dir = meta_obj[resume_utils.MODEL_DIR_ATTRIBUTE]
        if io_utils.PIPELINE_FILENAME not in io_utils.get_whole_filelist(path_str=model_dir):
            raise ValueError('Cannot recover pipeline originally used to predict these resumen directory texts.')
        pipeline = io_utils.load_model(filename_path=resume_dir+f'/{io_utils.PIPELINE_FILENAME}')
        
        if _is_features_extraction_finished_(resume_dir=resume_dir, check_tfidf_features=extract_tfidf):
            X_feats = get_processed_features(resume_dir)
        else:
            X_feats = extractor.transform(resume_dir=resume_dir, X=None)


    else: ##new run - no resume
        if texts is None or pipeline is None:
            raise ValueError("No resume directory provided. To run new predictions, please provide the 'texts' to predict on and the previously trained model pipeline.")
        if not isinstance(pipeline, SavingPipeline):
            raise TypeError('Input pipeline must be a SavingPipeline object')
        try:
            check_is_fitted(pipeline)
        except:
            raise NotFittedError('Current pipeline is not fitted yet. Cannot predict')
        texts = pd.Series(validate_text_input(texts))
        if not save:
            return pipeline.predict(texts)
            
        if (batch_size:=validate_batch_size(batch_size))==-1:
            batch_size = len(texts)
        if save:
            saving_directory = pipeline._handle_saving_directory(save=save, saving_directory=saving_directory, funct='predict')
            saving_directory = io_utils.prepare_new_directory(base_dir=saving_directory, funct='predict')
            print(f'Storing directory: {saving_directory}')
            pd.DataFrame(texts.rename(io_utils.TEXT_NAME_IN_STORED_DF)).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True)
    
            meta_obj = {}
            meta_obj[resume_utils.FUNCTION_ATTRIBUTE] = 'predict'
            meta_obj[resume_utils.NUMBER_OF_TEXTS_ATTRIBUTE] = len(texts)
            meta_obj[resume_utils.BATCH_SIZE_ATTRIBUTE] = batch_size
            if not pipeline.saving_directory:
                warnings.warn("The current pipeline used for predicting has no saving directory associated.\n\
                You should store it before predicting, otherwise it will be impossible to use that pipeline for resumes/predictions.")
            meta_obj[resume_utils.MODEL_DIR_ATTRIBUTE] = pipeline.saving_directory
            #meta_obj[resume_utils.TFIDF_BOOL_ATTRIBUTE] = pipeline.steps[0][1].extract_tfidf
            #meta_obj[resume_utils.MODEL_TYPE_ATTRIBUTE] = str(pipeline.steps[2][1])
            #meta_obj[resume_utils.SCALER_TYPE_ATTRIBUTE] = str(pipeline.steps[1][1])
            #meta_obj[resume_utils.FEATURE_EXTRACTOR_ATTRIBUTE] = str(pipeline.steps[0][1])
            store_meta_file(meta_obj, saving_dir=saving_directory)
    
       
        X_feats = pipeline.named_steps['features_extractor'].transform(
            X=texts, saving_directory=saving_directory, batch_size=batch_size
        )
    
    
    X_feats_scaled = pipeline['features_scaler'].transform(X=X_feats.reindex(pipeline['features_scaler'].get_feature_names_out(), axis=1) )
    predictions = pipeline.named_steps['classifier'].predict(X_feats_scaled)
    if save:
        pd.DataFrame(predictions).to_parquet(saving_directory+f'/{io_utils.PREDICTIONS_FILENAME}')
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
        if texts is None or categories is None:
            raise ValueError("No resume directory provided. To train a new model from scratch, please provide valid 'texts' and 'categories' inputs.")
 
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
    
    model = get_default_model()
    model.fit(x_train, categories)
   
    print('Model trained correctly')
    if save:
        print(f'Saving to {saving_directory}')
        io_utils.save_model(obj=scaler, filename_path=saving_directory + f'/{io_utils.SCALER_FILENAME}')
        io_utils.save_model(obj=model, filename_path=saving_directory + f'/{io_utils.MODEL_FILENAME}')
    return model
    
    
    
def predict(texts:  list|pd.Series = None , model_dir: str = None, save: bool = False, batch_size: int = False, resume_dir: str = None, ):
        
    if to_resume_flag(resume_dir):
        if not is_valid_resume_dir(resume_dir):
            raise ValueError('Cannot resume from chosen resume directory')
        if is_predictions_finished(resume_dir):
            print('Prediction was already successfully completed')
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
        if texts is None or model_dir is None:
            raise ValueError("No resume directory provided. To run new predictions, please provide the 'texts' to predict on and the previously trained model' directory.")
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
            saving_directory = io_utils.prepare_new_directory(parent_dir=model_dir + f'/{io_utils.PREDICTIONS_DIR}' + ('/tmp' if not save else ''), funct='predict')
            print(f'Storing directory: {saving_directory}')
            pd.DataFrame(texts.rename(io_utils.TEXT_NAME_IN_STORED_DF)).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True)
            
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
        #print('Features extracted correctly')
    
    
    scaler = io_utils.load_model(model_dir + f'/{io_utils.SCALER_FILENAME}')
    x_test = scaler.transform(features.reindex(scaler.get_feature_names_out(), axis=1)) ### normalization of features
    
    model = io_utils.load_model(model_dir + f'/{io_utils.MODEL_FILENAME}')
    predictions = model.predict(x_test)
    print("Finished to predict")
    
    if save:
        pd.DataFrame(predictions, columns=['predictions']).set_index(pd.DataFrame(texts).index).to_parquet(saving_directory +f'/{io_utils.PREDICTIONS_FILENAME}', index=True)
        #features.to_parquet(saving_directory +f'/{io_utils.FEATURES_FILENAME}'), index=True)
    elif saving_directory:
        shutil.rmtree(saving_directory)
    return predictions
    
def get_default_scaler():
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler()
    
    
def get_default_model():
    from sklearn.svm import LinearSVC
    return LinearSVC()