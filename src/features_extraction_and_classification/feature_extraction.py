import pandas as pd
import numpy as np
import default_config
import features_extraction_and_classification.default_resources as default_resources
from text_processing.text_utils import get_wordnet_pos
from text_processing.textractor import TexTractor
from text_processing.text_replacement import replace_features_in_text
from text_processing.LexiconMatcher import LexiconMatcher
from features_extraction_and_classification.contextual_features_extraction import *
from features_extraction_and_classification.validate_utils import validate_text_input, validate_batch_size



import warnings, sklearn

def __fillna__(feature_series, training_set):
    if training_set is None:
        return 0
    return feature_series.fillna(training_set[feature_series.name].mean())

def _map_emojis_to_count(emojis_series, include_positive_and_negative=True):
    if not include_positive_and_negative:
        return emojis_series.map(lambda x: len(x))
    emojis_counts = pd.json_normalize(emojis_series\
                                            .map(lambda emos: [get_emoji_sentiment(e['feature_value']) for e in emos])\
                                            .map(lambda sentiments: {sent: sentiments.count(sent) for sent in [-1,1,0]}))\
                            .rename({
                                1:'positive_emojis_count', 
                                -1:'negative_emojis_count',
                                0: 'neutral_emojis_count'},axis=1).set_index(emojis_series.index)
    emojis_counts['emojis_count'] = emojis_counts['positive_emojis_count']+emojis_counts['negative_emojis_count']+emojis_counts['neutral_emojis_count']
    return emojis_counts

def _map_emoticons_to_count(emoticons_series, include_positive_and_negative=True):
    if not include_positive_and_negative:
        return emoticons_series.map(lambda x: len(x))
    emoticons_counts = pd.json_normalize(emoticons_series\
                                        .map(lambda emos: [get_emoticon_sentiment(e['feature_value']) for e in emos])\
                                        .map(lambda sentiments: {sent: sentiments.count(sent) for sent in [-1,1,0]}))\
                        .rename({
                            1:'positive_emoticons_count', 
                            -1:'negative_emoticons_count',
                            0: 'neutral_emoticons_count'},axis=1).set_index(emoticons_series.index)
    emoticons_counts['emoticons_count'] = emoticons_counts['negative_emoticons_count']+emoticons_counts['positive_emoticons_count']+emoticons_counts['neutral_emoticons_count']
    return emoticons_counts


def _map_tags_to_count(tags_series, normalize=False):
    from nltk.corpus import wordnet
    tags_counts = pd.json_normalize(tags_series.map(lambda tags: [get_wordnet_pos(tag) for word,tag in tags])\
                                       .map(lambda tags: {tag: tags.count(tag) for tag in [wordnet.NOUN, wordnet.ADJ, wordnet.VERB, wordnet.ADV]}))\
                        .rename({
                            wordnet.NOUN:'names_count', 
                            wordnet.ADJ:'adjectives_count',
                            wordnet.VERB: 'verbs_count',
                            wordnet.ADV: 'adverbs_count'},axis=1).set_index(tags_series.index)
    if normalize:
        tags_counts = tags_counts.apply(lambda row: row / rowsum if (rowsum:=row.sum())>0 else row, axis=1)

    return tags_counts




    
def __extract_textual_features_single_text__(text):
    if not isinstance(text, (str, TexTractor)):
        raise ValueError('Wrong input, must pass either a string or a TexTractor object')
    if isinstance(text, str):
        text = TexTractor(text)
    text = text.process()
    text.get_sentences()
    return text    

def __extract_textual_features_to_df__(texts):
    texts = pd.Series(validate_text_input(texts))
    textual_df = texts.map(lambda t: __extract_textual_features_single_text__(t).to_pandas_row()).apply(pd.Series)
    return textual_df


def build_tfidf_matrix(tokenized_corpus, tfidf_extractor: sklearn.base.BaseEstimator, fit: bool, to_df: bool = False, y: list|np.ndarray|pd.Series = None):
    """
    Please notice that input MUST be already tokenized to work properly. 
    If you are trying to pass untokenized texts or using default tokenizer with tokenized corpus, it's very likely it will throw errors.
    If you want to use default vectorizers , try to detokenize input text before passing it to tfidf_extractor, but you will likely lose 
    much information from tokens such as emojis, mentions etc.
    """
    """
    if not isinstance(tfidf_extractor, sklearn.base.BaseEstimator):
        raise ValueError('Must pass a valid tfidf_extractor to build tf-idf matrix')

    ngram_range = tfidf_extractor['vectorizer'].ngram_range    
    top_k = tfidf_extractor['kbest'].k if ('kbest' in tfidf_extractor.named_steps) else False
    
    print(tokenized_corpus)
    if y is not None:
        print(y)
    print(tfidf_extractor)
    """
    if fit:
        """
        if top_k and y is None:
            raise ValueError('No y to fit tfidf extractor. Categories are needed to select top-k discriminant tokens. Use top_k=False if you want to extract tf-idf of each token or include categories of texts.')
        """
        tfidf_matrix = tfidf_extractor.fit_transform(tokenized_corpus, y=y)
    else:
        tfidf_matrix = tfidf_extractor.transform(tokenized_corpus)
    if to_df:
        tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray())
        if isinstance(tokenized_corpus, (pd.Series, pd.DataFrame)):
            tfidf_matrix.set_index(tokenized_corpus.index, inplace=True)
    return tfidf_matrix


def __extract_no_tfidf_features__(text, append_return_filtered_tokens=False):
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    tbwd = default_resources.get_detokenizer()
    ###Extracting textual features(emojis, emoticons, urls, tokens etc)
    textual_df = __extract_textual_features_to_df__(text)
    tagged_tokens = textual_df.apply(lambda row: extract_pos_tags(row, filter_nonstopwords_only=True), axis=1)
    if append_return_filtered_tokens:
        tfidf_tokens = tagged_tokens.map(lambda x: [tok for tok,tag in x]).rename('tokens')
    else:
        tfidf_tokens = None
    texts_no_symbols = textual_df.apply(lambda r: replace_features_in_text(r, text_col='tokens', columns_to_remove=['emoticons','emojis','urls'], columns_to_replace=[] , replace_mentions_with_twitter=True), axis=1)
    #Extracting NERs
    entities_df =  pd.json_normalize(texts_no_symbols.map(tbwd.detokenize).map(extract_entities_spacy).map(filter_entities_dict)).set_index(texts_no_symbols.index)
    entities_df = entities_df.map(len).rename(lambda x: f'spacy_{x}_count', axis=1)
    entities_df['spacy_entities_count'] = entities_df.sum(axis=1) 
    
    #Extracting moral foundations
    moral_foundations_df = extract_moral_foundations(texts_no_symbols.map(lambda tokens: ' '.join([t for t in tokens if t.lower().islower() and t.lower() not in default_resources.get_stopwords()])) )
    
    #Extracting lexicon features (Emotions with NRCLex, socialness, VAD scores)
    tokens_no_symbols = texts_no_symbols.map(lambda tok: LexiconMatcher.__validate_input_tokens__([t for t in tok if t.lower().islower()], delimiters='_#()[]')) ##validate_input_tokens splits tokens like ["a#b"] into ["a", "b"]
    social_df = pd.json_normalize(tokens_no_symbols.map(extract_socialness)).set_index(tokens_no_symbols.index)
    vad_df = pd.json_normalize(tokens_no_symbols.map(extract_vad)).set_index(tokens_no_symbols.index)
    emotions_df = pd.json_normalize(tokens_no_symbols.map(extract_emotions)).set_index(tokens_no_symbols.index)
    
    #Extracting sentiments, toxicities, subjectivity... (keeping texts with emojis and emoticons as they may be very useful for sentiment/toxicity detection)
    texts_with_emos = pd.Series(textual_df.apply(lambda row: replace_features_in_text(row, text_col='tokens', columns_to_remove=['mentions','urls'], columns_to_replace=[]), axis=1),name='text').map(tbwd.detokenize)
    sentiment_df = pd.json_normalize(texts_with_emos.map(extract_sentiment)).set_index(texts_with_emos.index)
    modality_df = pd.DataFrame(texts_with_emos.map(extract_modality), index=texts_with_emos.index).rename(lambda x: 'pattern_modality',axis=1)
    toxicity_df = pd.json_normalize(texts_with_emos.map(extract_toxicity)).set_index(texts_with_emos.index).rename(lambda x: f'detoxify_{x}',axis=1)
    populism_df = pd.json_normalize(texts_with_emos.map(extract_populism)).set_index(texts_with_emos.index)
    
    #Mapping textual features to counts in BOW style - Normalizing by number of tokens
    pos_tags_counts = _map_tags_to_count(tagged_tokens, normalize=True)
    emojis_counts = _map_emojis_to_count(textual_df.emojis)
    emoticons_counts = _map_emoticons_to_count(textual_df.emoticons)
    counts_df = pd.concat([entities_df, emojis_counts, emoticons_counts,
                           textual_df[['urls','mentions','repeatedPunctuation','hashtags','badwords','uppercaseWords', 'tokens', 'sentences']].map(lambda c: len(c)).rename(lambda c: c+'_count',axis=1)
                          ], axis=1, ignore_index=False)
    counts_df = pd.concat([counts_df[['tokens_count', 'sentences_count']],
                          counts_df.drop(['tokens_count', 'sentences_count'],axis=1).div(counts_df['tokens_count'],axis=0)
                          ], axis=1, ignore_index=False)
    
    #Putting all features together
    features_df = pd.concat([counts_df, pos_tags_counts, toxicity_df,modality_df,sentiment_df,emotions_df,vad_df,social_df,moral_foundations_df, populism_df],
          axis=1, ignore_index=False)    

    if features_df.isna().any().any():
        columns_with_nans = features_df.columns[features_df.isnull().any()]
        for column in columns_with_nans:
            features_df[column] = __fillna__(features_df[column], training_set=pd.read_parquet('C:/Users/onest/Documents/TextAn/tesi/code/data/models/ML/final/train.parquet'))
    return (features_df, tfidf_tokens) 
    
    
def extract_no_tfidf_features(text):
    return __extract_no_tfidf_features__(text, append_return_filtered_tokens=False)[0]

def _extract_features_in_batches(texts, batch_size: int, extract_tfidf: bool, 
tfidf_extractor: sklearn.base.BaseEstimator = None, fit_tfidf: bool = False, 
categories: list|np.ndarray|pd.Series = None,
saving_directory: str = None, already_processed_features_batches=None,
already_processed_tokens_batches=None, **kwargs):
    
    import os, features_extraction_and_classification.io_utils as io_utils
        
    features, tfidf_tokens = None, None
    if already_processed_features_batches is not None:
        if not isinstance(already_processed_features_batches, pd.DataFrame):
            raise ValueError("Wrong type for 'already_processed_features_batches'. It must be a pandas DataFrame")
        features = pd.DataFrame(already_processed_features_batches)
    if already_processed_tokens_batches is not None:
        if not isinstance(already_processed_tokens_batches, pd.Series):
            raise ValueError("Wrong type for 'already_processed_tokens_batches'. It must be a pandas Series")
        tfidf_tokens = already_processed_tokens_batches.rename('tokens')
    if saving_directory:
        saved_indexes = []
        saving_index = 0
        if 'saved_indexes' in kwargs:
            saved_indexes = kwargs['saved_indexes']
            saving_index = max(saved_indexes, default=-1)+1
        features_filename = io_utils.get_filename(io_utils.FEATURES_FILENAME, include_extension=False)
        features_file_extension = io_utils.get_file_extension(io_utils.FEATURES_FILENAME)
        tokens_filename = io_utils.get_filename(io_utils.TFIDF_TOKENS_FILENAME, include_extension=False)
        tokens_file_extension = io_utils.get_file_extension(io_utils.TFIDF_TOKENS_FILENAME)
        
    
    i=0
    while any(curr_batch:=texts[i:i+batch_size]):
        curr_features, tfidf_toks = __extract_no_tfidf_features__(curr_batch, append_return_filtered_tokens=extract_tfidf)
        features = pd.concat([features, curr_features], axis=0, ignore_index=False)
        tfidf_tokens = pd.concat([tfidf_tokens, tfidf_toks], axis=0, ignore_index=False) if extract_tfidf else tfidf_tokens
        if saving_directory:
            curr_features.to_parquet(saving_directory + f'/{features_filename}_batch{saving_index}{features_file_extension}', index=True)
            if extract_tfidf:
                pd.DataFrame(tfidf_toks, columns=['tokens']).to_parquet(saving_directory + f'/{tokens_filename}_batch{saving_index}{tokens_file_extension}', index=True)
            saved_indexes.append(saving_index)
            saving_index+=1
        i+=batch_size
    
    if extract_tfidf:
        if fit_tfidf:
            if isinstance(categories, (pd.Series, pd.DataFrame)) and isinstance(tfidf_tokens, (pd.Series, pd.DataFrame)):
                merged_toks_to_category = pd.concat([tfidf_tokens, categories], axis=1, ignore_index=False) ##merging tokens to relative category by DF indexing to ensure they are properly mapped
                tfidf_tokens, categories = merged_toks_to_category.iloc[:,0], merged_toks_to_category.iloc[:,-1]
        
        tfidf_features = build_tfidf_matrix(tokenized_corpus=tfidf_tokens, 
                                   fit=fit_tfidf, tfidf_extractor=tfidf_extractor, to_df=True, y=categories)\
                            .rename(lambda x: default_config.TFIDF_FEATURES_NAMES_LIKE.format(str(x)), axis=1)
        features = pd.concat([features, tfidf_features], axis=1, ignore_index=False)
            
    if saving_directory:
        features.to_parquet(saving_directory + f'/{io_utils.FEATURES_FILENAME}', index=True)
        if extract_tfidf:
            if fit_tfidf:
                io_utils.save_model(obj=tfidf_extractor, filename_path=saving_directory + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}') ##storing tfidf-extractor if fitted on these texts
            pd.DataFrame(tfidf_tokens).to_parquet(saving_directory + f'/{io_utils.TFIDF_TOKENS_FILENAME}', index=True)
            [os.remove(saving_directory + f'/tfidf_tokens_batch{saving_index}.parquet') for saving_index in saved_indexes]
        [os.remove(saving_directory + f'/features_batch{saving_index}.parquet') for saving_index in saved_indexes]
    return features


def extract_features(texts: str | list[str] = None, extract_tfidf: bool = False, tfidf_extractor: sklearn.base.BaseEstimator = None, fit_tfidf: bool = False, categories: list|np.ndarray|pd.Series = None, save: bool = False, saving_directory: str = None, resume_dir: str = None, batch_size: int = -1, raise_errors_on_wrong_indexes: bool = False, **kwargs):
    from features_extraction_and_classification.io_utils import prepare_new_directory
    import features_extraction_and_classification.io_utils as io_utils
    import features_extraction_and_classification.resume_utils as resume_utils
    from features_extraction_and_classification.resume_utils import store_meta_file, validate_meta_file, resume_extract_features, _is_features_extraction_finished_
    from features_extraction_and_classification.validate_utils import validate_x_y_inputs, to_resume_flag, validate_tfidf_user_inputs, validate_tfidf_parameters
    from features_extraction_and_classification.tfidf_utils import get_default_tfidf_extractor, get_ngram_topk_from_tfidf_extractor
    from sklearn.base import clone
    from pathlib import Path


    if not to_resume_flag(resume_dir): 
        if texts is None:
            raise ValueError('You must use valid "texts" to extract features from if not resuming from previously saved texts.')
        ngram_range = kwargs.get('ngram_range', None) ##validating parameters (texts, tfidf inputs...)
        top_k = kwargs.get('top_k', None)
        validate_tfidf_user_inputs(extract_tfidf=extract_tfidf, tfidf_extractor=tfidf_extractor, fit_tfidf=fit_tfidf, ngram_range=ngram_range, y=categories, top_k=top_k)
        if extract_tfidf and tfidf_extractor is None:
            tfidf_extractor = get_default_tfidf_extractor(ngram_range=ngram_range, top_k=top_k)
        validate_tfidf_parameters(extract_tfidf=extract_tfidf, tfidf_extractor=tfidf_extractor, fit_tfidf=fit_tfidf, y=categories)
        texts = pd.Series(validate_text_input(texts))
        
        
        if extract_tfidf:
            ngram_range, top_k = get_ngram_topk_from_tfidf_extractor(tfidf_extractor)
        if extract_tfidf and fit_tfidf and top_k is not False:
            texts, categories = validate_x_y_inputs(x=texts, y=categories, raise_errors_on_wrong_indexes=raise_errors_on_wrong_indexes)
        else:
            categories = None
        
        if (batch_size:=validate_batch_size(batch_size))==-1:
            batch_size = len(texts)
        
        if save:
            if saving_directory in [True,1, False, 0 , None]:
                saving_directory = prepare_new_directory(parent_dir=io_utils.DEFAULT_FEATURES_PATH, force_to_default_path=True, funct='extract_features')
                
            else:
                saving_directory = prepare_new_directory(base_dir=saving_directory, force_to_default_path=False, funct='extract_features')
         
            print(f'Storing directory: {saving_directory}')
            pd.concat(
                    [texts.rename(io_utils.TEXT_NAME_IN_STORED_DF), categories.rename(io_utils.CATEGORY_NAME_IN_STORED_DF)] if categories is not None else [texts.rename(io_utils.TEXT_NAME_IN_STORED_DF)],
                axis=1, ignore_index=False
            ).to_parquet(saving_directory + f'/{io_utils.TEXTS_FILENAME}', index=True) ###Storing original texts (and categories, if any)
            
            ### RESUMING UTILS ###
            meta_obj = {} 
            meta_obj[resume_utils.FUNCTION_ATTRIBUTE] = 'extract_features'
            meta_obj[resume_utils.NUMBER_OF_TEXTS_ATTRIBUTE] = len(texts)
            meta_obj[resume_utils.BATCH_SIZE_ATTRIBUTE] = batch_size
            meta_obj[resume_utils.TFIDF_BOOL_ATTRIBUTE] = extract_tfidf
            
            if extract_tfidf: ##if we extract tfidf matrix together with the features - we need to store the input parameters ('fit_tfidf' and 'tfidf_extractor') to eventually resume if anything goes wrong
                curr_extractor_filepath = saving_directory + f'/{io_utils.TFIDF_EXTRACTOR_FILENAME}'
                
                meta_obj[resume_utils.TFIDF_EXTRACTOR_ATTRIBUTE] = str(Path(curr_extractor_filepath).absolute()) ##adding info on the storing location of the tfidfextractor model - we will load it from file if need to resume
                meta_obj[resume_utils.FIT_TFIDF_ATTRIBUTE] = fit_tfidf
                
                if not fit_tfidf: 
                    io_utils.save_model(obj=tfidf_extractor, filename_path=curr_extractor_filepath, validate_input=False) #storing current tfidf_extractor
                    meta_obj[f'already_fitted_{resume_utils.TFIDF_EXTRACTOR_ATTRIBUTE}'] = True
                else: ## if we have to refit it on current texts, we just clone the tfidf_extractor and store it to ensure we dont store a fitted instance and to save space on disk.
                    io_utils.save_model(obj=clone(tfidf_extractor), filename_path=curr_extractor_filepath)
                    
            store_meta_file(meta_obj, saving_dir=saving_directory)
        
        batch_size = len(texts) if not batch_size else batch_size
        features = _extract_features_in_batches(texts=texts, categories=categories, extract_tfidf=extract_tfidf, tfidf_extractor=tfidf_extractor, fit_tfidf=fit_tfidf, saving_directory=saving_directory, batch_size=batch_size) ##feature extraction
        
    
    else: ##RESUMING
        features = resume_extract_features(resume_dir) ### IF WE HAVE TO RESUME, JUST USE RESUME_EXTRACT_FEATURES (handles all the needed checking for resuming and extracts features from unprocessed texts (if any)     
    
    
    return features

