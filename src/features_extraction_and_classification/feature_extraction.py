import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import default_config
import features_extraction_and_classification.default_resources as default_resources
from text_processing.text_utils import get_wordnet_pos
from text_processing.textractor import TexTractor
from text_processing.text_replacement import replace_features_in_text
from text_processing.LexiconMatcher import LexiconMatcher
from features_extraction_and_classification.contextual_features_extraction import *
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from features_extraction_and_classification.tfidf_utils import do_nothing
import warnings

def __fillna__(feature_series, training_set):
    if training_set is None:
        return 0
    return feature_series.fillna(training_set[feature_series.name].mean())

def map_emojis_to_count(emojis_series, include_positive_and_negative=True):
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

def map_emoticons_to_count(emoticons_series, include_positive_and_negative=True):
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


def map_tags_to_count(tags_series, normalize=False):
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


def __validate_text_input__(text):
    if isinstance(text, (str, TexTractor)):
        return [text]
    if isinstance(text, (list, np.ndarray, pd.Series)):
        if any(not isinstance(txt, (str, TexTractor)) for txt in text):
            raise ValueError("Input must be either a string, a TexTractor instance, or a collection (list, array, pandas.Series) of such types")
        return text
    raise ValueError( "Input must be either a string, a TexTractor instance, or a collection (list, array, pandas.Series) of such types")

    
def __extract_textual_features_single_text__(text):
    if not isinstance(text, (str, TexTractor)):
        raise ValueError('Wrong input, must pass either a string or a TexTractor object')
    if isinstance(text, str):
        text = TexTractor(text)
    text = text.process()
    text.get_sentences()
    return text    

def __extract_textual_features_to_df__(texts):
    texts = pd.Series(__validate_text_input__(texts))
    textual_df = texts.map(lambda t: __extract_textual_features_single_text__(t).to_pandas_row()).apply(pd.Series)
    return textual_df

def get_tfidf_extractor(ngram_range):
    tfidf_extractor = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=do_nothing, preprocessor=do_nothing, ngram_range=ngram_range)),  # frequencies
    ('tfidf', TfidfTransformer()),  # tfidf
    ('kbest', SelectKBest(score_func=chi2, k=20)),
])
    return tfidf_extractor

def extract_tfidf(tokenized_corpus, fit, tfidf_extractor, ngram_range=(1,1), to_df=False, **kwargs):
    if tfidf_extractor is None or not tfidf_extractor:
        tfidf_extractor=get_tfidf_extractor(ngram_range=ngram_range)   
    if fit:
        if 'y' not in kwargs:
            raise ValueError('No y_train to fit tfidf extractor')
        tfidf_matrix = tfidf_extractor.fit_transform(tokenized_corpus, y=kwargs['y'])
    else:
        tfidf_matrix = tfidf_extractor.transform(tokenized_corpus)
    if to_df:
        tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray())
        if isinstance(tokenized_corpus, (pd.Series, pd.DataFrame)):
            tfidf_matrix.set_index(tokenized_corpus.index, inplace=True)
    return tfidf_matrix


def __extract_no_tfidf_features__(text, yield_text_tfidf=True):
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    tbwd = default_resources.get_detokenizer()
    ###Extracting textual features(emojis, emoticons, urls, tokens etc)
    textual_df = __extract_textual_features_to_df__(text)
    tagged_tokens = textual_df.apply(lambda row: extract_pos_tags(row, filter_nonstopwords_only=True), axis=1)
    if yield_text_tfidf:
        tfidf_tokens = tagged_tokens.map(lambda x: [tok for tok,tag in x])
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
    pos_tags_counts = map_tags_to_count(tagged_tokens, normalize=True)
    emojis_counts = map_emojis_to_count(textual_df.emojis)
    emoticons_counts = map_emoticons_to_count(textual_df.emoticons)
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

def __extract_features_in_batches__(texts, batch_size, tfidf_extractor, fit_tfidf, saving_directory=False, already_processed_features=None,
    already_processed_tokens=None, **kwargs):
    import os, features_extraction_and_classification.io_utils as io_utils
        
    if fit_tfidf and ('y' not in kwargs):
        raise ValueError('Must pass categories through the "y" variable to fit tfidf_extractor')
    features, tfidf_tokens = pd.DataFrame(), pd.Series()
    if already_processed_features is not None:
        if not isinstance(already_processed_features, pd.DataFrame):
            raise ValueError("Wrong type for 'already_processed_features'. It must be a pandas DataFrame")
        features = pd.DataFrame(already_processed_features)
    if already_processed_tokens is not None:
        if not isinstance(already_processed_tokens, pd.Series):
            raise ValueError("Wrong type for 'already_processed_tokens'. It must be a pandas Series")
        tfidf_tokens = already_processed_tokens
    if saving_directory:
        saved_indexes = []
        saving_index = 0
        if 'saved_indexes' in kwargs:
            saved_indexes = kwargs['saved_indexes']
            saving_index = max(saved_indexes)+1
        features_filename = io_utils.get_filename(io_utils.FEATURES_FILENAME, include_extension=False)
        features_file_extension = io_utils.get_file_extension(io_utils.FEATURES_FILENAME)
        tokens_filename = io_utils.get_filename(io_utils.TFIDF_TOKENS_FILENAME, include_extension=False)
        tokens_file_extension = io_utils.get_file_extension(io_utils.TFIDF_TOKENS_FILENAME)
        
    i=0
    while any(curr_batch:=texts[i:i+batch_size]):
        curr_features, tfidf_toks = __extract_no_tfidf_features__(curr_batch, yield_text_tfidf=True)
        features = pd.concat([features, curr_features], axis=0, ignore_index=False)
        tfidf_tokens = pd.concat([tfidf_tokens, tfidf_toks], axis=0, ignore_index=False)
        if saving_directory:
            curr_features.to_parquet(os.path.join(saving_directory , f'{features_filename}_{saving_index}{features_file_extension}'), index=True)
            pd.DataFrame(tfidf_toks, index=tfidf_toks.index, columns=['tokens']).to_parquet(os.path.join(saving_directory , f'{tokens_filename}_{saving_index}{tokens_file_extension}'), index=True)
            saved_indexes.append(saving_index)
            saving_index+=1
        i+=batch_size
    if fit_tfidf:
        if isinstance(kwargs['y'], (pd.Series, pd.DataFrame)) and isinstance(tfidf_tokens, (pd.Series, pd.DataFrame)):
            merged_toks_to_category = pd.merge(tfidf_tokens, kwargs['y'], left_index=True, right_index=True)
            tfidf_tokens, kwargs['y'] = merged_toks_to_category[merged_toks_to_category.columns[0]], merged_toks_to_category[merged_toks_to_category.columns[-1]]

    tfidf_features = extract_tfidf(tokenized_corpus=tfidf_tokens, 
                                   fit=fit_tfidf, tfidf_extractor=tfidf_extractor, to_df=True, **kwargs)\
        .rename(lambda x: 'word_feat_'+str(x), axis=1)
    final_features = pd.concat([features, tfidf_features], axis=1, ignore_index=False)
    if saving_directory:
        final_features.to_parquet(os.path.join(saving_directory + f'/{io_utils.FEATURES_FILENAME}'), index=True)
        [os.remove(os.path.join(saving_directory, f'features_{saving_index}.parquet')) for saving_index in saved_indexes]
        [os.remove(os.path.join(saving_directory, f'tfidf_tokens_{saving_index}.parquet')) for saving_index in saved_indexes]
    return final_features

def extract_features(text, tfidf_extractor, fit_tfidf, saving_directory=False, overwrite=False, resume_mode=False, **kwargs):
    from features_extraction_and_classification.io_utils import prepare_new_directory
    #from pathlib import Path
    text = pd.Series(__validate_text_input__(text))
    batch_size = kwargs.pop("batch_size", False)
    batch_size = len(text) + 1 if not batch_size else batch_size
    if saving_directory:
        if not resume_mode:
            prepare_new_directory(base_dir=saving_directory, force_to_default_path=False, funct='extract_features', overwrite=overwrite)
            #Path(saving_directory).mkdir(exist_ok=overwrite, parents=True) ###TO IMPROVE!!! Should be mkdir(exist_ok=True) if calling_funct is train/predict... otherwise prepare_new_directory
    features = __extract_features_in_batches__(texts=text, tfidf_extractor=tfidf_extractor, fit_tfidf=fit_tfidf, saving_directory=saving_directory, batch_size=batch_size, **kwargs)
    return features
        
def _extract_features_old_(text, tfidf_extractor, fit_tfidf, **kwargs):
    if tfidf_extractor is None or not tfidf_extractor:
        if not fit_tfidf:
            raise ValueError('Cannot transform texts into tfidf vectors without a proper vectorizer in input')
        else:
            tfidf_extractor=get_tfidf_extractor(kwargs.get('ngram_range', (1,1)))
    
    tbwd = default_resources.get_detokenizer()
    ###Extracting textual features(emojis, emoticons, urls, tokens etc)
    textual_df = __extract_textual_features_to_df__(text)
    tagged_tokens = textual_df.apply(lambda row: extract_pos_tags(row, filter_nonstopwords_only=True), axis=1)
    #Extracting TF-IDF for non-stopwords non-symbols tokens only
    tfidf_features = extract_tfidf(tokenized_corpus=tagged_tokens.map(lambda x: [tok for tok,tag in x]), fit=fit_tfidf, tfidf_extractor=tfidf_extractor, to_df=True, **kwargs).rename(lambda x: 'word_feat_'+str(x), axis=1)
    
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

    #Mapping textual features to counts in BOW style - Normalizing by number of tokens
    pos_tags_counts = map_tags_to_count(tagged_tokens, normalize=True)
    emojis_counts = map_emojis_to_count(textual_df.emojis)
    emoticons_counts = map_emoticons_to_count(textual_df.emoticons)
    counts_df = pd.concat([entities_df, emojis_counts, emoticons_counts,
                           textual_df[['urls','mentions','repeatedPunctuation','hashtags','badwords','uppercaseWords', 'tokens', 'sentences']].map(lambda c: len(c)).rename(lambda c: c+'_count',axis=1)
                          ], axis=1, ignore_index=False)
    counts_df = pd.concat([counts_df[['tokens_count', 'sentences_count']],
                          counts_df.drop(['tokens_count', 'sentences_count'],axis=1).div(counts_df['tokens_count'],axis=0)
                          ], axis=1, ignore_index=False)

    #Putting all features together
    features_df = pd.concat([counts_df, pos_tags_counts, toxicity_df,modality_df,sentiment_df,emotions_df,vad_df,social_df,moral_foundations_df,
                            tfidf_features],
          axis=1, ignore_index=False)

    import random
    features_df['Populism'] = [random.random() for _ in range(len(features_df))]
    features_df['PeopleCentrism'] = [random.random() for _ in range(len(features_df))]
    features_df['AntiElitism'] = [random.random() for _ in range(len(features_df))]
    features_df['EmotionalAppeal'] = [random.random() for _ in range(len(features_df))]

    if features_df.isna().any().any():
        columns_with_nans = features_df.columns[features_df.isnull().any()]
        for column in columns_with_nans:
            features_df[column] = __fillna__(features_df[column], training_set=pd.read_parquet('C:/Users/onest/Documents/TextAn/tesi/code/data/models/ML/final/train.parquet'))
    return features_df

