import pandas as pd
import numpy as np
import emosent, warnings
from utils import flatten, filter_list
from nltk import ne_chunk , pos_tag#, word_tokenize, 
from nltk.corpus import wordnet
from pattern.en import sentiment as pattern_sentiment
from pattern.en import modality
from text_processing.textractor import TexTractor
from text_processing.text_replacement import replace_features_in_text
import default_config, features_extraction_and_classification.default_resources as default_resources


"""
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
"""

spacy_entities_cols = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP',
 'ORDINAL', 'ORG', 'PERCENT', 'PERSON','PRODUCT', 'QUANTITY','TIME', 'WORK_OF_ART']
nltk_entities_cols = ['facility', 'gpe', 'gsp', 'location', 'organization', 'person']


def extract_pos_tags(tokenized_text: TexTractor or list[str], filter_nonstopwords_only: bool = True, tokens_col: str = 'tokens') -> list[tuple[str,str]]:
    if not filter_nonstopwords_only:
        if isinstance(tokenized_text, TexTractor):
            tokenized_text = tokenized_text.process()
        if isinstance(tokenized_text, (TexTractor, pd.Series)):
            tokenized_text = tokenized_text[tokens_col]
        if not isinstance(tokenized_text, (list, np.ndarray)):
            raise ValueError('Wrong input - need a list of tokens to map to pos tags')
        tagged_tokens = pos_tag(tokenized_text)
        return tagged_tokens
    if not isinstance(tokenized_text, (TexTractor, pd.Series)):
        raise ValueError('You must pass a TexTractor instance to remove stopwords and urls/mentions/emojis/emoticons from tokens')
    if isinstance(tokenized_text, TexTractor):
        tokenized_text = tokenized_text.process()

    mytext = tokenized_text
    tokenized_text = replace_features_in_text(mytext, text_col=tokens_col, columns_to_remove=['emoticons','emojis','urls'], columns_to_replace=[] , replace_mentions_with_twitter=True)
    tagged_tokens = pos_tag(tokenized_text)
    
    filtered_tokens = replace_features_in_text(mytext, text_col=tokens_col, columns_to_replace=[], columns_to_remove=['urls','emoticons','emojis','mentions','repeatedPunctuation'])
    filtered_tokens = [t for token in filtered_tokens if (t:=token.lower()) not in default_resources.get_stopwords() and t.islower()]
    filtered_tags = [tagged_tokens[ind]
                     for ind in 
                         filter_list(list_to_filter=[ tok.lower() for tok,tag in  tagged_tokens],
                                     list_to_look=filtered_tokens,
                                     return_indexes=True) ]
    if filtered_tokens != [tok.lower() for tok,tag in filtered_tags]:
        warnings.warn('some tag was filtered out when removing stopwords')
        print([tok.lower() for tok,tag in filtered_tags if tok not in filtered_tokens])

    return filtered_tags

    
    

def _calculate_engspacy_sentiment_onescoreonly(curr_res_dct, weight_by_neutral=False):
    if 'negative' not in curr_res_dct or 'positive' not in curr_res_dct:
        raise ValueError('negative and positive must be in the current result')
    positive, negative = curr_res_dct['positive'], curr_res_dct['negative']
    abs_sentiment = positive-negative
    if not weight_by_neutral:
        return abs_sentiment
    neutral = curr_res_dct['neutral'] if 'neutral' in curr_res_dct else 1-(positive+negative)
    return abs_sentiment*(1-neutral)


def extract_entities_nltk(tagged_tokens):
    processed_tokens = [(token, pos) if not (token.startswith('#') or token.startswith('@')) else (token[1:], pos) for token, pos in tagged_tokens]
    entities_tree = ne_chunk(processed_tokens)
    
    # Dict to store the extracted nltk NERs -> will be structured as {'entity_type': list_of_tokens_of_entity_type}
    entities_dict = {k:[] for k in nltk_entities_cols}
    for subtree in entities_tree:
        if hasattr(subtree, 'label'):  # Controlla se è un chunk
            entity_type = subtree.label()  # Entity type
            entity_name = ' '.join([word for word, _ in subtree.leaves()])  # Name (i.e. curr token) of the entity
            try:
                entities_dict[entity_type.lower()].append(entity_name)
            except KeyError:
                entities_dict[entity_type.lower()] = [entity_name]
    return dict(entities_dict)


def extract_entities_spacy(text):
    # Dict to store the extracted spacy NERs -> will be structured as {'entity_type': list_of_tokens_of_entity_type}
    spacy_nlp = default_resources.get_spacy_nlp()
    entities = spacy_nlp(text).ents

    entities_dict = {k:[] for k in spacy_entities_cols}
    for ent in entities:
        entity_type = ent.label_#.lower()  # Entity type
        try:
            entities_dict[entity_type].append(ent.text)   # Name (i.e. curr token) of the entity
        except KeyError:
            entities_dict[entity_type] = [ent.text]
    return dict(entities_dict)
    

def filter_entities_dict(entities_dct: dict[str, list[str]], keys_to_join: list[tuple[list[str],str]]=[(['TIME','DATE'], 'TIME'),(['CARDINAL','ORDINAL','PERCENT','QUANTITY'], 'NUMERICAL')] , keys_to_drop: list[str]=[]) -> dict[str, list[str]]:
    """
    Re-structures the entities dict by either joining keys(i.e. entity types) into a single key or removing keys.
    @keys_to_join is a list of tuples, each containing the entity types to join together and the entity name of the joined type (e.g.: (['TIME','DATE'], 'TIME') )
    @keys_to_drop is a list of string, i.e. the entity types to remove from the entities found
    """
    if not keys_to_drop and not keys_to_join:
        return {k:v if v is not None else [] for k,v in entities_dct.items()}
    if len(features_to_join:=flatten(x[0] for x in keys_to_join)) != len(set(features_to_join)) or len(set(features_to_join) & set(keys_to_drop)):
        raise ValueError('Some key is used multiple times in either multiple features or both in removed features and joined ones')
    
    new_dct = {k:v if v is not None else [] for k,v in entities_dct.items() if k not in set(flatten(x[0] for x in keys_to_join)+keys_to_drop)}
    
    if not keys_to_join:
        return new_dct
    
    keys_to_join = {e[1]:e[0] for e in keys_to_join}
    for new_k in keys_to_join.keys():
        new_dct[new_k] = sum([],flatten([curr_value if (curr_value:=entities_dct.get(k, [])) is not None else [] for k in keys_to_join[new_k]]))
    return new_dct


def extract_moral_foundations(texts) -> pd.DataFrame:
    ### frameaxis.get_fa_scores() will split text by ' ' and calculate scores for each token
    fa = default_resources.get_frameaxis_instance()
    
    if isinstance(texts, (pd.DataFrame, pd.Series)):
        frameaxis_df = pd.DataFrame(index=texts.index)
    else:
        frameaxis_df = pd.DataFrame()
        texts = pd.Series(texts if isinstance(texts, (np.ndarray, list)) else [texts])
    frameaxis_df['frameaxis_text'] = texts
    frameaxis_df = frameaxis_df[['frameaxis_text']].reset_index()
    
    emfd_scores_clean_replacedmentions = fa.get_fa_scores(df=frameaxis_df, doc_colname='frameaxis_text', tfidf=False)
    ind_name = 'index' if texts.index.name is None else texts.index.name
    frameaxis_df = frameaxis_df.set_index(ind_name).drop('frameaxis_text',axis=1).\
                    join(emfd_scores_clean_replacedmentions.drop('frameaxis_text',axis=1).set_index(ind_name), how='left')#.fillna(0)
    del(emfd_scores_clean_replacedmentions)
    if frameaxis_df.index.name=='index':
        frameaxis_df.index.name=''
    return frameaxis_df

def extract_toxicity(text):
    detoxify_instance = default_resources.get_detoxify_instance()
    return detoxify_instance.predict(text)
    
def extract_sentiment(text, include_subjectivity=True):
    vader_sent = default_resources.get_vader_analyzer().polarity_scores(text)['compound']
    engspacy_sent = default_resources.get_engspacysentiment_nlp()(text)
    engspacy_sent = _calculate_engspacy_sentiment_onescoreonly({k:engspacy_sent.cats[k] for k in ['positive','negative']})
    pattern_sent, pattern_subjectivity = pattern_sentiment(text)
    scores = {'vader_sentiment':vader_sent, 'eng_spacysentiment':engspacy_sent, 'pattern_sentiment':pattern_sent}
    if include_subjectivity: 
        scores['pattern_subjectivity'] = pattern_subjectivity
    return scores

def extract_modality(text):
    return modality(text)

#from text_processing.vad_socialness_scoring import SocialnessCalculator
#socialness_extractor = SocialnessCalculator('C:/Users/onest/Documents/TextAn/tesi/code/data/utils/SocialnessNorms_DiveicaPexmanBinney2021.csv', min_max=(0,1), expand_lexicon=True, limit_expansion=True)
def extract_socialness(text):
    #strong_socialness_threshold = round(pd.Series(index=sc.scores_dict.keys(), data=sc.scores_dict.values(), name='socialness').quantile(0.8), 3) ---> 0.63
    socialness_values = default_resources.get_socialness_extractor().calculate_socialness(text, strong_socialness_threshold=0.63, fillna=False, lemmatize=True)
    try:
        socialness_values['strong_socialness_tokens_ratio'] = ((socialness_values['socialness_strong_tokens'] * (socialness_values['num_matched_tokens']/socialness_values['num_matches']))/ socialness_values['num_tokens'] )
    except ZeroDivisionError:
        socialness_values['strong_socialness_tokens_ratio'] = 0
    socialness_values = {k:socialness_values[k] for k in ['strong_socialness_tokens_ratio', 'socialness_mean']}
    return socialness_values

"""
def extract_socialness_df(texts):
    social_df = sc.calculate_socialness_df(texts, strong_socialness_threshold=strong_socialness_threshold, fillna=False, lemmatize=True)
    social_df['strong_socialness_tokens_ratio'] = ((social_df['socialness_strong_tokens'] * (social_df['num_matched_tokens']/social_df['num_matches']))/ social_df['num_tokens'] ).fillna(0)
    social_df = social_df[['strong_socialness_tokens_ratio', 'socialness_mean']]
    return social_df
"""
#from text_processing.vad_socialness_scoring import NRCVad
#vad_extractor = NRCVad('C:/Users/onest/Documents/TextAn/tesi/code/data/utils/NRC-VAD-Lexicon/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', expand_lexicon=True, limit_expansion=True, min_max=(-1,1))
def extract_vad(text):   
    vad_scores = default_resources.get_vad_extractor().calculate_vad_scores(text, fillna=False, lemmatize=True)
    for feat in ['valence', 'arousal', 'dominance']:
        try:   
            vad_scores[f'strong_{feat}_tokens_ratio'] = ((vad_scores[f'{feat}_strong_matches'] * (vad_scores['num_matched_tokens']/vad_scores['num_matches']))/ vad_scores['num_tokens'] )
        except ZeroDivisionError:
            vad_scores[f'strong_{feat}_tokens_ratio'] = 0
            
    return {k:vad_scores[k] for k in ['strong_valence_tokens_ratio',
 'valence_mean',
 'strong_arousal_tokens_ratio',
 'arousal_mean',
 'strong_dominance_tokens_ratio',
 'dominance_mean']}


"""
def extract_vad_df(texts):
    vad_df = vad.calculate_vad_df(texts, fillna=False, lemmatize=True)
    vad_cols = []
    for feat in ['valence', 'arousal', 'dominance']:
        vad_df[f'strong_{feat}_tokens_ratio'] = ((vad_df[f'{feat}_strong_matches'] * (vad_df['num_matched_tokens']/vad_df['num_matches']))/ vad_df['num_tokens'] ).fillna(0)
        vad_cols.extend([f'strong_{feat}_tokens_ratio', f'{feat}_mean'])
    vad_df = vad_df[vad_cols]
    return vad_df
"""

emotions_cols = ["trust", "positive", "negative", "joy", "surprise", "anger", "fear", "anticipation", "sadness", "disgust",]
def extract_emotions(tokenized_text):
    if not isinstance(tokenized_text, (list, np.ndarray, pd.Series)):
        raise ValueError('Input text must be tokenized')
    num_tokens = len(tokenized_text)
    text = default_resources.get_detokenizer().detokenize(tokenized_text)
    emotions_scores = default_resources.get_nrclex_extractor().load_raw_text(text).raw_emotion_scores
    emotions_scores = {k:emotions_scores.get(k,0)/num_tokens for k in emotions_cols}
    return emotions_scores

populism_cols = ['Populism', 'PeopleCentrism', 'AntiElitism', 'EmotionalAppeal']
def extract_populism(text): 
    """
    !!! UNDONE, SO FAR IT ASSIGNS RANDOM VALUES!!!
    """
    import random
    curr_populism = {feat: random.random() for feat in populism_cols}
    return curr_populism

def map_sentiment_score_to_category(curr_sentiment_score: float, negative_threshold: float, positive_threshold: float) -> int:
    if curr_sentiment_score>=positive_threshold:
        return 1
    if curr_sentiment_score<=negative_threshold:
        return -1
    return 0
    
def get_emoji_sentiment(emoji: str) -> int:
    """
    Returns sentiment associated to the current emoji : -1 if negative, 0 if neutral, 1 if positive.
    """
    curr_score = emosent.get_emoji_sentiment_rank(emoji)
    if curr_score is None:
        curr_score = 0
    else:
        curr_score = curr_score['sentiment_score']
    return map_sentiment_score_to_category(curr_score, negative_threshold=-0.1, positive_threshold=0.1)


def get_emoticon_sentiment(emoticon: str) -> int:
    """
    Returns sentiment associated to the current emoticon : -1 if negative, 0 if neutral, 1 if positive.
    """
    try:
        curr_score = default_config.emoticons_list_vader.get(emoticon)
    except KeyError:
        curr_score = 0
    return map_sentiment_score_to_category(curr_score, negative_threshold=-0.1, positive_threshold=0.1)