import default_config
import numpy as np
import pandas as pd
import re
import warnings

def get_default_replacing_dict() -> dict[str, str]:
    default_replacing_strings = {default_config.default_urls_colname: ' myUrlObj ' , 
                         default_config.default_emojis_colname: ' myEmojiObj ', 
                        default_config.default_emoticons_colname: ' myEmotObj ',
                        default_config.default_mentions_colname: {'user': ' myUsMentObj ', 'subreddit': ' myRdMentObj '},
                        default_config.default_repeated_punctuation_colname: ' myPunctObj ',
                        default_config.default_badwords_colname: ' myBadWObj ',
                        default_config.default_hashtags_colname:' myHashTObj ',
                        default_config.default_uppercase_words_colname:' myUpCaseObj '}
    return default_replacing_strings

def __get_standard_settings_for_replacement__(**kwargs) -> dict[str]:
    """
    Will return the default settings for replace_features_in_text function as a dict (kwargs)
    """
    if 'columns_to_remove' not in kwargs or kwargs['columns_to_remove'] is None:
        kwargs['columns_to_remove']=[]


    if 'columns_to_replace' not in kwargs or kwargs['columns_to_replace'] is None:
        def_columns = [default_config.default_urls_colname, default_config.default_emojis_colname,
                   default_config.default_emoticons_colname, default_config.default_mentions_colname,
                   default_config.default_repeated_punctuation_colname]
        columns_to_remove = kwargs.get('columns_to_remove')
        kwargs['columns_to_replace'] = [c for c in def_columns if c not in columns_to_remove]

    if 'replace_mentions_with_twitter' in kwargs and kwargs['replace_mentions_with_twitter'] and \
        default_config.default_mentions_colname not in kwargs['columns_to_remove']:
        kwargs['columns_to_replace'] = list (set(kwargs['columns_to_replace']) | set([default_config.default_mentions_colname]) )
    if 'strip' not in kwargs or kwargs['strip'] is None:
        if 'revert' in kwargs and kwargs ['revert']:
            kwargs['strip']=True
        else:
            kwargs['strip']=False
        



    return kwargs


def get_sorted_features(row: pd.Series, columns_to_replace: list[str], columns_to_remove: list[str], add_replacements: bool = True, **kwargs) -> list[dict]:
    """
    Will sort all of the features, based on their position in text, in the current row and will return them.
    If add_replacements is True, will add to each feature a 'replacement_string' attribute by assigning it the default value (e.g. 'MyEmotObj' for emoticons, etc)
    If 'replacements' in kwargs, will instead use such replacements as the 'replacement_string' attributes.
    If 'strip' in kwargs and it is set to True, the 'replacement_string' will be stripped.
    """
    if not add_replacements:
        columns = columns_to_replace+columns_to_remove
        return sorted(
            sum([list(row[c]) for c in columns], []),         
            key = lambda e: e['match_start'])
    
    replacements = kwargs['replacements'] if kwargs.get('replacements') else get_default_replacing_dict()
    if 'strip' in kwargs and kwargs['strip']:
        replacements = {k:v.strip() if isinstance(v,str) else {subk:subv.strip() for subk,subv in v.items()} for k,v in replacements.items()}

    features_with_replacements = []
    for c in columns_to_replace:
        if c!=default_config.default_mentions_colname:
            features_with_replacements.extend([{**t, 'replacement_string':replacements[c]} for t in row.copy()[c]])
        else:
            #curr_feature = default_config.single_feature_names_dict[default_config.default_mentions_colname]
            curr_feature = 'feature_value'
            if 'replace_mentions_with_twitter' in kwargs and kwargs['replace_mentions_with_twitter']:
                 features_with_replacements.extend([{**t,
                            'replacement_string': t[curr_feature].replace('u/', '@') if t[curr_feature].startswith('u/')
                            else t[curr_feature].replace('r/', '#') if t[curr_feature].startswith('r/')
                           else 'myMentObj'} for t in row.copy()[c]])
            else:
                features_with_replacements.extend([{**t,
                            'replacement_string':replacements[c]['user'] if t[curr_feature].startswith('u/')
                            else replacements[c]['subreddit'] if t[curr_feature].startswith('r/')
                           else 'myMentObj'} for t in row.copy()[c]])
    for c in columns_to_remove:
        features_with_replacements.extend([{**t, 'replacement_string':' '} for t in row.copy()[c]])
        
        
    try:
        return sorted(features_with_replacements, key=lambda x: x['match_start'])
    except TypeError:
        try:
            return sorted(features_with_replacements, key=lambda x: x['position_in_text'])
        except TypeError as e:
            raise e



def __replace_features_from_raw_text__(text: str, sorted_features: list[dict]) -> str:
    """
    Replaces the features found in text by their replacement string, and returns the replaced text after such replacements.
    sorted_features is a list of dict, each containing:
    - match_start: the starting index of a feature,
    - match_end: the ending index of a feature,
    - replacement_string: the replacement for a feature.
    """
    total_offset=0
    final_text = []
    last_index=0
    for curr_feat in sorted_features:
        start,end = curr_feat['match_start'], curr_feat['match_end']
        if start is None or end is None or not isinstance(start, int) or not isinstance(end, int):
            try:
                curr_feature_string = [v for k,v in curr_feat.items() if k in default_config.single_feature_names_dict.values()][0]
                curr_start_index = text[last_index:].find(curr_feature_string)
                start, end = curr_start_index+last_index, curr_start_index+last_index+len(curr_feature_string)
                warnings.warn("Invalid indexes provided for the current feature {}. Will be automatically extracted by the current text, but might generate mismatches when replacing. Please use explicit indexes to avoid unexpected behaviours during replacements".format(str(curr_feat)) )
            except:
                raise ValueError('Cannot replace feature {} with no explicit indexes, couldnt find a matching string in current text.'.format(str(curr_feat)))
        rep_str = curr_feat['replacement_string']
        curr_offset=max([0, len(rep_str)-(end-start)])
        final_text.append(text[last_index:start]+rep_str)
        total_offset+=curr_offset
        last_index = end
    
    final_text.append(text[last_index:])
    return ''.join(final_text)
    
    
def __replace_features_from_tokens__(tokens: list[str], sorted_features: list[dict], features_names_to_skip: list[str]) -> list[str]:
    """
    Starting from the @input tokens,
    1. replaces the tokens that are in sorted_features by their 'replacement_string' attribute,
    2. removes the tokens that are in features_names_to_skip,
    3. returns the final tokens after such replacements/removals
    """
    final_tokens = []
    try:
        curr_feat = next(sorted_features)
        curr_feat_type, curr_feat_str, curr_feat_repl = (curr_feat['feature_type'],curr_feat['feature_value'], curr_feat['replacement_string'])
    except: ###if no features_tokens, just returns the original ones
        final_tokens = tokens
        return final_tokens
    for token in tokens:
        if curr_feat is None: ###if no feature_tokens left
            final_tokens.append(token)
        elif token!=curr_feat_str:
            final_tokens.append(token) ###keeps original token if it is not a "feature" token
        elif curr_feat_type not in features_names_to_skip:
            final_tokens.append(curr_feat_repl) ###replaces original token with replacement if it's a feat to repl
            try:
                curr_feat = next(sorted_features)
                curr_feat_type, curr_feat_str, curr_feat_repl = (curr_feat['feature_type'], curr_feat['feature_value'], curr_feat['replacement_string'])
            except:
                curr_feat, curr_feat_type, curr_feat_str, curr_feat_replacement_str = None, None, None, None
        else: ###if curr token is a feature to remove, just skips it and moves on... 
            try:
                curr_feat = next(sorted_features) 
                curr_feat_type, curr_feat_str, curr_feat_repl = (curr_feat['feature_type'], curr_feat['feature_value'], curr_feat['replacement_string'])
            except:
                curr_feat, curr_feat_type, curr_feat_str, curr_feat_replacement_str = None, None, None, None

    return final_tokens


def __reset_features_from_replaced_text__(text: str, sorted_features: list[dict], columns_to_remove: list[str], **kwargs) -> str:
    """
        Starting from the @input text,
        1. looks for the text substrings that are in sorted_features by their 'replacement_string' attribute,
        2. replaces such substrings with original text substrings if they are not features to remove, otherwise replaces them with whitespace.
        3. returns the final text after such replacements

        If 'replace_mentions_with_twitter' (bool) is in kwargs and it is set to True, the reddit mentions "u/user" and "r/subreddit" will be replaced as "@user" and "#subreddit".
        """
    replace_mentions_with_twitter = kwargs.get('replace_mentions_with_twitter') if kwargs.get('replace_mentions_with_twitter') else False
    if not isinstance(replace_mentions_with_twitter, bool):
        raise ValueError("'replace_mentions_with_twitter' must be boolean")
    sorted_features = list(sorted_features)
    if not sorted_features:
        return text
    searching_features_substrings = [e['replacement_string'] for e in sorted_features]
    searching_features_patt = '|'.join(set(searching_features_substrings))
    sorted_features = map(lambda el: 
                            [(k,v) for k,v in el.items() if k in default_config.single_feature_names_dict.values()][0], sorted_features)
    features_to_remove = list(map(lambda c: default_config.single_feature_names_dict[c], columns_to_remove))

    def capture_features_and_kill(match):
        curr_el = next(sorted_features)
        curr_feat = curr_el[0]
        curr_orig_str = curr_el[1]
        if curr_feat in features_to_remove:
            return ' '
        elif replace_mentions_with_twitter and curr_feat==default_config.single_feature_names_dict[default_config.default_mentions_colname]:
            return curr_orig_str.replace('u/', '@') if curr_orig_str.startswith('u/') else curr_orig_str.replace('r/', '#') if curr_orig_str.startswith('r/') else curr_orig_str
        return curr_orig_str

    replaced_text = re.sub(searching_features_patt, capture_features_and_kill, text)#.strip()
    left_items, any_left = 0, True
    while any_left:
        try:
            next(sorted_features)
            left_items += 1
        except:
            any_left=False
    if left_items:
        print(left_items , 'unmatched...')
    return replaced_text#, left_items



def __reset_features_from_replaced_tokens__(tokens: list[str], sorted_features: list[dict], features_names_to_remove: list[str], **kwargs) -> list[str]:
    #dropping positions from features as we don't need them anymore -> sorted_features_replacement will be a list of tuples: (curr_replaced_str, feature_type, orig_str), e.g. (my_Emoji_obj, 'emoji',😃) for each feature of the curr row
    if 'replace_mentions_with_twitter' in kwargs and kwargs['replace_mentions_with_twitter']:
        sorted_features = map(lambda el: 
                            [(el['replacement_string'], el['feature_type'],(v:=el['feature_value']).replace('u/','@') if v.startswith('u/') else v.replace('r/', '#') if v.startswith('r/') else v) if el['feature_type']==default_config.single_feature_names_dict[default_config.default_mentions_colname]
                                else (el['replacement_string'], el['feature_type'], el['feature_value'])][0]
                             , 
                                           sorted_features)
    else:
        sorted_features = map(lambda el: 
                            [(el['replacement_string'], el['feature_type'],el['feature_value'])][0], sorted_features)
        
    split_repeated_punctuation=kwargs.get('split_repeated_punctuation') if kwargs.get('split_repeated_punctuation') else False
    if split_repeated_punctuation: 
        curr_tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=False, reduce_len=True)
    final_tokens = []
    try:
        next_feat_token = next(sorted_features) 
        next_feat_repl, next_feat_type, next_feat_str = next_feat_token
    except: ###if no features_tokens, just returns the original ones
        final_tokens = tokens
        return final_tokens
    for token in tokens:
        if next_feat_token is None: ###if no feature_tokens left
            final_tokens.append(token)
        elif token!=next_feat_repl:
            final_tokens.append(token) ###keeps original token if it is not a "feature" token
        elif next_feat_type not in features_names_to_remove:
            if split_repeated_punctuation and \
        next_feat_type == default_config.single_feature_names_dict[default_config.default_repeated_punctuation_colname]:
                final_tokens.extend(curr_tokenizer.tokenize(next_feat_str)) ###splitting multiple punctuation
            else:
                final_tokens.append(next_feat_str) ###replaces curr replacement with original token if it's a feat to repl
            try:
                next_feat_token = next(sorted_features) 
                next_feat_repl, next_feat_type, next_feat_str = next_feat_token
            except:
                next_feat_token, next_feat_repl, next_feat_type, next_feat_str = None, None, None, None
        else: ###if curr token is a feature to remove, just skips it and moves on... 
            try:
                next_feat_token = next(sorted_features) 
                next_feat_repl, next_feat_type, next_feat_str = next_feat_token
            except:
                next_feat_token, next_feat_repl, next_feat_type, next_feat_str = None, None, None, None

    return final_tokens



def replace_features_in_text(row: pd.Series,
                             columns_to_replace: list[str] = None,
                             columns_to_remove: list[str] =[],
                             text_col: str = 'text',
                             revert: bool = False,
                             strip: bool = None,
                             replace_mentions_with_twitter: bool = False,
                             split_repeated_punctuation: bool = False):
    """
    Replaces the current features in the row attributes by their default replacement. (e.g. emoticon feature -> 'MyEmotObj')

    @input text_col: string which indicates the current pandas column that contains the current text (might be a list of tokens as well).
    @input columns_to_replace: list containing the features to replace. If None, the default features will be 'emoticons', 'emojis', 'urls', 'mentions', 'repeatedPunctuations'
    @input columns_to_remove: list containing the features to remove. They will be replaced by ' ' in the current text
    @input replace_mentions_with_twitter will replace u/user with @user and r/subreddit with #subreddit
    @input revert: boolean, if True will look for replaced_features (e.g. myUrlObj) in the text and will replace them with original features in the text
    @input strip, if True the features will be kept with the same space characters in between, otherwise a space will be added before and after each feature... useful for tokenization. Strongly suggested to use it when reverse=True, in particular if the original string was replaced with those features by adding a space in bef/after (default behaviour).
    """
    curr_vars = __get_standard_settings_for_replacement__(**locals())
    
    columns_to_replace = curr_vars['columns_to_replace']
    columns_to_remove = curr_vars['columns_to_remove']
    strip = curr_vars['strip']
    if len(set(columns_to_remove))!=len(columns_to_remove) or len(set(columns_to_replace))!=len(columns_to_replace):
        raise ValueError('Input columns must be unique')
    elif len(set(columns_to_replace).intersection(set(columns_to_remove))):
        raise ValueError('Columns to remove cannot overlap columns to replace')
    
    orig_text = row[text_col]
    tokenized=False
    if not isinstance(orig_text, str):
        if isinstance(orig_text, (list,np.ndarray,pd.Series)):
            tokenized=True
        else:
            raise ValueError('Wrong text format for the text_col in input')
    
    
    
    def_replacements = get_default_replacing_dict()
    if not revert:
        sorted_features = (f for f in 
                       get_sorted_features(row=row, columns_to_replace=columns_to_replace, 
                                          columns_to_remove=columns_to_remove, add_replacements=True, 
                                         strip=strip, replace_mentions_with_twitter=replace_mentions_with_twitter)
                          )
        
        if tokenized:
            curr_features_names_to_remove = [v for k,v in default_config.single_feature_names_dict.items() if k in columns_to_remove]
            return __replace_features_from_tokens__(orig_text, sorted_features=sorted_features, features_names_to_skip=curr_features_names_to_remove)
        
        else: ###if orig_text is a string (not tokenized text in input)
            return __replace_features_from_raw_text__(text=orig_text, sorted_features=sorted_features)

    else:
        sorted_features = (f for f in 
                       get_sorted_features(row=row, **__get_standard_settings_for_replacement__(columns_to_replace=columns_to_replace+columns_to_remove,
                                                                                           columns_to_remove=[],
                                                                                           revert=True))
                          )
        if tokenized:
            curr_features_names_to_remove = [v for k,v in default_config.single_feature_names_dict.items() if k in columns_to_remove]
            return __reset_features_from_replaced_tokens__(tokens=orig_text, sorted_features=sorted_features, 
                                                       features_names_to_remove=curr_features_names_to_remove,
                                                       replacements = def_replacements, 
                                                       replace_mentions_with_twitter = replace_mentions_with_twitter
                                                      )
        else:
            return __reset_features_from_replaced_text__(text=orig_text, sorted_features=sorted_features,
                                                columns_to_remove=columns_to_remove,
                                                columns_to_replace=columns_to_replace,
                                                tokenized=tokenized,
                                                replacements = def_replacements,
                                                replace_mentions_with_twitter=replace_mentions_with_twitter)