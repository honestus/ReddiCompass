import re, warnings
from typing import Sequence
import pandas as pd
import numpy as np
from unidecode import unidecode

from ReddiCompass import default_config
from ReddiCompass.text_processing.text_replacement import replace_features_in_text, __replace_features_from_raw_text__
from ReddiCompass.text_processing.raw_text_features_extraction import extract_emojis, extract_emoticons, extract_mentions, extract_repeated_punctuation_marks, extract_url
#from feature_extraction import search_badwords_tokens_greedy
from ReddiCompass.text_processing.LexiconMatcher import LexiconMatcher
from ReddiCompass.text_processing.text_utils import is_upper_word
from ReddiCompass.utils import flatten

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize



class TextualFeature:
    def __init__(self, feature_type, feature_value, match_start=None, match_end=None, replacement_string=None, position_in_text=None, subcategory=None) -> None:
        """        
        :param feature_type: str 
        :param feature_value: str
        :param start_index: int -> the index of the first character of the current feature in the text
        :param end_index: int -> the index of the last character of the current feature in the text
        :param replacement_string: str -> the string to replace the current feature_value in text
        :param feature_number: int -> n meaning that this current feature is the n-th feature found in text
        :param subcategory: str -> category of the current feature (e.g. 'user mention' will be a subcategory of 'mention', 'positive_emoticon' will be a category of 'emoticon' etc)
        """
        if feature_type in ['match_start', 'match_end', 'position_in_text', 'replacement_string']:
            raise ValueError('Invalid feature type')
        self.feature_type = feature_type
        self.feature_value = feature_value
        self.match_start = match_start
        self.match_end = match_end
        self.replacement_string = replacement_string
        self.position_in_text = position_in_text   
        self.subcategory = subcategory
        
    def __str__(self) -> str:
        return  "'{}':'{}', match_start: {}, match_end: {}".format(self.feature_type, self.feature_value, self.match_start, self.match_end)
    
    def __repr__(self):
        return "'{}' , feature type: '{}'".format(self.feature_value, self.feature_type)

    def __eq__(self, other):
        return self.feature_type==other.feature_type and self.feature_value == other.feature_value

    def __lt__(self, other):
        if self.feature_type==other.feature_type:
            return self.feature_value < other.feature_value
        return self.feature_type < other.feature_type

    def get_value(self) -> str:
        return self.feature_value
    
    def get_type(self) -> str:
        return self.feature_type
    
    def get_indexes(self) -> tuple[int,int]:
        return (self.match_start, self.match_end)
    
    def get_replacement_string(self) -> str:
        return self.replacement_string
    
    def get_subcategory(self) -> str:
        return self.subcategory 
    
    ## Will be possible to set/edit replacement and subcategory attributes only
    def set_replacement_string(self, replacement_str) -> None:
        self.replacement_string = replacement_str
        
    def set_subcategory(self, subcategory) -> None:
        self.subcategory = subcategory
    
    def to_dict(self):
        return vars(self)
    
    def keys(self):
        return ['feature_type', 'feature_value', 'match_start', 'match_end', 'position_in_text', 'replacement_string']
    def __getitem__(self, key):
        return getattr(self, key if key!=self.feature_type else 'feature_value') 
    
    def copy(self) -> "TextualFeature":
        return TextualFeature(**vars(self))


class TexTractor:
    def __init__(self, text_data: "str | pd.Series | TexTractor", pre_sanitize: bool = False, tokenizer=None, *,
                 text_col: str = None, tokens_col: str = None, sentences_col: str = None) -> None:
        """
        :param text_data: Can either be a TexTractor instance, a string, or a row of a pandas dataframe, having columns for each feature
        :param tokenizer: The tokenizer to use (default is TweetTokenizer if None)
        :param text_col: The name of the column containing the text (necessary if input_data is a row of a dataframe)
        :param tokens_col: The name of the column containing the tokens
                (if None and text_data comes from a dataframe row, default='tokens'...
                if tokens_col not in text_data, self.tokens=None by default)
"""
        # If no tokenizer is given at init, it will use the TweetTokenizer by default
        self.tokenizer = tokenizer if tokenizer is not None else TweetTokenizer(strip_handles=False, reduce_len=True)

        if not isinstance(text_data, str) and pre_sanitize:
            warnings.warn("'pre_sanitize' is ignored for an already processed text. Please instantiate a new text from string to use it properly.")
            #pre_sanitize = False
        if not isinstance(text_data, TexTractor):
            self._features_names_ = [default_config.default_emojis_colname,
                                  default_config.default_emoticons_colname,
                                  default_config.default_urls_colname,
                                  default_config.default_mentions_colname,
                                  default_config.default_repeated_punctuation_colname,
                                  default_config.default_hashtags_colname,
                                  default_config.default_badwords_colname,
                                  default_config.default_uppercase_words_colname]
        if isinstance(text_data, str):  # from string
            self.text = text_data if not pre_sanitize else self.__sanitize__(text_data)
            for f in self._features_names_:
                setattr(self, f, None)
            self.is_processed = False  # Not processed...
            self.tokens = None  # Will extract all of the features/tokens/sentences when processing
            self.sentences = None
        
        elif isinstance(text_data, pd.Series):  # from already existing text (i.e. a DataFrame row containing all the features)
            if text_col is None:
                raise ValueError("If the input object is a pandas Series, it is mandatory to explicit define text_col.")
            if tokens_col is None:
                tokens_col = 'tokens' if text_col!='tokens' else None
            if sentences_col is None:
                sentences_col = 'sentences' if text_col!='sentences' else None
            self.text = text_data[text_col]
            for f in self._features_names_:
                curr_feature_list = list(map(lambda feat: dict_to_TextualFeature(feat) if isinstance(feat,dict) else dict_to_TextualFeature({default_config.single_feature_names_dict[f]:feat}) if isinstance (feat, str) else feat , text_data.get(f, None) ))
                setattr(self, f, curr_feature_list)
            self.tokens = text_data.get(tokens_col, None)  
            self.sentences = text_data.get(sentences_col, None)  

            if 'clean_text' in text_data:
                self.clean_text = text_data.get('clean_text', None)  

            if all(map(lambda x: x is None, [getattr(self, f) for f in self._features_names_]) ):
                   self.is_processed = False
            else:
                   self.is_processed = True  # Already processed

        elif isinstance(text_data, TexTractor):
            text_data_attributes = vars(text_data)
            for k, v in text_data_attributes.items():
                setattr(self, k, v)
        else:
            raise TypeError("text_data must be either a string or a pandas DataFrame row object")

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        """
        Will print the name of the current features and the number of extracted features for each feature type
        """
        features_dict = {attr: getattr(self, attr) for attr in self._features_names_ + ['tokens']}
        features_summary = "\n".join([f"{feature}: {len(value) if value is not None else 0}"
                                      for feature, value in features_dict.items()])

        return f"{self.text}\n\nFeatures:\n{features_summary}\nIs processed: {self.is_processed}"

    def __repr__(self):
        return self.text

    def __eq__(self, other: "TexTractor"):
        if not isinstance(other, TexTractor):
            raise TypeError('other must be an instance of TexTractor')
        return self.text == other.text


    def __sanitize__(self, text: str) -> str:
        """
        Removes multiple repeated characters (more than 3 times) and multiple repeated words (more than twice)
        """
        repeated_words_regex = r'(?<!\S)((\S+)(?:\s+\2))(?:\s+\2)+(?!\S)'
        repeated_chars_regex = r'(\D)\1{2,}'
        sanitized_text = re.sub(repeated_chars_regex, r'\1\1\1', text, flags=re.IGNORECASE)
        sanitized_text = re.sub(repeated_words_regex, r'\1', sanitized_text, flags=re.IGNORECASE)
        return sanitized_text
   
    
    def process(self, force=False) -> None:
        """
        Processes current TexTractor by extracting all features, extracting tokens and getting the clean text with no punctuations, no numbers, no features.
        """
        if force:
            for f in self._features_names_:
                setattr(self, f, None)
            setattr(self, 'tokens', None)
            setattr(self, 'is_processed', False)
        curr_text = self.text
        if getattr(self, default_config.default_emojis_colname) is None:
            curr_feat_name = default_config.single_feature_names_dict[default_config.default_emojis_colname]
            emojis_gen = extract_emojis(curr_text, retain_length=True)
            found_emojis = next(emojis_gen)
            emojis_features = list(map(lambda p: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_emojis_colname], 
                feature_value=p[curr_feat_name], match_start=p['match_start'], match_end=p['match_end'] ), 
                                       found_emojis) )
            setattr(self, default_config.default_emojis_colname, emojis_features)
            curr_text = next(emojis_gen)
        
        curr_text = unidecode(curr_text)

        if getattr(self, default_config.default_emoticons_colname) is None:
            emoticons_gen = extract_emoticons(curr_text, retain_length=True, position=True)
            found_emoticons = next(emoticons_gen)
            emoticons_features = list(map(lambda p: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_emoticons_colname], 
                feature_value=p[0], match_start=p[1][0], match_end=p[1][1] ), 
                                          found_emoticons) )
            setattr(self, default_config.default_emoticons_colname, emoticons_features)
            curr_text = next(emoticons_gen)
        if getattr(self, default_config.default_urls_colname) is None:
            urls_gen = extract_url(curr_text, retain_length=True, replace_string='', position=True)
            found_urls = next(urls_gen)
            urls_features = list(map(lambda p: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_urls_colname],
                feature_value=p[0], match_start=p[1][0], match_end=p[1][1] ), 
                                     found_urls) )
            setattr(self, default_config.default_urls_colname, urls_features)
            curr_text = next(urls_gen)
        mentions_gen = extract_mentions(curr_text, retain_length=False, position=False, replace_string={'u/': '@', 'r/': '#'})
        tmp_ment, clean_text = next(mentions_gen), next(mentions_gen)
        ### clean text ...
        ### will contain raw text without urls, emoticons, emojis
        ### it will be "sanitized" by removing multiple repeated chars/words
        ### todo: removing punctuation and all tokens that don't contain any alpha chars? -> easy to do but should be done after extracting sentences to ensure the extracted sentences will be consistent
        ### final clean text will be in lower case.
        self.clean_text = ' '.join( self.__sanitize__(clean_text).split() ).strip().lower() ###clean text  .... it will also call sanitize to handle repeated chars/words, it will lower the final text

        if getattr(self, default_config.default_repeated_punctuation_colname) is None:
            punct_gen = extract_repeated_punctuation_marks(curr_text, replace_string=' ', retain_length=True, position=True)
            found_punctuations = next(punct_gen)
            punctuations_features = list(map(lambda p: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_repeated_punctuation_colname], 
                feature_value=p[0], match_start=p[1][0], match_end=p[1][1] ), 
                                             found_punctuations) )
            setattr(self, default_config.default_repeated_punctuation_colname, punctuations_features)
            curr_text = next(punct_gen)
        if getattr(self, default_config.default_mentions_colname) is None:
            mentions_gen = extract_mentions(curr_text, retain_length=True, position=True)
            found_mentions = next(mentions_gen)
            mentions_features = list(map(lambda p: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_mentions_colname], 
                feature_value=p[0], match_start=p[1][0], match_end=p[1][1], subcategory='user' if p[0].startswith('u/') else 'subreddit' if p[0].startswith('r/') else None ), 
                                         found_mentions) )
            setattr(self, default_config.default_mentions_colname, mentions_features)
            curr_text = next(mentions_gen)
        if self.tokens is None and any([
            getattr(self, default_config.default_hashtags_colname) is None, 
            getattr(self, default_config.default_badwords_colname) is None, 
            getattr(self, default_config.default_uppercase_words_colname) is None]):
            #self.text_withreplacements = replace_features_in_text(row=self.to_pandas_row())
            text_with_replacement = replace_features_in_text(self)
            text_with_replacement = unidecode(text_with_replacement)
            self.tokens = self.tokenizer.tokenize(text_with_replacement)
            self.tokens = replace_features_in_text(self, text_col='tokens', revert=True)
        if self.hashtags is None:
            hashtags_found = [(p,t) for p,t in enumerate(self.tokens) if t.startswith('#') and len(t)>2 and any([str.isalpha(char) for char in t]) and t.count('#')<2 and t[1:6].lower()!='x200b']
            hashtags_features = list(map(lambda hasht: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_hashtags_colname], 
                feature_value=hasht[1], position_in_text=hasht[0] ),
                                         hashtags_found))
            setattr(self, default_config.default_hashtags_colname, hashtags_features)
        if self.badwords is None:
            badwords_matcher = LexiconMatcher(lexicon=default_config.badwords_LDNOOBW)
            badwords_found = badwords_matcher.get_matches(tokens=self.tokens, return_indexes=True, ignore_case=True) #search_badwords_tokens_greedy(badwords_list=default_config.badwords_LDNOOBW, tokens=self.tokens, ignore_case=True)
            badwords_features = list(map(lambda w: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_badwords_colname], 
                feature_value=w[2], position_in_text=w[0] ),
                                         badwords_found ))
            setattr(self, default_config.default_badwords_colname, badwords_features)
        if self.uppercaseWords is None:
            uppercase_found = [(p,t) for p,t in enumerate(self.tokens) if is_upper_word(t)]
            uppercase_features = list(map(lambda w: TextualFeature(
                feature_type=default_config.single_feature_names_dict[default_config.default_uppercase_words_colname], 
                feature_value=w[1], position_in_text=w[0] ),
                                         uppercase_found))
            setattr(self, default_config.default_uppercase_words_colname, uppercase_features)
        self.__map_index_to_features__()
        self.is_processed=True
        self.get_sentences()
        return self
    
    
    def get_clean_text(self) -> str:
        """
        Removes urls, emojis and emoticons from text. Returns the text made by tokens that contain at least one alpha char.
        """
        if not self.is_processed:
            self.process()
        if not 'clean_text' in vars(self) or self.clean_text is None:
            valid_tokens = replace_features_in_text(self, text_col='tokens', revert=False, strip=False, replace_mentions_with_twitter=True, columns_to_remove=['urls','emojis','emoticons'], columns_to_replace=[])
            text_after_remov = TreebankWordDetokenizer().detokenize( [tok for tok in valid_tokens if tok.lower().islower()] )
            self.clean_text = ' '.join( self.__sanitize__(text_after_remov).split() ).strip().lower()
        return self.clean_text
        
        
    def add_feature(self, feature_name: str, feature_values: Sequence[dict]=None) -> None:
        """
        Adds a new feature to this TexTractor instance
        
        :param feature_name: The name of the feature to add
        :param feature_values: optional. The values of the feature (must be in the form of 
        """
        # Checks that the feature name is valid (only numbers, letters, underscores, no spaces in between)
        if not re.match(r'^[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*$', feature_name):
            raise ValueError("The name of the new feature must be a single token string composed by alphanumeric chars only, eventually separated through '_'")
        if feature_name in vars(self):
            raise ValueError('Feature {} already exists'.format(feature_name))
        if feature_name in ['match_start', 'match_end', 'position_in_text', 'replacement_string']:
            raise ValueError('Invalid feature name')
        # Validates feature values (if feature_values is None, will assign an empty list as values)
        if feature_values is not None:
            
            feature_values = [dict_to_TextualFeature(v, feature_name_key=feature_name) 
                              if isinstance(v, dict) 
                              else TextualFeature(feature_type=feature_name, feature_value=v)
                             for v in feature_values]
        else:
            feature_values = []
        
        # Adds the current feature to the TexTractor instance
        setattr(self, feature_name, feature_values)
        self._features_names_.append(feature_name)
    

    
    def keys(self) -> Sequence[str]:
        return self._features_names_+['text','clean_text']
    
    def __getitem__(self, key: str):
        if not isinstance(key, str):
            return
        return getattr(self, key)
    
    def copy(self) -> "TexTractor":
        return TexTractor(self)
    
    def apply(self, func):
        """Applies any function to the current text attribute"""
        return func(self.text)
    
    def to_pandas_row(self) -> pd.Series:
        """
        Converts the current TexTractor instance to a pandas DataFrame row.
        Returns a pandas Series having as columns the feature_names and as values the features_values for each feature type
        """
        #Only builds columns for features names attribute, by excluding attributes such as self.tokenizer etc
        data = {attr:v for attr,v in vars(self).items() if not attr.startswith('_') and attr not in ['tokenizer', 'is_processed']}
        
        return pd.Series(data)
        ##self.map(lambda t: vars(t)).apply(pd.Series)


    def get_urls(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        if values_only:
            url_name = default_config.single_feature_names_dict[default_config.default_urls_colname]
            return list(map(lambda feat: feat[url_name], getattr(self, default_config.default_urls_colname)))
        return getattr(self, default_config.default_urls_colname)

    def get_emojis(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        if values_only:
            emoj_name = default_config.single_feature_names_dict[default_config.default_emojis_colname]
            return list(map(lambda feat: feat[emoj_name], getattr(self, default_config.default_emojis_colname)))
        return getattr(self, default_config.default_emojis_colname)

    def get_emoticons(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        if values_only:
            emot_name = default_config.single_feature_names_dict[default_config.default_emoticons_colname]
            return list(map(lambda feat: feat[emot_name], getattr(self, default_config.default_emoticons_colname)))
        return getattr(self, default_config.default_emoticons_colname)

    def get_mentions(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        if values_only:
            ment_name = default_config.single_feature_names_dict[default_config.default_mentions_colname]
            return list(map(lambda feat: feat[ment_name], getattr(self, default_config.default_mentions_colname)))
        return getattr(self, default_config.default_mentions_colname)

    def get_repeated_punctuations(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        if values_only:
            rep_punct_name = default_config.single_feature_names_dict[default_config.default_repeated_punctuation_colname]
            return list(map(lambda feat: feat[rep_punct_name], getattr(self, default_config.default_repeated_punctuation_colname)))
        return getattr(self, default_config.default_repeated_punctuation_colname)

    def get_hashtags(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        return getattr(self, default_config.default_hashtags_colname)
        if values_only:
            hasht_name = default_config.single_feature_names_dict[default_config.default_hashtags_colname]
            return list(map(lambda feat: feat[hasht_name], getattr(self, default_config.default_hashtags_colname)))
        return getattr(self, default_config.default_hashtags_colname)

    def get_badwords(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        return getattr(self, default_config.default_badwords_colname)
        if values_only:
            badw_name = default_config.single_feature_names_dict[default_config.default_badwords_colname]
            return list(map(lambda feat: feat[badw_name], getattr(self, default_config.default_badwords_colname)))
        return getattr(self, default_config.default_badwords_colname)
    
    def get_uppercase_words(self, values_only=False) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        return getattr(self, default_config.default_uppercase_words_colname)
        if values_only:
            uppword_name = default_config.single_feature_names_dict[default_config.default_uppercase_words_colname]
            return list(map(lambda feat: feat[uppword_name], getattr(self, default_config.default_uppercase_words_colname)))
        return getattr(self, default_config.default_uppercase_words_colname)

    def get_all_features(self, values_only=False) -> list[str]:
        if not self.is_processed:
            self.process()
        all_features = list(flatten(x for feature in self._features_names_ if (x:=getattr(self, feature)) ))
        if values_only:
            all_features = list(map(lambda myfeat: myfeat.feature_value, all_features))
        return all_features

    def __get_tokens_split_punctuation__(self) -> list[str]:
        punctuation_features = self.__get_sorted_features__(features=default_config.default_repeated_punctuation_colname)
        if not punctuation_features:
            return self.tokens
        punctuation_strings, punctuation_indexes = (p.feature_value for p in punctuation_features), list(map(lambda p: p.position_in_text, punctuation_features))
        final_tokens = [tok if tok_ind not in punctuation_indexes else self.tokenizer.tokenize(next(punctuation_strings)) for tok_ind, tok in enumerate(self.tokens) ]
        return list(flatten(final_tokens))
    
    def get_tokens(self, split_repeated_punctuation: bool = False) -> list[str]:
        if not self.is_processed:
            self.process()
        if split_repeated_punctuation:
            return self.__get_tokens_split_punctuation__()
        return self.tokens

    def __get_sentences_split_punctuation__(self) -> list[str]:
        """
        Sets multiple punctuation string to its first punctuation character (e.g. from !11!??! to !) in order to have better sentences splits!
        """
        punctuation_features = [f.copy() for f in self.__get_sorted_features__(features=default_config.default_repeated_punctuation_colname)]
        if not punctuation_features:
            if self.sentences is None:
                return sent_tokenize(self.text)
            return self.sentences

        for punct_feature in punctuation_features:
            punct_feature.set_replacement_string(next(e for e in punct_feature.feature_value if e in ['!','?'])) ##sets multiple punctuation string to its first punctuation character (e.g. from !11!??! to !)
        curr_text = __replace_features_from_raw_text__(self.text, sorted_features=punctuation_features)
        return sent_tokenize(curr_text)

    def get_sentences(self, ) -> list[str]:
        sanitize_multiple_punctuation: bool = True #to improve the logic here -> should we use it by default?
        if self.sentences is None:
            self.sentences = sent_tokenize(self.text) if not sanitize_multiple_punctuation else self.__get_sentences_split_punctuation__()
        return self.sentences
    
    def __map_index_to_features__(self, features=None) -> None:
        """
        Assigns index to each feature and sets it to its 'position_in_text' attribute. (i.e. the index of the features' tokens.)
        """
        if features is None:
            features = self._features_names_
        curr_tokens = enumerate(self.tokens)
        for feature_type in features:
            if all([f.position_in_text is not None for f in getattr(self, feature_type)]):
                continue #if all the features of type "feature_type" already have position_in_text validated, just moves on to the next feature type
            try:
                curr_features = sorted(getattr(self, feature_type), key=lambda f: f.match_start)
            except Exception as e:
                curr_features = getattr(self, feature_type)
            curr_offset, curr_tokens = 0, self.tokens
            for feat in curr_features:
                curr_index = curr_tokens.index(feat.feature_value)
                feat.position_in_text=curr_index+curr_offset
                curr_offset+=curr_index+1
                curr_tokens = self.tokens[curr_offset:]

    
    def __get_sorted_features__(self, features=None) -> list[TextualFeature]:
        if not self.is_processed:
            self.process()
        if features is None:
            features = self._features_names_
        if not isinstance(features, (np.ndarray, list, pd.Series)):
            features = [features]
        valid_features = [f for f in features if getattr(self, f) is not None]
        try:
            return sorted(sum([getattr(self, f) for f in valid_features], []), key=lambda e: e.match_start)
        except TypeError:
            try:
                return sorted(sum([getattr(self, f) for f in valid_features], []), key=lambda e: e.position_in_text)
            except TypeError:
                self.__map_index_to_features__()
                return sorted(sum([getattr(self, f) for f in valid_features], []), key=lambda e: e.position_in_text)


    

def dict_to_TextualFeature(curr_dict: dict, feature_name_key: str = None) -> TextualFeature:
    try:
        return TextualFeature(**curr_dict)
    except:
        curr_dict = curr_dict.copy()
        if feature_name_key is None:
            feature_name_key = [k for k in curr_dict.keys() if k in default_config.single_feature_names_dict.values()]
            if len(feature_name_key)!=1:
                raise ValueError('Please set feature_name_key with a proper value in order to understand the current feature name')
            else:
                feature_name_key = feature_name_key[0]

        print(feature_name_key)
        print(curr_dict)
        feature_value = curr_dict.pop(feature_name_key)

        new_dict = {'feature_type': feature_name_key,
                   'feature_value': feature_value}
        return TextualFeature(dict(**new_dict, **curr_dict))