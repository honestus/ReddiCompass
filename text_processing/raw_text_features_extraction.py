import re, regex, string, time
from unidecode import unidecode
import pandas as pd
import numpy as np
import emoji, emosent
from emoji.tokenizer import tokenize as emoji_tokenize
import gc
import default_config
from textblob import TextBlob
import nltk
from nltk.tokenize import TreebankWordDetokenizer
from utils import flatten
from text_processing.text_replacement import replace_features_in_text
from typing import Sequence


def get_length(text: str, tokenizer=None) -> int:
    """
    @return int: length of the input text
    If tokenizer is not None, will return the number of tokens
    """
    if tokenizer is None:
        return len(text)
    tokenized_text = tokenizer.tokenize(text.strip())
    return len(tokenized_text)


def get_emoticons_pattern(emoticons_list: list[str], pattern:int =0) -> re.Pattern:
    """
    @return re.Pattern: the compiled pattern to find emoticons from a string
    """
    ### EMOTICONS WILL BE FOUND ONLY IF PRECEDED/FOLLOWED BY A SPACE CHAR OR START/END OF STRING
    emoticon_joined_string = '|'.join(map(re.escape, emoticons_list))
    pattern_regex = ''
    if pattern==1:
        pattern_regex = re.compile("(?:^|(?<=\s))"+"("+emoticon_joined_string+")"+"(?=\s|$)") ### ONLY MATCHES SINGLE EMOTICONS 
    elif pattern==2:
        pattern_regex = re.compile("(?:^|(?<=\s))"+"("+emoticon_joined_string+")+"+"(?=\s|$)") ### MATCHES MULTIPLE EMOTICONS BUT ONLY KEEPS TRACK OF THE LAST ONE
    
    else:
        pattern_regex = regex.compile("(?:^|(?<=\s|" + emoticon_joined_string + "))" + "(" + emoticon_joined_string + ")"\
                                  +"(?=\s|$|" + emoticon_joined_string +")") ### MATCHES AND REPLACES ALL OF THE EMOTICONS (notice that a string such as "Nice:):P" will match :P as it matches any emoticons either followed/preceded by another emoticon or a space/end of line
    #re.sub(emoticon_reg3, r' :emoticon:\1:emoticon: ',txt)   
    return pattern_regex

def extract_emoticons(text: str, 
                      emoticons_list: list[str] or str = 'vader',
                      replace_string: str or bool = ', ',
                      position: bool = True, 
                      retain_length: bool = False) -> None:
    """
    @input replace_string: str or bool -> the string to replace any found emoticons in the input text (If True is passed, ', ' will be used as default value)
    @input position: bool -> if True, will return the positions (start_index, end_index) together with the found emoticons in the input text
    @input retain_length: bool -> if True, will replace the found emoticons by whitespaces (i.e. replace_string will be set to ' ') by preserving the length of the text
    @yield  list of extracted emoticons
    @yield replaced text after replacing/removing emoticons in the input text (if replace_string is False, will yield the input text as it is)
    """
    if not text:
        yield []
        if replace_string or replace_string=='':
            yield ''
        return

    if not isinstance(replace_string, (bool, str)):
        raise ValueError('Please use a valid value (either boolean or string) for replace_string')
        return
    ### Returns the list of emoticons found in the input @text

    if not isinstance(emoticons_list, (list, np.ndarray, pd.Series)) and emoticons_list not in ['default','vader']:
        raise ValueError('Please use a valid list of emoticons to extract')
        return

    if emoticons_list=='default':
        emoticon_reg = get_emoticons_pattern(default_config.emoticons_list)
    elif emoticons_list=='vader':
        emoticon_reg = get_emoticons_pattern(default_config.emoticons_list_vader)
    else:
        emoticon_reg = get_emoticons_pattern(emoticons_list)

    if replace_string is True:
        replace_string = ', '

    emoticons_found = []
    def capture_emoticons_and_kill(match):
        if position:
            emoticons_found.append((match.group(), match.span()))
        else:
            emoticons_found.append(match.group())
        if retain_length:
            return ' '*len(match.group())
        return match.group() if replace_string is False else replace_string

    replaced_text = emoticon_reg.sub(capture_emoticons_and_kill, text)#.strip()
    yield emoticons_found
    yield replaced_text
    return


import emoji, re
from unidecode import unidecode
from emoji.tokenizer import tokenize as emoji_tokenize

def extract_emojis(text: str, 
                   replace_string: str or bool = False,
                   retain_length: bool = False,
                  *, lang: str = 'en', ) -> None:
    """
    @input replace_string: str -> the string to replace any found emojis in the input text (If True is passed, ', ' will be used as default value)
    @input retain_length: bool -> if True, will replace the found emojis by whitespaces (i.e. replace_string will be set to ' ') by preserving the length of the text
    @yield list of extracted emojis
    @yield replaced text after replacing/removing emojis in the input text
    """

    matches = [{
        'match_start': m.value.start,
        'match_end': m.value.end,
        default_config.single_feature_names_dict[default_config.default_emojis_colname]: m.value.emoji,
        'emoji_alias': emoji.EMOJI_DATA[m.value.emoji][lang]
    } for m in emoji_tokenize(text, keep_zwj=False) if isinstance(m.value, emoji.EmojiMatch)]
    yield matches
    if replace_string or retain_length:
        if not retain_length and replace_string is True:
            replace_string=' , '
        for e in matches:
            start,end = e['match_start'], e['match_end']
            text = text[:start]+(' '*(end-start) if retain_length else replace_string) +text[end:]
    yield text




def get_url_pattern(pattern=0) -> re.Pattern:
    url_regex_django = re.compile (
r'(?:http|ftp)s?://'# http:// or https://
r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
r'localhost|'  # localhost...
r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
r'(?::\d+)?'  # optional port
r'(?:/?|[/?]\S+)', re.IGNORECASE )

    url_regex_custom_user = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

    url_regex_emosol = r'(?:(?:https?|ftp)://)(?:\S{1,50}(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]{1,100}-?)*[a-z\u00a1-\uffff0-9]{1,60})(?:\.(?:[a-z\u00a1-\uffff0-9]{1,60}-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?'

    url_pattern = r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4})'
    
    url_regex_perini = r'(?:(?:(?:https?|ftp):)?\/\/)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z0-9\u00a1-\uffff][a-z0-9\u00a1-\uffff_-]{0,62})?[a-z0-9\u00a1-\uffff]\.)+(?:[a-z\u00a1-\uffff]{2,}\.?))(?::\d{2,5})?(?:[/?#]\S*)?'
    
    if pattern==1 or 'django' in str(pattern).lower():
        return url_regex_django
    if pattern==2 or 'emosol'  in str(pattern).lower():
        return re.compile(url_regex_emosol, re.IGNORECASE)
    if pattern==3 or 'perini'  in str(pattern).lower():
        return re.compile(url_regex_perini, re.IGNORECASE)
    if pattern==4:
        return re.compile(url_pattern, re.IGNORECASE)
    return re.compile(url_regex_custom_user, re.IGNORECASE)  


def extract_url(text: str,
                replace_string: str or bool = False,
                position: bool = False,
                retain_length: bool = False) -> None:
    """
        @input replace_string: str or bool -> the string to replace any found urls in the input text (If True is passed, ' ' will be used as default value)
        @input position: bool -> if True, will return the positions (start_index, end_index) together with the found urls in the input text
        @input retain_length: bool -> if True, will replace the found urls by whitespaces (i.e. replace_str will be set to ' ') by preserving the length of the text
        @yield the list of extracted urls
        @yield replaced text after replacing/removing urls in the input text (if replace_string is False, will yield the input text as it is)
        """
    if not text:
        yield []
        if replace_string or replace_string=='':
            yield ''
        return

    if not isinstance(replace_string, (bool, str)):
        raise ValueError('Please use a valid value (either boolean or string) for replace_string')
        return
    
    url_regex = get_url_pattern()

    if replace_string is True:
        replace_string=' '

    urls_found = []
    def capture_urls_and_kill(match):
        if position:
            urls_found.append((match.group(), match.span()))
        else:
            urls_found.append(match.group())
        if retain_length:
            return ' '*len(match.group())
        return match.group() if replace_string is False else replace_string

    replaced_text = url_regex.sub(capture_urls_and_kill, text)
    yield urls_found
    yield replaced_text
    return


    
def get_badwords_pattern(bad_words_list: list[str]) -> re.Pattern:
    ### BAD WORDS WILL BE FOUND ONLY IF PRECEDED/FOLLOWED BY A SPACE, PUNCTUATION CHAR OR START/END OF STRING

    badwords_joined_string = '|'.join(map(re.escape, map(str.lower, bad_words_list)))
    ### WILL ONLY MATCH SINGLE BADWORDS (I.E. NOT CONCATENATED ONES LIKE "FUCKFUCKFUCK", BUT ONLY "FUCK" WITH WHITESPACES/PUNCTUATION DELIMITERS 
    badwords_regex = re.compile(r"(?:^|(?<=[\s{}]))".format(string.punctuation)+"("+badwords_joined_string+")"+"(?=\W|$)")
    #badwords_regex_old = re.compile("(?:^|(?<=\W))"+"("+badwords_joined_string+")"+"(?=\W|$)") ###not matching _fuck
    return badwords_regex


def get_repeated_punctuation_marks_pattern(keep_distinct: bool = False, use_ones: bool = True, punctuation: list[str] = ['!','?']) -> re.Pattern:
    """
    @input keep_distinct: bool. If True, the returned pattern won't match punctuation containing both ! and ? (e.g. !!? or ?!?)
    @input use_ones: bool. If True, "1" will be considered as part of the punctuation found
    @input punctuation: list of string -> the punctuation characters to look for.
    """
    #new_pat = r"([" + ''.join(punctuation) + r"])\1{1,}"
    #new_pat_working = r"(([" + ''.join(punctuation) + r"])(\2|1*\2){1,})"
    #new_pat_withones_fp = r"(([" + ''.join(punctuation) + r"])(\2|1){1,})" #matcha ogni ?1 o !1 e quindi restituisce false positives...
    if keep_distinct:        
        pat = r"(([{}])(1*\2)+1*)".format(''.join(punctuation)) #matcha solo !1! o !1!1 o ?1? (anche con 1 o ! o ? ripetuti) 
    else:
        pat = "(([{}])(1*[{}])+1*)".format(''.join(punctuation), ''.join(punctuation)) #uguale al pattern con keep_distinct ma matcha qualsiasi !? ?! !1? etc. -> cioè tutte le sottostringhe che iniziano con ! o ? e che sono seguite da almeno un 1 o ! o ?
    return re.compile(pat)
                                           

def extract_repeated_punctuation_marks(text, keep_distinct=False, replace_string=False, position=False, retain_length=False):
    """
        @input keep_distinct: bool -> if True, punctuation containing both "!" and "?" won't match (e.g. "!?!")
        @input replace_string: str or bool -> the string to replace any found punctuations in the input text (If True is passed, ' ' will be used as default value)
        @input position: bool -> if True, will return the positions (start_index, end_index) together with the found punctuations in the input text
        @input retain_length: bool -> if True, will replace the found punctuations by whitespaces (i.e. replace_str will be set to ' ') by preserving the length of the text
        @yield  list of extracted punctuations
        @yield replaced text after replacing/removing punctuations in the input text (if replace_string is False, will yield the input text as it is)
    """

    if not text:
        yield []
        if replace_string or replace_string=='':
            yield ''
        return

    if not isinstance(replace_string, (bool, str)):
        raise ValueError('Please use a valid value (either boolean or string) for replace_string')
        return

    repeated_punctuation_pattern = get_repeated_punctuation_marks_pattern(keep_distinct=keep_distinct, use_ones=True, punctuation=['!','?'])
    
    if replace_string is True:
        replace_string = ' '

    punctuations_found = []
    def capture_punctuations_and_kill(match):
        if position:
            punctuations_found.append((match.group(), match.span()))
        else:
            punctuations_found.append(match.group())
        if retain_length:
            return ' '*len(match.group())
        return match.group() if replace_string is False else replace_string

    replaced_text = repeated_punctuation_pattern.sub(capture_punctuations_and_kill, text)#.strip()
    yield punctuations_found
    yield replaced_text
    return


def extract_mentions(text: str,
                     replace_string: dict[str, str] or str or list[str, str] or bool = {'u/': '@', 'r/': '#'},
                     keep_mention: bool = True,
                     position: bool = True,
                     retain_length: bool = False) -> None:
    """
            @input replace_string: dict[str] or str or bool -> the string to replace any found mentions in the input text (If True is passed, the default dict will be used as default value)
            @input position: bool -> if True, will return the positions (start_index, end_index) together with the found mentions in the input text
            @input retain_length: bool -> if True, will replace the found mentions by whitespaces (i.e. replace_str will be set to ' ') by preserving the length of the text
            @input keep_mention: bool -> if True, the mentioned user/subreddit will be kept in the replaced string (i.e. from u/user the new string will be -> replace_string+'user')
            @yield list of extracted mentions
            @yield replaced text after replacing/removing mentions in the input text (if replace_string is False, will yield the input text as it is)
    """
    if not text:
        yield []
        if replace_string or replace_string == '':
            yield ''
        return

    if not isinstance(replace_string, (str, bool, dict, list, np.ndarray, pd.Series)):
        raise ValueError('Please use a valid value (either boolean or dict or listwise) for replace_string. \
                         Notice that if you use a list, it must have exactly two values (the string to replace user mentions "u/" and the string to replace subreddit mentions "r/"; if you use a dict instead, it must contain "u/" and "r/" among its keys')
        return
    if isinstance(replace_string, dict):
        if 'u/' not in replace_string or 'r/' not in replace_string:
            raise ValueError('Please specify replacements for both user mentions and reddit mentions')
            return
    if isinstance(replace_string, (list, np.ndarray, pd.Series)):
        if len(replace_string) != 2:
            raise ValueError(
                'The input list for @replace_string must have exactly two values: the string to replace user mentions "u/" and the string to replace subreddit mentions "r/")')
            return
        else:
            replace_string = {'u/': replace_string[0], 'r/': replace_string[1]}

    if replace_string is True:
        replace_string = {'u/': '@', 'r/': '#'}
    if isinstance(replace_string, str):
        replace_string = {'u/': replace_string, 'r/': replace_string}

    mentions_regex = re.compile(r"(?:^|(?<=\W))[ur]/(?=\w*[a-zA-Z])[a-zA-Z0-9]+\w*[a-zA-Z0-9]")


    mentions_found = []

    def capture_mentions_and_kill(match):
        curr_match_str = match.group()
        if position:
            mentions_found.append((curr_match_str, match.span()))
        else:
            mentions_found.append(curr_match_str)
        if replace_string is False:
            return curr_match_str
        if retain_length and not keep_mention:
            return ' ' * len(curr_match_str)
        for k in replace_string.keys():
            if k in curr_match_str:  ##probably better to use match.group().startswith(k)?
                if not retain_length:
                    return replace_string[k] + (curr_match_str if keep_mention else '').split(k)[-1]

                curr_whole_match_len = len(curr_match_str)
                curr_mention_only = curr_match_str.split(k)[-1]
                return ' ' * (curr_whole_match_len - len(curr_mention_only)) + curr_mention_only
        return match.group()  ##if match != replacing_dict_keys...

    replaced_text = re.sub(mentions_regex, capture_mentions_and_kill, text)  # .strip()
    yield mentions_found
    yield replaced_text
    return



"""
def __extract_textual_features__(df, text_col='text', keep_orig=True, verbose=False, urls=True, emoticons='vader', emojis=True, repeated_punctuation=True, mentions=True, hashtags=False, uppercase_words=False, badwords=False, remove_repeated_chars=True,):
    
    get_clean_texts = True
    repeated_words_regex_words_only = r'(?<!\S)((\w+)(?:\W*)\2)((?:\W*)\2)+(?!\w)' ###replaces 
    # "word word word non-words-char(punctuation, whitespaces..) word word" with "word word"
    uppercase_regex = r"(?:^|(?<=\W))[A-Z]{2,}[^a-z,.;?:!()\[\]\s]*(?=\W|$)"
    hashtags_regex = r"(?:^|(?<=\W))(?<![@#])[#@](?=\d*[a-zA-Z])[\w]{2,}"

    repeated_words_regex = r'(?<!\S)((\S+)(?:\s+\2))(?:\s+\2)+(?!\S)'
    repeated_chars_regex = r'(\D)\1{2,}'
    
    clean_texts = df.copy()[text_col]
    features_cols_to_replace = []
    
    if keep_orig:
        new_df = df.copy()
    else:
        new_df = pd.DataFrame()
    removed_urls, removed_emoticons = False, False
    
    if remove_repeated_chars:
        clean_texts = clean_texts.map(lambda t: re.sub(repeated_words_regex, r'\1',\
                                                                   re.sub(repeated_chars_regex, r'\1\1\1', t, flags=re.IGNORECASE)\
                                                                   , flags=re.IGNORECASE ))
        if verbose:
            print('Multiple repeated characters removed...')
            
    if emojis:
        features_cols_to_replace.append(default_config.default_emojis_colname)
        emojis_gem = clean_texts.map(lambda t: extract_emojis_new(t, retain_length=True))
        new_df[default_config.default_emojis_colname] = emojis_gem.map(next)
        clean_texts = emojis_gem.map(next) ###removing emojis after extracting them in order to find all the emoticons/hashtags without "noise"
        if verbose:
            print('Emojis extracted...')

    clean_texts = clean_texts.map(unidecode)
    #clean_texts = texts.map(lambda x: replace_emojis(x, replace_string=' , ')) 
    if emoticons:
        if emoticons is True:
            emoticons='vader'
        features_cols_to_replace.append(default_config.default_emoticons_colname)
        emoticons_gen = clean_texts.map(lambda x: extract_emoticons(x, emoticons_list=emoticons, retain_length=True, position=True))
        emoticons_found = emoticons_gen.map(next)
        clean_texts = emoticons_gen.map(next)
        new_df[default_config.default_emoticons_colname] = emoticons_found.map(lambda x: [{default_config.single_feature_names_dict[default_config.default_emoticons_colname]:e[0], 'match_start':e[1][0], 'match_end':e[1][1]} for e in x])
        if verbose:
            print('Emoticons extracted...')
        removed_emoticons=True
    if urls:
        features_cols_to_replace.append(default_config.default_urls_colname)
        urls_gen = clean_texts.map(lambda text: extract_url(text, retain_length=True, replace_string='', position=True))
        new_df[default_config.default_urls_colname] = urls_gen.map(next).map(lambda x: [{default_config.single_feature_names_dict[default_config.default_urls_colname]:e[0], 'match_start':e[1][0], 'match_end':e[1][1]} for e in x])
        clean_texts = urls_gen.map(next)
        removed_urls = True
        if verbose:
            print('Urls extracted...')
        
    if any([repeated_punctuation, mentions, hashtags, remove_repeated_chars,uppercase_words,badwords]):
        if not removed_urls:
            clean_texts = clean_texts.map(lambda x: replace_url(x, remove=True))
        if not removed_emoticons:
            clean_texts = clean_texts.map(lambda x: extract_emoticons(x, emoticons_list='vader', replace_string=', ')).map(lambda gen: next(x for i,x in enumerate(gen) if i==1))#
        ###removing urls/emoticons from texts in order to find real uppercase_words/hashtags/badwords (false positive ones may arise from urls' strings)
    
    if repeated_punctuation:
        features_cols_to_replace.append(default_config.default_repeated_punctuation_colname)
        repeated_punctuation_gen = clean_texts.map(lambda text: extract_repeated_punctuation_marks(text, replace_string=' ', retain_length=True, position=True))
        new_df[default_config.default_repeated_punctuation_colname] = repeated_punctuation_gen.map(next).map(lambda x: [{default_config.single_feature_names_dict[default_config.default_repeated_punctuation_colname]:e[0], 'match_start':e[1][0], 'match_end':e[1][1]} for e in x])
        clean_texts = repeated_punctuation_gen.map(next)
        if verbose:
            print('Repeated_punctuation extracted...')
    if mentions:
        features_cols_to_replace.append(default_config.default_mentions_colname)
        mentions_gen = clean_texts.map(lambda text: extract_mentions(text, retain_length=True, position=True))
        new_df[default_config.default_mentions_colname] = mentions_gen.map(next).map(lambda x: [{default_config.single_feature_names_dict[default_config.default_mentions_colname]:e[0], 'match_start':e[1][0], 'match_end':e[1][1]} for e in x])
        clean_texts = mentions_gen.map(next)
        if verbose:
            print('Mentions extracted...')
    if badwords:
        badwords_regex = get_badwords_pattern(default_config.badwords_LDNOOBW)
        features_cols_to_replace.append(default_config.default_badwords_colname)
        new_df[default_config.default_badwords_colname] = clean_texts.map(lambda text: 
                                            list(map(lambda t: ( t.group(), (t.start(), t.end())), badwords_regex.finditer(text))))\
                                .map(lambda x: [{default_config.single_feature_names_dict[default_config.default_badwords_colname]:e[0], 'match_start':e[1][0], 'match_end':e[1][1]} for e in x])
        if verbose:
            print('Badwords extracted...')
    
    if uppercase_words:
        features_cols_to_replace.append(default_config.default_uppercase_words_colname)
        new_df[default_config.default_uppercase_words_colname] = clean_texts.map(lambda text: 
                                            list(map(lambda t: ( t.group(), (t.start(), t.end())), re.compile(uppercase_regex).finditer(text))))\
                                .map(lambda x: [{default_config.single_feature_names_dict[default_config.default_uppercase_words_colname]:e[0], 'match_start':e[1][0], 'match_end':e[1][1]} for e in x])
        
        if verbose:
            print('Uppercase words extracted...')
    
    if hashtags:
        features_cols_to_replace.append(default_config.default_hashtags_colname)
        new_df[default_config.default_hashtags_colname] = clean_texts.replace({'x200B':''}, regex=True).map(lambda text: 
                                            list(map(lambda t: ( t.group(), (t.start(), t.end())), re.compile(hashtags_regex).finditer(text))))\
                                .map(lambda x: [{default_config.single_feature_names_dict[default_config.default_hashtags_colname]:e[0], 'match_start':e[1][0], 'match_end':e[1][1]} for e in x])
        
        if verbose:
            print('Hashtags extracted...')

    if get_clean_texts:
        new_df['clean_text'] = clean_texts
        new_df['clean_text'] = new_df.apply(lambda row: replace_features_in_text(row, text_col='clean_text', columns_to_replace=features_cols_to_replace, columns_to_remove=[]), axis=1).map(lambda t: " ".join(t.split()))

        #['clean_text'] = new_df.apply(lambda row: replace_text(row, text_col='clean_text'), axis=1)#.map(lambda t: " ".join(t.split()))
        new_df['clean_text_with_emoticons'] = texts.map(lambda x: replace_url(x, remove=True)).map( \
                                                               extract_mentions).map(lambda gen: next(x for i,x in enumerate(gen) if i==1)).map(lambda x: \
                                                                re.sub(repeated_words_regex, r'\1',\
                                                                   re.sub(repeated_chars_regex, r'\1\1\1', x, flags=re.IGNORECASE)\
                                                                   , flags=re.IGNORECASE )).map(lambda t: " ".join(t.split()))

        
    
        
    return new_df

"""


    




def clean_text(text):
    repeated_chars_regex = r'(\D)\1{2,}'
    repeated_words_regex = r'(?<!\S)((\S+)(?:\s+\2))(?:\s+\2)+(?!\S)'

    t = replace_emojis(text, replace_string=' ')
    t = unidecode(t)
    t = replace_url(t, remove=True)
    t = extract_mentions(t, remove=True)[1]
    t = extract_emoticons(t, replace_string=' ', emoticons_list='vader')[1]

    chars_to_remove = ''.join(map(str, range(10)))+string.punctuation
    t = t.translate(str.maketrans(chars_to_remove, ' '*len(chars_to_remove)))

    t = re.sub(repeated_chars_regex, r'\1\1\1', t, flags=re.IGNORECASE)
    t = re.sub(repeated_words_regex, r'\1', t, flags=re.IGNORECASE)

    final_text = ' '.join(t.split())
    return final_text


def clean_texts(texts):
    repeated_chars_regex = r'(\D)\1{2,}'
    repeated_words_regex = r'(?<!\S)((\S+)(?:\s+\2))(?:\s+\2)+(?!\S)'
    if isinstance(texts, (list, np.ndarray)):
        texts = pd.Series(texts)
    texts = texts.map(lambda t: replace_emojis(t, replace_string=' '))\
                            .map(lambda t: replace_url(unidecode(t),remove=True))
    texts = texts.map(lambda t: extract_mentions(t, remove=True)).map(lambda x: x[1])
    texts = texts.map(lambda t: extract_emoticons(t, emoticons_list='vader', replace_string=' ')).map(lambda x: x[1])
    
    chars_to_remove = ''.join(map(str, range(10)))+string.punctuation
    texts_no_punctuation_no_numbers = texts.map(lambda t: t.translate(str.maketrans(chars_to_remove, ' '*len(chars_to_remove))) )
    final_texts = texts_no_punctuation_no_numbers.map(lambda t: re.sub(repeated_words_regex, r'\1',\
                                    re.sub(repeated_chars_regex, r'\1\1\1', t, flags=re.IGNORECASE)\
                          , flags=re.IGNORECASE ) ).map(lambda t: " ".join(t.split()))

    #chars_to_remove = string.punctuation
    #texts_no_punctuation = texts.map(lambda t: " ".join (t.translate(str.maketrans(chars_to_remove,' '*len(chars_to_remove))).split() ))
    
    return final_texts

