import string
import re

def is_upper_custom(char: str) -> bool:
    if not char.isalpha():
        return True
    return char.isupper()

def is_upper_word(word: str, min_alphachar: int = 2) -> bool:
    return all(map(is_upper_custom, word)) and len(list(filter(str.isalpha, word)))>=min_alphachar


def is_valid_url_new(url: str, replace_malformatted: bool = False, add_scheme: bool = False,
                     use_validator: bool = True, use_urlparse: bool = False, use_tldextract: bool = False) -> bool:
    
    if not use_validator+use_urlparse+use_tldextract:
        raise ValueError('Either one of validator or urlparse must be True')
        return
    from urllib.parse import urlparse 
    import validators, tldextract
    def isvalid_validator_old(url):
        if validators.url(url):
            return True
        extracted = tldextract.extract(url)
        if extracted.domain and extracted.suffix:
            return True
        return False
    def isvalid_validator(url):
        if validators.url(url):
            return True
        return False
    def isvalid_tldextract(url):
        extracted = tldextract.extract(url)
        if extracted.domain and extracted.suffix:
            return True
        return False
    def isvalid_urlparse(url):
        #print(url)
        parsed_url = urlparse(url)
        if parsed_url.scheme and parsed_url.netloc:
            return True
        return False 
    
    url_valid_validator, url_valid_urlparse, url_valid_tldextract = False, False, False
    if use_validator:
        url_valid_validator = isvalid_validator(url)
    if use_urlparse:
        url_valid_urlparse = isvalid_urlparse(url)
    if use_tldextract:
        url_valid_tldextract = isvalid_tldextract(url)
    
    
    if url_valid_validator==use_validator and url_valid_urlparse==use_urlparse and url_valid_tldextract==use_tldextract:
        return bool(url_valid_validator+url_valid_urlparse+use_tldextract)
    elif replace_malformatted or add_scheme:
        well_formatted_url = url.strip()
        if replace_malformatted:
            well_formatted_url = re.sub(r'\\', '/', well_formatted_url)
            well_formatted_url = re.sub(r'(https?:)[/]+', r'\1'+'//', well_formatted_url)
        if add_scheme:
            extracted = tldextract.extract(well_formatted_url)
            if extracted.domain and extracted.suffix:
                # Aggiungi schema http se manca
                if not well_formatted_url.startswith(('http://', 'https://', 'ftp://')):
                    well_formatted_url = 'http://' + well_formatted_url
        return is_valid_url_new(url=well_formatted_url, replace_malformatted=False, add_scheme=False, use_validator=use_validator, use_urlparse=use_urlparse, use_tldextract=use_tldextract)
    return False


from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def custom_features_pos_tag(feature: str) -> str:
    if feature in ['emoji', 'emoticon']:
        return 'UH'
    if feature in ['mention', 'hashtag']:
        return 'NN'
    if feature == ['url']:
        return 'NN'
    if feature == ['badword']:
        return 'JJ'        
    if feature == 'repeatedPunct':
        return '.'


    
from nltk.tag.mapping import map_tag
#tagset = 'brown'
def map_pos_tag(tag: str, curr_tagset='en-ptb', tagset='universal') -> str:
    #{tag:map_tag("en-ptb", tagset, tag) for tag in all_tags_found}
    return map_tag(source=curr_tagset, source_tag=tag, target=tagset)