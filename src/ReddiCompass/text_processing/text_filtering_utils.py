from ReddiCompass.text_processing.textractor import TexTractor
from ReddiCompass.text_processing.text_replacement import replace_features_in_text

def get_nonstopwords_tokens(text: str | TexTractor, remove_features_tokens=True, stopwords=[]):
    text = TexTractor(text).process()
    if remove_features_tokens:
        filtered_tokens = replace_features_in_text(text, text_col='tokens', columns_to_replace=[], columns_to_remove=['urls','emoticons','emojis','mentions','repeatedPunctuation'])
    else:
        filtered_tokens = text[tokens_col]
    filtered_tokens = [t for token in filtered_tokens if (t:=token.lower()) not in stopwords and t.islower()]
    return filtered_tokens
	