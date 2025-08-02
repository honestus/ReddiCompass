from text_processing.LexiconMatcher import LexiconMatcher
    
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from numpy import interp
import numbers
from typing import Sequence
from utils import flatten


class NRCVad(LexiconMatcher):
    def __init__(self,
                 lexicon_file: str,
                 expand_lexicon: bool = True,
                 limit_expansion: bool = True,
                 min_max: tuple[int, int] = (0,1)) -> None:
        self.scores_dict = self.__load_from_lexicon_file__(lexicon_file)
        self._expand_lexicon_ = expand_lexicon
        self._limit_expansion_ = limit_expansion
        super().__init__(lexicon=self.scores_dict.keys(), add_only=True)
        self.lexicon = self.scores_dict.keys()
        self.min_max = min_max
        if self.min_max!=(0,1):
            normalized_df = self.__normalize__(new_min=min_max[0], new_max=min_max[1])
            self.scores_dict = dict(normalized_df.apply(lambda s: s.to_list(), axis=1))
        if expand_lexicon:
            self.curr_dict = self.scores_dict.copy() if limit_expansion else self.scores_dict
        
    def __str__(self):
        return super().__str__().replace('LexiconMatcher','NRC-VAD')
    
    def __load_from_lexicon_file__(self, lexicon_file: str) -> dict[str, list [float]]:
        columns = ['word', 'valence', 'arousal', 'dominance']
        with open(lexicon_file, 'r') as f:
            whole_l = [row.split('\t') for row in f.readlines()]
        nrcvad_dict = {e[0]:list(map(float, e[1:])) for e in whole_l}
        #nrcvad_df = {columns[i]: list(map(lambda e: e[i], whole_l)) for i in range(len(columns))}
        #nrcvad_df = pd.DataFrame.from_dict(nrcvad_df).set_index('word').astype(float)
        return nrcvad_dict

    def __normalize__(self, *, new_min, new_max):
        scores_df = self.get_scores_df()
        curr_min, curr_max = scores_df.min(axis=1).min(), scores_df.max(axis=1).max()
        normalized_scores_df = scores_df.apply(lambda value: (new_max-new_min)*(value-curr_min)/(curr_max-curr_min) + new_min)
        return normalized_scores_df
    
    def __update_starting_tokens__(self) -> None:
        """
        Will update useful custom attributes for the current lexicon, such as max n of tokens in a lexicon word, starting tokens of a word, etc.
        Max n of tokens will be the max number of tokens to generate new words from a text.
        Possible starting tokens is a set containing all the starting tokens composing the words in current lexicon
        They are both useful for efficiency reasons: saving time when generating words(i.e. multiple tokens) from a new text
        """
        #self.lexicon = self.get_lexicon()
        super().__update_starting_tokens__()
        
        if self._expand_lexicon_:
            self._max_n_of_tokens_ = max(3, self._max_n_of_tokens_) ###setting it to default 3 as it may eventually match for synonyms for words like "New york city" that are not currently in lexicon even if some synonym (e.g. new york) might be in there
        
        
    def __generate_all_possible_words__(self, tokens_list: list[str]) -> set[str]:
        """
        Will generate all the possible n-grams from the input tokens_list, with n<=self.max_n_of_tokens
        @return set of strings: all the distinct generated words from the input tokens
        """
        self.__update_starting_tokens__() 
        all_possible_words = set()
        n_words = len(tokens_list)
        max_n_tokens = 3
        ###generating all possible words having n_of_tokens<=max_n_of_tokens
        for i in range(n_words):
            for span in range(max_n_tokens, 0, -1):
                if i + span <= n_words:
                    all_possible_words.add(' '.join(tokens_list[i:i+span]))

        #all_possible_words_with_counts = {w: all_possible_words.count(w) for w in set(all_possible_words)}
        return all_possible_words
        
    
    def __get_synonym_in_lexicon__(self, curr_w: str) -> str:
        """
        @input curr_w: string, a word composed by one or more tokens (e.g. 'get up')
        @return string: the most similar synonym to curr_w in the curr dict. If no synonyms in dict, will return False.
        If curr_w starts with a negation, will return the antonym of the word: e.g. curr_w="not be", will return the antonym of be
        """
        if curr_w in self.scores_dict:
            return curr_w
            #return False ?
        
        curr_tokens = curr_w.split()
        use_antonyms = False

        if len(curr_tokens) > 1 and (curr_tokens[0] == 'not' or "n't" in curr_tokens[0]):
            use_antonyms = True  ###handling negations with antonynms
            context = '_'.join(curr_tokens[1:])
        else:
            context = curr_w.replace(' ', '_')
        for synonym in wordnet.synsets(context): ###looking for synonyms if curr_w is not in dict
            for lemma in set(synonym.lemmas()):
                meaning = None
                if use_antonyms:
                    antonyms = lemma.antonyms()
                    if antonyms:
                        meaning = antonyms[0].name().replace('_', ' ')
                else:
                    meaning = lemma.name().replace('_', ' ')

                if meaning in self.scores_dict:                            
                    return meaning
        return False
    
    
    def  __get_matches__(self,
                    tokens: list[str],
                    return_indexes: bool = False,
                    ignore_case: bool = False,
                    use_synonyms: bool = False) -> list[str]:
        """
        @return list of strings: the list of tokens/words (i.e. multiple tokens joined) from the input tokens in the current lexicon.
        If use_synonyms, will look for synonyms of the words in the current dict and will assign the same score
        """
        self.__update_starting_tokens__()

        if not use_synonyms:
            return super().get_matches(tokens, ignore_case=ignore_case, return_indexes=return_indexes, split_tokens_delimiters = False )
        
        if ignore_case:
            curr_lexicon = list(map(str.lower, self.get_lexicon() ))

        n = len(tokens)
        final_tokens=[]
        i = 0  
        while i < n:
            longest_match = None
            longest_match_score = None
            # Starting from the longest possible word (i.e. max n of tokens)
            curr_token = tokens[i]
            if not use_synonyms and not self._trie_.search(curr_token) and (not ignore_case or self._trie_.search(curr_token.lower())):
                i+=1
                continue
            for num_tokens in range(min(self._max_n_of_tokens_, n - i), 0, -1):
                candidate_word = ' '.join(tokens[i:i + num_tokens])
                if candidate_word in self.scores_dict or (ignore_case and candidate_word.lower() in curr_lexicon):
                    longest_match = candidate_word  if not return_indexes else (i, i+num_tokens-1, candidate_word)
                    #longest_match_score = words_dict[candidate]
                    break
                elif use_synonyms:
                    curr_synonym = self.__get_synonym_in_lexicon__(candidate_word) or (self.__get_synonym_in_lexicon__(candidate_word.lower()) if ignore_case else False)
                    if curr_synonym:
                        #print('Added new word {} as it is a synonym of {}'.format(candidate_word, curr_synonym))
                        self.curr_dict[candidate_word] = self.scores_dict[curr_synonym]
                        longest_match = candidate_word  if not return_indexes else (i, i+num_tokens-1, candidate_word)
                        break
            # If any match, updating the index and adding curr_match to final_tokens
            if longest_match:
                final_tokens.append(longest_match)
                i += num_tokens  # By updating i with num_tokens we skip all the tokens which are in this curr_word
            else: 
                i += 1

        return final_tokens
    
    
    
    def __update_global_lexicon__(self, tokens_list: list[str]) -> None:
        """
        Updates self.curr_dict and self.new_words
        Will generate all possible words from tokens_list, will look for synonyms and if any synonym in dict, will add words to dict by using the same score as the synonym 
        """
        all_possible_words = self.__generate_all_possible_words__(tokens_list) #generating all possible words having n_of_tokens<=max_n_of_tokens
        ###for each possible word, checking if it already exists in lexicon or if it has any synonym in lexicon
        ###if synonym(curr_word) in lexicon, assigning score of the synonym to curr_word
        for word in set(all_possible_words):
            word_synonym = self.__get_synonym_in_lexicon__(word)
            if word_synonym and word_synonym!=word:
                #print('Added new word {} as it is a synonym of {}'.format(word, word_synonym))
                self.curr_dict[word] = self.scores_dict[word_synonym] #assigning the score of the synonym to the new word
                self._new_words_.add(word)
            self._is_updated_ = False
            
    
    def add_words(self, words: dict[str, float]) -> None:
        if not isinstance(words, dict):
            raise ValueError("Please use a dict to add new word(s). Otherwise the new words will have default score=nan")
            if not isinstance(words, (set, list, np.ndarray, pd.Series)):
                words = [words]
            words = set(words)
            curr_new_words = words.difference(set(self.scores_dict.keys()))
            if curr_new_words:
                self.scores_dict |= {w:np.nan for w in curr_new_words}
                self._new_words_.update(curr_new_words)
                self._is_updated_ = False
        else:
            scores_df = self.get_scores_df()
            curr_min, curr_max = self.min_max
            if all([isinstance(v,(list, np.ndarray,pd.Series)) and len(v)==3 and all([isinstance(e, numbers.Number) and curr_min<=e<=curr_max for e in v]) for v in words.values()]):
                self.scores_dict|=words
            else:
                raise ValueError("Invalid format for the dict in input")

    
    def calculate_vad_scores(self, text, lemmatize=True, strong_feeling_threshold: dict[str, float] or list[float] or float={'valence': 0.376, 'arousal': 0.294, 'dominance': 0.31}, fillna=False):
        """
                Will calculate the scores  of a text by the scores of the words that are in the text.
                Text score will aggregate individual words scores (i.e. mean, median, total, std deviation, n° of words having score>=threshold)
                Will return such scores as a tuple
        """
        if isinstance(strong_feeling_threshold, (int,float)):
            strong_feeling_threshold = {'valence':strong_feeling_threshold, 'arousal':strong_feeling_threshold, 'dominance':strong_feeling_threshold}
        elif isinstance(strong_feeling_threshold, (list, np.ndarray, pd.Series)):
            if len(strong_feeling_threshold!=3):
                raise ValueError('wrong thresholds!')
            strong_feeling_threshold = list(strong_feeling_threshold)
            strong_feeling_threshold = {'valence':strong_feeling_threshold[0], 'arousal':strong_feeling_threshold[1], 'dominance':strong_feeling_threshold[2]}
        def get_single_feature_scores(curr_scores, curr_threshold):
            return [len(np.where(curr_scores>curr_threshold)[0]), np.sum(curr_scores), 
                    np.mean(curr_scores), np.std(curr_scores), np.median(curr_scores)]

        if not isinstance(text, (list,np.ndarray)):
            from nltk.tokenize import WordPunctTokenizer
            tokenizer = WordPunctTokenizer()
            tokens = tokenizer.tokenize(text)
        else:
            tokens = self.__validate_input_tokens__(tokens=text, delimiters='')

        tokens = [tok:=t.lower() if not lemmatize else WordNetLemmatizer().lemmatize(t.lower()) for t in tokens] ##lemmas and lowercases
        #tokens = [tok.lower() for tok in text if tok.lower().islower()] ###removing non-words tokens and lowering all tokens
        num_tokens = len(tokens)
        num_found_tokens, num_matches = 0, 0
        if self._expand_lexicon_:
            if self._limit_expansion_:
                self.curr_dict = self.scores_dict.copy()
                matching_tokens = self.__get_matches__(tokens, use_synonyms=True, return_indexes=True)
            else:
                self.curr_dict = self.scores_dict ###can just update scores_dict without any problem if not limit_expansion 
                self.__update_global_lexicon__(tokens) ##will update lexicon with all the possible phrases
                matching_tokens = self.__get_matches__(tokens, use_synonyms=False, return_indexes=True) #avoid generating synonyms as it's already done in update_global_lexicon
        else:
            matching_tokens = self.__get_matches__(tokens, use_synonyms=False, return_indexes=True)

        indexes, matching_tokens = (zip(* [(a[:-1], a[-1]) for a in matching_tokens])) if matching_tokens else ([],[])
        indexes = set(flatten(indexes))
        valences, arousals, dominances = [], [], []
        tokens_counts = {l: matching_tokens.count(l) for l in set(matching_tokens)}
        if not hasattr(self,'curr_dict'):
            self.curr_dict = self.scores_dict.copy()
        for t in tokens_counts:            
            curr_valence, curr_arousal, curr_dominance = self.curr_dict[t]
            valences.extend([curr_valence]*tokens_counts[t])
            arousals.extend([curr_arousal]*tokens_counts[t])
            dominances.extend([curr_dominance]*tokens_counts[t])
            num_found_tokens+=(tokens_counts[t]*len(t.split())) ##multiplying by len(t.split) in order to count "multiple joined tokens" as they are multiple tokens
                                                                ## eg -> word match= 'water polo', original tokens=['water','polo'], but with get_tokens_list they will become a single token: ['water polo'] -> in order to match properly n of tokens I will count the length of it
            num_matches+=tokens_counts[t]
            #valences.append(curr_valence)
            #arousals.append(curr_arousal)
            #dominances.append(curr_dominance)
            #num_found_tokens+=1
        if fillna is not False and fillna is not None:
            n_unmatched_tokens = num_tokens - len(matching_tokens)
            if n_unmatched_tokens:
                if fillna is not True and isinstance(fillna, (float, int)):
                    curr_default_value = fillna
                else:
                    curr_min, curr_max = self.min_max
                    curr_default_value = (curr_max + curr_min)/2
                valences.extend([curr_default_value] * n_unmatched_tokens)
                arousals.extend([curr_default_value] * n_unmatched_tokens)
                dominances.extend([curr_default_value] * n_unmatched_tokens)

        valence_scores = get_single_feature_scores(np.array(valences), strong_feeling_threshold['valence'])
        arousal_scores = get_single_feature_scores(np.array(arousals), strong_feeling_threshold['arousal'])
        dominance_scores = get_single_feature_scores(np.array(dominances), strong_feeling_threshold['dominance'])
        
        del(self.curr_dict)
        if hasattr(self, '_new_words_'):
            self._new_words_=set()
        columns = ['num_tokens', 'num_matched_tokens', 'num_matches',
                   'valence_strong_matches', 'valence_total', 'valence_mean', 'valence_std', 'valence_median',
                  'arousal_strong_matches', 'arousal_total', 'arousal_mean', 'arousal_std', 'arousal_median',
                  'dominance_strong_matches', 'dominance_total', 'dominance_mean', 'dominance_std', 'dominance_median']
        scores = (num_tokens, num_found_tokens, num_matches, valence_scores, arousal_scores, dominance_scores)
        return {columns[i]:flatten(scores)[i] for i in range(len(columns))}
    
    def calculate_vad_df(self, df, text_col=None, **kwargs):
        
        if not isinstance(df, pd.Series) and text_col is None:
            raise ValueError('Please set a text_col')
            return
        if text_col is not None:
            df = df[text_col]
        all_single_text_scores = df.map(lambda t: self.calculate_vad_scores(t, **kwargs))
        all_single_text_scores =  all_single_text_scores.apply(pd.Series).apply(lambda c: c.explode(), axis=1)
        #all_single_text_scores.columns = columns
        return all_single_text_scores

    
    def get_lexicon(self):
        return set(self.scores_dict.keys())
    
    def get_scores(self):
        return self.scores_dict
    
    def get_scores_df(self):
        return pd.DataFrame.from_dict(self.scores_dict).T.rename({0:'valence',1:'arousal',2:'dominance'},axis=1)
    
    
    
    
    
    
    
class SocialnessCalculator(LexiconMatcher):
    def __init__(self,lexicon_file,use_median=False,min_max=(0,1),expand_lexicon=True, limit_expansion=True):
        self.scores_dict = self.__load_from_lexicon_file__(lexicon_file, min_max=min_max, use_median=use_median)
        self._expand_lexicon_ = expand_lexicon
        self._limit_expansion_ = limit_expansion
        super().__init__(lexicon=self.scores_dict.keys(), add_only=True)
        self.lexicon = self.scores_dict.keys()
        self.min_max = min_max
        
        
    def __str__(self):
        return super().__str__().replace('LexiconMatcher','SocialnessCalculator')
    
    def __load_from_lexicon_file__(self, lexicon_file: str, min_max: tuple[int, int], use_median: bool = False) -> dict[str, float]:
        social_df = pd.read_csv(lexicon_file).rename(lambda c: c.lower(),axis=1).set_index('word')
        if use_median:
            curr_col = 'median'
        else:
            curr_col = 'mean'
        social_df = social_df[curr_col]
        if min_max:
            social_df = social_df.map(lambda v: interp(v,[1,7],min_max))
        return social_df.to_dict()
    
    def __update_starting_tokens__(self):
        """
            Will update useful custom attributes for the current lexicon, such as max n of tokens in a lexicon word, starting tokens of a word, etc.
            Max n of tokens will be the max number of tokens to generate new words from a text.
            Possible starting tokens is a set containing all the starting tokens composing the words in current lexicon
            They are both useful for efficiency reasons: saving time when generating words(i.e. multiple tokens) from a new text
        """
        super().__update_starting_tokens__()
        
        if self._expand_lexicon_:
            self._max_n_of_tokens_ = max(3, self._max_n_of_tokens_) # setting it to 3 as it may eventually match for synonyms for words like "New york city" that are not currently in lexicon even if some synonym (e.g. new york) might be in there
        
        
    def __generate_all_possible_words__(self, tokens_list):
        """
        Will generate all the possible n-grams from the input tokens_list, with n<=self.max_n_of_tokens
        @return set of strings: all the distinct generated words from the input tokens
        """
        self.__update_starting_tokens__() 
        all_possible_words = set()
        n_words = len(tokens_list)
        max_n_tokens = 3
        ###generating all possible words having n_of_tokens<=max_n_of_tokens
        for i in range(n_words):
            for span in range(max_n_tokens, 0, -1):
                if i + span <= n_words:
                    all_possible_words.add(' '.join(tokens_list[i:i+span]))

        #all_possible_words_with_counts = {w: all_possible_words.count(w) for w in set(all_possible_words)}
        return all_possible_words

    def __get_matches__(self,
                        tokens: list[str],
                        return_indexes: bool = False,
                        ignore_case: bool = False,
                        use_synonyms: bool = False) -> list[str]:
        """
        @return list of strings: the list of tokens/words (i.e. multiple tokens joined) from the input tokens in the current lexicon.
        If use_synonyms, will look for synonyms of the words in the current dict and will assign the same score
        """
        self.__update_starting_tokens__()

        if not use_synonyms:
            return super().get_matches(tokens, ignore_case=ignore_case, return_indexes=return_indexes,
                                       split_tokens_delimiters=False)

        if ignore_case:
            curr_lexicon = list(map(str.lower, self.get_lexicon()))

        n = len(tokens)
        final_tokens = []
        i = 0
        while i < n:
            longest_match = None
            longest_match_score = None
            # Starting from the longest possible word (i.e. max n of tokens)
            curr_token = tokens[i]
            if not use_synonyms and not self._trie_.search(curr_token) and (
                    not ignore_case or self._trie_.search(curr_token.lower())):
                i += 1
                continue
            for num_tokens in range(min(self._max_n_of_tokens_, n - i), 0, -1):
                candidate_word = ' '.join(tokens[i:i + num_tokens])
                if candidate_word in self.scores_dict or (ignore_case and candidate_word.lower() in curr_lexicon):
                    longest_match = candidate_word if not return_indexes else (i, i + num_tokens - 1, candidate_word)
                    # longest_match_score = words_dict[candidate]
                    break
                elif use_synonyms:
                    curr_synonym = self.__get_synonym_in_lexicon__(candidate_word) or (
                        self.__get_synonym_in_lexicon__(candidate_word.lower()) if ignore_case else False)
                    if curr_synonym:
                        # print('Added new word {} as it is a synonym of {}'.format(candidate_word, curr_synonym))
                        self.curr_dict[candidate_word] = self.scores_dict[curr_synonym]
                        longest_match = candidate_word if not return_indexes else (i, i + num_tokens - 1, candidate_word)
                        break
            # If any match, updating the index and adding curr_match to final_tokens
            if longest_match:
                final_tokens.append(longest_match)
                i += num_tokens  # By updating i with num_tokens we skip all the tokens which are in this curr_word
            else:
                i += 1

        return final_tokens

    
    def __get_synonym_in_lexicon__(self, curr_w: str) -> str:
        """
        @input curr_w: string, a word composed by one or more tokens (e.g. 'get up')
        @return a string: the most similar synonym to curr_w in the curr dict. If no synonyms in dict, will return False.
        If curr_w starts with a negation, will return the antonym of the word: e.g. curr_w="not be", will return the antonym of be
        """
        if curr_w in self.scores_dict:
            return curr_w
            #return False ?

        curr_tokens = curr_w.split()
        use_antonyms = False

        if len(curr_tokens) > 1 and (curr_tokens[0] == 'not' or "n't" in curr_tokens[0]):
            use_antonyms = True  ###handling negations with antonynms
            context = '_'.join(curr_tokens[1:])
        else:
            context = curr_w.replace(' ', '_')
        for synonym in wordnet.synsets(context): ###looking for synonyms if curr_w is not in dict
            for lemma in set(synonym.lemmas()):
                meaning = None
                if use_antonyms:
                    antonyms = lemma.antonyms()
                    if antonyms:
                        meaning = antonyms[0].name().replace('_', ' ')
                else:
                    meaning = lemma.name().replace('_', ' ')

                if meaning in self.scores_dict:                            
                    return meaning
        return False
    
    
    def __update_global_lexicon__(self, tokens_list) -> None:
        """
        Updates self.curr_dict and self.new_words
        Will generate all possible words from tokens_list, will look for synonyms and if any synonym in dict, will add words to dict by using the same score as the synonym
        """
        all_possible_words = self.__generate_all_possible_words__(tokens_list) #generating all possible words having n_of_tokens<=max_n_of_tokens
        ###for each possible word, checking if it already exists in lexicon or if it has any synonym in lexicon
        ###if synonym(curr_word) in lexicon, assigning score of the synonym to curr_word
        for curr_w in set(all_possible_words):
            curr_synonym = self.__get_synonym_in_lexicon__(curr_w)
            if curr_synonym and curr_synonym!=curr_w:
                self.curr_dict[curr_w] = self.scores_dict[curr_synonym] #assigning the score of the synonym to the new word
                self._new_words_.add(curr_w)
            self._is_updated_ = False
    
    
    def add_words(self, words: dict[str, float]) -> None:
        if not isinstance(words, dict):
            raise ValueError("Please use a dict to add new word(s). Otherwise the new words will have default score=nan")
            if not isinstance(words, (set, list, np.ndarray, pd.Series)):
                words = [words]
            words = set(words)
            curr_new_words = words.difference(set(self.scores_dict.keys()))
            if curr_new_words:
                self.scores_dict |= {w:np.nan for w in curr_new_words}
                self._new_words_.update(curr_new_words)
                self._is_updated_ = False
        else:
            #curr_min = min([min(e[i] for i in range(3) for e in tmp.scores_dict.values())])
            #curr_max = max([max(e[i] for i in range(3) for e in tmp.scores_dict.values())])
            curr_min, curr_max = 0, 1
            if all([isinstance(v,numbers.Number) and all([curr_min<=v<=curr_max]) for v in words.values()]):
                self.scores_dict|=words
            else:
                raise ValueError("Invalid format for the dict in input")
    
    
    def calculate_socialness(self, text, lemmatize=True, strong_socialness_threshold=0.63, fillna=False):
        """
            @return tuple: (num of tokens, num of matched tokens, mean, median, std dev, sum of scores of the words)
            Will calculate the scores  of a text by the scores of the words that are in the text.
            Text score will aggregate individual words scores (i.e. mean, median, total, std deviation, n° of words having score>=threshold)
            Will return such scores as a tuple
            """
        if not isinstance(text, (list,np.ndarray)):
            from nltk.tokenize import WordPunctTokenizer
            tokenizer = WordPunctTokenizer()
            tokens = tokenizer.tokenize(text)
        else:
            tokens = self.__validate_input_tokens__(tokens=text, delimiters='')
        tokens = [t.lower() if not lemmatize else WordNetLemmatizer().lemmatize(t.lower()) for t in tokens] ##lemmas and lowercases

        #tokens = [tok.lower() for tok in text if tok.lower().islower()] ###removing non-words tokens and lowering all tokens
        num_tokens = len(tokens)
        num_social_tokens, num_matches = 0, 0
        if self._expand_lexicon_:
            if self._limit_expansion_:
                self.curr_dict = self.scores_dict.copy()
                matching_tokens = self.__get_matches__(tokens, use_synonyms=True, return_indexes=True)
            else:
                self.curr_dict = self.scores_dict ###can just update scores_dict without any problem if not limit_expansion 
                self.__update_global_lexicon__(tokens) ##will update lexicon with all the possible phrases
                matching_tokens = self.__get_matches__(tokens, use_synonyms=False, return_indexes=True) #avoid generating synonyms as it's already done in update_global_lexicon
        else:
            matching_tokens = self.__get_matches__(tokens, use_synonyms=False, return_indexes=True)

        indexes, matching_tokens = (zip(* [(a[:-1], a[-1]) for a in matching_tokens])) if matching_tokens else ([],[])
        indexes = set(flatten(indexes))
        tokens_counts = {l: matching_tokens.count(l) for l in set(matching_tokens)}
        if not hasattr(self,'curr_dict'):
            self.curr_dict = self.scores_dict.copy()
            
        socialness_scores = []    
        for t in tokens_counts:
            curr_score = self.curr_dict[t]
            socialness_scores.extend([curr_score]*tokens_counts[t])
            num_matches+=tokens_counts[t]
            num_social_tokens+=(tokens_counts[t]*len(t.split())) ##multiplying by len(t.split) in order to count "multiple joined tokens" as they are multiple tokens
                                                                ## eg -> word match= 'water polo', original tokens=['water','polo'], but with get_tokens_list they will become a single token: ['water polo'] -> in order to match properly n of tokens I will count the length of it
        if fillna is not False and fillna is not None:
            n_unmatched_tokens = num_tokens - len(indexes)
            if n_unmatched_tokens:
                if fillna is not True and isinstance(fillna, (float, int)):
                    curr_default_value = fillna
                else:
                    curr_min, curr_max = self.min_max
                    curr_default_value = (curr_max + curr_min)/2
                socialness_scores.extend([curr_default_value] * n_unmatched_tokens)

        del(self.curr_dict)
        if hasattr(self, '_new_words_'):
            self._new_words_ = set()
            
        socialness_scores = np.array(socialness_scores)
        socialness_scores = (num_tokens, num_social_tokens, num_matches, len(np.where(socialness_scores>strong_socialness_threshold)[0]),
                np.sum(socialness_scores), np.mean(socialness_scores), 
                np.std(socialness_scores), np.median(socialness_scores))
        columns = ['num_tokens', 'num_matched_tokens', 'num_matches',
                   'socialness_strong_tokens', 'socialness_total', 'socialness_mean', 'socialness_std', 'socialness_median']
        return {columns[i]:flatten(socialness_scores)[i] for i in range(len(columns))}

    
    def calculate_socialness_df(self, df, text_col=None, **kwargs):
        columns = ['num_tokens', 'num_matched_tokens', 'num_matches',
                   'socialness_strong_tokens', 'socialness_total', 'socialness_mean', 'socialness_std', 'socialness_median']
        if not isinstance(df, pd.Series) and text_col is None:
            raise ValueError('Please set a text_col')
            return
        if text_col is not None:
            df = df[text_col]
        all_single_text_scores = df.map(lambda t: self.calculate_socialness(text=t, **kwargs))
        all_single_text_scores =  all_single_text_scores.apply(pd.Series)
        #all_single_text_scores.columns = columns
        return all_single_text_scores
    
    def get_lexicon(self):
        return self.scores_dict.keys()
    
    def get_scores(self):
        return self.scores_dict