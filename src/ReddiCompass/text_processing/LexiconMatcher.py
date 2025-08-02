import warnings
import pandas as pd
import numpy as np
from utils import flatten
from typing import Sequence




class LexiconMatcher:
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False

    class Trie:
        def __init__(self):
            self.root = LexiconMatcher.TrieNode()

        def insert(self, word: str) -> None:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = LexiconMatcher.TrieNode()
                node = node.children[char]
            node.is_end_of_word = True

        def search(self, s: str) -> int:
            node = self.root
            for char in s:
                if char not in node.children:
                    return False
                node = node.children[char]

            is_prefix = True
            is_exact = node.is_end_of_word
            return is_prefix + is_exact



    
    def __init__(self, lexicon:Sequence[str], add_only:bool = True) -> None:
        curr_lexicon = set(lexicon)
        self.lexicon = curr_lexicon
        self._add_only_ = add_only
        self._max_n_of_tokens_= 0
        self._trie_ = LexiconMatcher.Trie()
        self._is_updated_ = False
        self._new_words_ = curr_lexicon.copy()
            
    def __repr__(self):
        return self.__str__()
    
    def __str__(self) -> str:
        return 'LexiconMatcher containing {} words'.format(len(self.lexicon))

    @staticmethod
    def __validate_input_tokens__(tokens: Sequence[str], delimiters: str = '-_#()[]', keep_original_indexes: bool = False) -> list[str]:
        """
        Will return the list of tokens after splitting each token by white space or any of the delimiters in input.
        This way, a list of tokens such as ['t1', 't2 t3', 't4#t5'] will become: ['t1','t2','t3','t4','t5']
        """
        if not delimiters and not isinstance(delimiters, str):
            return tokens if not keep_original_indexes else list(enumerate(tokens))

        splitted_tokens_list = [t.translate(str.maketrans(delimiters, ' ' * len(delimiters))).strip().split() for t in flatten(tokens)]
        if not keep_original_indexes:
            return list(flatten(splitted_tokens_list))
        else:
            tokens_with_positions = [(idx, token) for idx, token_list in enumerate(splitted_tokens_list) for token in token_list]
            return tokens_with_positions
        
    def __update_starting_tokens__(self) -> None:
        """
        Max n of tokens will be the max number of tokens to generate new words from a text.
        Possible starting tokens is a set containing all the starting tokens composing the words in current lexicon
        They are both useful for efficiency reasons: saving time when generating words(i.e. multiple tokens) from a new text"""
        if self._is_updated_:
            return
        elif self._add_only_: 
            """ 
            Will only calculate max_n_of_tokens and starting_tokens for new words added to lexicon and will merge the results with the already existing max and starting_tokens
            """
            curr_max_n_of_tokens, possible_starting_tokens = 0, set()
            if hasattr(self, '_new_words_') and self._new_words_:
                n_of_tokens, possible_starting_tokens = list(zip (*map (lambda tokens: \
                                                                     (len(tokens), tokens[0]), map(str.split, self._new_words_))))
                for word in self._new_words_:
                    self._trie_.insert(word)
                curr_max_n_of_tokens = max(n_of_tokens)
            self._max_n_of_tokens_ = max(self._max_n_of_tokens_, curr_max_n_of_tokens)
            #self._starting_tokens_ = self._starting_tokens_.union(set(possible_starting_tokens))
            self._new_words_ = set()
            self._is_updated_ = True
        
        else:    
            n_of_tokens, starting_tokens = list(zip( *map(lambda tokens: \
                                                     (len(tokens), tokens[0]), map(str.split, self.lexicon) ) ))
            self._trie_ = Trie()
            for word in self.lexicon:
                self._trie_.insert(word)
            self._max_n_of_tokens_ = max(n_of_tokens) ##setting useful attributes
            #self._starting_tokens_ = set(starting_tokens)
            self._new_words_ = set()
            self._is_updated_ = True
    
    
    def add_words(self, words: Sequence[str]) -> None:
        """
        Will update self.lexicon by adding the @input words
        """
        if not isinstance(words, (set, list, np.ndarray, pd.Series)):
            words = [words]
        curr_new_words = set(words).difference(self.lexicon)
        if curr_new_words:
            self.lexicon.update(curr_new_words)
            self._new_words_.update(curr_new_words)
            self._is_updated_ = False ##if any new word added, is_updated=False in order to update_starting_tokens properly
        
    def remove_words(self, words: Sequence[str]) -> None:
        """
        Will update self.lexicon by removing the @input words
        """
        if self._add_only_:
            warnings.warn("Cannot remove on a add_only element. Will ignore the removal operation and the lexicon will remain the same")
            return 
        if not isinstance(words, (set, list, np.ndarray, pd.Series)):
            words = [words]
        curr_words_length = len(self.lexicon.copy())
        self.lexicon.difference_update(words)
        
        if curr_words_length!=len(self.lexicon):
            self._is_updated_ = False ##if any word removed, is_updated=False in order to update_starting tokens properly


    def get_matches(self, tokens: Sequence[str], return_indexes: bool = False, ignore_case: bool = False, split_tokens_delimiters: str = '') -> list[str]:
        """
        @return the list of words of self.lexicon which match the input tokens.
        If @input return_indexes is True, will return the tokens positions together  with the matches.
        """
        self.__update_starting_tokens__() ##calling update_starting_tokens to be sure the current starting tokens and max tokens are properly set
        if ignore_case:
            curr_lexicon = list(map(str.lower, self.lexicon))
        if not isinstance(tokens, (set, list, np.ndarray, pd.Series)):
            tokens = [tokens]
        if not isinstance(split_tokens_delimiters, str) and split_tokens_delimiters is not False:
            split_tokens_delimiters = ''
        tokens = self.__validate_input_tokens__(tokens, delimiters=split_tokens_delimiters, keep_original_indexes=return_indexes)
        if not tokens:
            return []
        if return_indexes:
            indexes, tokens = zip(*tokens)
        n = len(tokens)
        final_tokens=[]
        i = 0  
        while i < n:
            longest_match = None
            curr_token = tokens[i]
            # Starting from the longest possible word (i.e. max n of tokens)
            if not self._trie_.search(curr_token) and (not ignore_case or not self._trie_.search(curr_token.lower())): ###curr token is not a match/subword match for any of the words in lexicon
                i+=1
                continue
            for num_tokens in range(min(self._max_n_of_tokens_, n - i), 0, -1): ##starting from the longest possible word (i.e. word made by token, token+1,..., token+n)
                candidate_word = ' '.join(tokens[i:i + num_tokens])
                if candidate_word in self.lexicon or (ignore_case and candidate_word.lower() in curr_lexicon): ##if curr word is a valid word in lexicon, we have a match and we skip all of the tokens in this current word... otherwise we decrease n and look for the word composed by token, token+1,..., token+n-1
                    longest_match = candidate_word  if not return_indexes else (indexes[i], indexes[i+num_tokens-1], candidate_word)
                    break
                
            # If any match, updating the index and adding curr_match to final_tokens
            if longest_match: ###if we have any match, we add it to the tokens 
                final_tokens.append(longest_match)
                i += num_tokens  # By updating i with num_tokens we skip all the tokens which are in this curr_word
            else: 
                i += 1
                
        return final_tokens