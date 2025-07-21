# ReddiCompass
Reddit-specific NLP pipeline with tailored preprocessing, combined syntactic/semantic/lexical feature extraction, structured multi-run training/prediction, and full resume support.

## Reddit NLP Pipeline

A modular, extensible NLP pipeline focused on Reddit data, combining specialized preprocessing, multi-level feature extraction, and structured handling of multi-run training and prediction workflows.

## Overview

This project implements:

- A Reddit-specific preprocessing pipeline tailored to the linguistic and structural peculiarities of Reddit posts and comments.
- A flexible feature extraction system that combines:
  - *Reddit and online SNSs textual features* (mentions, hashtags, emojis, emoticons, exclamation marks, UPPERWORDS and so on)
    - ***They are handled as individual tokens!***
  - *Syntactic and Lexical features* (token frequencies, n-grams, POS tagging, NERs)
  - *Lexicon matching logic and lexicon extracted features* (NRCLex, NRCVad, Socialness)
  - *Semantic features* (embeddings, sentiment, emotions, toxicity levels)
- A systematic approach to filesystem organization for multi-model and multi-run training and prediction workflows.
- Full resume support across all stages: training, prediction, and feature extraction.

## Usage Examples

### Example 1: Text Processing

The following example demonstrates how to process a single text and extract various features from it:

```python
from text_processing.textractor import TexTractor

# Example Reddit text
t = TexTractor("This is a r/reddit text and it's amazing! 😀 emoji emoji Try it out 🚀 #reddit www.reddit.com")


# Process the text to extract all features
t.process()

# Get features from the text
mentions = t.get_mentions()
print(mentions) ## will print ['r/reddit', feature_type: 'mention']
emojis = t.get_emojis() ## will contain ['😀', '🚀']


# Get all features from the processed text
t.get_all_features()
```

