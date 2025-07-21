# ReddiCompass
Reddit-specific NLP pipeline with tailored preprocessing, combined syntactic/semantic/lexical feature extraction, structured multi-run training/prediction, and full resume support.


## Reddit NLP Pipeline

A modular, extensible NLP pipeline focused on Reddit data, combining specialized preprocessing, multi-level feature extraction, and structured handling of multi-run training and prediction workflows.

## Overview

This project implements:

- A Reddit-specific preprocessing pipeline tailored to the linguistic and structural peculiarities of Reddit posts and comments.
- A flexible feature extraction system that combines:
  - **Reddit and online SNSs textual features** (mentions, hashtags, emojis, emoticons, exclamation marks, UPPER words and so on) - **handled as individual tokens!**
  - **Lexical features** (e.g., token frequencies, n-grams)
  - **Syntactic features** (e.g., POS tagging, dependency parsing)
  - **Semantic features** (e.g., embeddings, sentiment, topic distributions)
- A systematic approach to filesystem organization for multi-model and multi-run training and prediction workflows.
- Full resume support across all stages: training, prediction, and feature extraction.

## Filesystem Structure

The pipeline writes and reads from a structured directory layout, allowing multiple models and runs to coexist independently.

