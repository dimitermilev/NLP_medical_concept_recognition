import numpy as np
import pandas as pd

def get_corpus_vocabulary(token_context_dicts):
    '''Function creates a vocabulary of unique (stemmed) 
    words in the corpus, and returns the list, as well as index encodings crosswalks'''
    
    '''Temporary transformation into pandas df to easily pull the unique values'''
    df_temp = pd.DataFrame(token_context_dicts)
    words = list(set(df_temp["curr_stem"].values))
    words.extend(["_unknown","_padding"])
    n_words = len(words)
    
    '''Dictionary crosswalks that return a word's encoding, or the word itself from its index '''
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for i, w in enumerate(words)}

    return words, n_words, word2idx, idx2word

def get_corpus_tags(token_context_dicts):
    df_temp = pd.DataFrame(token_context_dicts)
    tags = sorted(list(set(df_temp["curr_targetlabel"].values)))
    n_tags = len(tags)
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for i, t in enumerate(tags)}
    return tags, n_tags, tag2idx, idx2tag
