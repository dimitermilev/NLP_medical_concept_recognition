import numpy as np 
import pandas as pd
import gensim
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.utils import class_weight

def create_embeddings_matrix(glove_file_loc, glove_file_dims, n_words, idx2word):
    '''Use the GloVe pre-trained 300 dimension word vectors to create an embedding matrix for the vocab in the corpus'''
    '''Specify location of the GloVe file, which can be downloaded here: https://nlp.stanford.edu/projects/glove/'''
    w2v_output_file = glove_file_loc
    glovemodel = gensim.models.KeyedVectors.load_word2vec_format(w2v_output_file, binary=False)
    '''Pass the GloVe pretrained model weights into the vocabulary list'''
    '''Create an empty matrix with the dimensions of the vocab size and the embeddings size'''
    embedding_matrix = np.zeros((n_words , glove_file_dims))
    missing = []
    for i, w in idx2word.items():
        try:
            embedding_vector = glovemodel[w]
            embedding_matrix[i] = embedding_vector
        except: 
            '''Assess the size of the vocabulary unsuccessfully matched to the GloVe word vectors'''
            missing.append(w)
            pass
    return embedding_matrix

def prepare_sequences(all_tokens_tags):
    '''Function takes tokenized corpus and creates equal-length sequences of encoded words and tags'''
    seq_len = int(np.ceil(np.percentile([len(s) for s in all_tokens_tags], 99.5)))    
    '''Pads sentences to a length matching up to 99.5 percentile of sequences length'''
    X = [[word2idx[PorterStemmer().stem(word[0]).lower()] for word in sentence] for sentence in all_tokens_tags]
    X = pad_sequences(maxlen=seq_len, sequences=X, padding="post",value=n_words - 1)
    X_df = pd.DataFrame(X).reset_index()
    y = [[tag2idx[word[2]] for word in sentence] for sentence in all_tokens_tags]
    y = pad_sequences(maxlen=seq_len, sequences=y, padding="post", value=tag2idx["O"])   
    '''Due to imbalanced target classes, class weights are derived and returned'''
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y.flatten())
    '''Return the y variable as an array for each time step, with array length matching the number of classes'''
    y_cat = [to_categorical(i, num_classes=n_tags) for i in y]
    return X, y_cat, class_weights
