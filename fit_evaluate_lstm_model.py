import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

def seqs_train_test_split(X,y,test_size, random_state):
    '''Split prepared sequences into train and test sets'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test

def score_lstm_model(model, X_test, y_test, idx2word, idx2tag):
    '''Function outputs model predictions and comparisons to true values;
    Model also outputs classification report of model performance by simple class'''    
    predictions = model.predict(X_test)
    '''Get highest probability value from predictions output'''
    predictions_argmax=[]
    for pred in predictions:
        p_arg = np.argmax(pred, axis=-1)
        predictions_argmax.append(p_arg)
    '''Create final comparison table between predicted and actual values'''
    res_word_by_word=[]
    for i, sentence in enumerate(predictions_argmax):
        for j, word in enumerate(X_test_final[i]):
            true = np.argmax(y_test[i], axis=-1)
            nn=[]
            nn.append(idx2word[word])
            nn.append(idx2tag[true[j]][2:])
            nn.append(idx2tag[predictions_argmax[i][j]][2:])
            res_word_by_word.append(nn)  
    result_comparison = pd.DataFrame(res_word_by_word, columns=['word','label','guess'])
    '''Print classification report'''
    print(classification_report(df_res['guess'], df_res['label']))
    return result_comparison