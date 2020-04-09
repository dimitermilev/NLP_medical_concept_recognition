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


def fit_lstm_model(X_train, y_train, n_words, n_tags, seq_len, class_weights, epochs):
    '''Set up LSTM model with one input - equal length sequences of encoded text'''
    input_seq = Input(shape=(seq_len,))

    '''Pass the GloVe pretrained model weights into the embedding layer'''
    embedding = Embedding(input_dim=n_words, output_dim=300, 
                  weights=[embedding_matrix], 
                  trainable=True)(input_seq)
    embedding = Dropout(0.1)(embedding)
    
    '''Add Bidirectional LSTM layer, dense hidden layer, and final output layer'''
    model = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1))(embedding)
    model = TimeDistributed(Dense(64, activation='relu'))(model)
    output = Dense(n_tags, activation="softmax")(model)
    
    '''Compile and fit deep neural network'''
    model = Model(inputs=input_seq, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train
                          , epochs=epochs, batch_size=32, validation_split=0.1, verbose=1, class_weight = [class_weights])
    
    '''Create simple performance report for the model'''
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f'Model validation loss was {val_loss}')
    print(f'Model validation accuracy was {val_acc}')
    return model, history


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