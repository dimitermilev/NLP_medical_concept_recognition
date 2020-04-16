import string
import re
import numpy as np
import pandas as pd
import json
import pickle

import nltk
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tag import pos_tag
from nltk.tag import str2tuple
from nltk.chunk import ne_chunk
import spacy
from spacy import displacy

from gensim import corpora, models, similarities, matutils
from gensim.models.keyedvectors import KeyedVectors

from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import tensorflow
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from flask import Flask,url_for,render_template,request
from flaskext.markdown import Markdown

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

'''Load word to word encoding crosswalks'''
for file in ['word2idx','idx2word']:
    with open(f"{file}.pickle", "rb") as pfile:
        exec(f"{file} = pickle.load(pfile)")

'''Load created embeddings matrix'''
with open("bio_embeddings.pickle", "rb") as pfile:
    bio_embbeding_matrix = pickle.load(pfile)

'''Get pretrained BiLSTM modelmodel'''
global model
model = tensorflow.keras.models.load_model('hospitalnotes_model.h5')
    
app = Flask(__name__)
Markdown(app)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/extract',methods=['GET','POST'])
def extract():
    '''Function to intake raw hospital note text, process it and output HTML tagged text with problems, treatments and tests'''
        
    '''Define basic string functions to perform through processing'''
    stemmer=PorterStemmer()
    spaces = re.compile(r'\ +')
    stemfunc = lambda x: stemmer.stem(x).lower()
    puncfix = lambda x: x.replace("’","").replace(":"," : ").replace("“"," “ ").replace("?”"," ? ” ").replace("?"," ? ").replace("!"," ! ").replace("."," . ").replace(","," , ")

    '''Tokenize raw text'''
    raw_text = request.form['rawtext']
    note_tokens_func = raw_text.split('\n')
    all_note_tokens_func = [re.split('\ +', puncfix(ln)) for ln in note_tokens_func]
    
    '''Turn tokens into sequences of tuples - word and its part of speech'''
    data_func= []
    data_stem_func = []
    for i, line in enumerate(all_note_tokens_func):
        ln = []
        ln_stem = []
        for j, word in enumerate(line):
            if all_note_tokens_func[i][j]=='':
                all_note_tokens_func[i][j]=' '
            ln.append((all_note_tokens_func[i][j], pos_tag([all_note_tokens_func[i][j]])[0][1]))
            ln_stem.append((stemfunc(all_note_tokens_func[i][j]), pos_tag([all_note_tokens_func[i][j]])[0][1]))
        '''Retain both stemmed and full versions of each word token'''
        data_func.append(ln)
        data_stem_func.append(ln_stem)
        
    '''Create index number encoded copy of the tokenized document'''
    document_idx = []
    for s in data_stem_func:
        line_idx=[]
        for w in s:
            try:
                line_idx.append(word2idx[w[0]] )
            except:
                line_idx.append(word2idx['_padding'] )
        document_idx.append(line_idx)
        
    '''Turn document line sequences into uniform 60 token padded sequences'''
    document_idx_eval = pad_sequences(maxlen=60, sequences=document_idx, padding="post",value=11555 - 2)
    document_idx_df=pd.DataFrame(document_idx).reset_index()
    document_idx_eval = np.array(document_idx_eval)
                 
    '''Use BiLSTM model to recognize the medical concepts in the uniform sequences'''
    res_func_sentence=[]
    for i, doc in enumerate((document_idx_eval)):
        line_res=[]
        p = model.predict(np.array([document_idx_eval[i]]))

        p_arg = np.argmax(p, axis=-1)

        for j, word in enumerate(document_idx_eval[i]): 
            nn=[]
            nn.append(idx2word[word])
            nn.append(p_arg[0][j])
            line_res.append(nn)
        res_func_sentence.append(line_res)
        
    '''Prepare predictions for display purposes - first, create the tokenized list of original words'''
    f_display_text = []
    for i in list(document_idx_df['index']):
        words_only=[]
        for item in (data_func[i][0:60]):
            words_only.append(item[0])
        f_display_text.append(words_only)

    '''Turn predictions into simple problem, treatment and test'''
    idx2tag_simple = {0: 'problem', 1: 'test', 2: 'treatment', 3: 'problem', 4: 'test'
                  , 5: 'treatment', 6: 'outside'}

    '''Initialize sentence and entity lists that will be used in the HTML tagging'''
    f_test_set_translate=[]
    f_tags_translate=[]
    f_test_spans = []
    f_full_sents = []
    f_entities = []
    
    '''Specify the character by character position of predicted tags'''
    for i, line in enumerate(document_idx_eval):
        line_translate = []
        line_tags = []
        sent_pos = 0
        s = []
        for j, word in enumerate(f_display_text[i]):
            if word!='_padding':
                line_translate.append(word)
                line_tags.append(idx2tag_simple[res_func_sentence[i][j][1]])
                length = len(word)
                if j == 0:
                    s.append((0,0+length))
                    sent_pos+=length
                else:
                    s.append((sent_pos+1,sent_pos+1+length))
                    sent_pos+=length+1
        f_test_set_translate.append(line_translate)
        f_full_sents.append(' '.join(f_display_text[i]))
        f_tags_translate.append(line_tags)
        f_test_spans.append(s)        

    '''Create a structure linking the predicted entities to the specific sentence character spans'''
    f_ents=[]
    for i, line in enumerate(f_test_spans):
        sent_ents = []
        for j, span in enumerate(line):
            if f_tags_translate[i][j] !='outside':
                sent_ents.append({'start': span[0], 'end':span[1], 'label': f_tags_translate[i][j]})
        f_ents.append(sent_ents)
    
    '''Final linkage of sentences to entity character spans'''
    f_spacy_sentences=[]
    for i, sent in enumerate(f_full_sents):
        newsent=[{'text':sent, 'ents':f_ents[i],'title':None}]
        f_spacy_sentences.append(newsent)

    '''Use spaCy display html tagged sentences along entity character spans'''
    html_text=''
    for sentence in f_spacy_sentences:  
        html_text+=displacy.render(sentence, style='ent', manual=True, options= {'colors': {
                                                                                "PROBLEM": "#7aecec",
                                                                                "TREATMENT": "#bfeeb7",
                                                                                "TEST": "#aa9cfc"
                                                                                   }})

    html_text = html_text.replace("\n\n","\n")
    result = HTML_WRAPPER.format(html_text)

    return render_template('result.html',rawtext=raw_text,result=result)        

if __name__ == '__main__':
    app.run(debug=True, threaded=False)