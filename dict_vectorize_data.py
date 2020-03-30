import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from sklearn.feature_extraction import DictVectorizer

def tokens_to_dict(all_tokens_tags):
    '''Transform text/tag tokens into word-level trigram window:
    Include features (like prefix/suffix, title, digit, etc) for prev/current/next words'''
    
    '''Initialize list to hold trigram context dictionaries for each word'''
    token_context_dicts=[]
    
    '''Add previous/current/next word features to each dictionary'''
    for j, doc in enumerate(all_tokens_tags):
        features_doc=[]
        len_doc=len(doc)
        for i, elem in enumerate(doc):
            word = doc[i][0]
            postag = doc[i][1]
            label = doc[i][2]           
            '''Common features for all words'''
            features = {
                    'sentence' : j+1,
                    'word_order' : i+1,
                    'curr': word,
                    'curr_stem': stem_word(word),
                    'curr_bow': word[:3],
                    'curr_eow': word[-3:],
                    'curr_isupper': word.isupper(),
                    'curr_istitle': word.istitle(),
                    'curr_isdigit' : word.isdigit(),
                    'curr_postag' : postag,
                    'curr_sentencepos' : 'middle',
                    'curr_targetlabel': label
                    }
            '''Features for all words not at the beginning of a line of text'''
            if i > 0:
                prev_word = doc[i-1][0]
                prev_tag = doc[i-1][1]
                prev_label = doc[i-1][2]
                features.update({
                        'prev' : prev_word,
                        'prev_stem': stem_word(prev_word),
                        'prev_eow': prev_word[-3:],
                        'prev_isupper': prev_word.isupper(),
                        'prev_istitle': prev_word.istitle(),
                        'prev_isdigit': prev_word.isdigit(),      
                        'prev_postag' : prev_tag,
                        'prev_targetlabel' : prev_label
                        })
            else:
                '''Features for beginning of a line of text'''
                features.update({'curr_sentencepos' : 'beginning'})

            '''Features for all words not at the end of a line of text'''
            if i < len(doc)-1:
                next_word = doc[i+1][0]
                next_postag = doc[i+1][1]
                next_label = doc[i+1][2]        
                features.update({
                        'next' : next_word,
                        'next_stem': stem_word(next_word),
                        'next_eow': next_word[-3:],
                        'next_isupper': next_word.isupper(),
                        'next_istitle': next_word.istitle(),
                        'next_isdigit': next_word.isdigit(),      
                        'next_postag' : next_postag,
                        'next_targetlabel' : next_label
                        })
            else:
                '''Features for end of a line of text'''
                features.update({'curr_sentencepos' : 'end'})
            features_doc.append(features)
        token_context_dicts.extend(features_doc)
    return token_context_dicts
    

def dict_vectorize_trigrams(token_context_dicts, list_of_features):
    '''Function takes list of dictionaries and list of desired dictionary keys,
       and outputs a dictionary vectorized one hot encodings of features'''
    dict_vec = DictVectorizer(sparse=False)
    
    '''Temporary transformation into pandas df to easily select features and fill in NA values as False'''
    df_temp = pd.DataFrame(token_context_dicts)[list_of_features].fillna(False)
    '''Also return the target label'''
    y = [ word['curr_targetlabel'] for word in token_context_dicts] 

    return dict_vec.fit_transform(df_temp.to_dict('records')), y
