import os
import re
from nltk.tag import pos_tag

def process_text_tags(hospital_datasets, folder_types):
    '''Read in and tokenize all hospital notes text and their tags; 
       Function takes in two lists of folders:
       1. List of hospitals that have provided notes
       2. List of data types - for example text files and concept files'''
    
    '''Set file paths to note text'''
    cwd = os.getcwd()
    base = cwd+ '/' + 'concept_assertion_relation_training_data'

    '''Set regexes for processing tags'''
    name = re.compile(r'c="(.*?)" \d')
    tag = re.compile(r't="(.*?)"$')
    position = re.compile(r'(\d+:\d+\ \d+:\d+)')

    '''Initialize lists that will hold notes and tags'''
    all_note_tokens = []
    all_note_tags = []

    for db in hospital_datasets:
        '''Process each NOTE file and append to note list'''
        filenames = sorted([file for file in (os.listdir(base+db+folder_types[0])) if file.endswith('.txt')])
        for file in filenames:
            text = open(base+db+folder_types[0]+file, 'r').read()
            
            '''Parse notes into individual lines and then individual words'''
            note_tokens = text.split('\n')
            note_tokens = [re.split('\ +', ln) for ln in note_tokens]            
            all_note_tokens += note_tokens
            
            '''Process each TAG file and append to tag list'''
            cons = open(base+db+folder_types[1]+file[:-3] + 'con', 'r').read()
            cons = cons.split('\n')
            cons = [con for con in cons if len(con)>0]
            
            '''Set list framework mirroring the note text framework:
               Will be filled in with concept tags'''
            con_frame = [['O'] * len(ln) for ln in note_tokens]
            
            '''Use regex to parse tags into their individual components'''
            for concept in cons:
                concept_name = name.findall(concept) #Extracts the concept name
                concept_tag = tag.findall(concept)[0] #Extracts the concept type
                concept_span = position.findall(concept) #Extracts the concept position
                
                '''Identify word positions for each concept in the text line'''
                span_1, span_2 = concept_span[0].split(' ')
                spans = [int(i) for i in (span_1.split(':')+ span_2.split(':'))]
                original_text = ' '.join(note_tokens[spans[0] - 1][spans[1]:spans[3] + 1])

                '''Create BIO tags: Begin, Inside or Outside 
                   based on the position of the word within the tag'''
                order = 1
                for start_index in range(spans[1], spans[3] + 1):
                    if order == 1:
                        con_frame[spans[0] - 1][start_index] = 'B_'+concept_tag
                    else:
                        con_frame[spans[0] - 1][start_index] = 'I_'+concept_tag
                    order +=1
                    
            all_note_tags += (con_frame)
            
    return all_note_tokens, all_note_tags

def stem_word(word):
    return PorterStemmer().stem(word).lower()

def combine_text_tags(all_note_tokens):
    '''Combine the identical structures of the text / tag tokens into tuples;
       Also add POS tag as tuple values''' 
        
    '''Initialize list of tuples'''
    all_tokens_tags = []
    
    for i, line in enumerate(all_note_tokens):
        ln_stem = []
        for j, word in enumerate(line):
            if all_note_tokens[i][j]=='':
                all_note_tokens[i][j]=' '
            ln_stem.append((all_note_tokens[i][j], pos_tag([all_note_tokens[i][j]])[0][1], all_note_tags[i][j]))
        all_tokens_tags.append(ln_stem)
        
    return all_tokens_tags