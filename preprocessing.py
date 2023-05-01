'''
Preprocessing and Feature Engeneering
---------------------
Kilian Lüders & Bent Stohlmann
1.5.2023 (Draft Version)

Input:
Word Embedding Model: we_model.model
Text Data: 2023_3_7_vhmk_data.csv

Output:
Training Data: training_data.pkl
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import spacy
from gensim.models import Word2Vec

#feature engineering: Word Embedding
wm_model = Word2Vec.load("data/we_model.model")
w2v = dict(zip(wm_model.wv.index_to_key, [wm_model.wv[i] for i in wm_model.wv.index_to_key]))

def w2v_transform(words):
    words = [word for word in words.split(" ") if word in w2v]
    if len(words) == 0:
        return np.zeros(200)
    else:
        return np.mean([w2v[w] for w in words], axis=0)


#feature engineering: token and lemma
nlp = spacy.load("de_core_news_lg")

def preprocess_token(text):
    '''
    returns tokenized text
    '''
    doc = nlp(text, disable=['attribute_ruler','ner'])
    return " ".join([token.text.lower() for token in doc if token.is_alpha == True]) + " "

def preprocess_lemma(text):
    '''
    returns lemmatized text
    '''
    doc = nlp(text, disable=['attribute_ruler','ner'])
    return " ".join([token.lemma_.lower() for token in doc if token.is_alpha == True]) + " "



def main():
    # load data
    data = pd.read_csv("data/2023_3_7_vhmk_data.csv", sep=",", index_col=0)
    # load case metadata and prepare merge column
    metadata = pd.read_csv("data/Metadaten2.6.1.csv", sep="\t").rename(columns={'dateiname': 'entscheidung'})
    # check number of cases (n = 300)
    # len(data.entscheidung.unique())


    # filter relevant sents
    # subset: only part of decision which is supposed to be annotated 
    data = data[data.teiltext.isin(['zulaessigkeit','ueberschneidung', 'begruendetheit'])]

    # recode vars from str to int
    for var in ['zweck', 'geeignetheit', 'erforderlichkeit', 'angemessenheit','unspezifisch']:
        data[var] = data[var].astype(int)

    # new varible
    # prop -> any proportionality coding
    data['prop'] = np.where((data.zweck > 0) | (data.geeignetheit > 0) | (data.erforderlichkeit > 0) | (data.angemessenheit > 0) | (data.unspezifisch > 0), 1, 0)
    data = data[data.teiltext.isin(['zulaessigkeit','ueberschneidung', 'begruendetheit'])]
    print("number decisions:" + str(data.entscheidung.nunique()))
    print("shape dataframe:" + str(data.shape))
    print("numbre of codings:\n" + str(data.prop.value_counts()))

    print('Features:')
    print('Lemma')
    data['X_lemma'] = data['text'].progress_apply(lambda x: preprocess_lemma(x))

    print('Token')
    data['X_token'] = data['text'].progress_apply(lambda x: preprocess_token(x))

    print('X_we')
    data['X_we'] = data['X_lemma'].progress_apply(lambda x: w2v_transform(x))

    # save training data sent level
    data[['entscheidung', 'prop', 'X_lemma', 'X_token', 'X_we']].reset_index(drop=True).to_pickle('data/training_data_sent.pkl')
    print('save:\tdata/training_data_sent.pkl')

    # add aditional space to merge dec text in a simple way
    data['text'] = data.text.apply(lambda x: x + " ")

    # aggregation 
    data_dec = data.groupby(['entscheidung']).agg({
            'id': 'count',
            'prop': 'sum',
            'text': 'sum',
            'X_lemma': 'sum',
            #'X_lemma_stop': 'sum',
            'X_token': 'sum',
            #'X_token_stop': 'sum'
            }).reset_index().rename(columns={'id': 'len'})

    print('X_we - Dec')
    data_dec['X_we'] = data_dec['X_lemma'].progress_apply(lambda x: w2v_transform(x))

    # save training data dec level
    data_dec.reset_index(drop=True).to_pickle('data/training_data.pkl')
    print('save:\tdata/training_data.pkl')
    return

if __name__ == "__main__":
    main()
