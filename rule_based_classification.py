''''
Rule Based-Classification
---------------------
Kilian Lüders & Bent Stohlmann
1.5.2023 (Draft Version)


Code for the rule-based classifier. The classifier checks whether proportionality tests occur in GFCC decisions or not.

usage:
import rule_based_classification as rbc
rbc.check_entscheidung(entscheidung_name: *decision name*, entscheidungs_text: *decision text*)

version note:
This version of the code has been slightly modified to make the classifier work with the same data as the other ML classifiers.
New is therefore the function prepare_text().
'''

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
nlp = spacy.load("de_core_news_lg") # language model

# Definition of search patterns
#################

# regex pattern
pattern_space = re.compile(r'\s+')
pattern_vhmk = re.compile(r"[Vv]erhältnismä")               # To check if sentence is relevant -> is_vhmk_stichwort()
pattern_stufe_geeignet = re.compile(r"[Gg]eeignet")         # To check if sentence is relevant -> is_stufen_stichwort()
pattern_stufe_erforderlich = re.compile(r"[Ee]rforderlich") # To check if sentence is relevant -> is_stufen_stichwort()
pattern_stufe_angemessen = re.compile(r"[Aa]ngemessen")     # To check if sentence is relevant -> is_stufen_stichwort()
pattern_uebermass = re.compile(r"Überma.*verbo.*")          # To check if sentence is relevant -> is_uebermass_stichwort()

# spacy matcher settings
matcher_kon = Matcher(nlp.vocab) # Searches subjunctive
matcher_kon.add("MOOD_SUB", [[{"MORPH": {"REGEX": ".*Mood=Sub.*"}}]])

matcher_vhmk_adjd = Matcher(nlp.vocab) # seeks adverbial use of 'verhältnismäßig'
matcher_vhmk_adjd.add("VHMK_ADJD", [[{"TEXT": {"REGEX": ".*erhältnismä.*"}, "TAG": "ADJD"},{"TAG":"ADJD"}],
                                    [{"TEXT": {"REGEX": ".*erhältnismä.*"}, "TAG": "ADJD"},{"TAG":"ADJA"}],
                                    [{"TEXT": {"REGEX": ".*erhältnismä.*"}, "TAG": "ADJD"},{"TAG":"ADV"}],
                                    [{"TEXT": {"REGEX": ".*erhältnismä.*"}, "TAG": "ADJD"},{"TAG":"PIAT"}],
                                    [{"TEXT": {"REGEX": ".*erhältnismä.*"}, "TAG": "ADJD"},{"TAG":"PIS"}]])

matcher_vhmk_erschwert = Matcher(nlp.vocab) # Seeks wording 'unverhältnismässig erschwert' (disproportionately difficult)
matcher_vhmk_erschwert.add("VHMK_erschwert", [[{"TEXT": {"REGEX": ".nverhältnismä.*"}},{"TEXT": {"REGEX": ".*rschw.*"}}]])

matcher_erfge_adja = Matcher(nlp.vocab) # Looks for instances of 'geeignet/erfordlich' as an adjective followed by a noun.
matcher_erfge_adja.add("GE_ADJA", [[{"TEXT": {"REGEX": "geeigne.*"}, "TAG": "ADJA"}, {"TAG":"NN"}]])
matcher_erfge_adja.add("ERF_ADJA", [[{"TEXT": {"REGEX": "erforderli.*"}, "TAG": "ADJA"}, {"TAG":"NN"}]])

matcher_erfge_adja_mittel = Matcher(nlp.vocab) # Seeks wording 'geeignetes/erforderliches Mittel/Maß' (appropriate/required means/measure)
matcher_erfge_adja_mittel.add("ERFGE_ADJA_mittel", [[{"TEXT": {"REGEX": "geeigne.*"}, "TAG": "ADJA"}, {"TEXT":"Mittel", "TAG":"NN"}],
                                                    [{"TEXT": {"REGEX": "erforderli.*"}, "TAG": "ADJA"}, {"TEXT":"Mittel", "TAG":"NN"}],
                                                    [{"TEXT": {"REGEX": "geeigne.*"}, "TAG": "ADJA"}, {"TEXT":"Maß", "TAG":"NN"}],
                                                    [{"TEXT": {"REGEX": "erforderli.*"}, "TAG": "ADJA"}, {"TEXT":"Maß", "TAG":"NN"}]])


# helper functions for the search
#################

def is_vhmk_stichwort(rn_txt: str) ->bool:
    '''
    Identifies phrase with proportionality keyword.
    '''
    return bool(pattern_vhmk.findall(rn_txt)) 


def is_stufen_stichwort(rn_txt: str) ->bool:
    '''
    Identifies sentence that refer to at least two different proportionality steps.
    '''
    match_geeignet = bool(pattern_stufe_geeignet.findall(rn_txt))
    match_erforderlich = bool(pattern_stufe_erforderlich.findall(rn_txt))
    match_angemessen = bool(pattern_stufe_angemessen.findall(rn_txt))
    return sum([match_geeignet, match_erforderlich, match_angemessen]) > 1

def is_uebermass_stichwort(rn_txt: str) ->bool:
    '''
    Identifies phrase with 'Übermaß' keyword.
    '''
    return bool(pattern_uebermass.findall(rn_txt))

def is_vhmk_erschwert(sent_doc: spacy.tokens.span.Span) -> bool:
    '''
    Seeks wording 'unverhältnismässig erschwert' (disproportionately difficult)
    '''
    return bool(matcher_vhmk_erschwert(sent_doc))

def is_vhmk_adjd(sent_doc: spacy.tokens.span.Span) -> bool:
    '''
    Check whether "verhältnismässig" (proportionate) is an adverbially used adjective.
    '''
    return bool(matcher_vhmk_adjd(sent_doc))

def is_erfge_adja(sent_doc: spacy.tokens.span.Span) -> bool:
    '''
    Checks if "erforderlich" or "geeignet" is used as an adjective followed by a noun.
    '''
    return bool(matcher_erfge_adja(sent_doc))

def is_erfge_adja_mittel(sent_doc: spacy.tokens.span.Span) -> bool:
    '''
    Seeks wording 'geeignetes/erforderliches Mittel/Maß' (appropriate/required means/measure)
    '''
    return bool(matcher_erfge_adja_mittel(sent_doc))

def is_konjunktiv(sent_doc: spacy.tokens.span.Span) -> bool:
    '''
    Checks if it is a sentence in the subjunctive mood.
    '''
    return bool(matcher_kon(sent_doc))

# helper functions for text handling
#################

#def get_tbeg(eb_attr):
#    '''
#    Function to retrieve the information about the decision partition from the xml tags
#    '''
#    if 'tbeg' in eb_attr.keys():
#        return eb_attr['tbeg']
#    else:
#        return np.nan


def clean_string(text):
    '''
    Function to clean up strings
    '''
    return text.replace('\n','').replace("&lt;","[").replace("&gt;","]").replace("#160"," ").replace(u'\xa0', u' ')


#def clearn_rn_tags(rn_attr):
#    '''
#    Function to search the paragraph tags of the xml file
#    '''
#    if rn_attr == dict():
#        return np.nan
#    else: 
#        return rn_attr['rn']

def prepare_text(entscheidung_name: str, entscheidungs_text: str) -> pd.DataFrame:
    '''
    Function to get text from VMHK annotation into the right format for the check_entscheidung() function.

    As input it gets decision name (entscheidung_name) and decision text (entscheidungs_text).
    The language model is applied -> nlp()
    The output is a data frame in which each row represents one sentence. 
    '''
    sent_i = 0 #count sentences
    data_entscheidung = list()
    entscheidungs_text = clean_string(entscheidungs_text) #clean string
    entscheidungs_text = re.sub(pattern_space, " ",entscheidungs_text).strip() #clean raw text from spaxes
    doc = nlp(entscheidungs_text) # spacy language model
    for sent in doc.sents: # iterate over sents
        sent_i += 1
        data_entscheidung.append({
            'id': entscheidung_name + "_" + str(sent_i), 
            'file': entscheidung_name,
            'ebene': np.nan,
            'ebene_nr': np.nan,
            'tbeg': "eg",
            'rn': np.nan,
            'sent_i': sent_i,
            'text_raw': sent.text.strip(),
            'doc': sent,
            'ebenen_tag': False})
    return pd.DataFrame(data_entscheidung)


# main function to check decisions
#################


def check_entscheidung(entscheidung_name: str, entscheidungs_text: str, return_df = False):
    '''
    Essential function where everything comes together. 

    As input it gets decision name (entscheidung_name) and decision text (entscheidungs_text).

    Additionally there is the option return_df:
    If false: the function returns only the result (bool).
    If True: the function returns a tuple of result and data frame.
    
    The dataframe contains the essential sentences and all found decision categories.
    This is very useful when you want to understand individual decisions.
    '''
    # prepare text -> get df where each row represents a sent
    e_df = prepare_text(entscheidung_name, entscheidungs_text)
    ## Exclude circumstances
    #if not all(e_df.tbeg.isna()):
    #    e_df = e_df[e_df.tbeg == "eg"]

    # From here the searching starts:
    # first step: identify relevant sentences
    e_df['relevant_vhmk'] = e_df['text_raw'].apply(is_vhmk_stichwort) # proportionality keyword
    e_df['relevant_stufen'] = e_df['text_raw'].apply(is_stufen_stichwort) # keywords of prop steps
    e_df['relevant_uebermass'] = e_df['text_raw'].apply(is_uebermass_stichwort) # Übermaß keyword
    # drop irrelevant sents
    e_df = e_df.drop(e_df[((e_df.relevant_vhmk == False) & (e_df.relevant_stufen == False)) & (e_df.relevant_uebermass == False)].index)

    # second step: verify found sentences
    e_df['is_konjunktiv'] = e_df['doc'].apply(is_konjunktiv)
    e_df['is_erfge_adja'] = e_df['doc'].apply(is_erfge_adja)
    e_df['is_erfge_adja_mittel'] = e_df['doc'].apply(is_erfge_adja_mittel)
    e_df['is_vhmk_adjd'] = e_df['doc'].apply(is_vhmk_adjd)
    e_df['is_vhmk_erschwert'] = e_df['doc'].apply(is_vhmk_erschwert)

    # finally:inference of the decision.
    output = False
    for i in e_df.index:
        if not e_df['is_konjunktiv'][i]: # no subjunctive
            if e_df['relevant_vhmk'][i]: # is relevant because of prop keyword
                if not e_df['is_vhmk_adjd'][i] and not e_df['is_vhmk_erschwert'][i]: # Neither adjd use nor 'erschwert' wording
                    output = True
            if e_df['relevant_stufen'][i]: # is relevant because of steps keyword
                if e_df['is_erfge_adja'][i]:
                    if e_df['is_erfge_adja_mittel'][i]: # if erforderlich/geeignet adj.-> check "Mittel"/"Maß" are the nouns
                        output = True
                else:
                    output = True
            if e_df['relevant_uebermass'][i]: # is relevant because of übermaß keyword
                output = True
    # decision tree is the perfect place for shade and that's just how I feel
    if return_df:
        return output, e_df
    else:
        return output
