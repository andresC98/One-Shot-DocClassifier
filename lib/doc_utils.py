#################################################################
# Utils library with useful functions developed for the Project #
#                                                               #
# Author: Andres Carrillo Lopez                                 #
# GitHub: AndresC98@github.com                                  #
#                                                               #
#  -> Dependencies:                                             #
#            - numpy                                            #
#            - nltk                                             #
#            - gensim                                           #
#            - keras                                            #
#            - string                                           #
#            - wikipedia                                        #                                       
#################################################################


import wikipedia
import numpy as np
import nltk, gensim
import string

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


ALL_TOPICS = ["Chemical engineering", "Biomedical engineering","Civil engineering", "Electrical engineering", "Mechanical engineering", "Aerospace engineering", "Financial engineering", "Software engineering" ,"Industrial engineering", "Materials engineering","Computer engineering"]


def wiki_parse(target_article = None, topics = all_topics, split_on_words = True):
    '''
    Downloads and parses all summary definitions of the <topics> list specified.
    If a target article is specified, also returns its corresponding summary.
    By defautl, 
    '''

    summaries = list()
    for i, topic in enumerate(topics):
        print("Obtaining wikipedia summary for the topic: {}. (Class #[{}])".format(topic,i))
        summaries.append( wikipedia.summary(topic))
    if(target_article):
        #Also return target article requested.
        print("\nObtaining wikipedia summary for target article:",target_article)
        target = wikipedia.summary(target_article)
        return target, summaries
    else:
        return summaries

def clean_text(text):
    '''
    Returns cleaned version of text.
    Note that text, initially divided in parrgrfs, loses structure and is grouped.
    '''
    nltk.download('stopwords',quiet=True)
    nltk.download('punkt',quiet=True) #obtaining punctuation library

    n_words = 0

    corpus = list()

    for topic in text:
        corpus.append(word_tokenize(topic))

    stop_words = set(stopwords.words('english'))
    punct_exclusions = set(string.punctuation)

    cleaned_corpus = list()

    for topic in corpus:
        cleaned_corpus_topic = list()
        for word in topic:
            if( (word not in stop_words) and word not in punct_exclusions):
                if '.' in word: #solving wiki bug
                    for w in word.split('.'):    
                        cleaned_corpus_topic.append(w)
                        n_words += 1
                else:
                    cleaned_corpus_topic.append(word)
                    n_words += 1
        cleaned_corpus.append(cleaned_corpus_topic)
    
    #print("Total number of words in corpus: ",n_words )
    return cleaned_corpus

def vect_seq(sequences, max_dims=10000):
    '''
    Source: "Deep Learning with Python - François Cholet"
    Returns vectorized version of sequence text data.
    '''

    results = np.zeros((len(sequences),max_dims))
    
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1
    
    return results

