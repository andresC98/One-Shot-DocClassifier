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

import keras
from keras.utils import to_categorical

ALL_TOPICS = ["Chemical engineering", "Biomedical engineering","Civil engineering", "Electrical engineering", "Mechanical engineering", "Aerospace engineering", "Financial engineering", "Software engineering" ,"Industrial engineering", "Materials engineering","Computer engineering"]


def wikiParse(target_article = None, topics = ALL_TOPICS, split_on_words = True):
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


def cleanText(text):
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


def vectSeq(sequences, max_dims=10000):
    '''
    Source: "Deep Learning with Python - François Cholet"
    Returns vectorized version of sequence text data.
    '''

    results = np.zeros((len(sequences),max_dims))
    
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1
    
    return results


def prepareNeuralNetData(target_article_name, topic_definitions = ALL_TOPICS):
    '''
    Given a target article name from wikipedia, and a list of topics,
    retrieves from wikipedia their definitions, preprocess a training dataset
    suitable for Keras neural net input.

    Returns data for neural network training and testing 
    '''
    target_article, summaries = wikiParse(target_article_name,topics = topic_definitions)

    cleaned_corpus = cleanText(summaries)
    cleaned_target = cleanText([target_article])

    foo = summaries.copy() #placeholder memory allocation
    foo.append(target_article)

    cleaned_total_corpus = cleanText(foo) #for building dictionary
    
    #Doc2Bow dictionary of full corpus
    dictionary = gensim.corpora.Dictionary(cleaned_total_corpus)

    #Preparing test text data: 
    test_model_input = list()
    test_model_input.append(dictionary.doc2idx(cleaned_target[0]))
    test_model_input = np.array(test_model_input)
    
    #Preparing train text data:
    model_input = list()
    for topic in cleaned_corpus:
        model_input.append(dictionary.doc2idx(topic))
    model_input = np.array(model_input)

    #Data sequencing  
    x_train = vectSeq(model_input)
    x_test = vectSeq(test_model_input)

    #Generating labels (one hot encoding)
    cat_topics = list()
    for i, topic in enumerate(topic_definitions):
        cat_topics.append(i)

    y_train = to_categorical(cat_topics)

    return x_train, y_train, x_test
