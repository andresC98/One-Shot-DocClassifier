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
import wikipediaapi
import numpy as np
import nltk, gensim
import string, time

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import keras
from keras.utils import to_categorical

ALL_TOPICS = ["Chemical engineering",
              "Biomedical engineering",
              "Civil engineering", 
              "Electrical engineering", 
              "Mechanical engineering", 
              "Aerospace engineering", 
              "Financial engineering", 
              "Software engineering",
              "Industrial engineering", 
              "Materials engineering",
              "Computer engineering"]

WIKI = wikipediaapi.Wikipedia( language='en',
                            extract_format=wikipediaapi.ExtractFormat.WIKI)

    
def getWikiSummaries(target_article = None, topics = ALL_TOPICS, split_on_words = True):
    '''
    Downloads and parses all summary definitions of the <topics> list specified.
    If a target article is specified, also returns its corresponding summary.
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

def getWikiFullPage(target_article = None, topics = ALL_TOPICS, split_on_words = True):
    '''
    Downloads and parses the full page of definitions of the <topics> list specified.
    If a target article is specified, also returns its corresponding summary.
    '''
    full_pages = list()
    for i, topic in enumerate(topics):
        print("Obtaining full wikipedia page for the topic: {}. (Definition of Class #[{}])".format(topic,i))
        full_pages.append(wikipedia.page(topic))
    if(target_article):
        #Also return target article requested.
        print("\nObtaining wikipedia summary for target article:",target_article)
        target = wikipedia.summary(target_article)
        return target, full_pages
    else:
        return full_pages

    return 

def getCatMembersList(topic):
    '''
    Returns for a given topic a list of its category members title pages.
    '''
    category = WIKI.page("Category:"+ topic)

    cat_members_list = []
    for c in category.categorymembers.values():
        if "Category:" in c.title:
            break
        elif c.ns==0:
            cat_members_list.append(c.title)
    
    return cat_members_list

def getCatMembersTexts(cat_members_list, section = "Summary"):
    '''
    Retrieves either the summaries or the full wiki text of 
    all pages in a given category members list.
    '''
    c_members_texts = []

    for c_member in cat_members_list: 

        c_page = WIKI.page(c_member)
        if "all" in section:
            #Obtain full wikipedia text from page
            c_members_texts.append(c_page.text)
        else:
            #Obtain only Summary section of wiki article
            c_members_texts.append(c_page.summary)

    return c_members_texts

def getAllCatArticles(topics_list):
    '''
    Retrieves all articles from categories pages given a list of topics.
    Raw text Dataset structure: [ [topic_j_cat_pages], topic_j_label]

    Returns raw text dataset and the total number of articles retrieved.
    '''

    raw_dataset = list()
    total_num_articles = 0

    for topic_id, topic in enumerate(topics_list):
            
        cat_members_list = getCatMembersList(topic)
        
        page_summaries = getCatMembersTexts(cat_members_list)
        print("Retrieved {} articles from category topic '{}'[TopicID:{}]".format(len(page_summaries), topic, topic_id))
        total_num_articles += (len(page_summaries) - 1)

        raw_dataset.append( (page_summaries[1:], topic_id)) #first summary is the topic definition, needs to be exluded


    return raw_dataset, total_num_articles

def cleanText(text, full_page=False):
    '''
    Given a raw text input , tokenizes into words and performs stopword
    and punctuation removal operations; text thus loses structure and is grouped.
    If 'full_page' specified, takes into account cleaning full content.

    Returns cleaned version of text (list of cleaned words).
    '''
    nltk.download('stopwords',quiet=True)
    nltk.download('punkt',quiet=True) #obtaining punctuation library

    n_words = 0

    corpus = list()

    if full_page:
        for topic in text:
            corpus.append(word_tokenize(topic.content))
    else:
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
    Vectorizes a sequence of text data (supposed cleaned).

    Returns numpy vector version of sequence text data, ready 
    for Feedforward Neural Network input.
    '''

    results = np.zeros((len(sequences),max_dims))
    
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1
    
    return results

def data_preprocessing(train_data, test_data, debug = False):
    '''
    Given raw wikipedia content pages cleans training and testing sets.
    Creates doc2bow dictionary of full corpus.

    Returns dictionary, cleaned dataset and pairs.
    '''
    test_data_clean_pairs = list() #has labels too
    test_data_clean = list()

    for topic_cat in test_data:
        topic_id = topic_cat[1]
        cleaned_test_corpus = cleanText(topic_cat[0])
        if debug:
            print("Cleaning all articles from TopicID:", topic_id)
            print(cleaned_test_corpus)
        for article in cleaned_test_corpus:
            test_data_clean_pairs.append((article,topic_id))
            test_data_clean.append(article)

    #Clean topic defs (train data) and obtain dictionary of full corpus
    train_data_clean = cleanText(train_data, full_page=True)

    foo = train_data_clean.copy() #placeholder memory allocation
    for page in test_data_clean: #appending test data for dictionary creation
        foo.append(page)

    #Doc2Bow dictionary of full corpus
    dictionary = gensim.corpora.Dictionary(foo)

    if debug:
        print(dictionary.token2id)
        print("Total number of unique words in corpus:",len(dictionary))


    return dictionary, train_data_clean, test_data_clean, test_data_clean_pairs

def processNeuralNetData (train_data_clean, test_data_clean,test_data_clean_pairs , dictionary ,topics = ALL_TOPICS, debug = False):
    '''
    Given a set of testing data (articles to categorize) and
    train data (topic definitions), process -->cleaned<-- text until obtaining
    NeuralNet-ready encoded vectors. 

    Returns training and test vectors.
    '''

    #Data sequencing/encoding  
    train_model_input = list()
    test_model_input = list()

    for topic in train_data_clean:
        train_model_input.append(dictionary.doc2idx(topic))

    for test_page in test_data_clean:
        test_model_input.append(dictionary.doc2idx(test_page))
     
    train_model_input = np.array(train_model_input)
    test_model_input = np.array(test_model_input)

    x_train = vectSeq(train_model_input, max_dims=len(dictionary))
    x_test = vectSeq(test_model_input, max_dims=len(dictionary))

    #Generating labels (one hot encoding)
    train_labels = list()
    test_labels  = list()

    for i, topic in enumerate(ALL_TOPICS):
        train_labels.append(i)

    for test_page in test_data_clean_pairs:
        test_labels.append(test_page[1])
        
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels) 

    return x_train, y_train, x_test, y_test