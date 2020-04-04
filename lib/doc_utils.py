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

# Wiki API libraries
import wikipedia
import wikipediaapi

# Data processing
import numpy as np
import nltk, gensim
import string, time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import download
download('punkt')
download('wordnet')
download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER =WordNetLemmatizer()

# Model evaluation and Visualization
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# NN Preprocessing
import keras
from keras.utils import to_categorical

# Multithreading
import threading

# Removed topics (unavailable in wiki)
# "Materials engineering",
# "Financial engineering", 

ALL_TOPICS = ["Chemical engineering",
              "Biomedical engineering",
              "Civil engineering",
              "Electrical engineering",
              "Mechanical engineering",
              "Aerospace engineering",
              "Software engineering",
              "Industrial engineering",
              "Computer engineering"]

# "Mat",
# "Fin", 

ENG_TOPICS_ABVR = ["Chem",
                   "Biomd",
                   "Civil",
                   "Elec",
                   "Mech",
                   "Aero",
                   "SW",
                   "Ind",
                   "Comp"]

WIKI = wikipediaapi.Wikipedia(language='en',
                              extract_format=wikipediaapi.ExtractFormat.WIKI)


def getWikiSummaries(target_article=None, topics_list=ALL_TOPICS, split_on_words=True):
    '''
    Downloads and parses all summary definitions of the <topics_list> list specified.
    If a target article is specified, also returns its corresponding summary.
    '''

    summaries = list()
    for i, topic in enumerate(topics_list):
        print("Obtaining wikipedia summary for the topic: {}. (Class #[{}])".format(topic, i))
        summaries.append(wikipedia.summary(topic))
    if (target_article):
        # Also return target article requested.
        print("\nObtaining wikipedia summary for target article:", target_article)
        target = wikipedia.summary(target_article)
        return target, summaries
    else:
        return summaries


def getWikiFullPage(target_article=None, topics_list=ALL_TOPICS, split_on_words=True):
    '''
    Downloads and parses the full page of definitions of the <topics_list> list specified.
    If a target article is specified, also returns its corresponding summary.
    '''
    full_pages = list()
    for i, topic in enumerate(topics_list):
        print("Obtaining full wikipedia page for the topic: {}. (Definition of Class #[{}])".format(topic, i))
        full_pages.append(wikipedia.page(topic))
    if (target_article):
        # Also return target article requested.
        print("\nObtaining wikipedia summary for target article:", target_article)
        target = wikipedia.summary(target_article)
        return target, full_pages
    else:
        return full_pages

    return


def concurrentGetWikiFullPage(target_article=None, topics_list=ALL_TOPICS, split_on_words=True):
    '''
    MULTITHREADING VERSION

    Downloads and parses the full page of definitions of the <topics> list specified.
    If a target article is specified, also returns its corresponding summary.
    '''
    global lock
    global raw_dataset

    lock = threading.Lock()
    full_pages = ["" for elem in topics_list]

    def getWikiDefinitionPage(topic_id, topic):
        """wrapper function to start the job in the child process"""
        print("Obtaining full wikipedia page for the topic: {}. (Definition of Class #[{}])".format(topic, topic_id))
        lock.acquire()
        full_pages[topic_id] = wikipedia.page(topic)
        lock.release()

    thread_list = []

    for topic_id, topic in enumerate(topics_list):
        thread = threading.Thread(target=getWikiDefinitionPage, args=(topic_id, topic,))
        thread.daemon = True  # so that closes when disc
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    return full_pages


def getCatMembersList(topic):
    '''
    Returns for a given topic a list of its category members title pages.
    '''
    category = WIKI.page("Category:" + topic)

    cat_members_list = []
    for c in category.categorymembers.values():
        if "Category:" in c.title:
            break
        elif c.ns == 0:
            cat_members_list.append(c.title)

    return cat_members_list


def getCatMembersTexts(cat_members_list, section="Summary"):
    '''
    Retrieves either the summaries or the full wiki text of 
    all pages in a given category members list.
    '''
    c_members_texts = []

    for c_member in cat_members_list:

        c_page = WIKI.page(c_member)
        if "all" in section:
            # Obtain full wikipedia text from page
            c_members_texts.append(c_page.text)
        else:
            # Obtain only Summary section of wiki article
            c_members_texts.append(c_page.summary)

    return c_members_texts


def getAllCatArticles(topics_list, full_text_test=False):
    '''
    Retrieves all articles from categories pages given a list of topics.
    Raw text Dataset structure: [ [topic_j_cat_pages], topic_j_label]

    Returns raw text dataset and the total number of articles retrieved.
    '''

    raw_dataset = list()
    total_num_articles = 0

    for topic_id, topic in enumerate(topics_list):

        cat_members_list = getCatMembersList(topic)

        if full_text_test:
            test_pages = getCatMembersTexts(cat_members_list, section="all")
        else:
            test_pages = getCatMembersTexts(cat_members_list)

        print("Retrieved {} articles from category topic '{}'[TopicID:{}]".format(len(test_pages), topic, topic_id))
        total_num_articles += (len(test_pages) - 1)

        raw_dataset.append((test_pages[1:], topic_id))  # first summary is the topic definition, needs to be exluded

    return raw_dataset, total_num_articles


def concurrentGetAllCatArticles(topics_list, full_text_test=True):
    '''
    MULTITHREADED VERSION. Faster, but may contain bugs.

    Retrieves all articles from categories pages given a list of topics.
    Raw text Dataset structure: [ [topic_j_cat_pages], topic_j_label]

    Returns raw text dataset and the total number of articles retrieved.
    '''
    global lock
    global raw_dataset

    lock = threading.Lock()

    total_num_articles = 0
    raw_dataset = ["" for elem in topics_list]

    def getCategoryArticles(topic_id, topic):
        """wrapper function to start the job in the child process"""
        cat_members_list = getCatMembersList(topic)

        if full_text_test:
            test_pages = getCatMembersTexts(cat_members_list, section="all")
        else:
            test_pages = getCatMembersTexts(cat_members_list)

        if (len(test_pages) == 0):
            print("Could not retrieve articles from category topic:'{}'[TopicID:{}]\n".format(topic, topic_id))
        else:
            print("Retrieved {} articles from category topic '{}'[TopicID:{}]".format(len(test_pages) - 1, topic,
                                                                                      topic_id))
            lock.acquire()
            raw_dataset[topic_id] = (
            test_pages[1:], topic_id)  # first summary is the topic definition, needs to be exluded
            lock.release()

    thread_list = []

    for topic_id, topic in enumerate(topics_list):
        thread = threading.Thread(target=getCategoryArticles, args=(topic_id, topic,))
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    for topic in raw_dataset:
        if len(topic) != 0:
            total_num_articles += len(topic[0])

    return raw_dataset, total_num_articles


def cleanText(text, full_page=False, topic_defs=True):
    '''
    Given a raw text input , tokenizes into words and performs stopword
    and punctuation removal operations; text thus loses structure and is grouped.
    If 'full_page' specified, takes into account cleaning full content.

    Returns cleaned version of text (list of cleaned words).
    '''
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)  # obtaining punctuation library

    n_words = 0

    corpus = list()

    if topic_defs:  # processing topic definitions
        if full_page:
            for topic in text:
                corpus.append(word_tokenize(topic.content))
        else:
            for topic in text:
                corpus.append(word_tokenize(topic))

    if not topic_defs:  # processing test data
        for topic in text:
            corpus.append(word_tokenize(topic))

    stop_words = set(stopwords.words('english'))
    punct_exclusions = set(string.punctuation)

    cleaned_corpus = list()

    for topic in corpus:
        cleaned_corpus_topic = list()
        for word in topic:
            if ((word not in stop_words) and word not in punct_exclusions):
                if '.' in word:  # solving wiki bug
                    for w in word.split('.'):
                        cleaned_corpus_topic.append(w)
                        n_words += 1
                else:
                    cleaned_corpus_topic.append(word)
                    n_words += 1
        cleaned_corpus.append(cleaned_corpus_topic)

    # print("Total number of words in corpus: ",n_words )
    return cleaned_corpus


def vectSeq(sequences, max_dims=10000):
    '''
    Source: "Deep Learning with Python - François Cholet"
    Vectorizes a sequence of text data (supposed cleaned).

    Returns numpy vector version of sequence text data, ready 
    for Feedforward Neural Network input.
    '''

    results = np.zeros((len(sequences), max_dims))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results


def dataPreprocessing(train_data, test_data, full_page=False, debug=False):
    '''
    Given raw wikipedia content pages cleans training and testing sets.
    Creates doc2bow dictionary of full corpus.

    Returns dictionary, cleaned dataset and pairs.
    '''
    test_data_clean_pairs = list()  # has labels too
    test_data_clean = list()

    for topic_cat in test_data:
        if not topic_cat:
            # for empty (not found) topics:
            continue
        topic_id = topic_cat[1]

        cleaned_test_corpus = cleanText(topic_cat[0], full_page, topic_defs=False)

        if debug:
            print("Cleaning all articles from TopicID:", topic_id)
            print(cleaned_test_corpus)
        for article in cleaned_test_corpus:
            test_data_clean_pairs.append((article, topic_id))
            test_data_clean.append(article)

    # Clean topic defs (train data) and obtain dictionary of full corpus
    train_data_clean = cleanText(train_data, full_page=True)

    foo = train_data_clean.copy()  # placeholder memory allocation
    for page in test_data_clean:  # appending test data for dictionary creation
        foo.append(page)

    # Doc2Bow dictionary of full corpus
    dictionary = gensim.corpora.Dictionary(foo)

    if debug:
        print(dictionary.token2id)
        print("Total number of unique words in corpus:", len(dictionary))

    return dictionary, train_data_clean, test_data_clean, test_data_clean_pairs


def processNeuralNetData(train_data_clean, test_data_clean, test_data_clean_pairs, dictionary, topics=ALL_TOPICS,
                         debug=False):
    '''
    Given a set of testing data (articles to categorize) and
    train data (topic definitions), process -->cleaned<-- text until obtaining
    NeuralNet-ready encoded vectors. 

    Returns training and test vectors.
    '''

    # Data sequencing/encoding
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

    # Generating labels (one hot encoding)
    train_labels = list()
    test_labels = list()

    for i, topic in enumerate(ALL_TOPICS):
        train_labels.append(i)

    for test_page in test_data_clean_pairs:
        test_labels.append(test_page[1])

    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    return x_train, y_train, x_test, y_test


def processClassifierData(train_raw_data, test_raw_data, topics=ALL_TOPICS):
    """
    Simple data conversion for Sklearn classifiers input.
    """
    x_train = []
    # Note: this supposes topic definition is full page
    for wikipage in train_raw_data:
        x_train.append(wikipage.content)

    y_train = [i for i in range(len(topics))]

    y_test = []
    x_test = []

    for article_class in test_raw_data:
        for article in article_class[0]:
            x_test.append(article)
            y_test.append(article_class[1])

    return x_train, y_train, x_test, y_test


def plotConfMatrix(y_test, predictions, model):
    '''
    Given a one-hot encoded test labels and predictions [class labels]
    computes and plots confusion matrix of model classification result.
    '''

    if model in "NN":  # onehot encoded output of NN
        conf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions)
    else:
        conf_matrix = confusion_matrix(y_test, predictions)

    df_cm = pd.DataFrame(conf_matrix, index=[top for top in ENG_TOPICS_ABVR],
                         columns=[top for top in ENG_TOPICS_ABVR])

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion matrix of topic classification")
    plt.show()

    return

def custom_preprocess(doc):
    '''
    TODO: Document
    '''
    tokenized_doc=word_tokenize(doc)
    lemmatized_doc = [LEMMATIZER.lemmatize(word) for word in tokenized_doc]
    #tokens= [word for word in lemmatized_doc if word.isalnum()]
    tokens= [word.lower() for word in lemmatized_doc if word.isalnum() and not word in STOP_WORDS]

    return tokens

def prepare_corpus(raw_text, train_data=True, preprocess='simple'):
    '''
    Given a raw array of texts (either test data or training topics),
    performs text preprocessing and outputs processed text.
    '''
    if not train_data:  # data is a list of tuples (2nd element being the class)
        for i, topic in enumerate(raw_text):
            for raw_article in topic[0]:
                if preprocess in 'simple':
                    tokens = gensim.utils.simple_preprocess(raw_article)
                else:
                    tokens = custom_preprocess(raw_article)
                yield tokens
    else:
        for i, raw_topic_def in enumerate(raw_text):
            if preprocess in 'simple':
                tokens = gensim.utils.simple_preprocess(raw_topic_def)
            else:
                tokens = custom_preprocess(raw_topic_def)
            # we also add topic class id for training data
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def evaluate_model(model, test_corpus, test_labels, eval="binary"):
    '''
    Given a doc2vec trained model grom GENSIM and a test labeled corpus,
    performs similarity queries of test corpus vs topic definitions.
    
    Returns predictions array and accuracy list.
    '''
    accuracy_list = list()
    predictions = list()
    for doc_id, doc in enumerate(test_corpus):

        inferred_vector = model.infer_vector(doc)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        most_similar_label = sims[0][0]  # index 0 === most similar
        predictions.append(most_similar_label)
        second_most_similar_label = sims[1][0]
        if most_similar_label == test_labels[doc_id]:
            accuracy_list.append(1)
        elif (second_most_similar_label == test_labels[doc_id] and "weighted" in eval):
            accuracy_list.append(0.5)
        else:
            accuracy_list.append(0)

    accuracy_list = np.array(accuracy_list)
    print("Model {} accuracy over {} test documents: {}%.".format(eval, len(test_labels), np.mean(accuracy_list) * 100))

    return predictions, accuracy_list
