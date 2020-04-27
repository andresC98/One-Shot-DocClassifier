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
#################################################################

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
LEMMATIZER = WordNetLemmatizer()

# Model evaluation and Visualization
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# NN Preprocessing
from keras.utils import to_categorical

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

# For Arxiv parser (Topics)
ARXIV_SUBJECTS = ["computer_science",
                  "economics",
                  "eess",
                  "mathematics",
                  "physics",
                  "q_biology",
                  "q_finance",
                  "statistics"]


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


def processNeuralNetData(train_data, test_data, full_page=False, topics=ALL_TOPICS, debug=False):
    '''
    Given raw wikipedia content pages  (topics and articles) cleans training and testing sets.
    Creates doc2bow dictionary of full corpus, and sequences input data into suitable form for NeuralNet Classifier.

    Returns training and test vectors.
    TODO: Adapt for ArXiv dataset.
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


def processClassifierData(train_raw_data, test_raw_data, topics, dataset_type="wiki"):
    """
    Given raw wikipedia pages (topic defs) as train data, and raw string articles as test data,
    Generates (unprocessed text) train / test pairs suitable for Sklearn-compatible Classifiers.
    """
    x_train = []
    y_test = []
    x_test = []

    # Note: this supposes topic definition is full page
    if dataset_type in "wiki":
        for wikipage in train_raw_data:
            x_train.append(wikipage.content)

        y_train = [i for i in range(len(topics))]

        for article_class in test_raw_data:
            for article in article_class[0]:
                x_test.append(article)
                y_test.append(article_class[1])
    else:  # arxiv dataset
        for wikipage in train_raw_data:  # also gets topics defs form wiki
            x_train.append(wikipage.content)

        y_train = [i for i in range(len(topics))]
        for subject in test_raw_data:
            for paper in subject["papers"]:
                x_test.append(paper["title"] + " : " + paper["abstract"])
                y_test.append(subject["label"])

    return x_train, y_train, x_test, y_test


def plotConfMatrix(y_test, predictions, model, dataset_type="wiki"):
    '''
    Given a one-hot encoded test labels and predictions [class labels]
    computes and plots confusion matrix of model classification result.
    '''

    if model in "NN":  # onehot encoded output of NN
        conf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions)
    else:
        conf_matrix = confusion_matrix(y_test, predictions)

    if dataset_type in "wiki":
        df_cm = pd.DataFrame(conf_matrix, index=[top for top in ENG_TOPICS_ABVR],
                             columns=[top for top in ENG_TOPICS_ABVR])
    else:  # arxiv
        df_cm = pd.DataFrame(conf_matrix, index=[top for top in ARXIV_SUBJECTS],
                             columns=[top for top in ARXIV_SUBJECTS])

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion matrix of topic classification")
    plt.show()

    return


def custom_preprocess(doc):
    '''
    TODO: Document
    '''
    tokenized_doc = word_tokenize(doc)
    lemmatized_doc = [LEMMATIZER.lemmatize(word) for word in tokenized_doc]
    # tokens= [word for word in lemmatized_doc if word.isalnum()]
    tokens = [word.lower() for word in lemmatized_doc if word.isalnum() and not word in STOP_WORDS]

    return tokens


def prepare_corpus(raw_text, train_data=True, preprocess='simple', dataset_type="wiki"):
    '''
    Given a raw array of texts (either test data or training topics),
    performs text preprocessing and outputs processed text.
    '''
    if not train_data:
        if dataset_type in "wiki":  # data is a list of tuples (2nd element being the class)
            for i, topic in enumerate(raw_text):
                for raw_article in topic[0]:
                    if preprocess in 'simple':
                        tokens = gensim.utils.simple_preprocess(raw_article)
                    else:
                        tokens = custom_preprocess(raw_article)
                    yield tokens
        else:  # arxiv
            for subject in raw_text:
                for paper in subject["papers"]:
                    if preprocess in 'simple':
                        tokens = gensim.utils.simple_preprocess(paper["title"] + " : " + paper["abstract"])
                    else:
                        tokens = custom_preprocess(paper["title"] + " : " + paper["abstract"])
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
        elif (second_most_similar_label == test_labels[doc_id] and "top2" in eval):
            accuracy_list.append(1)
        else:
            accuracy_list.append(0)

    accuracy_list = np.array(accuracy_list)
    print("Model {} accuracy over {} test documents: {}%.".format(eval, len(test_labels), np.mean(accuracy_list) * 100))

    return predictions, accuracy_list
