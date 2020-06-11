import numpy as np
from gensim.models import doc2vec
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels

import doc_utils

class MaxSimClassifier(ClassifierMixin, BaseEstimator):
    """ 
    Implementation of Maximum similarity classifier.
    Parameters
    ----------
    dataset_type: str
        Either "wiki" or "arxiv"
    vector_size : int
        TODO
    min_count : int
        TODO
    epochs : int
        TODO
    Attributes
    ----------
    X_ : List of wikipedia pages defining topics
        The input passed during `fit`.
    y_ : Topic labels
        The labels passed during `fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    model: doc2vec base model from gensim
    """

    def __init__(self, dataset_type, preprocess = "simple" ,vector_size=50, min_count=2, epochs=50, dm=1, window=2):
        # doc2vec parameters
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.window = window
        
        #Classifier data parameters
        self.dataset_type = dataset_type
        self.preprocess = preprocess

        self.model = doc2vec.Doc2Vec(vector_size=self.vector_size,
                                     min_count=self.min_count,
                                     epochs=self.epochs,
                                     dm=self.dm,
                                     window=self.window)

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Adequating corpus for inference later
        X = list(doc_utils.prepare_corpus(X, train_data=True, preprocess = self.preprocess,dataset_type=self.dataset_type))

        self.model.build_vocab(X)
        self.model.train(X, total_examples=self.model.corpus_count, epochs=self.model.epochs)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """ 
        Prediction function for max. similarity classifier.
        Parameters
        ----------
        X : list 
            The input dataset: Arxiv dataset to classify
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Prediction output .
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        # Input validation 
        input_articles = list(doc_utils.prepare_corpus(X, train_data=False,preprocess = self.preprocess,dataset_type=self.dataset_type))

        outputs = list()
        for doc in input_articles:
            inferred_vector = self.model.infer_vector(doc)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
            most_similar_label = sims[0][0]  # index 0 === most similar
            outputs.append(most_similar_label)

        pred_labels = np.array(outputs)

        return pred_labels

    def score(self, X, y, eval="weighted"):
        """
        TODO:  Document
        """
        accuracy_list = list()
        outputs = list()

        check_is_fitted(self, ['X_', 'y_'])

        input_articles = list(doc_utils.prepare_corpus(X, train_data=False,preprocess=self.preprocess ,dataset_type=self.dataset_type))

        for i, doc in enumerate(input_articles):
            inferred_vector = self.model.infer_vector(doc)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
            most_similar_label = sims[0][0]  # index 0 === most similar
            outputs.append(most_similar_label)
            second_most_similar_label = sims[1][0]
            if most_similar_label == y[i]:
                accuracy_list.append(1)
            elif (second_most_similar_label == y[i] and "weighted" in eval):
                accuracy_list.append(0.5)
            elif (second_most_similar_label == y[i] and "top2" in eval):
                accuracy_list.append(1)
            else:
                accuracy_list.append(0)

        accuracy_list = np.array(accuracy_list)
        # print("Model {} accuracy over {} test documents: {}%.".format(eval, len(y), np.mean(accuracy_list) * 100))
        return np.mean(accuracy_list)

    def pseudo_label(self, x_train, dataset, paperslist = None ,result="extended" ,top_n=1, debug = False ):
        '''
        Given a set of topic definitions and documents (x_train, dataset),
        it performs inference by comparing docs against the topic definitions
        and obtain the most similar papers/articles predicted per topic.
        
        Currently only supports top1 (i.e best prediction) addition.

        Resulting new training data either consisting of:
            - Original definitions + best article/paper
            - best article/paper matching the topic
        '''
        if(self.dataset_type == "arxiv"):
            return self.arxiv_pseudo_label(x_train, dataset, paperslist ,result ,top_n, debug)
        else:  #wikipedia data
            return self.wiki_pseudo_label(x_train, dataset, result, top_n, debug )

    def arxiv_pseudo_label(self, x_train, dataset,paperslist ,result="extended" ,top_n=1, debug = False ):

            if paperslist is None:
                print("ERROR: paperlist from Arxivparser needed")
                return -1

            input_articles = list(doc_utils.prepare_corpus(dataset, train_data=False, 
                                                        preprocess=self.preprocess,dataset_type=self.dataset_type))

            doc_topics_sims = [ [],[],[],[],[],[],[],[]]

            for doc_id, doc in enumerate(input_articles):
                inferred_vector = self.model.infer_vector(doc)
                sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
                top_n_sims = sims[:top_n]

                for i in range(top_n):
                    topic_id = top_n_sims[i][0] #predicted topic id of paper
                    topic_sim = top_n_sims[i][1] #similarity
                    doc_topics_sims[topic_id].append((topic_sim, doc_id))

            best_papers_per_topic = [-1,-1,-1,-1,-1,-1,-1,-1]

            n_papers_per_topic = len(paperslist)//len(doc_utils.ARXIV_WIKI_TOPICS)

            for i,topic in enumerate(doc_topics_sims):
                paper_id = (max(topic, key = lambda i : i[0])[1])
                best_papers_per_topic[i] = paper_id #TODO: add topN papers and not just best one

                if debug:
                    true_label = paper_id//n_papers_per_topic
                    print("Topic {} ({}) best matching paper: id #{}".format(i,doc_utils.ARXIV_WIKI_TOPICS[i],paper_id))
                    print("\t--->True label:[",str(true_label), "](",doc_utils.ARXIV_WIKI_TOPICS[true_label] ,
                            ") \t\tPaper title:",paperslist[paper_id]['title'])

            x_train_ext = ["", "", "", "", "", "", "", ""]
            x_train_papers = ["", "", "", "", "", "", "", ""]

            for topic_id, train_sample in enumerate(x_train):
                best_paper_id = best_papers_per_topic[topic_id]
                #Creating extended train data
                if result in "extended":
                    x_train_ext[topic_id] = " . ".join([train_sample, paperslist[best_paper_id]["title"], paperslist[best_paper_id]["abstract"]])
                elif result in "bestpapers": #train samples are the best papers
                    x_train_papers[topic_id] = paperslist[best_paper_id]["title"] + " : " +paperslist[best_paper_id]["abstract"] 
                else:
                    print("[ERROR] Result argument can be only 'extended' or 'bestpapers'.")
                    return -1

            if result in "extended":
                new_x_train = x_train_ext
            else: #best papers
                new_x_train = x_train_papers

            return dataset, new_x_train

    def wiki_pseudo_label(self, x_train, dataset,result="extended" ,top_n=2, debug = False ):

        input_articles = list(doc_utils.prepare_corpus(dataset, train_data=False,preprocess=self.preprocess,dataset_type=self.dataset_type))

        doc_topics_sims = [ [],[],[],[],[],[],[],[],[]]
        
        article_list = list()
        for topic in dataset:
            for article in topic[0]:
                article_list.append(article)

        for doc_id, doc in enumerate(input_articles):
            inferred_vector = self.model.infer_vector(doc)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
            top_n_sims = sims[:top_n]

            for i in range(top_n):
                topic_id = top_n_sims[i][0] #predicted topic id of paper
                topic_sim = top_n_sims[i][1] #similarity
                doc_topics_sims[topic_id].append((topic_sim, doc_id))

        best_articles_per_topic = [-1,-1,-1,-1,-1,-1,-1,-1,-1]

        for i,topic in enumerate(doc_topics_sims):
            article_id = (max(topic, key = lambda i : i[0])[1])
            if debug: 
                print("For topic {}, best article id: {}".format(i, article_id))
            best_articles_per_topic[i] = article_id #TODO: add topN papers and not just best one

        x_train_ext = ["", "", "", "", "", "", "", "", ""]
        x_train_articles = ["", "", "", "", "", "", "", "", ""]

        for topic_id, train_sample in enumerate(x_train):
            best_article_id = best_articles_per_topic[topic_id]
            if debug:
                print("For topic id {}, Adding article id {} to dataset".format(topic_id,best_article_id))
                print(article_list[best_article_id])
            #Creating extended train data
            if result in "extended":    
                x_train_ext[topic_id] = " . ".join([article_list[best_article_id],train_sample])
            elif result in "bestpapers": #train samples are the best articles
                x_train_articles[topic_id] = article_list[best_article_id]
            else:
                print("[ERROR] Result argument can be only 'extended' or 'bestpapers'.")
                return -1

        if result in "extended":
            new_x_train = x_train_ext
        else: #best articles
            new_x_train = x_train_articles

        return dataset ,new_x_train
