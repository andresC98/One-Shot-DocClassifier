import numpy as np
from gensim.models import doc2vec
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels

from doc_utils import prepare_corpus

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
    def __init__(self, dataset_type, vector_size=50, min_count=2, epochs= 50):
        #TODO: Add more doc2vec model parameters 
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.dataset_type = dataset_type

        self.model = doc2vec.Doc2Vec(vector_size=self.vector_size, 
                                min_count=self.min_count, 
                                epochs=self.epochs)


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
        #Adequating corpus for inference later
        X = list(prepare_corpus(X,train_data=True, dataset_type = self.dataset_type))

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
        input_articles = list(prepare_corpus(X, train_data=False, dataset_type=self.dataset_type))

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
        
        input_articles = list(prepare_corpus(X, train_data=False, dataset_type = self.dataset_type))

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
        #print("Model {} accuracy over {} test documents: {}%.".format(eval, len(y), np.mean(accuracy_list) * 100))
        return np.mean(accuracy_list)