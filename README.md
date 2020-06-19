# One-Shot-DocClassifier
*Bachelor Thesis Project - Document classification based in topic definitions*
***
This repository holds the contents developed for my bachelor thesis, titled: *"Clasificación de documentos basada en definiciones de categorías"*. Is structured in **2 main folders**:
  
  > ***lib***: Contains the python modules developed for the library. Logic and functions for text preprocessing, data extraction and models. Inside, the modules: *doc_utils.py*,  *arxiv_parser.py*,  *wiki_parser.py*,  *max_sim_classifier.py*.
  
  > ***notebooks***: Contains .ipynb notebooks with practical examples on using the library as well as some of the results obtained. For the best execution ease, run them inside *Google Colaboratory* and just upload the modules contained inside "lib" folder.
***

**How to visualize or run the sample notebooks from Colaboratory**

1. Download the notebooks from: `Notebooks\********.ipynb`

2. Go to [Google Colaboratory](https://colab.research.google.com)

3. Go to File > Open notebook > Upload

4. Select the desired notebook and launch it


**List of modules and contents**


**Text preprocessing and vectorization module: "doc_utils.py":**

Contains the following utilities and auxiliary functions for the project (documented inside code library):

* Data cleaning / vectorization / BoW
  - `prepare_corpus()`
  - `prepare_train_articles()`
  - `cleanText()`
  - `vectSeq()`

* Data processing for classifier inputs
  - `processNeuralNetData()`
  - `processClassifierData()`
  
* Classification metrics / evaluation
  - `top2acc()` 
  - `plotConfMatrix()`
  - `plotDefinitionsLength()`

**Building the datasets: Wikipedia and Arxiv crawlers**

* Wikipedia module: "wiki_parser.py":* 
  - `getWikiSummaries()`
  - `getWikiFullPage()`
  - `concurrentGetWikiFullPage()`
  - `getCatMembersList()`
  - `getCatMembersTexts()`
  - `getAllCatArticles()`
  - `concurrentGetAllCatArticles()`
  
 * ArXiv module: "arxiv_parser.py"
    - `init_parser()`
    - `arxiv_parser()`
 
**Maximum Similarity Classifier module: "max_sim_classifier.py"**
  - `MaxSimClassifier()`: Classifier class compatible with *sklearn* , fitting, inference and label propagation functions.
  With the following methods:
    - `fit()`
    - `fit_articles()`
    - `predict()`
    - `score()` 
    - `pseudo_label()`
    
 
