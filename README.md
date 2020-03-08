# One-Shot-DocClassifier
Work In Progress - Bachelor Thesis Project - Document classification based in topic definitions

**Example using the library:**

Look at `Notebooks\Topic Classification using Topic Definitions - Baseline.ipynb`

**Main library file: "doc_utils.py":**

Contains the following utilities for the project:

* Parsing / Dataset building functions: 
  - `getWikiSummaries()`
  - `getWikiFullPage()`
  - `concurrentGetWikiFullPage()`
  - `getCatMembersList()`
  - `getCatMembersTexts()`
  - `getAllCatArticles()`
  - `concurrentGetAllCatArticles()`

* Data cleaning / vectorization / BoW
  - `cleanText()`
  - `vectSeq()`
  - `dataPreprocessing()`

* Data processing for classifier inputs
  - `processNeuralNetData()`
  - `processClassifierData()`
  
* Classification metrics / evaluation
  - `plotConfMatrix()`
