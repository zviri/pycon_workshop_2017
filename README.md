# Abstract
This Scikit-learn tutorial will introduce you to the basics of Python machine learning with text data in a step by step manner. You will learn how to transform raw text into data that is usable by machine learning models, build and evaluate your models on some real-world text data. In this tutorial we will answer the following questions:
* What is a vectorizer and which one should I use?
* What is the difference between methods “fit” and “transform”?
* What is a document-term matrix and what are its properties?
* What is the appropriate machine learning model to use?
* And much more...

Attendees of this tutorial should be comfortable with working in Python. Understanding of basic machine learning principles is welcomed, but not required. At least basic experience with Pandas and Scikit-learn is a plus.

Attendees who want to follow this tutorial need to bring a laptop with Scikit-learn, Pandas and IPython (Jupyter) notebook already installed. Installing the Anaconda distribution of Python is the easiest way to accomplish this. Both Python 2 and 3 are acceptable.

# Outline
* Introduction to supervised learning in scikit-learn
* Converting text to feature vectors
* Classifying creditors from the Czech Insolvency Register
  * Loading and preprocessing the dataset
  * Vectorizing the dataset
  * Building and evaluating the model
  * Examining the model
* Topics not covered

# Files
* [tutorial.ipynb](tutorial.ipynb) - IPython/Jupyter notebook for this tutorial.
* [data/receivables.tsv](data/receivables.tsv) - Real world dataset used in this tutorial.
* [data/stopwords_cz.txt](data/stopwords_cz.txt) - List of czech stopwords.

# Resources
### Scikit-learn
* [Working With Text Data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) tutorial from the scikit-learn community itself.
* [Text Feature Extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) more about converting text into features from scikit-learn.

### Classification tutorials
* [Sentiment Classification Using scikit-learn](https://www.youtube.com/watch?v=y3ZTKFZ-1QQ) video.
* [Tutorial from Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words).
* [Tutorial from fastml](http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/) as a follow-up to the one from Kaggle.
* Popular Machine Learning on [https://www.coursera.org/learn/machine-learning](Coursera).
