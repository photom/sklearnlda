#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, time, string, random, csv, argparse
import scipy
import numpy as np
from scipy.special import psi
from nltk.tokenize import wordpunct_tokenize
from utils import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
from nltk.corpus import wordnet
# from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from functools import partial
np.set_printoptions(threshold=np.inf)

n.random.seed(10000001)
meanchangethresh = 1e-3
MAXITER = 10000

token_pattern = r"(?u)\b[^\W\d_][^\W\d_]+\b"
lemmatizer = WordNetLemmatizer()
analyzer = CountVectorizer().build_analyzer()


def analyze_word(doc, stopwords):
    analyzer(doc)
    lemmatized = [lemmatizer.lemmatize(w) for w in analyzer(doc)]
    return [w for w in lemmatized if re.match(token_pattern, w) and w not in stopwords]
    # return (lemmatizer.lemmatize(w) if re.match(token_pattern, w) and w not in stopwords else "" for w in analyzer(doc))


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-K', '--topics', help='number of topics, defaults to 10', required=True)
    #parser.add_argument('-m', '--mode', help='mode, test | normal | stock', required=True)
    #parser.add_argument('-v', '--vocab', help='Vocab file name, .csv', default="dictionary.csv", required=False)
    parser.add_argument('-d', '--docs', help='file with list of docs, .txt', default="alldocs.txt", required=False)
    #parser.add_argument('-a', '--alpha', help='alpha parameter, defaults to 0.2', default=0.8, required=False)
    #parser.add_argument('-e', '--eta', help='eta parameter, defaults to 0.2', default=0.2, required=False)
    #parser.add_argument('-t', '--tau', help='tau parameter, defaults to 1024', default=1024, required=False)
    #parser.add_argument('-k', '--kappa', help='kappa parameter, defaults to 0.7', default=0.7, required=False)
    #parser.add_argument('-n', '--iterations', help='number of iterations, defaults to 10000', default=10000, required=False)
    parser.add_argument('-s', '--stopwords', help='stopwords, defaults to stopwords.txt', default='stopwords.txt',
                        required=False)

    args = parser.parse_args()

    K = 10
    #mode = str(args.mode)
    #K = int(args.topics)
    #alpha = float(args.alpha)
    #eta = float(args.eta)
    #tau = float(args.tau)
    #kappa = float(args.kappa)
    #iterations = int(args.iterations)
    with open(args.stopwords) as f:
        stopwords = set(f.read().split())
    doc_files = str(args.docs)
    assert doc_files is not None, "no docs"
    docs = getalldocs(doc_files)
    D = len(docs)
    n_features = 1000
    n_top_words = 20

    analyze_word_partial = partial(analyze_word, stopwords=stopwords)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=0.01, max_features=n_features,
                                    strip_accents='ascii',
                                    stop_words=stopwords, token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b",
                                    preprocessor=None, analyzer=analyze_word_partial)

    tf = tf_vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_topics=K, learning_method='online',
                                    learning_decay=0.7,
                                    learning_offset=1024,
                                    max_iter=200,
                                    total_samples=D,
                                    random_state=1234,
                                    )
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    # print("names: " + str(tf_feature_names))
    print_top_words(lda, tf_feature_names, n_top_words)

    testdocs = getalldocs("testdocs_googl.txt")
    test_tf = tf_vectorizer.fit_transform(testdocs)
    ret = lda.transform(test_tf)
    print(len(ret))
    sys.stdout.write(str(ret))

if __name__ == '__main__':
    main()
