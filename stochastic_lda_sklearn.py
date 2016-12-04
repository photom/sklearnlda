#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, time, string, random, csv, argparse
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.special import psi
from nltk.tokenize import wordpunct_tokenize
from utils import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
# import matplotlib.pyplot as plt
from nltk.corpus import wordnet
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
    parser.add_argument('-d', '--docs', help='file with list of docs, .txt', default="alldocs.txt", required=False)
    parser.add_argument('-s', '--stopwords', help='stopwords, defaults to stopwords.txt', default='stopwords.txt',
                        required=False)

    args = parser.parse_args()

    K = 10
    with open(args.stopwords) as f:
        stopwords = set(f.read().split())
    doc_files = str(args.docs)
    assert doc_files is not None, "no docs"
    docs = getalldocs(doc_files)
    D = len(docs)
    n_features = 1024
    n_top_words = 20

    analyze_word_partial = partial(analyze_word, stopwords=stopwords)
    tf_vectorizer = CountVectorizer(min_df=0.01, max_df=0.9, max_features=n_features,
                                    strip_accents='ascii',
                                    stop_words=stopwords, token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b",
                                    preprocessor=None, analyzer=analyze_word_partial)
    tf = tf_vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_topics=K, learning_method='online',
                                    learning_decay=0.9,
                                    learning_offset=1024,
                                    max_iter=500,
                                    total_samples=D,
                                    )
    transform_result = lda.fit_transform(tf)
    trained_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, trained_feature_names, n_top_words)
    print("mean")
    print(np.mean(transform_result, axis=0))
    print("std")
    print(np.std(transform_result, axis=0))
    print(transform_result)
    hist, bin_edges = np.histogram(transform_result[:, 0], bins=10)
    plt.bar(bin_edges[:-1], hist, width=1)
    # plt.show()
    testdoc = getfiles("hogehoge.txt")
    test_tf = tf_vectorizer.transform([testdoc])
    tf_feature_names = tf_vectorizer.get_feature_names()
    n_test_samples, n_test_features = test_tf.shape
    print("testfeturenum:" + str(n_test_features) +
          " tf_featre_names:" + str(len(tf_feature_names)))
    result = lda.transform(test_tf)
    print(result)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)


if __name__ == '__main__':
    main()
