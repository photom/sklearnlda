#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.tag import pos_tag
import nltk

PUBDATE = "pubDate"
CONTENT = "content"
NEWS_DIR = "news"
ticker = "GOOGL"

stemmer = LancasterStemmer()

# nltk.download('all')
english_stopwords = set(stopwords.words('english'))
english_stopwords.remove('up')
english_stopwords.remove('down')

alnum = string.lowercase + string.digits
words = set()

for picker_file in os.listdir(NEWS_DIR):
    if not picker_file.find(ticker) == 0:
        continue
    news_path = os.path.join(NEWS_DIR, picker_file)
    with open(news_path, mode='r') as f:
        article_map = pickle.load(f)
        for link, date_content in article_map.iteritems():
            text = date_content[CONTENT]
            tokens = wordpunct_tokenize(text)
            word_tags = pos_tag(tokens)
            for raw_word, tag in word_tags:
                if tag == 'NNP':
                    continue
                word = raw_word.lower()
                while len(word) > 0 and not word[0] in alnum:
                    word = word[1:]
                while len(word) > 0 and not word[-1] in alnum:
                    word = word[:-1]

                if len(word) > 1 and \
                    not word in english_stopwords and \
                    not word[0] in string.digits and \
                    all(ch in alnum for ch in word):
                    #stemmed = stemmer.stem(word)
                    normed = wordnet.morphy(word)
                    if normed is not None:
                        # words.add(",".join([normed, tag]))
                        words.add(normed)
                    #if stemmed is not None:
                    #    words.add(stemmed)
                    else:
                        # words.add(",".join([word, tag]))
                        words.add(word)


dictionary_path = "dictionary_{ticker}.csv".format(ticker=ticker.lower())
with open(dictionary_path, "wb") as f:
    content = u"\n".join(sorted(list(words)))
    f.write(content.encode('utf-8'))
