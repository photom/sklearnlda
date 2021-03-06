#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import string

PUBDATE = "pubDate"
CONTENT = "content"
DOCS_DIR = "docs"
NEWS_DIR = "news"

ticker_raws = ["GOOGL", "NVDA", "AMZN", "NFLX"]
for ticker in ticker_raws:
    alnum = string.lowercase + string.digits
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    for picker_file in os.listdir(NEWS_DIR):
        if not picker_file.find(ticker) == 0:
            continue
        news_path = os.path.join(NEWS_DIR, picker_file)
        with open(news_path, mode='r') as f:

            output_body = picker_file.replace(".pickle", "")
            article_map = pickle.load(f)
            for idx, link in enumerate(article_map):
                date_content = article_map[link]
                text = date_content[CONTENT]
                output_path = output_body + "_" + str(idx) + ".txt"
                filename = "{dir}/{output}".format(dir=DOCS_DIR, output=output_path)
                print (filename)
                with open(filename, "wb") as f:
                    f.write(text.encode('utf-8'))
