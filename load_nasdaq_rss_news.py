#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import pickle
import feedparser
import dateutil.parser

from load_news import *

NASDAQ_RSS_URL_TMP = "https://www.google.com/finance/company_news?q={ticker}&output=rss"
PATH_TMP = "news/{ticker}_{date}.pickle"
PUBDATE = "pubDate"
CONTENT = "content"


def main():
    ticker_raws = ["GOOGL", "NVDA", "AMZN", "NFLX"]
    for ticker_raw in ticker_raws:
        ticker = "NASDAQ:{raw}".format(raw=ticker_raw)
        response = feedparser.parse(NASDAQ_RSS_URL_TMP.format(ticker=ticker))

        link_dates = [(entry.link, entry.published) for entry in response.entries \
                      if entry.link.find("https://www.google.com") < 0]

        article_map = {}

        for link_date in link_dates:
            link = link_date[0]
            date_str = link_date[1]
            date = dateutil.parser.parse(date_str)
            date_key = date.strftime("%Y%m%d")
            if link.startswith("https"):
                link = link.replace("https", "http")
            for url_header in KNOWN_PAGES.keys():
                if link.find(url_header) >= 0:
                    text = KNOWN_PAGES[url_header](link)
                    if not article_map.has_key(date_key):
                        article_map[date_key] = {}
                    article_map[date_key][link] = {PUBDATE: date, CONTENT: text}
                    break
            else:
                if not any([link.find(url) >= 0 for url in SKIP_PAGES]):
                    print("unknown link. link: " + link)

        for date_key, new_map in article_map.iteritems():
            path = PATH_TMP.format(ticker=ticker_raw, date=date_key)
            if os.path.exists(path):
                with open(path, mode='r') as f:
                    old_article_map = pickle.load(f)
            else:
                old_article_map = {}
            old_article_map.update(new_map)
            with open(path, mode='wb') as f:
                pickle.dump(old_article_map, f, protocol=2)


if __name__ == '__main__':
    main()
