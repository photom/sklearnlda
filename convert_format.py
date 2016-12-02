#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dateutil.parser
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
import urllib2
import re
from bs4 import BeautifulSoup
import pickle
import feedparser

PUBDATE = "pubDate"
CONTENT = "content"

with open('module1_cs1.2.abstracts.pickle', mode='r') as f:
  absmap = pickle.load(f)

files = os.listdir('news')
if not os.path.exists('news'):
  os.makedirs('news')
for file in files:
  with open('news/' + file, mode='r') as f:
    # load
    pair_map = pickle.load(f)
    new_map = {}
    for key, pair in pair_map.iteritems():
      new_map[key] = {PUBDATE: pair[0], CONTENT: pair[1]}
    # write
    path = "news/" + file
    with open(path, mode='wb') as write_file:
      pickle.dump(new_map, write_file, protocol=2)
