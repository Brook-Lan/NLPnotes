#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-26

@author:Brook
基于信息熵的新词发现算法

"""
import re
from collections import Counter


class WordCount:
    def __init__(self, max_length, min_freq=0):
        assert isinstance(max_length, int), 'max_length must be int'
        assert isinstance(min_freq, int), 'min_freq must be int'
        self.max_length = max_length
        self.min_freq = min_freq

    def preprocess(self, text):
        drop_chars = ['=','，', '\s', '。', '、', '：', '\(', '\)', '\[', '\]', '\.', ',', '”', '“', '？', '?', '！', '‘', '’', '…']
        p = '[' + "".join(drop_chars) + ']'
        return re.sub(p, '', text)

    def count_text(self, text):
        new_text = self.preprocess(text)
        count = Counter(new_text)
        for length in range(2, self.max_length+1):
            for i in range(length):
                count.update(re.findall(".{%s}" % length, new_text[i:]))
        return {k:v for k,v in count.items() if v > self.min_freq}


class NewWord:
    def __init__(self):
        self.wc = WordCount(4)







if __name__ == "__main__":
    s = "录取词语最低信息熵越大说明越有可能独立成词"
    wc = WordCount(4, 0)
    print(wc.count_text(s))
