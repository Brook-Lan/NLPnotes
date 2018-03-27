#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-26

@author:Brook
基于信息熵和凝固度的新词发现算法
参考链接：http://www.matrix67.com/blog/archives/5044
"""
import re
from collections import Counter
import numpy as np


class WordCount:
    def __init__(self, max_length, text):
        assert isinstance(max_length, int), 'max_length must be int'
        self.max_length = max_length
        self._word_dict = {}
        self.text = self.preprocess(text)
        self.count()

    @property
    def word_dict(self):
        return dict(self._word_dict)
    
    def preprocess(self, text):
        drop_chars = ['=','，', '\s', '\d', '。', '、', '：', '\(', '\)', '\[', '\]', '\.', ',', '”', '“', '？', '?', '！', '‘', '’', '…']
        p = '[' + "".join(drop_chars) + ']'
        return re.sub(p, '', text)

    def count(self):
        """生成候选词，并计数
        """
        new_text = self.text
        self._word_dict = Counter(new_text)
        for length in range(2, self.max_length+1):
            for i in range(length):
                self._word_dict.update(re.findall(".{%s}" % length, new_text[i:]))

    def get_neighbors(self, word):
        """左右邻词
        """
        left = Counter(re.findall("(.)%s" % word, self.text))
        right = Counter(re.findall("%s(.)" % word, self.text))
        return (left, right)


class Words:
    def __init__(self, text, min_support=30, min_ent=3):
        self.words = WordCount(4, text)
        self.word_dict = self.words.word_dict
        self.min_support = min_support
        self.min_ent = min_ent
        self.min_count = 10
        
    def cal_freeze(self, word):
        """计算凝固度
        """
        word_len = len(word)
        if word_len == 1:
            return 0
        word_dict = self.word_dict
        size = sum(word_dict.values()) # 哪个size, 待确认
        frzs = []
        for i in range(1, word_len):
            left_w, right_w = word[:i], word[i:]
            frz = word_dict[word] * size /(word_dict[left_w] * word_dict[right_w])
            frzs.append(frz)
        return min(frzs)

    def cal_entropy(self, word):
        """计算信息熵
        """
        neighbors = self.words.get_neighbors(word)
        ents = []
        for neb in neighbors:
            seq = np.array(list(neb.values()))
            p = seq/seq.sum()
            ent = -(p *np.log2(p)).sum()
            ents.append(ent)
        return min(ents)
    
    def gen_word(self):
        for word, count in self.word_dict.items():
            if len(word) < 2 or count < self.min_count:
                continue
            frz = self.cal_freeze(word)
            if frz < self.min_support:
                continue
            ent = self.cal_entropy(word)
            if ent < self.min_ent:
                continue
            print("word: %s  count: %s  frz: %s  ent: %s" % (word, count, frz, ent))
            yield word
            

if __name__ == "__main__":
    path = 'data/陈二狗.txt'
    with open(path, encoding='gbk') as f:
        text = f.read()
        
   
    words = Words(text)
    for word in words.gen_word():
        pass
    
    
