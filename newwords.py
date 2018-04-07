#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-26

@author:Brook
基于信息熵和互信息的新词发现算法
原理参考链接：http://www.matrix67.com/blog/archives/5044
"""
import re
from collections import Counter
import numpy as np


class Words:
    def __init__(self, text, max_length=4, min_support=30, min_ent=3, min_count=5):
        self.max_length = max_length
        self.min_support = min_support
        self.min_ent = min_ent
        self.min_count = min_count
        self.text = self.preprocess(text)
        self.wordcount = self._build_words()

    @staticmethod
    def preprocess(text):
        """文本去标点
        """
        drop_chars = ['=','，', '\s', '\d', '。', '、', '：', '\(', '\)', '\[', '\]', '\.', ',', '”', '“', '？', '?', '！', '‘', '’', '…']
        p = '[' + "".join(drop_chars) + ']'
        return re.sub(p, '', text)

    def _build_words(self):
        """生成候选词，并计数
        """
        print("正在生成候选词...")
        text = self.text
        wordcount = Counter(text)
        for length in range(2, self.max_length+1):
            for i in range(length):
                wordcount.update(re.findall(".{%s}" % length, text[i:]))
        return wordcount

    def find_neighbors(self, word):
        """查找左右邻词
        """
        left = Counter(re.findall("(.)%s" % word, self.text))
        right = Counter(re.findall("%s(.)" % word, self.text))
        return (left, right)

    def _cal_pmi(self, word):
        """计算凝固度, 点互信息PMI
        """
        word_len = len(word)
        if word_len == 1:
            return 0
        wordcount = self.wordcount
        size = sum(wordcount.values()) # 哪个size, 待确认
        #size = len(self.text)
        pmis = []
        for i in range(1, word_len):
            left_w, right_w = word[:i], word[i:]
            pmi = wordcount[word] * size /(wordcount[left_w] * wordcount[right_w])
            pmis.append(pmi)
        return min(pmis)

    def _cal_entropy(self, word):
        """计算信息熵
        """
        neighbors = self.find_neighbors(word)
        ents = []
        for neb in neighbors:
            seq = np.array(list(neb.values()))
            p = seq/seq.sum()
            ent = -(p *np.log2(p)).sum()
            ents.append(ent)
        return min(ents)

    def generate_words(self, filter_func=None):
        """这边遍历所有的候选词，实际应用时，可以把不存在词库里的候选词挑出来成不成词
        """
        print("正在生成新词...")
        if filter_func is None:
            filter_func = lambda x: len(x[0]) >= 2 and x[1] > self.min_count

        for word, count in filter(filter_func, self.wordcount.items()):
            pmi = self._cal_pmi(word)
            if pmi < self.min_support:
                continue
            ent = self._cal_entropy(word)
            if ent < self.min_ent:
                continue
            print("word: %s  count: %s  pmi: %.2f  ent: %.2f" % (word, count, pmi, ent))
            yield word


            

if __name__ == "__main__":
    path = 'data/陈二狗.txt'
    with open(path, encoding='gbk') as f:
        text = f.read()
        
   
    words = Words(text)
    for word in words.generate_words():
        pass
    
    
