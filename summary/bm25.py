#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-29

@author:Brook

bm25算法是用来评价搜索词和文档之间的相关性的算法



"""
from collections import Counter
import math


class BM25:
    def __init__(self, docs):
        self.docs = docs
        self.docs_len = len(docs)  # 文档句子数
        self.avgdl = sum([len(doc) for doc in docs]) / self.docs_len  # 文档句子平均长度
        self.f = []  # 词频
        self.df = Counter()  # 词的文档频率
        self.idf = {}  # 逆文档频率
        self.k1 = 1.5  # 调节因子
        self.b = 0.75  # 调节因子
        self.init()

    def init(self):
        for doc in self.docs:
            tf = Counter(doc)
            self.f.append(tf)
            self.df.update(tf.keys())

        for word, freq in self.df.items():
            self.idf[word] = math.log(self.docs_len - freq + 0.5) - math.log(freq + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            wf = self.f[index][word]
            score += (self.idf[word] * wf * (self.k1 + 1)
                    / (wf + self.k1 * (1 - self.b + self.b * d
                    / self.avgdl)))
        return score
        
    def simall(self, doc):
        scores = []
        for index in range(self.docs_len):
            score = self.sim(doc, index)
            scores.append(score)
        return scores
