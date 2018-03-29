#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-29

@author:Brook
"""
import re
from collections import defaultdict

import jieba

from bm25 import BM25


class SentenceTextRank:
    def __init__(self, docs):
        self.d = 0.85
        self.max_iter = 200
        self.min_diff = 0.001
        self.docs = docs
        self.bm25 = BM25(docs)
        self.D = len(docs)
        self.weight = []
        self.weight_sum = []
        self.vertex = defaultdict(lambda: 1)
        self.top = {}
        self.solve()

    def solve(self):
        for i, sentence in enumerate(self.docs):
            scores = self.bm25.simall(sentence)
            self.weight.append(scores)
            self.weight_sum.append(sum(scores) - scores[i])

        for _ in range(self.max_iter):
            vertex_tmp = defaultdict(lambda: 1-self.d)
            max_diff = 0
            for i in range(self.D):
                for j in range(self.D):
                    if i==j or self.weight_sum[j]==0:
                        continue
                    vertex_tmp[i] += (self.d * self.weight[j][i] / self.weight_sum[j] * self.vertex[j])

                diff = abs(self.vertex[i] - vertex_tmp[i])
                max_diff = max(diff, max_diff)
            self.vertex = vertex_tmp
            if max_diff <= self.min_diff:
                break

    def get_top_index(self):
        top = sorted(self.vertex.items(), key=lambda x: x[1], reverse=True)
        index_top = [index for index, value in top]
        return index_top

    @staticmethod
    def split_sentence(text):
        sentences = []
        for line in re.split("[\r\n]", text):
            line = line.strip()
            if len(line) == 0:
                continue
            for sent in re.split( "[，,。:：“”？?！!；;]", line):
                sent = sent.strip()
                if len(sent) == 0:
                    continue
                sentences.append(sent)
        return sentences

    @staticmethod
    def convert_sents2docs(sentences):
        docs = []
        for sent in sentences:
            term = [w for w in jieba.cut(sent) if len(w) > 0] 
            docs.append(term)
        return docs

    @staticmethod
    def get_summary(text, n):
        sentences = SentenceTextRank.split_sentence(text)
        docs = SentenceTextRank.convert_sents2docs(sentences)
        textrank = SentenceTextRank(docs)
        index_top = textrank.get_top_index()
        sentences_top = [sentences[i] for i in index_top[:n]]
        summary = sorted(sentences_top, key=lambda x:sentences.index(x))
        return "。".join(summary)


        
if __name__ == "__main__":
    text = """原标题：今年继续承接各项重大工程建设。记者从日前举行的徐汇区2018年重大工程建设项目推进会获悉，以精品文化资源闻名沪上的徐汇区今年将继续承接各项重大工程建设，回应人民群众对高品质生活的需求，助力打响“上海文化”品牌。据介绍，今年徐汇区涉及的市级重大工程基本确定为15个，其中13个正式项目、2个预备项目。
    　　市民文化体育类“明星项目”持续推进。上海传统戏剧院团设施进一步提升，包括宛平剧场和越剧艺术演艺传习中心。徐家汇体育公园预计今年年底前完成体育馆游泳馆加固。上海音乐学院歌剧院继续完成续建。此外，轨道交通15号线站体施工全面开展，龙华机场消防站、长桥水厂提标、社会租赁住宅等三大新增项目也将陆续开启。
    　　据了解，今年徐汇区区级重大工程共将涉及建筑面积约485.9万平方米，道路及架空线整治27公里，河道治理约8公里，绿化近23.1公顷。"""

    summary = SentenceTextRank.get_summary(text, 5)

    print("==================原文=======================\n")
    print(text)
    print("\n")
    print("==================摘要======================\n")
    print(summary)







