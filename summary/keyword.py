#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-28

@author:Brook
"""
from collections import deque, defaultdict

import jieba


class KeyWordTextRank:
    def __init__(self):
        self.words = defaultdict(set)
        self.score = {}
        self.max_iter = 200
        self.d = 0.85
        self.min_diff = 0.001

    def build_neighbors(self, docs):
        """生成邻词
        """
        neighbors = defaultdict(set)
        que = deque(maxlen=5)
        for doc in docs:
            que.clear()
            for w in doc:
                que.append(w)
                for qw in que:
                    if qw == w:
                        continue
                    neighbors[qw].add(w)
                    neighbors[w].add(qw)
        return neighbors

    def get_rank(self, docs):
        """textrank(pagerank)算法
        """
        neighbors = self.build_neighbors(docs)
        score = defaultdict(lambda: 1)
        for _ in range(self.max_iter):
            max_diff = 0
            score_tmp = defaultdict(lambda: 1-self.d)
            for w, neighbor_words in neighbors.items():
                size = len(neighbor_words)
                for nbw in neighbor_words:
                    if w == nbw:
                        continue
                    score_tmp[nbw] += (self.d/size * score[w])
            for w in score_tmp:
                max_diff = max(abs(score_tmp[w] - score[w]), max_diff)
            score = score_tmp
            if max_diff <= self.min_diff:
                break
        return score

    @staticmethod
    def get_keywords(text, n=10):
        termlist = [[w for w in jieba.cut(text) if len(w) > 1]]

        textrank = KeyWordTextRank()
        score = textrank.get_rank(termlist)
        top = sorted(score.items(), key=lambda x: x[1], reverse=True)
        top = [w for w, value in top]
        return top[:n]


if __name__ == "__main__":
    import jieba
    text = "春回大地、万物复苏的美好时节，习近平总书记同首次访问中国的金正恩委员长就发展中朝两党两国关系、维护朝鲜半岛和平稳定进行坦诚友好会谈，并从战略高度提出四点重要主张。这次时机特殊、意义重大的历史性会晤，是中朝传统友好合作关系在新时代得以继承和发展的生动写照，是坚持通过对话协商解决半岛问题的中国方案带来的关键成效，必将有力推动中朝传统友谊在新的历史时期迈上新台阶，为朝鲜半岛局势的进一步转圜注入关键性暖流，对推动地区乃至世界和平稳定发展产生历史性影响。"
    top = KeyWordTextRank.get_keywords(text)
    print(top[:8])




