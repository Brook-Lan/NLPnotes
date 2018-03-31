#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-31

@author:Brook
"""
from collections import defaultdict
import math

class ProbStat:
    """
    概率统计
    """
    def __init__(self):
        """
        为了防止在后续的联合概率计算时，因为其中
        一个词的概率为0而造成乘积也为0，这里将所有
        词的出现数初始为1，总数初始为2
        """
        self.wordcount = defaultdict(lambda: 1)
        self._total = 2

    @property
    def total(self):
        return self._total + sum(self.wordcount.values())

    def get_freq(self, word):
        return self.wordcount[word]

    def get_prob(self, word):
        """获取概率
        """
        return self.get_freq(word)/self.total

    def feed_word(self, word, value=1):
        self.wordcount[word] += value

    def feed_words(self, words):
        for w in words:
            self.feed_word(w)


class Bayes:
    def __init__(self):
        self.total = 0
        self.label_ps = {}
        self.words = set()
    
    def train(self, data):
        """贝叶斯训练
        """
        for words, label in data:
            if label not in self.label_ps:
                self.label_ps[label] = ProbStat()
            self.label_ps[label].feed_words(words)
            self.words.update(words)
        self.total += sum([ps.total for ps in self.label_ps.values()])

    @property
    def prior_p(self):
        """计算先验概率
        """
        prior_p = {}
        for label, prob in self.label_ps.items():
            prior_p[label] = math.log(prob.total) - math.log(self.total)
        return prior_p

    def likelihood(self, words):
        """计算似然概率,为了防止联合概率过小而下溢出，采用log
        """
        words = [w for w in words if w in self.words]
        likelihood = {}
        for label, prob in self.label_ps.items():
            likelihood[label] = sum([math.log(prob.get_prob(w)) for w in words])
        return likelihood

    def classify(self, words):
        prior_p = self.prior_p
        likelihood = self.likelihood(words)
        last_p, last_label = "unknowon", "unknowon"
        for i, label in enumerate(self.label_ps):
            p = prior_p[label] + likelihood[label]
            if i == 0:
                last_label, last_p = label, p
            if p > last_p:
                last_p, last_label = p, label
        return last_label


def test():
    docs = [
        "my dog has flea problems help please",
        "maybe not take him to dog park stupid",
        "my dalmation is so cute I love hime",
        "stop posting stupid worthless garbage",
        "mr licks ate my steak how to stop him",
        "quit buying worthless dog food stupid"
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    docs = [s.lower().split() for s in docs]
    data = zip(docs, classVec)

    bayes = Bayes()
    bayes.train(data)

    s = "love my dalmation".split()
    print("classified as", bayes.classify(s))
    s = "stupid garbage".split()
    print("classified as", bayes.classify(s))
    
    
if __name__ == "__main__":
    test()
    







            




