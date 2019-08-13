from itertools import chain, islice
from collections import Counter, defaultdict

import numpy as np


def build_n_gram(words, n=2):
    """
    给定词语列表，得出相邻词语的组合
    args
    ----
    words: list, 词语序列, 通常由对句子分词得到
    n: 每个组合项的成员个数

    Returns
    -------
    groups: iterable, 所有组合的迭代器
    """
    iters = []
    for i in range(n):
        # a_iter = iter(words[i:])
        a_iter = islice(words, i, None)
        it = [a_iter] * n
        iters.append(zip(*it))
    return chain(*iters)


def count_neighbor_words(words):
    """统计每个词的左右邻词
    Args
    ----
    words: list, 词语序列, 通常由对句子分词得到

    Returns
    ------
    left_neighbor_wordcount: dict, key为word, value为该word的左邻词的wordcount (dict)
    right_neighbor_wordcount: dict, key为word, value为该word的右邻词的wordcount (dict)
    """
    left_neighbor_wordcount = defaultdict(lambda: defaultdict(int))
    right_neighbor_wordcount = defaultdict(lambda: defaultdict(int))
    word_pairs = build_n_gram(words, 2)
    for w in word_pairs:
        w_a, w_b = w[:1], w[1:]
        left_neighbor_wordcount[w_b][w_a] += 1
        right_neighbor_wordcount[w_a][w_b] += 1
    return left_neighbor_wordcount, right_neighbor_wordcount


def calculate_pmi(raw_word, wordcount):
    """计算点互信息
    Args
    ----
    raw_word : tuple, 候选词/词组
    wordcount : dict, 词频统计,除了有word的词频，对任意i(0 < i < len(word)-1),
                word[:i] 和 word[i:] 对瓷瓶也记录在wordcount里
    Returns
    -------
    pmi : float, word的点互信息
    """
    length = len(raw_word)
    if length == 1:
        return 0
    size = sum(wordcount.values())
    pmis = []
    for i in range(1, length):
        word_left, word_right = raw_word[:i], raw_word[i:]
        pmi = wordcount[raw_word] * size / (wordcount[word_left] * wordcount[word_right])
        pmis.append(pmi)
    return min(pmis)


def calculate_entropy(raw_word, neighbor_wordcount):
    """计算信息熵
    Args
    ----
    raw_word : tuple, 候选词/词组
    neighbors_wordcount : dict

    Returns
    -------
    """
    neighbors = neighbor_wordcount[raw_word]
    seq = np.array(list(neighbors.values()))
    p = seq / seq.sum()
    ent = -(p * np.log2(p)).sum()
    return ent


class WordFinder:
    """词语发现
    根据文本的统计规律，发现有较强统计规律的字符组合（即词语）
    用法:
    ```
    >> wf = WordFinder()
    >> wf.add_text(text1)
    >> wf.add_text(text2)
    >> words = wf.find()
    ```

    Attributes
    ----------
    n_gram: int, 
    text_processor: callale
    word_count: dict
    left_neighbor_wordcount: dict
    right_neighbor_wordcount: dict
    """
    def __init__(self, n_gram=2, text_processor=None):
        self.n_gram = n_gram
        if text_processor is None:
            text_processor = list
        self.text_processor = text_processor

        self.raw_wordcount = Counter()
        self.left_neighbor_wordcount = defaultdict(lambda: defaultdict(int))
        self.right_neighbor_wordcount = defaultdict(lambda: defaultdict(int))

    def update_neighbors_wordcount(self, words):
        word_pairs = build_n_gram(words, 2)
        for w in word_pairs:
            w_a, w_b = w[:1], w[1:]
            self.left_neighbor_wordcount[w_b][w_a] += 1
            self.right_neighbor_wordcount[w_a][w_b] += 1

    def update_wordcount(self, words):
        self.raw_wordcount.update([(w,) for w in words])
        for n in range(2, self.n_gram+1):
            self.raw_wordcount.update(build_n_gram(words, n))

    def add_text(self, text):
        words = self.text_processor(text)
        self.update_neighbors_wordcount(words)
        self.update_wordcount(words)

    def find(self, min_pmi=50, min_ent=3.5, min_freq=10):
        for w, freq in self.raw_wordcount.items():
            if len(w) < 2 or freq <= min_freq:
                continue
            pmi = calculate_pmi(w, self.raw_wordcount)
            left_part, right_part = w[:1], w[-1:]
            ent_left = calculate_entropy(left_part, self.left_neighbor_wordcount)
            ent_right = calculate_entropy(right_part, self.right_neighbor_wordcount)
            if min(ent_left, ent_right) > min_ent and pmi > min_pmi:
                print("ent:{ent:.2f}, pmi: {pmi:.2f}, freq: {freq}, w:{w}".format(ent=min(ent_left, ent_right),
                                                                      pmi=pmi,
                                                                      freq=freq,
                                                                      w="".join(w)))


if __name__ == "__main__":
    import jieba
    import json

    wf = WordFinder(text_processor=jieba.lcut)
    path = 'data/articles.json'
    texts = []
    with open(path) as f:
        data = json.load(f)
        for k, txt in data.items():
            texts.append(txt)
            wf.add_text(txt)
    wf.find()
