#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2018-03-30

@author:Brook
由bm25寻找相似的句子

"""
import re
import jieba

from bm25 import BM25


class Sim:
    def __init__(self, text):
        self.sentences = self.split_text(text)
        self.docs = self.sentences2docs(self.sentences)
        self.bm25 = BM25(self.docs)

    def split_text(self, text):
        """分割文本
        """
        sentences = []
        for line in re.split("[\r\n]", text):
            line = line.strip()
            if len(line) == 0:
                continue
            for sent in re.split( "[，,。:：？?！!；;]", line):
                sent = sent.strip()
                if len(sent) == 0:
                    continue
                sentences.append(sent)
        return sentences

    def split_sentence(self, sentence):
        """句子分词
        """
        doc = [w for w in jieba.cut(sentence) if len(w) > 1] 
        return doc

    def sentences2docs(self, sentences):
        """句子集转换成文档集
        """
        docs = []
        for sent in sentences:
            doc = self.split_sentence(sent)
            docs.append(doc)
        return docs

    def find_similar(self, sentence, k=5):
        """找相似的句子
        """
        doc = self.split_sentence(sentence)
        scores = self.bm25.simall(doc)
        scores = [(i, score) for i, score in enumerate(scores)]
        top = sorted(scores, key=lambda x:x[1], reverse=True)
        for i, score in top[:k]:
            sim_sent = self.sentences[i]
            if sentence != sim_sent:
                yield sim_sent 


if __name__ == "__main__":
    text = """
        国有企业是关系到国计民生的重要领域。我们党历来重视国有企业的发展与改革问题。中共十八大以来，随着反腐败力度的不断增强，随着针对国有企业的专项巡视的开展，国有企业腐败问题不断引起关注，其中尤其是国企高管的腐败问题成为较为突出的关键性问题。
        习近平总书记在十八届中央纪委五次全会上指出：“要着力完善国有企业监管制度，加强党对国有企业的领导，加强对国企领导班子的监督，搞好对国企的巡视，加大审计监督力度。要完善国有资产资源监管制度，强化对权力集中、资金密集、资源富集的部门和岗位的监管。”可见，解决好国企腐败问题，尤其是高层管理人员的腐败问题，对于继续深化国有企业改革具有重要意义。
        制度经济理论认为，社会制度可分为两个层面或类别，即制度逻辑和治理结构。研究发现，这两个层面或类别都可能影响社会的腐败水平。具体到国有企业，作为现代企业组织，它需要建立适应市场经济的治理结构以完成其经济性目标；与此同时，作为涉及到国计民生的重要领域，它则需要强调社会目标和政治目标。相应地，这两种属性实际体现在具体的制度逻辑和治理结构上。一方面，制度逻辑强调信仰体系和相关做法，即在国有企业内部所建立和维系的组织文化、价值信仰等软性机制；另一方面，治理结构则强调所有的组织安排或机制。在国有企业中，基于企业经营和管理的各种制度建构和权力运行机制都包含于其中。在国有企业内部，当制度逻辑中的价值信仰等软性机制出现问题时，就会出现个人腐化堕落等现象的发生；当治理结构所强调的组织安排或配套机制偏离规范时，那么就会导致权力滥用、一把手缺乏监督等腐败现象的频发。
        基于这样两个维度，来自于中国政法大学的李莉副教授和中国财富传媒集团中国财富研究院的云翀研究员根据公开信息整理出十八大以来厅局级以上170名高层官员腐败案例数据库（本案例库的数据搜集从2012年12月到2016年3月，数据来源为中纪委监察部网站以及新华网等官方媒体）。通过对于这一数据库的深入分析，该研究显示十八大以来，厅局级以上落马的国企高管人员的数量在逐年攀升。2013年，厅局级以上落马高管有15人（8.8%），2014年和2015年分别增长到73人（42.9%），2016年截止到2月份，也已经达到9人（5.3%）（见图1）。总体而言，自2013年以来，落马的厅局级以上国企高管人员共计170名，其中男性164人（96.5%），女性6人（3.5%）。可见，十八大以来短短三年左右的时间内国企落马的高层官员数量已经超过了十八大之前十年的总和，这充分显示出十八大以来国有企业反腐败决心之强、力度之大、成效之显著。除数量上的鲜明特点外，该研究进一步依据建立的一手数据库总结出了十八大以来国企落马官员的三个特征。
        """

    sim = Sim(text)

    sent = "国有企业腐败问题不断引起关注"
    print("\n==========原句==================\n")
    print(sent)
    print("\n===========相似的句子============\n")
    for s in sim.find_similar(sent):
        print(s)




