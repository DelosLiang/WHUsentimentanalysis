#!/usr/bin/python                                                                                                                                                                                                                    
#coding=utf-8

from bayes import Bayes
import jieba
import pandas as pd
from numpy import *

# 1. 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stopwords

# 2. 对句子进行分词
def wordCut(sentence):
    words = jieba.cut(sentence.strip())
    stopwords = stopwordslist('C:\\Users\\liang\\Desktop\\机器学习\\朴素贝叶斯分类器\\停用词表\\cn_stopwords.txt') # 这里加载停用词的路径
    outstr = []
    for word in words:
        if word not in stopwords:
            if word != '\t':
                outstr.append(word)
    return outstr

def DataHandle(filename,flag):
    out = []
    #lines = pd.read_table("C:\\Users\\John\\Desktop\\emotion Analysis\\goods.txt", header=None, encoding='utf-8', names=['评论'])
    lines = pd.read_table(filename,header=None,encoding='utf-8',names=['评论'])
    for line in lines['评论']:
        line = str(line)
        outStr = wordCut(line) # 这里的返回值是字符串
        out.append(outStr)

    if flag:
        Vec = [1] * lines.shape[0]
    else:
        Vec = [0] * lines.shape[0]

    return Vec, out

if __name__ == '__main__':
    goodDataPath = 'C:\\Users\\liang\\Desktop\\机器学习\\朴素贝叶斯分类器\\good.txt'
    badDataPath = 'C:\\Users\\liang\\Desktop\\机器学习\\朴素贝叶斯分类器\\bad.txt'

    # 1 好评    0 差评
    goodVec, goodList = DataHandle(goodDataPath, 1)
    badVec, badList = DataHandle(badDataPath, 0)

    listClasses = goodVec + badVec
    listOPosts = goodList + badList
    print(listClasses)
    print(listOPosts)

    myVocabList = Bayes.createVocabList(listOPosts) # 构造词表，获取训练集中所有不重复的词语构成列表
    print(myVocabList)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(Bayes.setOfWords2Vec(myVocabList, postinDoc))
    # 4. 训练数据
    p0V, p1V, pAb = Bayes.trainNB0(array(trainMat), array(listClasses))
    # 5. 测试数据
    while True:
        inputS = input(u'请输入您对本商品的评价：')

        testEntry = wordCut(inputS)
        thisDoc = array(Bayes.setOfWords2Vec(myVocabList, testEntry))
        print('评价: ', Bayes.classifyNB(thisDoc, p0V, p1V, pAb))
    
    
    
