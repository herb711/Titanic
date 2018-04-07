#!/usr/bin/python  
# encoding:utf-8  
"""  
@author: Bruce  
@contact: isunrise@foxmail.com  
@file: dtrain.py  
@time: 2017/4/3 $ {TIME}  
此决策树使用pandas中的dataFrame数据实现，要求dataFrame中指定一列为y标签。调用createtree方法训练。createtree有2个参数  
第一个参数中数据DataFrame，第二个参数指定dataFrame中y标签的列名为str类型。  
"""  
  
from math import log  
import pandas as pd  
import operator  
import numpy as np  
import matplotlib.pyplot as plt  
  
#计算信息熵  
def calcshannonent(dataset, re):  
    numentries = len(dataset)  
    #计算数据集每一分类数量  
    l = dataset.columns.tolist()  
    k = l.index('Survived')-1  
    s_k = pd.pivot_table(dataset, index=re, values=l[k], aggfunc=len)  
    #classlabel = set(dataset[re])  
    shannonent = 0.0  
  
    #每一分类的信息熵  
    for i in list(s_k):  
        prob = i / float(numentries)  
        shannonent += prob * log(prob, 2)  
    return shannonent  
  
#对给定的数据集的指定特征的分类值进行分类  
def splitdataset(dataset, axis, value):  
    retdataset = dataset[dataset[axis] == value]  
    del retdataset[axis]  
    return retdataset  
  
#选择最佳分裂特征：  
def chooseBestFeatureToSplit(dataset,re):  
    #分裂前的信息熵  
    baseEntroy = calcshannonent(dataset,re)  
  
    #信息增益及最佳分裂特征初始：  
    bestinfogain = 0.0  
    bestfeature = dataset.columns[1]  
  
    #对每一特征进行循环  
    for f in dataset.columns:  
        if f == 'Survived': continue  
  
        #获取当前特征的列表  
        featlist = dataset[f]  
        #确定有多少分裂值  
        uniqueVals = set(featlist)  
  
        #初始化当前特征信息熵为0  
        newEntrypoy = 0.0  
  
        #对每一分裂值计算信息熵  
        for value in uniqueVals:  
            #分裂后的数据集  
            subdataset = splitdataset(dataset, f ,value)  
            #计算分支的概率  
            prob = len(dataset[dataset[f]==value])/float(len(dataset))  
  
            #分裂后信息熵  
            newEntrypoy += prob*calcshannonent(subdataset, re)  
  
        # 计算信息增益  
        infogain =  newEntrypoy - baseEntroy  
  
        #如果信息增益最大，则替换原信息增益，并记录最佳分裂特征  
        if f != 'Survived':  
            if (infogain > bestinfogain):  
                bestinfogain = infogain  
                bestfeature = f  
  
    #返回最佳分裂特征号  
    return bestfeature  
  
#确定分枝的主分类  
def majority(labellist):  
    classcount = {}  
    #分类列表中的各分类进行投票  
    for vote in labellist:  
        if vote not in classcount.keys():  
            classcount[vote] =0  
        classcount[vote] += 1  
    #排序后选择最多票数的分类  
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)  
    #返回最多票数的分类的标签  
    return sortedclasscount[0][0]  
  
def createtree(dataset, result):  
    '有2个参数，第一个是dataFrame类型的数据，第个是字符串类型的y标签'  
    #如果数据集的分类只有一个，则返回此分类，无需子树生长  
    classlist = list(dataset[result].values)  
    if classlist.count(classlist[0]) == len(classlist):  
        return classlist[0]  
  
    #如果数据集仅有1列变量，加y标签有2列，则返回此数据集的分类  
    if len(dataset.columns) == 2:  
        return majority(classlist)  
  
    bestfeat = chooseBestFeatureToSplit(dataset,result)  
    mytree = {bestfeat: {}}  
  
    #此节点分裂为哪些分枝  
    uniquevals = set(dataset[bestfeat])  
  
    #对每一分枝，递归创建子树  
    for value in uniquevals:  
        mytree[bestfeat][value] = createtree(splitdataset(dataset, bestfeat,value),result)  
  
    #完成后，返回决策树  
    return mytree  